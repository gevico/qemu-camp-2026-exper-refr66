# GPGPU 实验调试思路笔记

## 概述

GPGPU 方向共 17 个子测试，覆盖 PCI 设备建模、MMIO 寄存器、VRAM 访问、SIMT 上下文、RV32I 指令解释器、RV32F 浮点指令、低精度浮点转换（BF16/FP8/FP4）等功能。

调试过程中遇到一个隐蔽的指令编码 bug，本文记录完整的分析过程。

---

## 现象

运行全部 17 个测试，16 个通过，1 个失败：

```
PASS: device-id
PASS: vram-size
...
PASS: lp-convert
FAIL: lp-convert-e5m2-e2m1  ← 唯一失败
PASS: lp-convert-saturate
```

失败信息：

```
ERROR:../tests/qtest/gpgpu-test.c:857:
  gpgpu_test_lp_convert_e5m2_e2m1: assertion failed
  ((int32_t)val == expected[i]): (0 == -3)
```

测试的第 3 个断言失败。测试内核期望：
- output[0] = e5m2 round-trip(4) → 4
- output[1] = e2m1 round-trip(2) → 2
- output[2] = bf16 round-trip(-3) → -3  ← 实际得到 0
- output[3] = e4m3 round-trip(-2) → -2

特点：**只有负数转换失败，正数全部通过**。

---

## 第一轮排查：软硬件边界划分

### 可能的原因

1. **BF16 转换函数有 bug**：`f32_to_bf16()` 或 `bf16_to_f32()` 对负数处理不正确
2. **fcvt.w.s 指令解释器有 bug**：`exec_one_inst()` 中浮点转整数对负数处理不对
3. **测试内核指令编码有 bug**：写入 VRAM 的二进制指令本身就不正确

### 排除法

先看「为什么正数能通过」：

```
42 → fcvt.s.w → f1=42.0 → fcvt.bf16.s → f2=bf16(42) → fcvt.s.bf16 → f3=42.0 → fcvt.w.s → x10=42
```

这条链路正数能拿到正确结果，说明：
- `f32_to_bf16()` / `bf16_to_f32()` 工作正常
- `fcvt.s.w` / `fcvt.s.bf16` / `fcvt.bf16.s` 指令解释正常
- `fcvt.w.s` 对正整数的转换正常

如果这些函数对负数也正常工作（BF16(-3.0) 的位模式是 `0xC040`，转换回来应该是 -3.0），那问题很可能出在**指令编码**而不是软硬件实现。

---

## 第二轮排查：指令编码逐位解码

### 定位可疑指令

失败位置是 output[2]，对应内核中 `fcvt.w.s x12, f3, rtz`。测试文件中的编码是：

```c
0xC0019653,  /* fcvt.w.s x12, f3, rtz      ; x12 = -3 */
```

### 逐位解码

将 `0xC0019653` 展开为二进制：

```
1100 0000 0000 0001 1001 0110 0101 0011
```

按 RISC-V R-type 指令格式提取字段：

| 字段 | 位域 | 值 | 期望值 | 结论 |
|------|------|----|--------|------|
| opcode | [6:0] | `1010011` = 0x53 | 0x53 (OP-FP) | ✅ |
| rd | [11:7] | `01100` = 12 (x12) | 12 (x12) | ✅ |
| funct3 | [14:12] | `001` = 1 | 0 (fcvt.w.s) | ❌ **错误** |
| rs1 | [19:15] | `00011` = 3 (f3) | 3 (f3) | ✅ |
| rs2 | [24:20] | `00001` = 1 (RTZ) | 1 (RTZ) | ✅ |
| funct7 | [31:25] | `1100000` = 0x60 | 0x60 (FCVT_W_S) | ✅ |

**关键发现**：`funct3 = 1`，不是预期的 `funct3 = 0`。

### funct3 的含义

RISC-V 浮点转整数指令的 funct3 编码：

| funct3 | 指令 |
|--------|------|
| 000 | fcvt.w.s（有符号） |
| 001 | fcvt.wu.s（无符号） |

主模拟器代码中 `exec_one_inst()` 的处理：

```c
case FUNCT7_FCVT_W_S:
    if (funct3 == 0) {
        /* fcvt.w.s - signed conversion */
        lane->gpr[rd] = float32_to_int32_round_to_zero(...);
    } else {
        /* fcvt.wu.s - unsigned conversion */
        lane->gpr[rd] = float32_to_uint32(...);
        // 负数浮点转无符号整数 → QEMU softfloat 返回 0
    }
```

**真相**：编码 `0xC0019653` 的 funct3=1，被解码为 `fcvt.wu.s`（无符号转换）。对负数 -3.0 执行 `float32_to_uint32()`，QEMU 的 softfloat 库返回 0。

### 为什么其他测试通过了

检查其他 `fcvt.w.s` 指令编码：

```
fp_kernel:      0xC00293D3  → funct3 = 1, rs1 = 6 (f6), rs2 = 0
lp_convert:     0xC0019553  → funct3 = 1, rs1 = 1 (f1), rs2 = 0
                 0xC00195D3  → funct3 = 1, rs1 = 1 (f1), rs2 = 0
```

**所有** `fcvt.w.s` 编码都有 funct3=1 的问题！但因为：

1. **正数值**：`float32_to_uint32(42.0)` = 42，结果与 signed 一致
2. **寄存器巧合**：`rs1=1 (f1)` 读取的是 f1 而非注释中的 f3。但各段代码之间 f1 恰好保留了正确的值（每个 round-trip 段都会写 f1）

所以其他测试**凭巧合通过**，只有负数 -3 暴露了问题。

---

## 第三轮：正确编码推导

RISC-V `fcvt.w.s rd, rs1, rtz` 的正确编码：

```
funct7[31:25] = 1100000  (FCVT_W_S)
rs2[24:20]    = 00001    (RTZ rounding mode)
rs1[19:15]    = 00011    (f3)
funct3[14:12] = 000      (fcvt.w.s, signed)
rd[11:7]      = varies   (目标整数寄存器)
opcode[6:0]   = 1010011  (OP-FP)
```

以 `fcvt.w.s x12, f3, rtz`（rd=12）为例：

```
1100000 00001 00011 000 01100 1010011
= 1100 0000 0001 0001 1000 0110 0101 0011
= 0xC0118653
```

全部修正对照表：

| 目标 | 旧编码 `0xC00XXXXX` | 新编码 `0xC011XXXX` |
|------|-------------------|-------------------|
| x7, f5 | `0xC00293D3` | `0xC01283D3` |
| x10, f3 | `0xC0019553` | `0xC0118553` |
| x11, f3 | `0xC00195D3` | `0xC01185D3` |
| x12, f3 | `0xC0019653` | `0xC0118653` |
| x13, f3 | `0xC00196D3` | `0xC01186D3` |
| x14, f3 | `0xC0019753` | `0xC0118753` |

---

## 教训与最佳实践

### 1. 指令编码的「三重验证」

手写二进制指令时，永远要从三个角度交叉验证：

```python
# 1. 汇编器验证
# riscv64-unknown-elf-gcc -c -march=rv32if test.s && objdump -d test.o

# 2. 位域解码
funct3 = (inst >> 12) & 0x7
rs1    = (inst >> 15) & 0x1F
rs2    = (inst >> 20) & 0x1F

# 3. 执行跟踪
# 在 exec_one_inst 中打印每条指令的解码结果
```

### 2. 测试输入要覆盖边界

这个 bug 之所以存活，是因为所有测试数据都是**正数**。如果 lp-convert 或 fp-kernel-exec 中混入一个负数，早就暴露了。

### 3. 「巧合通过」的危险信号

多个测试都调用了同一个有 bug 的代码路径但都通过，说明测试之间存在**相关性**——它们测试的是同一组巧合条件，而不是真正的功能正确性。一个测试用例应该尽量：

- 覆盖不同符号（正、负、零）
- 覆盖边界值（最大、最小、溢出、Inf、NaN）
- 覆盖不同寄存器分配

### 4. 调试方法总结

```
观察现象 → 定位失败点 → 假设可能原因 → 排除法缩小范围
→ 检查边界条件（正负号！）→ 逐位解码二进制数据
→ 定位到 funct3 位 → 验证正确编码 → 修复并回归
```

---

## 最终结果

修复后所有 17/17 个 GPGPU 子测试通过，得分 100/100。

```
ok 1 - device-id
ok 1 - vram-size
ok 1 - global-ctrl
ok 1 - dispatch-regs
ok 1 - vram-access
ok 1 - dma-regs
ok 1 - irq-regs
ok 1 - simt-thread-id
ok 1 - simt-block-id
ok 1 - simt-warp-lane
ok 1 - simt-thread-mask
ok 1 - simt-reset
ok 1 - kernel-exec
ok 1 - fp-kernel-exec
ok 1 - lp-convert
ok 1 - lp-convert-e5m2-e2m1   ← 此前失败
ok 1 - lp-convert-saturate
```
