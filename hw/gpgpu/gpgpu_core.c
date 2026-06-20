/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/* ===================================================================
 * Field extraction macros for RISC-V 32-bit instructions
 * =================================================================== */
#define INST_OPCODE(inst)       ((inst) & 0x7F)
#define INST_RD(inst)           (((inst) >> 7) & 0x1F)
#define INST_RS1(inst)          (((inst) >> 15) & 0x1F)
#define INST_RS2(inst)          (((inst) >> 20) & 0x1F)
#define INST_FUNCT3(inst)       (((inst) >> 12) & 0x7)
#define INST_FUNCT7(inst)       (((inst) >> 25) & 0x7F)
#define INST_I_IMM(inst)        ((int32_t)(inst) >> 20)  /* sign-extended */
#define INST_S_IMM(inst)        (((int32_t)(inst) >> 25 << 5) | \
                                 (((inst) >> 7) & 0x1F))
#define INST_U_IMM(inst)        ((int32_t)((inst) & 0xFFFFF000))

/* Sign-extend a 12-bit immediate */
#define IMM_SEXT12(x)   ((int32_t)((x) & 0x800 ? (x) | 0xFFFFF000 : (x)))

/*
 * ===================================================================
 * RV32F opcode/funct3/funct7 constants
 * =================================================================== */
/* Standard */
#define OPCODE_OP_FP        0x53
#define FUNCT7_FADD_S       0x00
#define FUNCT7_FSUB_S       0x04
#define FUNCT7_FMUL_S       0x08
#define FUNCT7_FDIV_S       0x0C
#define FUNCT7_FSQRT_S      0x2C
#define FUNCT7_FSGNJ_S      0x10
#define FUNCT7_FMIN_MAX_S   0x14
#define FUNCT7_FCVT_W_S     0x60  /* float -> int */
#define FUNCT7_FCVT_S_W     0x68  /* int -> float */
#define FUNCT7_FMV_X_W      0x70
#define FUNCT7_FMV_W_X      0x78

/* Custom low-precision conversions (funct7 values from kernel encoding) */
#define FUNCT7_FCVT_BF16    0x22  /* BF16 conversions */
#define FUNCT7_FCVT_FP8     0x24  /* FP8 (E4M3 / E5M2) conversions */
#define FUNCT7_FCVT_FP4     0x26  /* FP4 (E2M1) conversions */

/* rs2 values for BF16 */
#define RS2_FCVT_S_BF16     0     /* fcvt.s.bf16  - BF16 -> f32 */
#define RS2_FCVT_BF16_S     1     /* fcvt.bf16.s  - f32 -> BF16 */

/* rs2 values for FP8 */
#define RS2_FCVT_S_E4M3     0     /* fcvt.s.e4m3   - E4M3 -> f32 */
#define RS2_FCVT_E4M3_S     1     /* fcvt.e4m3.s   - f32 -> E4M3 */
#define RS2_FCVT_S_E5M2     2     /* fcvt.s.e5m2   - E5M2 -> f32 */
#define RS2_FCVT_E5M2_S     3     /* fcvt.e5m2.s   - f32 -> E5M2 */

/* rs2 values for FP4 */
#define RS2_FCVT_S_E2M1     0     /* fcvt.s.e2m1   - E2M1 -> f32 */
#define RS2_FCVT_E2M1_S     1     /* fcvt.e2m1.s   - f32 -> E2M1 */

/* rs2 for rounding mode in fcvt.w.s / fcvt.s.w */
#define RS2_RTZ             1     /* round-toward-zero */

/* ------------------------------------------------------------------ */
/* Helper: round-to-nearest-even on a value with guard & sticky bits   */
/* Returns the rounded value; `*carry` is set if rounding overflowed.  */
/* ------------------------------------------------------------------ */
static inline uint32_t rne_round(uint32_t val, int guard, int sticky,
                                 int *carry)
{
    *carry = 0;
    if (!guard) {
        return val;                     /* no rounding needed */
    }
    if (sticky || (val & 1)) {
        /* round up */
        val++;
        if (val == 0) {
            *carry = 1;                 /* overflow */
        }
    }
    return val;
}

/* ------------------------------------------------------------------ */
/* Low-precision format helpers                                        */
/* ------------------------------------------------------------------ */

/*
 * E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa  (8 bits)
 * All exponent values 0..15 are used (no Inf/NaN reserved).
 */
#define E4M3_EXP_BIAS   7
#define E4M3_MAX_EXP    15

/*
 * E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa (8 bits)
 * Like IEEE half, but exponent 31 reserved for Inf/NaN.
 */
#define E5M2_EXP_BIAS   15
#define E5M2_INF_EXP    31

/*
 * E2M1: 1 sign, 2 exponent (bias 1), 1 mantissa (4 bits)
 * All exponent values 0..3 are used (no Inf/NaN reserved).
 * Stored in low 4 bits of uint8_t.
 */
#define E2M1_EXP_BIAS   1
#define E2M1_MAX_EXP    3

/* ---------- float32 -> bfloat16 (round-to-nearest-even) ---------- */
static uint16_t f32_to_bf16(float32 f)
{
    uint32_t bits = float32_val(f);
    uint16_t upper = (bits >> 16) & 0xFFFF;
    uint16_t lower = bits & 0xFFFF;

    /* Round-to-nearest-even on the lower 16 bits */
    int guard = (lower >> 15) & 1;
    int sticky = (lower & 0x7FFF) != 0;
    int carry;
    uint32_t rounded = rne_round(upper, guard, sticky, &carry);
    if (carry) {
        /* Increment exponent — handled by carry into upper bits */
        rounded = (upper + 1) & 0xFFFF;
    }
    return (uint16_t)rounded;
}

/* ---------- bfloat16 -> float32 ---------- */
static float32 bf16_to_f32(uint16_t bf16)
{
    return make_float32((uint32_t)bf16 << 16);
}

/* ---------- float32 -> E4M3 (FP8) ---------- */
static uint8_t float32_to_e4m3(float32 f)
{
    uint32_t bits = float32_val(f);
    int sign = (bits >> 31) & 1;
    int exp_f32 = (bits >> 23) & 0xFF;
    int mant_f32 = bits & 0x7FFFFF;

    /* Zero */
    if (exp_f32 == 0 && mant_f32 == 0) {
        return (uint8_t)(sign << 7);
    }

    /* Infinity / NaN -> saturate to max representable value */
    if (exp_f32 == 0xFF) {
        /* Max E4M3 positive: 0x7E = sign=0, exp=15, mant=6 -> 448 */
        /* For negative: 0xFE */
        return (uint8_t)((sign << 7) | 0x7E);
    }

    int exp_unbiased = exp_f32 - 127;
    int exp_e4m3 = exp_unbiased + E4M3_EXP_BIAS;

    /* Overflow -> saturate */
    if (exp_e4m3 >= E4M3_MAX_EXP) {
        return (uint8_t)((sign << 7) | 0x7E);
    }

    /* Subnormal (too small) -> flush to zero */
    if (exp_e4m3 < 0) {
        return (uint8_t)(sign << 7);
    }

    /* Extract top 3 mantissa bits and round */
    int mant_e4m3 = (mant_f32 >> 20) & 0x7;
    int guard = (mant_f32 >> 19) & 1;
    int sticky = (mant_f32 & 0x7FFFF) != 0;
    int carry;
    uint32_t mant_rnd = rne_round(mant_e4m3, guard, sticky, &carry);
    if (carry) {
        mant_rnd = 0;
        exp_e4m3++;
        if (exp_e4m3 > E4M3_MAX_EXP) {
            return (uint8_t)((sign << 7) | 0x7E);
        }
    }

    return (uint8_t)((sign << 7) | (exp_e4m3 << 3) | (uint8_t)mant_rnd);
}

/* ---------- E4M3 -> float32 ---------- */
static float32 e4m3_to_float32(uint8_t e4m3)
{
    int sign = (e4m3 >> 7) & 1;
    int exp_e4m3 = (e4m3 >> 3) & 0xF;
    int mant_e4m3 = e4m3 & 0x7;

    if (exp_e4m3 == 0) {
        if (mant_e4m3 == 0) {
            return make_float32((uint32_t)sign << 31);
        }
        /* Subnormal */
        int exp_f32 = 127 - E4M3_EXP_BIAS + 1;  /* = 121 */
        uint32_t mant_f32 = (uint32_t)mant_e4m3 << (23 - 3);
        return make_float32(((uint32_t)sign << 31) |
                            ((uint32_t)exp_f32 << 23) | mant_f32);
    }

    int exp_unbiased = exp_e4m3 - E4M3_EXP_BIAS;
    int exp_f32 = exp_unbiased + 127;
    uint32_t mant_f32 = (uint32_t)mant_e4m3 << (23 - 3);

    return make_float32(((uint32_t)sign << 31) |
                        ((uint32_t)exp_f32 << 23) | mant_f32);
}

/* ---------- float32 -> E5M2 (FP8) ---------- */
static uint8_t float32_to_e5m2(float32 f)
{
    uint32_t bits = float32_val(f);
    int sign = (bits >> 31) & 1;
    int exp_f32 = (bits >> 23) & 0xFF;
    int mant_f32 = bits & 0x7FFFFF;

    if (exp_f32 == 0 && mant_f32 == 0) {
        return (uint8_t)(sign << 7);
    }

    /* Inf/NaN */
    if (exp_f32 == 0xFF) {
        /* Preserve NaN vs Inf based on mantissa */
        if (mant_f32 == 0) {
            /* Inf */
            return (uint8_t)((sign << 7) | (E5M2_INF_EXP << 2));
        }
        /* NaN: propagate mantissa top 2 bits if non-zero */
        uint8_t mant_e5m2 = (mant_f32 >> 21) & 0x3;
        if (mant_e5m2 == 0) mant_e5m2 = 1; /* ensure non-zero */
        return (uint8_t)((sign << 7) | (E5M2_INF_EXP << 2) | mant_e5m2);
    }

    /* Handle subnormal float32 inputs */
    if (exp_f32 == 0) {
        /* Subnormal float32, this is very small */
        /* Normalize it */
        int shift = 0;
        uint32_t m = mant_f32;
        while ((m & 0x800000) == 0 && shift < 23) {
            m <<= 1;
            shift++;
        }
        m &= 0x7FFFFF;
        exp_f32 = 1 - shift;
        mant_f32 = m;
        /* After normalization it's a normal value */
    }

    int exp_unbiased = exp_f32 - 127;
    int exp_e5m2 = exp_unbiased + E5M2_EXP_BIAS;

    /* Overflow -> saturate to Inf */
    if (exp_e5m2 >= E5M2_INF_EXP) {
        return (uint8_t)((sign << 7) | (E5M2_INF_EXP << 2));
    }

    /* Too small -> flush to zero */
    if (exp_e5m2 < 0) {
        return (uint8_t)(sign << 7);
    }

    /* Extract top 2 mantissa bits and round */
    int mant_e5m2 = (mant_f32 >> 21) & 0x3;
    int guard = (mant_f32 >> 20) & 1;
    int sticky = (mant_f32 & 0xFFFFF) != 0;
    int carry;
    uint32_t mant_rnd = rne_round(mant_e5m2, guard, sticky, &carry);
    if (carry) {
        mant_rnd = 0;
        exp_e5m2++;
        if (exp_e5m2 >= E5M2_INF_EXP) {
            return (uint8_t)((sign << 7) | (E5M2_INF_EXP << 2));
        }
    }

    return (uint8_t)((sign << 7) | (exp_e5m2 << 2) | (uint8_t)mant_rnd);
}

/* ---------- E5M2 -> float32 ---------- */
static float32 e5m2_to_float32(uint8_t e5m2)
{
    int sign = (e5m2 >> 7) & 1;
    int exp_e5m2 = (e5m2 >> 2) & 0x1F;
    int mant_e5m2 = e5m2 & 0x3;

    if (exp_e5m2 == 0) {
        if (mant_e5m2 == 0) {
            return make_float32((uint32_t)sign << 31);
        }
        /* Subnormal */
        int exp_f32 = 127 - E5M2_EXP_BIAS + 1;
        uint32_t mant_f32 = (uint32_t)mant_e5m2 << (23 - 2);
        return make_float32(((uint32_t)sign << 31) |
                            ((uint32_t)exp_f32 << 23) | mant_f32);
    }

    if (exp_e5m2 == E5M2_INF_EXP) {
        if (mant_e5m2 == 0) {
            /* Inf */
            return make_float32(((uint32_t)sign << 31) | 0x7F800000);
        }
        /* NaN */
        return make_float32(((uint32_t)sign << 31) | 0x7FC00000 |
                            ((uint32_t)mant_e5m2 << 21));
    }

    int exp_unbiased = exp_e5m2 - E5M2_EXP_BIAS;
    int exp_f32 = exp_unbiased + 127;
    uint32_t mant_f32 = (uint32_t)mant_e5m2 << (23 - 2);

    return make_float32(((uint32_t)sign << 31) |
                        ((uint32_t)exp_f32 << 23) | mant_f32);
}

/*
 * ---------- float32 -> E2M1 (FP4) ----------
 * Hand-written conversion (not using softfloat).
 * E2M1: 1 sign, 2 exponent (bias 1), 1 mantissa (4 bits total)
 */
static uint8_t float32_to_e2m1(float32 f)
{
    uint32_t bits = float32_val(f);
    int sign = (bits >> 31) & 1;
    int exp_f32 = (bits >> 23) & 0xFF;
    int mant_f32 = bits & 0x7FFFFF;

    /* Zero */
    if (exp_f32 == 0 && mant_f32 == 0) {
        return (uint8_t)(sign << 3);
    }

    /* Infinity / NaN -> saturate to max */
    if (exp_f32 == 0xFF) {
        return (uint8_t)((sign << 3) | 0x7);
    }

    int exp_unbiased = exp_f32 - 127;
    int exp_e2m1 = exp_unbiased + E2M1_EXP_BIAS;

    /* Overflow -> saturate to max */
    if (exp_e2m1 >= E2M1_MAX_EXP) {
        return (uint8_t)((sign << 3) | 0x7);
    }

    /* Too small -> flush to zero */
    if (exp_e2m1 < 0) {
        return (uint8_t)(sign << 3);
    }

    /* Extract top 1 mantissa bit */
    int mant_e2m1 = (mant_f32 >> 22) & 1;
    int guard = (mant_f32 >> 21) & 1;
    int sticky = (mant_f32 & 0x1FFFFF) != 0;

    if (guard && (sticky || mant_e2m1)) {
        mant_e2m1++;
        if (mant_e2m1 >= 2) {
            mant_e2m1 = 0;
            exp_e2m1++;
            if (exp_e2m1 > E2M1_MAX_EXP) {
                return (uint8_t)((sign << 3) | 0x7);
            }
        }
    }

    return (uint8_t)((sign << 3) | (exp_e2m1 << 1) | mant_e2m1);
}

/* ---------- E2M1 -> float32 ---------- */
static float32 e2m1_to_float32(uint8_t e2m1)
{
    int sign = (e2m1 >> 3) & 1;
    int exp_e2m1 = (e2m1 >> 1) & 0x3;
    int mant_e2m1 = e2m1 & 0x1;

    if (exp_e2m1 == 0) {
        if (mant_e2m1 == 0) {
            return make_float32((uint32_t)sign << 31);
        }
        /* Subnormal */
        int exp_f32 = 127 - E2M1_EXP_BIAS + 1;
        uint32_t mant_f32 = (uint32_t)mant_e2m1 << (23 - 1);
        return make_float32(((uint32_t)sign << 31) |
                            ((uint32_t)exp_f32 << 23) | mant_f32);
    }

    int exp_unbiased = exp_e2m1 - E2M1_EXP_BIAS;
    int exp_f32 = exp_unbiased + 127;
    uint32_t mant_f32 = (uint32_t)mant_e2m1 << (23 - 1);

    return make_float32(((uint32_t)sign << 31) |
                        ((uint32_t)exp_f32 << 23) | mant_f32);
}

/* ===================================================================
 * Warp initialization
 * =================================================================== */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));
    warp->warp_id = warp_id;
    warp->thread_id_base = thread_id_base;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    for (uint32_t lane = 0; lane < num_threads && lane < GPGPU_WARP_SIZE;
         lane++) {
        GPGPULane *ln = &warp->lanes[lane];
        ln->gpr[0] = 0; /* x0 is always zero */
        ln->pc = pc;
        ln->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, lane);
        ln->active = true;
        ln->fp_status.float_rounding_mode = float_round_nearest_even;
        ln->fp_status.flush_to_zero = false;
    }

    /* Active mask: first num_threads lanes active */
    if (num_threads >= GPGPU_WARP_SIZE) {
        warp->active_mask = 0xFFFFFFFF;
    } else {
        warp->active_mask = (1u << num_threads) - 1;
    }
}

/* ===================================================================
 * Single-instruction execution for one lane
 * Returns 0 on success, -1 on ebreak, -2 on error
 * =================================================================== */
static int exec_one_inst(GPGPUState *s, GPGPULane *lane, uint32_t inst)
{
    uint32_t opcode = INST_OPCODE(inst);
    uint32_t rd = INST_RD(inst);
    uint32_t rs1 = INST_RS1(inst);
    uint32_t rs2 = INST_RS2(inst);
    uint32_t funct3 = INST_FUNCT3(inst);
    uint32_t funct7 = INST_FUNCT7(inst);

    switch (opcode) {
    /* ==================== LUI ==================== */
    case 0x37: /* LUI */
        if (rd != 0) {
            lane->gpr[rd] = INST_U_IMM(inst);
        }
        lane->pc += 4;
        break;

    /* ==================== AUIPC ==================== */
    case 0x17: /* AUIPC */
        if (rd != 0) {
            lane->gpr[rd] = lane->pc + INST_U_IMM(inst);
        }
        lane->pc += 4;
        break;

    /* ==================== JAL ==================== */
    case 0x6F: {
        /* JAL: rd = pc+4, pc += offset */
        int32_t imm = 0;
        imm |= ((inst >> 21) & 0x3FF) << 1;   /* imm[10:1] */
        imm |= ((inst >> 20) & 1) << 11;        /* imm[11] */
        imm |= ((inst >> 12) & 0xFF) << 12;     /* imm[19:12] */
        imm |= (inst & 0x80000000) ? 0xFFF00000 : 0; /* sign extend */
        if (rd != 0) {
            lane->gpr[rd] = lane->pc + 4;
        }
        lane->pc += imm;
        break;
    }

    /* ==================== JALR ==================== */
    case 0x67: /* JALR */
        if (rd != 0) {
            lane->gpr[rd] = lane->pc + 4;
        }
        lane->pc = (lane->gpr[rs1] + INST_I_IMM(inst)) & ~1;
        break;

    /* ==================== BRANCH ==================== */
    case 0x63: {
        int32_t imm = 0;
        imm |= ((inst >> 8) & 0xF) << 1;        /* imm[4:1] */
        imm |= ((inst >> 25) & 0x3F) << 5;       /* imm[10:5] */
        imm |= ((inst >> 7) & 1) << 11;           /* imm[11] */
        imm |= (inst & 0x80000000) ? 0xFFFFF000 : 0;
        bool taken = false;
        switch (funct3) {
        case 0: taken = (lane->gpr[rs1] == lane->gpr[rs2]); break;      /* BEQ */
        case 1: taken = (lane->gpr[rs1] != lane->gpr[rs2]); break;      /* BNE */
        case 4: taken = ((int32_t)lane->gpr[rs1] < (int32_t)lane->gpr[rs2]); break; /* BLT */
        case 5: taken = ((int32_t)lane->gpr[rs1] >= (int32_t)lane->gpr[rs2]); break; /* BGE */
        case 6: taken = (lane->gpr[rs1] < lane->gpr[rs2]); break;       /* BLTU */
        case 7: taken = (lane->gpr[rs1] >= lane->gpr[rs2]); break;       /* BGEU */
        }
        if (taken) {
            lane->pc += imm;
        } else {
            lane->pc += 4;
        }
        break;
    }

    /* ==================== LOAD ==================== */
    case 0x03: {
        uint32_t addr = lane->gpr[rs1] + INST_I_IMM(inst);
        uint32_t data = 0;
        if (addr + 4 <= s->vram_size) {
            data = ldl_le_p(s->vram_ptr + addr);
        }
        switch (funct3) {
        case 0: /* LB */
            data = (int32_t)(int8_t)(data & 0xFF);
            break;
        case 1: /* LH */
            data = (int32_t)(int16_t)(data & 0xFFFF);
            break;
        case 2: /* LW */
            break;
        case 3: /* LD (not in RV32I but for completeness) */
            break;
        case 4: /* LBU */
            data &= 0xFF;
            break;
        case 5: /* LHU */
            data &= 0xFFFF;
            break;
        }
        if (rd != 0) {
            lane->gpr[rd] = data;
        }
        lane->pc += 4;
        break;
    }

    /* ==================== STORE ==================== */
    case 0x23: {
        uint32_t addr = lane->gpr[rs1] + IMM_SEXT12(INST_S_IMM(inst));
        uint32_t data = lane->gpr[rs2];
        if (addr + 4 <= s->vram_size) {
            switch (funct3) {
            case 0: /* SB */
                stb_p(s->vram_ptr + addr, data & 0xFF);
                break;
            case 1: /* SH */
                stw_le_p(s->vram_ptr + addr, data & 0xFFFF);
                break;
            case 2: /* SW */
                stl_le_p(s->vram_ptr + addr, data);
                break;
            }
        }
        lane->pc += 4;
        break;
    }

    /* ==================== OP-IMM (integer immediate) ==================== */
    case 0x13: {
        uint32_t imm = INST_I_IMM(inst);
        uint32_t src = lane->gpr[rs1];
        uint32_t res = 0;
        switch (funct3) {
        case 0: res = src + imm; break;                              /* ADDI */
        case 1: /* SLLI */
            res = src << (imm & 0x1F);
            break;
        case 2: res = (int32_t)src < (int32_t)imm; break;           /* SLTI */
        case 3: res = src < imm; break;                              /* SLTIU */
        case 4: res = src ^ imm; break;                              /* XORI */
        case 5: /* SRLI / SRAI */
            if (imm & 0x400) {
                res = (uint32_t)((int32_t)src >> (imm & 0x1F));      /* SRAI */
            } else {
                res = src >> (imm & 0x1F);                           /* SRLI */
            }
            break;
        case 6: res = src | imm; break;                              /* ORI */
        case 7: res = src & imm; break;                              /* ANDI */
        }
        if (rd != 0) {
            lane->gpr[rd] = res;
        }
        lane->pc += 4;
        break;
    }

    /* ==================== OP (integer register) ==================== */
    case 0x33: {
        uint32_t src1 = lane->gpr[rs1];
        uint32_t src2 = lane->gpr[rs2];
        uint32_t res = 0;
        switch (funct3) {
        case 0: /* ADD / SUB / MUL */
            if (funct7 == 0x00) {
                res = src1 + src2;                                    /* ADD */
            } else if (funct7 == 0x20) {
                res = src1 - src2;                                    /* SUB */
            } else if (funct7 == 0x01) {
                res = src1 * src2;                                    /* MUL */
            }
            break;
        case 1: /* SLL */
            res = src1 << (src2 & 0x1F);
            break;
        case 2: /* SLT */
            res = ((int32_t)src1 < (int32_t)src2);
            break;
        case 3: /* SLTU */
            res = (src1 < src2);
            break;
        case 4: /* XOR */
            res = src1 ^ src2;
            break;
        case 5: /* SRL / SRA */
            if (funct7 == 0x00) {
                res = src1 >> (src2 & 0x1F);                          /* SRL */
            } else if (funct7 == 0x20) {
                res = (uint32_t)((int32_t)src1 >> (src2 & 0x1F));     /* SRA */
            }
            break;
        case 6: /* OR */
            res = src1 | src2;
            break;
        case 7: /* AND */
            res = src1 & src2;
            break;
        }
        if (rd != 0) {
            lane->gpr[rd] = res;
        }
        lane->pc += 4;
        break;
    }

    /* ==================== FENCE / FENCE.I ==================== */
    case 0x0F:
        lane->pc += 4;
        break;

    /* ==================== SYSTEM (CSR*, EBREAK, ECALL) ==================== */
    case 0x73: {
        uint32_t csr = (inst >> 20); /* csr address */
        uint32_t uimm = rs1;         /* for CSR ops with rs1=0, uimm */
        uint32_t src = lane->gpr[rs1];

        switch (funct3) {
        case 0: /* ECALL / EBREAK */
            if ((inst >> 20) == 0 && rs1 == 0) {
                /* ECALL */
                lane->pc += 4;
            } else if ((inst >> 20) == 1 && rs1 == 0) {
                /* EBREAK - stop execution */
                lane->pc += 4;
                return -1;
            } else {
                lane->pc += 4;
            }
            break;

        case 1: /* CSRRW */
            if (rd != 0) {
                lane->gpr[rd] = lane->gpr[rs1]; /* simplified, no real CSR */
            }
            lane->pc += 4;
            break;

        case 2: /* CSRRS */
            {
                (void)src;
                uint32_t csr_val = 0;
                if (csr == CSR_MHARTID) {
                    csr_val = lane->mhartid;
                } else if (csr == CSR_FCSR) {
                    csr_val = lane->fcsr;
                } else if (csr == CSR_FFLAGS) {
                    csr_val = lane->fcsr & 0x1F;
                } else if (csr == CSR_FRM) {
                    csr_val = (lane->fcsr >> 5) & 0x7;
                }
                if (rd != 0) {
                    lane->gpr[rd] = csr_val;
                }
                /* CSRRS: set bits where rs1 has 1 */
                /* We don't implement real CSR writes for mhartid */
            }
            lane->pc += 4;
            break;

        case 3: /* CSRRC */
            if (rd != 0) {
                lane->gpr[rd] = 0;
            }
            lane->pc += 4;
            break;

        case 5: /* CSRRWI */
            if (rd != 0) {
                lane->gpr[rd] = uimm;
            }
            lane->pc += 4;
            break;

        case 6: /* CSRRSI */
            if (rd != 0) {
                lane->gpr[rd] = 0;
            }
            lane->pc += 4;
            break;

        default:
            lane->pc += 4;
            break;
        }
        break;
    }

    /* ==================== OP-FP (RV32F + custom conversions) ==================== */
    case OPCODE_OP_FP: {
        uint32_t rs1_f = rs1;
        uint32_t rs2_f = rs2;
        float32 res_f = 0;

        switch (funct7) {
        /* ---- RV32F standard ---- */
        case FUNCT7_FADD_S:     /* fadd.s */
            res_f = float32_add(lane->fpr[rs1_f], lane->fpr[rs2_f],
                                &lane->fp_status);
            if (rd != 0) lane->fpr[rd] = res_f;
            break;

        case FUNCT7_FSUB_S:     /* fsub.s */
            res_f = float32_sub(lane->fpr[rs1_f], lane->fpr[rs2_f],
                                &lane->fp_status);
            if (rd != 0) lane->fpr[rd] = res_f;
            break;

        case FUNCT7_FMUL_S:     /* fmul.s */
            res_f = float32_mul(lane->fpr[rs1_f], lane->fpr[rs2_f],
                                &lane->fp_status);
            if (rd != 0) lane->fpr[rd] = res_f;
            break;

        case FUNCT7_FDIV_S:     /* fdiv.s */
            res_f = float32_div(lane->fpr[rs1_f], lane->fpr[rs2_f],
                                &lane->fp_status);
            if (rd != 0) lane->fpr[rd] = res_f;
            break;

        case FUNCT7_FSQRT_S:    /* fsqrt.s */
            res_f = float32_sqrt(lane->fpr[rs1_f], &lane->fp_status);
            if (rd != 0) lane->fpr[rd] = res_f;
            break;

        case FUNCT7_FSGNJ_S:    /* fsgnj / fsgnjn / fsgnjx */
            {
                float32 a = lane->fpr[rs1_f];
                float32 b = lane->fpr[rs2_f];
                switch (funct3) {
                case 0: /* fsgnj: copy sign of b */
                    res_f = make_float32((float32_val(a) & 0x7FFFFFFF) |
                                         (float32_val(b) & 0x80000000));
                    break;
                case 1: /* fsgnjn: negate sign of b */
                    res_f = make_float32((float32_val(a) & 0x7FFFFFFF) |
                                         ((~float32_val(b)) & 0x80000000));
                    break;
                case 2: /* fsgnjx: xor signs */
                    res_f = make_float32(float32_val(a) ^
                                         (float32_val(b) & 0x80000000));
                    break;
                }
                if (rd != 0) lane->fpr[rd] = res_f;
            }
            break;

        case FUNCT7_FMIN_MAX_S: /* fmin.s / fmax.s */
            if (funct3 == 0) {
                res_f = float32_min(lane->fpr[rs1_f], lane->fpr[rs2_f],
                                    &lane->fp_status);
            } else {
                res_f = float32_max(lane->fpr[rs1_f], lane->fpr[rs2_f],
                                    &lane->fp_status);
            }
            if (rd != 0) lane->fpr[rd] = res_f;
            break;

        case FUNCT7_FCVT_W_S:   /* fcvt.w.s / fcvt.wu.s */
            {
                FloatRoundMode rm = rs2_f & 0x7;
                uint32_t rs1_int = rs1_f;
                if (funct3 == 0) {
                    /* fcvt.w.s */
                    if (rm == RS2_RTZ) {
                        if (rd != 0) {
                            lane->gpr[rd] = (uint32_t)float32_to_int32_round_to_zero(
                                lane->fpr[rs1_int], &lane->fp_status);
                        }
                    } else {
                        lane->fp_status.float_rounding_mode = rm;
                        if (rd != 0) {
                            lane->gpr[rd] = (uint32_t)float32_to_int32(
                                lane->fpr[rs1_int], &lane->fp_status);
                        }
                        lane->fp_status.float_rounding_mode = float_round_nearest_even;
                    }
                } else {
                    /* fcvt.wu.s */
                    if (rm == RS2_RTZ) {
                        if (rd != 0) {
                            lane->gpr[rd] = float32_to_uint32_round_to_zero(
                                lane->fpr[rs1_int], &lane->fp_status);
                        }
                    } else {
                        lane->fp_status.float_rounding_mode = rm;
                        if (rd != 0) {
                            lane->gpr[rd] = float32_to_uint32(
                                lane->fpr[rs1_int], &lane->fp_status);
                        }
                        lane->fp_status.float_rounding_mode = float_round_nearest_even;
                    }
                }
            }
            break;

        case FUNCT7_FCVT_S_W:   /* fcvt.s.w / fcvt.s.wu */
            {
                if (funct3 == 0) {
                    /* fcvt.s.w */
                    res_f = int32_to_float32((int32_t)lane->gpr[rs1_f],
                                              &lane->fp_status);
                } else {
                    /* fcvt.s.wu */
                    res_f = uint32_to_float32(lane->gpr[rs1_f],
                                               &lane->fp_status);
                }
                if (rd != 0) lane->fpr[rd] = res_f;
            }
            break;

        case FUNCT7_FMV_X_W:    /* fmv.x.w */
            if (rd != 0) {
                lane->gpr[rd] = float32_val(lane->fpr[rs1_f]);
            }
            break;

        case FUNCT7_FMV_W_X:    /* fmv.w.x */
            if (rd != 0) {
                lane->fpr[rd] = make_float32(lane->gpr[rs1_f]);
            }
            break;

        /* ---- Custom BF16 conversions ---- */
        case FUNCT7_FCVT_BF16:
            if (rs2_f == RS2_FCVT_S_BF16) {
                /* fcvt.s.bf16: BF16 -> float32 (rs1 = f reg with BF16 in low 16) */
                if (rd != 0) {
                    lane->fpr[rd] = bf16_to_f32(
                        (uint16_t)(float32_val(lane->fpr[rs1_f]) & 0xFFFF));
                }
            } else if (rs2_f == RS2_FCVT_BF16_S) {
                /* fcvt.bf16.s: float32 -> BF16 (rd = f reg, BF16 in low 16) */
                if (rd != 0) {
                    uint16_t bf16 = f32_to_bf16(lane->fpr[rs1_f]);
                    lane->fpr[rd] = make_float32(bf16);
                }
            }
            break;

        /* ---- Custom FP8 conversions (E4M3 / E5M2) ---- */
        case FUNCT7_FCVT_FP8:
            if (rs2_f == RS2_FCVT_S_E4M3) {
                /* fcvt.s.e4m3: E4M3 (low 8 bits) -> float32 */
                if (rd != 0) {
                    uint8_t e4m3 = float32_val(lane->fpr[rs1_f]) & 0xFF;
                    lane->fpr[rd] = e4m3_to_float32(e4m3);
                }
            } else if (rs2_f == RS2_FCVT_E4M3_S) {
                /* fcvt.e4m3.s: float32 -> E4M3 */
                if (rd != 0) {
                    uint8_t e4m3 = float32_to_e4m3(lane->fpr[rs1_f]);
                    lane->fpr[rd] = make_float32(e4m3);
                }
            } else if (rs2_f == RS2_FCVT_S_E5M2) {
                /* fcvt.s.e5m2: E5M2 (low 8 bits) -> float32 */
                if (rd != 0) {
                    uint8_t e5m2 = float32_val(lane->fpr[rs1_f]) & 0xFF;
                    lane->fpr[rd] = e5m2_to_float32(e5m2);
                }
            } else if (rs2_f == RS2_FCVT_E5M2_S) {
                /* fcvt.e5m2.s: float32 -> E5M2 */
                if (rd != 0) {
                    uint8_t e5m2 = float32_to_e5m2(lane->fpr[rs1_f]);
                    lane->fpr[rd] = make_float32(e5m2);
                }
            }
            break;

        /* ---- Custom FP4 conversions (E2M1) ---- */
        case FUNCT7_FCVT_FP4:
            if (rs2_f == RS2_FCVT_S_E2M1) {
                /* fcvt.s.e2m1: E2M1 (low 4 bits) -> float32 */
                if (rd != 0) {
                    uint8_t e2m1 = float32_val(lane->fpr[rs1_f]) & 0xF;
                    lane->fpr[rd] = e2m1_to_float32(e2m1);
                }
            } else if (rs2_f == RS2_FCVT_E2M1_S) {
                /* fcvt.e2m1.s: float32 -> E2M1 */
                if (rd != 0) {
                    uint8_t e2m1 = float32_to_e2m1(lane->fpr[rs1_f]);
                    lane->fpr[rd] = make_float32(e2m1);
                }
            }
            break;

        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu_core: unknown OP-FP funct7=0x%02x\n",
                          funct7);
            lane->pc += 4;
            break;
        }
        lane->pc += 4;
        break;
    }

    /* ==================== LOAD-FP / STORE-FP ==================== */
    case 0x07: /* FLW */
        {
            uint32_t addr = lane->gpr[rs1] + INST_I_IMM(inst);
            if (addr + 4 <= s->vram_size) {
                float32 val = make_float32(ldl_le_p(s->vram_ptr + addr));
                if (rd != 0) {
                    lane->fpr[rd] = val;
                }
            }
            lane->pc += 4;
        }
        break;

    case 0x27: /* FSW */
        {
            uint32_t addr = lane->gpr[rs1] + IMM_SEXT12(INST_S_IMM(inst));
            if (addr + 4 <= s->vram_size) {
                stl_le_p(s->vram_ptr + addr,
                         float32_val(lane->fpr[rs2]));
            }
            lane->pc += 4;
        }
        break;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu_core: unknown opcode 0x%02x at pc=0x%x\n",
                      opcode, lane->pc);
        lane->pc += 4;
        return -2;
    }

    return 0;
}

/* ===================================================================
 * Warp execution: execute all active lanes of a warp until ebreak
 * =================================================================== */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;

    while (cycles < max_cycles) {
        /* Fetch instruction from VRAM at warp's first active lane's PC */
        uint32_t pc = 0;
        bool first = true;
        for (uint32_t lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
            if (warp->active_mask & (1u << lane)) {
                if (first) {
                    pc = warp->lanes[lane].pc;
                    first = false;
                }
            }
        }

        if (first) {
            /* No active lanes */
            break;
        }

        /* Fetch instruction */
        uint32_t inst = 0;
        if (pc + 4 <= s->vram_size) {
            inst = ldl_le_p(s->vram_ptr + pc);
        }

        /* Execute for each active lane */
        bool all_done = true;
        for (uint32_t lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
            if (warp->active_mask & (1u << lane)) {
                GPGPULane *ln = &warp->lanes[lane];
                ln->pc = pc; /* synchronize PC */
                int ret = exec_one_inst(s, ln, inst);
                if (ret == -1) {
                    /* ebreak: mark lane inactive */
                    warp->active_mask &= ~(1u << lane);
                    ln->active = false;
                } else {
                    all_done = false;
                }
            }
        }

        if (all_done) {
            break;
        }

        cycles++;
    }

    return 0;
}

/* ===================================================================
 * Kernel execution: dispatch all blocks/warps
 * =================================================================== */
int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t grid_x = s->kernel.grid_dim[0];
    uint32_t grid_y = s->kernel.grid_dim[1];
    uint32_t grid_z = s->kernel.grid_dim[2];
    uint32_t block_x = s->kernel.block_dim[0];
    uint32_t block_y = s->kernel.block_dim[1];
    uint32_t block_z = s->kernel.block_dim[2];
    uint64_t kernel_addr = s->kernel.kernel_addr;

    if (grid_x == 0) grid_x = 1;
    if (grid_y == 0) grid_y = 1;
    if (grid_z == 0) grid_z = 1;
    if (block_x == 0) block_x = 1;
    if (block_y == 0) block_y = 1;
    if (block_z == 0) block_z = 1;

    uint32_t total_threads = block_x * block_y * block_z;
    uint32_t num_warps = (total_threads + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;

    /* Iterate over all blocks */
    for (uint32_t bz = 0; bz < grid_z; bz++) {
        for (uint32_t by = 0; by < grid_y; by++) {
            for (uint32_t bx = 0; bx < grid_x; bx++) {
                uint32_t block_id[3] = { bx, by, bz };
                uint32_t block_id_linear = bz * grid_x * grid_y + by * grid_x + bx;

                /* Iterate over all warps in this block */
                for (uint32_t wi = 0; wi < num_warps; wi++) {
                    GPGPUWarp warp;
                    uint32_t thread_id_base = wi * GPGPU_WARP_SIZE;
                    uint32_t remaining = total_threads - thread_id_base;
                    uint32_t warp_threads = (remaining > GPGPU_WARP_SIZE) ?
                                            GPGPU_WARP_SIZE : remaining;

                    gpgpu_core_init_warp(&warp, (uint32_t)kernel_addr,
                                         thread_id_base, block_id,
                                         warp_threads, wi, block_id_linear);

                    gpgpu_core_exec_warp(s, &warp, 10000);
                }
            }
        }
    }

    return 0;
}
