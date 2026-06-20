/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"
#include "gpgpu_core.h"

static void gpgpu_reset(DeviceState *dev);

/* MMIO control register read */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque;
    uint32_t ret = 0;

    switch (addr) {
    /* Device info (0x0000-0x00FF) */
    case GPGPU_REG_DEV_ID:
        ret = GPGPU_DEV_ID_VALUE;
        break;
    case GPGPU_REG_DEV_VERSION:
        ret = GPGPU_DEV_VERSION_VALUE;
        break;
    case GPGPU_REG_DEV_CAPS:
        ret = 0;
        break;
    case GPGPU_REG_VRAM_SIZE_LO:
        ret = (uint32_t)(s->vram_size & 0xFFFFFFFF);
        break;
    case GPGPU_REG_VRAM_SIZE_HI:
        ret = (uint32_t)((s->vram_size >> 32) & 0xFFFFFFFF);
        break;

    /* Global control (0x0100-0x01FF) */
    case GPGPU_REG_GLOBAL_CTRL:
        ret = s->global_ctrl;
        break;
    case GPGPU_REG_GLOBAL_STATUS:
        ret = s->global_status;
        break;
    case GPGPU_REG_ERROR_STATUS:
        ret = s->error_status;
        break;

    /* IRQ (0x0200-0x02FF) */
    case GPGPU_REG_IRQ_ENABLE:
        ret = s->irq_enable;
        break;
    case GPGPU_REG_IRQ_STATUS:
        ret = s->irq_status;
        break;
    case GPGPU_REG_IRQ_ACK:
        ret = 0;
        break;

    /* Kernel dispatch (0x0300-0x03FF) */
    case GPGPU_REG_KERNEL_ADDR_LO:
        ret = (uint32_t)(s->kernel.kernel_addr & 0xFFFFFFFF);
        break;
    case GPGPU_REG_KERNEL_ADDR_HI:
        ret = (uint32_t)((s->kernel.kernel_addr >> 32) & 0xFFFFFFFF);
        break;
    case GPGPU_REG_KERNEL_ARGS_LO:
        ret = (uint32_t)(s->kernel.kernel_args & 0xFFFFFFFF);
        break;
    case GPGPU_REG_KERNEL_ARGS_HI:
        ret = (uint32_t)((s->kernel.kernel_args >> 32) & 0xFFFFFFFF);
        break;
    case GPGPU_REG_GRID_DIM_X:
        ret = s->kernel.grid_dim[0];
        break;
    case GPGPU_REG_GRID_DIM_Y:
        ret = s->kernel.grid_dim[1];
        break;
    case GPGPU_REG_GRID_DIM_Z:
        ret = s->kernel.grid_dim[2];
        break;
    case GPGPU_REG_BLOCK_DIM_X:
        ret = s->kernel.block_dim[0];
        break;
    case GPGPU_REG_BLOCK_DIM_Y:
        ret = s->kernel.block_dim[1];
        break;
    case GPGPU_REG_BLOCK_DIM_Z:
        ret = s->kernel.block_dim[2];
        break;
    case GPGPU_REG_SHARED_MEM_SIZE:
        ret = s->kernel.shared_mem_size;
        break;
    case GPGPU_REG_DISPATCH:
        ret = 0;
        break;

    /* DMA (0x0400-0x04FF) */
    case GPGPU_REG_DMA_SRC_LO:
        ret = (uint32_t)(s->dma.src_addr & 0xFFFFFFFF);
        break;
    case GPGPU_REG_DMA_SRC_HI:
        ret = (uint32_t)((s->dma.src_addr >> 32) & 0xFFFFFFFF);
        break;
    case GPGPU_REG_DMA_DST_LO:
        ret = (uint32_t)(s->dma.dst_addr & 0xFFFFFFFF);
        break;
    case GPGPU_REG_DMA_DST_HI:
        ret = (uint32_t)((s->dma.dst_addr >> 32) & 0xFFFFFFFF);
        break;
    case GPGPU_REG_DMA_SIZE:
        ret = s->dma.size;
        break;
    case GPGPU_REG_DMA_CTRL:
        ret = s->dma.ctrl;
        break;
    case GPGPU_REG_DMA_STATUS:
        ret = s->dma.status;
        break;

    /* SIMT thread context (0x1000-0x1FFF) */
    case GPGPU_REG_THREAD_ID_X:
        ret = s->simt.thread_id[0];
        break;
    case GPGPU_REG_THREAD_ID_Y:
        ret = s->simt.thread_id[1];
        break;
    case GPGPU_REG_THREAD_ID_Z:
        ret = s->simt.thread_id[2];
        break;
    case GPGPU_REG_BLOCK_ID_X:
        ret = s->simt.block_id[0];
        break;
    case GPGPU_REG_BLOCK_ID_Y:
        ret = s->simt.block_id[1];
        break;
    case GPGPU_REG_BLOCK_ID_Z:
        ret = s->simt.block_id[2];
        break;
    case GPGPU_REG_WARP_ID:
        ret = s->simt.warp_id;
        break;
    case GPGPU_REG_LANE_ID:
        ret = s->simt.lane_id;
        break;

    /* Sync registers (0x2000-0x2FFF) */
    case GPGPU_REG_BARRIER:
        ret = 0;
        break;
    case GPGPU_REG_THREAD_MASK:
        ret = s->simt.thread_mask;
        break;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu_ctrl_read: unhandled addr 0x%"HWADDR_PRIx"\n",
                      addr);
        ret = 0;
        break;
    }

    return ret;
}

/* MMIO control register write */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;

    switch (addr) {
    /* Global control */
    case GPGPU_REG_GLOBAL_CTRL:
        s->global_ctrl = val;
        if (val & GPGPU_CTRL_RESET) {
            /* Device reset via GLOBAL_CTRL.RESET */
            gpgpu_reset(DEVICE(s));
            /* Clear self-clearing reset bit */
            s->global_ctrl &= ~GPGPU_CTRL_RESET;
        }
        break;

    case GPGPU_REG_ERROR_STATUS:
        /* Write 1 to clear */
        s->error_status &= ~(uint32_t)val;
        break;

    /* IRQ */
    case GPGPU_REG_IRQ_ENABLE:
        s->irq_enable = val;
        break;
    case GPGPU_REG_IRQ_ACK:
        /* Write 1 to clear IRQ status */
        s->irq_status &= ~(uint32_t)val;
        break;

    /* Kernel dispatch registers */
    case GPGPU_REG_KERNEL_ADDR_LO:
        s->kernel.kernel_addr = (s->kernel.kernel_addr & 0xFFFFFFFF00000000ULL) |
                                (uint64_t)(uint32_t)val;
        break;
    case GPGPU_REG_KERNEL_ADDR_HI:
        s->kernel.kernel_addr = (s->kernel.kernel_addr & 0x00000000FFFFFFFFULL) |
                                ((uint64_t)(uint32_t)val << 32);
        break;
    case GPGPU_REG_KERNEL_ARGS_LO:
        s->kernel.kernel_args = (s->kernel.kernel_args & 0xFFFFFFFF00000000ULL) |
                                (uint64_t)(uint32_t)val;
        break;
    case GPGPU_REG_KERNEL_ARGS_HI:
        s->kernel.kernel_args = (s->kernel.kernel_args & 0x00000000FFFFFFFFULL) |
                                ((uint64_t)(uint32_t)val << 32);
        break;
    case GPGPU_REG_GRID_DIM_X:
        s->kernel.grid_dim[0] = val;
        break;
    case GPGPU_REG_GRID_DIM_Y:
        s->kernel.grid_dim[1] = val;
        break;
    case GPGPU_REG_GRID_DIM_Z:
        s->kernel.grid_dim[2] = val;
        break;
    case GPGPU_REG_BLOCK_DIM_X:
        s->kernel.block_dim[0] = val;
        break;
    case GPGPU_REG_BLOCK_DIM_Y:
        s->kernel.block_dim[1] = val;
        break;
    case GPGPU_REG_BLOCK_DIM_Z:
        s->kernel.block_dim[2] = val;
        break;
    case GPGPU_REG_SHARED_MEM_SIZE:
        s->kernel.shared_mem_size = val;
        break;

    case GPGPU_REG_DISPATCH:
        /* Trigger kernel dispatch */
        s->global_status = GPGPU_STATUS_BUSY;
        gpgpu_core_exec_kernel(s);
        s->global_status = GPGPU_STATUS_READY;
        break;

    /* DMA registers */
    case GPGPU_REG_DMA_SRC_LO:
        s->dma.src_addr = (s->dma.src_addr & 0xFFFFFFFF00000000ULL) |
                          (uint64_t)(uint32_t)val;
        break;
    case GPGPU_REG_DMA_SRC_HI:
        s->dma.src_addr = (s->dma.src_addr & 0x00000000FFFFFFFFULL) |
                          ((uint64_t)(uint32_t)val << 32);
        break;
    case GPGPU_REG_DMA_DST_LO:
        s->dma.dst_addr = (s->dma.dst_addr & 0xFFFFFFFF00000000ULL) |
                          (uint64_t)(uint32_t)val;
        break;
    case GPGPU_REG_DMA_DST_HI:
        s->dma.dst_addr = (s->dma.dst_addr & 0x00000000FFFFFFFFULL) |
                          ((uint64_t)(uint32_t)val << 32);
        break;
    case GPGPU_REG_DMA_SIZE:
        s->dma.size = val;
        break;
    case GPGPU_REG_DMA_CTRL:
        s->dma.ctrl = val;
        if (val & GPGPU_DMA_START) {
            s->dma.status = GPGPU_DMA_BUSY;
            timer_mod(s->dma_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
                      100); /* ~negligible delay */
        }
        break;

    /* SIMT thread context */
    case GPGPU_REG_THREAD_ID_X:
        s->simt.thread_id[0] = val;
        break;
    case GPGPU_REG_THREAD_ID_Y:
        s->simt.thread_id[1] = val;
        break;
    case GPGPU_REG_THREAD_ID_Z:
        s->simt.thread_id[2] = val;
        break;
    case GPGPU_REG_BLOCK_ID_X:
        s->simt.block_id[0] = val;
        break;
    case GPGPU_REG_BLOCK_ID_Y:
        s->simt.block_id[1] = val;
        break;
    case GPGPU_REG_BLOCK_ID_Z:
        s->simt.block_id[2] = val;
        break;
    case GPGPU_REG_WARP_ID:
        s->simt.warp_id = val;
        break;
    case GPGPU_REG_LANE_ID:
        s->simt.lane_id = val;
        break;

    /* Sync registers */
    case GPGPU_REG_THREAD_MASK:
        s->simt.thread_mask = val;
        break;
    case GPGPU_REG_BARRIER:
        /* Barrier not implemented in basic tests */
        break;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu_ctrl_write: unhandled addr 0x%"HWADDR_PRIx
                      " val 0x%"PRIx64"\n",
                      addr, val);
        break;
    }
}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* VRAM read (BAR2) */
static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque;
    uint64_t ret = 0;

    if (addr + size > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu_vram_read: out of bounds addr 0x%"HWADDR_PRIx
                      " size %u\n", addr, size);
        return 0;
    }

    switch (size) {
    case 1:
        ret = s->vram_ptr[addr];
        break;
    case 2:
        ret = lduw_le_p(s->vram_ptr + addr);
        break;
    case 4:
        ret = ldl_le_p(s->vram_ptr + addr);
        break;
    case 8:
        ret = ldq_le_p(s->vram_ptr + addr);
        break;
    }

    return ret;
}

/* VRAM write (BAR2) */
static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;

    if (addr + size > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu_vram_write: out of bounds addr 0x%"HWADDR_PRIx
                      " size %u\n", addr, size);
        return;
    }

    switch (size) {
    case 1:
        s->vram_ptr[addr] = val;
        break;
    case 2:
        stw_le_p(s->vram_ptr + addr, val);
        break;
    case 4:
        stl_le_p(s->vram_ptr + addr, val);
        break;
    case 8:
        stq_le_p(s->vram_ptr + addr, val);
        break;
    }
}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)size;
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)val;
    (void)size;
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* DMA completion handler */
static void gpgpu_dma_complete(void *opaque)
{
    GPGPUState *s = opaque;

    s->dma.status = GPGPU_DMA_COMPLETE;

    /* Generate IRQ if enabled */
    if (s->dma.ctrl & GPGPU_DMA_IRQ_ENABLE) {
        s->irq_status |= GPGPU_IRQ_DMA_DONE;
    }
}

/* Kernel completion handler */
static void gpgpu_kernel_complete(void *opaque)
{
    GPGPUState *s = opaque;

    s->global_status = GPGPU_STATUS_READY;

    /* Generate IRQ if enabled */
    if (s->irq_enable & GPGPU_IRQ_KERNEL_DONE) {
        s->irq_status |= GPGPU_IRQ_KERNEL_DONE;
    }
}

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr) {
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }

    /* BAR 0: control registers */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: VRAM */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: doorbell registers */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0, 0xFE000,
                  &s->ctrl_mmio, 0, 0xFF000,
                  0, errp)) {
        g_free(s->vram_ptr);
        return;
    }

    msi_init(pdev, 0, 1, true, false, errp);

    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);
    s->kernel_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,
                                   gpgpu_kernel_complete, s);

    s->global_status = GPGPU_STATUS_READY;
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    timer_free(s->dma_timer);
    timer_free(s->kernel_timer);
    g_free(s->vram_ptr);
    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));
    memset(&s->simt, 0, sizeof(s->simt));
    timer_del(s->dma_timer);
    timer_del(s->kernel_timer);
    if (s->vram_ptr) {
        memset(s->vram_ptr, 0, s->vram_size);
    }
}

static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()
    }
};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { }
    },
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)
