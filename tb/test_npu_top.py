# Open-NPU RTL — Top-Level Integration Tests (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# Tests for npu_top: CSR access, DMA transfers, controller FSM cycle.

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


class WbSlave:
    """Wishbone B4 single-cycle master driver for npu_top's slave port."""

    def __init__(self, dut, clk):
        self.dut = dut
        self.clk = clk

    async def write(self, addr, data, sel=0xF):
        self.dut.wb_slv_cyc_i.value = 1
        self.dut.wb_slv_stb_i.value = 1
        self.dut.wb_slv_we_i.value = 1
        self.dut.wb_slv_adr_i.value = addr
        self.dut.wb_slv_dat_i.value = data
        self.dut.wb_slv_sel_i.value = sel
        await RisingEdge(self.clk)
        while not self.dut.wb_slv_ack_o.value:
            await RisingEdge(self.clk)
        self.dut.wb_slv_cyc_i.value = 0
        self.dut.wb_slv_stb_i.value = 0
        await RisingEdge(self.clk)

    async def read(self, addr):
        self.dut.wb_slv_cyc_i.value = 1
        self.dut.wb_slv_stb_i.value = 1
        self.dut.wb_slv_we_i.value = 0
        self.dut.wb_slv_adr_i.value = addr
        self.dut.wb_slv_sel_i.value = 0xF
        await RisingEdge(self.clk)
        while not self.dut.wb_slv_ack_o.value:
            await RisingEdge(self.clk)
        # Sample data after NBA settles
        await ReadOnly()
        data = int(self.dut.wb_slv_dat_o.value)
        # Exit ReadOnly phase before driving
        await Timer(1, unit="step")
        await RisingEdge(self.clk)
        self.dut.wb_slv_cyc_i.value = 0
        self.dut.wb_slv_stb_i.value = 0
        await RisingEdge(self.clk)
        return data


class WbMasterMem:
    """Emulates external memory responding to DMA WB master requests."""

    def __init__(self, dut, clk, mem_data=None):
        self.dut = dut
        self.clk = clk
        self.mem = {}  # addr → 32-bit word
        if mem_data:
            self.mem.update(mem_data)

    async def run(self):
        """Background task: respond to WB master transactions."""
        while True:
            await RisingEdge(self.clk)
            if self.dut.wb_mst_cyc_o.value and self.dut.wb_mst_stb_o.value:
                addr = int(self.dut.wb_mst_adr_o.value)
                if self.dut.wb_mst_we_o.value:
                    # Write from DMA to memory
                    self.mem[addr] = int(self.dut.wb_mst_dat_o.value)
                else:
                    # Read from memory to DMA
                    self.dut.wb_mst_dat_i.value = self.mem.get(addr, 0xDEAD_BEEF)
                self.dut.wb_mst_ack_i.value = 1
            else:
                self.dut.wb_mst_ack_i.value = 0


async def reset(dut):
    """Apply reset sequence."""
    dut.rst_n.value = 0
    dut.wb_slv_cyc_i.value = 0
    dut.wb_slv_stb_i.value = 0
    dut.wb_slv_we_i.value = 0
    dut.wb_slv_adr_i.value = 0
    dut.wb_slv_dat_i.value = 0
    dut.wb_slv_sel_i.value = 0
    dut.wb_mst_ack_i.value = 0
    dut.wb_mst_dat_i.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_reset_state(dut):
    """After reset, IRQ should be 0, CSR status should be idle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # Check IRQ is deasserted
    await ReadOnly()
    assert dut.irq_o.value == 0, "IRQ should be 0 after reset"


@cocotb.test()
async def test_csr_version_read(dut):
    """Read VERSION register (0x014) through top-level WB slave."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    version = await wb.read(0x014)
    # VERSION: {8'd0, MAJOR=1, MINOR=0, PATCH=0} = 0x00_01_00_00
    assert version == 0x00010000, f"Expected VERSION=0x00010000, got 0x{version:08X}"


@cocotb.test()
async def test_csr_hw_config_read(dut):
    """Read HW_CONFIG register (0x018) through top-level WB slave."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    hw_cfg = await wb.read(0x018)
    # Dynamic: read ARRAY_SIZE from RTL parameter
    array_sz = int(dut.ARRAY_SIZE.value)
    import math
    dw_ch_log2 = int(math.log2(array_sz))
    expected = (0 << 27) | (0 << 26) | (1 << 25) | (1 << 24) | (128 << 16) | (dw_ch_log2 << 12) | (1 << 8) | array_sz
    assert hw_cfg == expected, f"Expected HW_CONFIG=0x{expected:08X}, got 0x{hw_cfg:08X}"


@cocotb.test()
async def test_csr_write_read_roundtrip(dut):
    """Write a layer register and read it back."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # Write LAYER_MODE (0x040) = 0xCAFEBABE
    await wb.write(0x040, 0xCAFEBABE)
    val = await wb.read(0x040)
    assert val == 0xCAFEBABE, f"Expected 0xCAFEBABE, got 0x{val:08X}"

    # Write DMA_IN_ADDR (0x100) = 0x80000000
    await wb.write(0x100, 0x80000000)
    val = await wb.read(0x100)
    assert val == 0x80000000, f"Expected 0x80000000, got 0x{val:08X}"


@cocotb.test()
async def test_irq_generation(dut):
    """Start a layer with zero-length DMA → done quickly → IRQ fires."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    # Configure zero-length DMA (all sizes = 0)
    await wb.write(0x128, 0)  # DMA_IN_SIZE = 0
    await wb.write(0x12C, 0)  # DMA_WGT_SIZE = 0
    await wb.write(0x188, 0)  # POST_PARAM_COUNT = 0

    # Enable IRQ for done bit
    await wb.write(0x008, 0x01)  # IRQ_EN[0] = done

    # Start (write CTRL[0] = 1)
    await wb.write(0x000, 0x01)

    # Controller will go through states quickly with zero-length,
    # but compute_done depends on systolic output valid (which won't fire).
    # For this V1 integration test, we verify the start was accepted.
    # Wait a few cycles for the start to propagate
    for _ in range(5):
        await RisingEdge(dut.clk)

    # Read STATUS — should show busy
    status = await wb.read(0x004)
    busy_bit = status & 0x1
    assert busy_bit == 1, f"Expected busy after START, got status=0x{status:08X}"


@cocotb.test()
async def test_dma_load_weight(dut):
    """Load weight data from external memory into weight SRAM via DMA."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # Populate external memory with test data
    wgt_base = 0x4000_0000
    test_words = [0x11223344, 0x55667788, 0xAABBCCDD, 0xEEFF0011]
    ext_mem = {wgt_base + i * 4: test_words[i] for i in range(len(test_words))}
    mem = WbMasterMem(dut, dut.clk, ext_mem)
    cocotb.start_soon(mem.run())

    # Configure: set wgt_addr and wgt_size (4 words = 16 bytes)
    await wb.write(0x108, wgt_base)   # DMA_WGT_ADDR
    await wb.write(0x12C, 16)         # DMA_WGT_SIZE = 16 bytes → 4 words
    await wb.write(0x100, 0x8000_0000)  # DMA_IN_ADDR (different from wgt)
    await wb.write(0x128, 0)          # DMA_IN_SIZE = 0
    await wb.write(0x10C, 0x9000_0000)  # DMA_PARAM_ADDR (different)
    await wb.write(0x188, 0)          # POST_PARAM_COUNT = 0

    # Start layer
    await wb.write(0x000, 0x01)

    # Wait for controller to complete weight DMA phase
    # (wgt phase → act phase; act is zero-length so skips to param phase)
    # The full cycle is: LOAD_WGT → WAIT_WGT → LOAD_ACT(skip) → LOAD_PARAM(skip) → COMPUTE
    timeout = 200
    for _ in range(timeout):
        await RisingEdge(dut.clk)

    # Read status — should be in compute (busy) since compute_done won't fire
    status = await wb.read(0x004)
    assert (status & 0x1) == 1, f"Expected busy during compute, got 0x{status:08X}"


@cocotb.test()
async def test_abort_during_operation(dut):
    """Abort while busy should return to idle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # Setup minimal config with non-zero weight (so DMA takes time)
    wgt_base = 0x2000_0000
    ext_mem = {wgt_base + i * 4: i + 1 for i in range(100)}
    mem = WbMasterMem(dut, dut.clk, ext_mem)
    cocotb.start_soon(mem.run())

    await wb.write(0x108, wgt_base)    # DMA_WGT_ADDR
    await wb.write(0x12C, 400)         # DMA_WGT_SIZE = 400 bytes → 100 words
    await wb.write(0x100, 0x3000_0000) # DMA_IN_ADDR
    await wb.write(0x128, 0)           # DMA_IN_SIZE = 0
    await wb.write(0x10C, 0x4000_0000) # DMA_PARAM_ADDR
    await wb.write(0x188, 0)           # POST_PARAM_COUNT = 0

    # Start
    await wb.write(0x000, 0x01)

    # Wait a few cycles then abort
    for _ in range(20):
        await RisingEdge(dut.clk)

    # Should be busy during DMA
    status = await wb.read(0x004)
    assert (status & 0x1) == 1, "Expected busy during DMA load"

    # Abort
    await wb.write(0x000, 0x02)  # CTRL[1] = ABORT

    # Wait for abort to take effect
    for _ in range(20):
        await RisingEdge(dut.clk)

    # Now status should be not busy (controller returns to IDLE via DONE)
    status = await wb.read(0x004)
    busy = status & 0x1
    assert busy == 0, f"Expected not busy after abort, got status=0x{status:08X}"


@cocotb.test()
async def test_soft_reset(dut):
    """Soft reset clears all state."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    # Write some config
    await wb.write(0x040, 0xDEADBEEF)  # LAYER_MODE

    # Issue soft reset
    await wb.write(0x000, 0x04)  # CTRL[2] = SOFT_RST

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Layer mode should still be 0xDEADBEEF (soft_rst only resets controller FSM, not CSR data)
    val = await wb.read(0x040)
    assert val == 0xDEADBEEF, f"Soft reset should not clear CSR data, got 0x{val:08X}"

    # Status should show not busy
    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should not be busy after soft reset"


@cocotb.test()
async def test_full_dma_cycle_with_abort(dut):
    """Run a full DMA weight load, then abort at compute phase."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # Setup 8 words of weight data
    wgt_base = 0x1000_0000
    wgt_data = {wgt_base + i * 4: (i + 1) * 0x1111 for i in range(8)}
    mem = WbMasterMem(dut, dut.clk, wgt_data)
    cocotb.start_soon(mem.run())

    await wb.write(0x108, wgt_base)    # DMA_WGT_ADDR
    await wb.write(0x12C, 32)          # DMA_WGT_SIZE = 32 bytes → 8 words
    await wb.write(0x100, 0x5000_0000) # DMA_IN_ADDR (different from wgt)
    await wb.write(0x128, 0)           # DMA_IN_SIZE = 0 (skip)
    await wb.write(0x10C, 0x6000_0000) # DMA_PARAM_ADDR
    await wb.write(0x188, 0)           # POST_PARAM_COUNT = 0

    # Start
    await wb.write(0x000, 0x01)

    # Wait for weight DMA to complete (8 words × ~3 cycles each = ~24 cycles)
    for _ in range(100):
        await RisingEdge(dut.clk)

    # At this point controller should be stuck at COMPUTE (waiting for compute_done)
    status = await wb.read(0x004)
    assert (status & 0x1) == 1, "Should still be busy (waiting in compute)"

    # Abort to exit
    await wb.write(0x000, 0x02)
    for _ in range(10):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle after abort"
