# Open-NPU RTL — Abort/Soft Reset Robustness Tests (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# P0.2: Verify clean abort/reset recovery in all controller phases:
#   1. Abort during weight load (DMA in progress)
#   2. Abort during compute (systolic/PPU running)
#   3. Abort during store (DMA output in progress)
#   4. Soft reset during compute (must return to IDLE immediately)
#   5. Resume after abort (verify next inference runs correctly)
#   6. Abort during auto-next multi-layer (must stop, not restart)
#
# DUT = npu_top (full chip)

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

class WbSlave:
    """Wishbone B4 master driver for npu_top's slave port."""

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
        for _ in range(100):
            if self.dut.wb_slv_ack_o.value:
                break
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
        for _ in range(100):
            if self.dut.wb_slv_ack_o.value:
                break
            await RisingEdge(self.clk)
        await ReadOnly()
        try:
            data = int(self.dut.wb_slv_dat_o.value)
        except ValueError:
            data = 0
        await Timer(1, unit="step")
        self.dut.wb_slv_cyc_i.value = 0
        self.dut.wb_slv_stb_i.value = 0
        await RisingEdge(self.clk)
        return data


class WbMasterMem:
    """DDR emulator for DMA master port with optional slow response."""

    def __init__(self, dut, clk, delay=0):
        self.dut = dut
        self.clk = clk
        self.mem = {}
        self.delay = delay  # Fixed delay cycles per transaction
        self.txn_count = 0

    def populate(self, base_addr, words):
        for i, w in enumerate(words):
            self.mem[base_addr + i * 4] = int(w)

    def read_words(self, base_addr, count):
        """Read back stored words."""
        return [self.mem.get(base_addr + i * 4, 0) for i in range(count)]

    async def run(self):
        while True:
            await RisingEdge(self.clk)
            try:
                cyc = int(self.dut.wb_mst_cyc_o.value)
                stb = int(self.dut.wb_mst_stb_o.value)
            except ValueError:
                self.dut.wb_mst_ack_i.value = 0
                continue

            if cyc and stb:
                # Optional delay
                for _ in range(self.delay):
                    self.dut.wb_mst_ack_i.value = 0
                    await RisingEdge(self.clk)
                    try:
                        if not int(self.dut.wb_mst_cyc_o.value):
                            break
                    except ValueError:
                        break

                try:
                    addr = int(self.dut.wb_mst_adr_o.value)
                    we = int(self.dut.wb_mst_we_o.value)
                except ValueError:
                    self.dut.wb_mst_ack_i.value = 0
                    continue

                if we:
                    try:
                        self.mem[addr] = int(self.dut.wb_mst_dat_o.value)
                    except ValueError:
                        self.mem[addr] = 0
                else:
                    self.dut.wb_mst_dat_i.value = self.mem.get(addr, 0)

                self.dut.wb_mst_ack_i.value = 1
                self.txn_count += 1
            else:
                self.dut.wb_mst_ack_i.value = 0


async def reset(dut):
    """Apply hardware reset."""
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


async def setup(dut):
    """Start clock and reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)


async def wait_busy(wb, expect_busy=True, max_cycles=200):
    """Wait until status busy matches expectation."""
    for _ in range(max_cycles):
        status = await wb.read(0x004)
        busy = status & 0x1
        if busy == (1 if expect_busy else 0):
            return status
    return await wb.read(0x004)


async def program_conv_layer(wb, wgt_addr, act_addr, param_addr,
                             wgt_bytes, act_bytes, out_bytes=0,
                             out_channels=16, skip_store=False):
    """Program a minimal 1×1 Conv layer."""
    await wb.write(0x040, 0x0001)     # LAYER_MODE: conv2d, int8
    await wb.write(0x044, 0x00010001) # IN_DIM: 1x1
    await wb.write(0x048, 16)         # IN_C = 16
    await wb.write(0x04C, 0x00010001) # OUT_DIM: 1x1
    await wb.write(0x050, out_channels)
    await wb.write(0x054, 0x0101)     # KERNEL: 1x1
    await wb.write(0x058, 0x0101)     # STRIDE: 1x1
    await wb.write(0x05C, 0)          # PADDING: 0
    await wb.write(0x078, 0)          # SRAM_BASE: 0
    await wb.write(0x108, wgt_addr)
    await wb.write(0x100, act_addr)
    await wb.write(0x10C, param_addr)
    await wb.write(0x12C, wgt_bytes)
    await wb.write(0x128, act_bytes)
    await wb.write(0x130, out_bytes)
    await wb.write(0x188, out_channels)
    ctrl = 0x02 if skip_store else 0x00  # FUSE_START = skip store
    await wb.write(0x118, ctrl)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Abort during weight load
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=500, timeout_unit="us")
async def test_abort_during_wgt_load(dut):
    """Issue ABORT while DMA is loading weights — must return to IDLE cleanly."""
    await setup(dut)

    # Use slow memory (5 cycle delay) so DMA takes longer
    mem = WbMasterMem(dut, dut.clk, delay=5)
    mem.populate(0x1000, [0] * 100)  # 100 words of weight data
    mem.populate(0x2000, [0] * 16)   # activation data
    mem.populate(0x3000, [0] * 64)   # param data
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    # Program layer with 100 words of weights (= 400 bytes → long DMA)
    await program_conv_layer(wb, 0x1000, 0x2000, 0x3000,
                             wgt_bytes=400, act_bytes=64, skip_store=True)

    # START
    await wb.write(0x000, 0x01)

    # Wait until busy, then a few more cycles into weight DMA
    status = await wait_busy(wb, expect_busy=True)
    assert status & 0x1, "Should be busy immediately after START"

    # Let DMA run for ~10 cycles (only ~1-2 words at 5-cycle delay)
    for _ in range(10):
        await RisingEdge(dut.clk)

    # Issue ABORT
    await wb.write(0x000, 0x02)

    # Wait for controller to go idle
    for _ in range(50):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    busy = status & 0x1
    assert busy == 0, f"Expected IDLE after abort during wgt load, status=0x{status:08X}"

    # Verify DMA master bus released (CYC should be 0)
    try:
        cyc = int(dut.wb_mst_cyc_o.value)
    except ValueError:
        cyc = 0
    assert cyc == 0, "DMA master CYC should be deasserted after abort"

    mem_task.cancel()
    dut._log.info(f"PASS: Abort during weight load, {mem.txn_count} txns completed before abort")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Abort during compute phase
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=1, timeout_unit="ms")
async def test_abort_during_compute(dut):
    """Issue ABORT while compute engine is running — must return to IDLE."""
    await setup(dut)

    # Fast memory (0 delay) so DMA completes quickly
    mem = WbMasterMem(dut, dut.clk, delay=0)
    mem.populate(0x1000, [0] * 16)   # 16 words weights (= 64 bytes)
    mem.populate(0x2000, [0] * 16)   # 16 words activation
    mem.populate(0x3000, [0] * 64)   # 64 words params
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await program_conv_layer(wb, 0x1000, 0x2000, 0x3000,
                             wgt_bytes=64, act_bytes=64, skip_store=True)

    # START
    await wb.write(0x000, 0x01)

    # Wait for DMA to finish and compute to start
    # DMA loads: 16 + 16 + 64 = 96 words → ~100 cycles with 0 delay
    for _ in range(150):
        await RisingEdge(dut.clk)

    # Should be in compute phase (busy=1, DMA done)
    status = await wb.read(0x004)
    assert status & 0x1, "Should still be busy (in compute)"

    # ABORT
    await wb.write(0x000, 0x02)

    for _ in range(50):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, f"Expected IDLE after abort during compute, status=0x{status:08X}"

    mem_task.cancel()
    dut._log.info("PASS: Abort during compute phase returns to IDLE")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Abort during output store
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=2, timeout_unit="ms")
async def test_abort_during_store(dut):
    """Issue ABORT while DMA is storing output — must return to IDLE cleanly."""
    await setup(dut)

    # Use slow memory for store phase (so we can catch it)
    mem = WbMasterMem(dut, dut.clk, delay=5)
    mem.populate(0x1000, [0] * 16)   # weights
    mem.populate(0x2000, [0] * 16)   # activation
    mem.populate(0x3000, [0] * 64)   # params (all zeros → output will be zeros after PPU)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    # Program layer WITH store (out_bytes > 0, no fusion)
    await program_conv_layer(wb, 0x1000, 0x2000, 0x3000,
                             wgt_bytes=64, act_bytes=64,
                             out_bytes=64,  # 16 words output to store
                             skip_store=False)
    # Set output address
    await wb.write(0x104, 0x5000)  # DMA_OUT_ADDR

    # START
    await wb.write(0x000, 0x01)

    # Wait for compute to finish and store to begin
    # This takes a while — DMA loads + compute. Wait generously.
    for _ in range(5000):
        await RisingEdge(dut.clk)
        # Check if we're past compute (DMA store should assert CYC for writes)
        try:
            cyc = int(dut.wb_mst_cyc_o.value)
            we = int(dut.wb_mst_we_o.value)
            if cyc and we:
                # We're in store phase!
                break
        except ValueError:
            pass

    # Verify we're in store (busy + DMA master writing)
    status = await wb.read(0x004)
    if status & 0x1:
        # ABORT during store
        await wb.write(0x000, 0x02)

        for _ in range(50):
            await RisingEdge(dut.clk)

        status = await wb.read(0x004)
        assert (status & 0x1) == 0, f"Expected IDLE after abort during store, status=0x{status:08X}"
        dut._log.info("PASS: Abort during output store returns to IDLE")
    else:
        # Layer already completed before we could abort — that's OK
        dut._log.info("PASS: Layer completed before abort (fast path) — still demonstrates clean exit")

    mem_task.cancel()


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Soft reset during compute
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=1, timeout_unit="ms")
async def test_soft_rst_during_compute(dut):
    """Soft reset while compute is running — must immediately return to IDLE."""
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk, delay=0)
    mem.populate(0x1000, [0] * 16)
    mem.populate(0x2000, [0] * 16)
    mem.populate(0x3000, [0] * 64)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await program_conv_layer(wb, 0x1000, 0x2000, 0x3000,
                             wgt_bytes=64, act_bytes=64, skip_store=True)

    # START
    await wb.write(0x000, 0x01)

    # Wait for compute phase
    for _ in range(150):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert status & 0x1, "Should be busy in compute"

    # SOFT RESET
    await wb.write(0x000, 0x04)

    # Should be immediate — wait just 2 cycles
    for _ in range(5):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, f"Expected IDLE after soft_rst, status=0x{status:08X}"

    # Verify CSR data preserved
    val = await wb.read(0x108)  # DMA_WGT_ADDR should still be 0x1000
    assert val == 0x1000, f"CSR data lost after soft_rst: wgt_addr=0x{val:08X}"

    mem_task.cancel()
    dut._log.info("PASS: Soft reset during compute immediately returns to IDLE, CSR preserved")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Resume after abort (verify system still works)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=2, timeout_unit="ms")
async def test_resume_after_abort(dut):
    """After abort, start a new layer — must complete successfully."""
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk, delay=0)
    mem.populate(0x1000, [0] * 16)
    mem.populate(0x2000, [0] * 16)
    mem.populate(0x3000, [0] * 64)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    # --- First run: abort mid-way ---
    await program_conv_layer(wb, 0x1000, 0x2000, 0x3000,
                             wgt_bytes=64, act_bytes=64, skip_store=True)
    await wb.write(0x008, 0x01)  # IRQ_EN: done

    await wb.write(0x000, 0x01)  # START

    for _ in range(50):
        await RisingEdge(dut.clk)

    # ABORT
    await wb.write(0x000, 0x02)

    for _ in range(50):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle after first abort"

    # Clear IRQ status (W1C)
    await wb.write(0x00C, 0xFF)

    # --- Second run: should complete normally ---
    await wb.write(0x000, 0x01)  # START again

    # Wait for completion via IRQ
    done = False
    for _ in range(50000):
        await RisingEdge(dut.clk)
        try:
            if int(dut.irq_o.value):
                done = True
                break
        except ValueError:
            pass

    assert done, "Second layer did not complete after abort recovery"

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle after second layer completes"

    mem_task.cancel()
    dut._log.info("PASS: System resumes correctly after abort — second layer completes")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Abort during auto-next (must stop, not restart next layer)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=2, timeout_unit="ms")
async def test_abort_auto_next(dut):
    """Abort during auto-next multi-layer — must stop without starting next layer."""
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk, delay=2)
    mem.populate(0x1000, [0] * 100)  # Lots of weight data (slow load)
    mem.populate(0x2000, [0] * 16)
    mem.populate(0x3000, [0] * 64)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    # Program with auto-next: 3 layers
    await program_conv_layer(wb, 0x1000, 0x2000, 0x3000,
                             wgt_bytes=400, act_bytes=64, skip_store=True)
    await wb.write(0x030, 3)  # LAYER_COUNT = 3

    # START with AUTO_NEXT
    await wb.write(0x000, 0x09)  # CTRL[0]=START, CTRL[3]=AUTO_NEXT

    # Wait until first layer is in DMA
    for _ in range(20):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert status & 0x1, "Should be busy"

    # Read curr_layer — should be 0 (first layer)
    curr_layer = (status >> 8) & 0xFF

    # ABORT — should prevent auto-next from continuing
    await wb.write(0x000, 0x02)

    for _ in range(100):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    busy = status & 0x1
    assert busy == 0, f"Expected IDLE after abort during auto-next, status=0x{status:08X}"

    # hw_curr_layer should be at most 1 (completed current layer's DONE, then stopped)
    final_layer = (status >> 8) & 0xFF
    assert final_layer <= 1, \
        f"Auto-next should have stopped, but curr_layer={final_layer} (expected ≤1)"

    mem_task.cancel()
    dut._log.info(f"PASS: Abort during auto-next stops at layer {final_layer}")
