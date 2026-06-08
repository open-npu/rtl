# Open-NPU RTL — Boundary Parameter / Edge Case Tests (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# P0.3: Verify correct behavior at parameter boundaries:
#   1. OUT_C=0 (zero output channels — layer should still complete)
#   2. OUT_C=1 (single output channel — minimum valid config)
#   3. OUT_C=256 (maximum output channels — large DMA + many PPU passes)
#   4. 1×1 input with 1×1 kernel (smallest possible spatial)
#   5. Large stride (stride > kernel, stride=4 with kernel=1)
#   6. All DMA sizes = 0 (no data to transfer — should still trigger done)
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
    """DDR emulator for DMA master port."""

    def __init__(self, dut, clk):
        self.dut = dut
        self.clk = clk
        self.mem = {}
        self.txn_count = 0

    def populate(self, base_addr, words):
        for i, w in enumerate(words):
            self.mem[base_addr + i * 4] = int(w)

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


async def wait_done(dut, max_cycles=100000):
    """Wait for IRQ (done) or timeout. Returns True if done."""
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        try:
            if int(dut.irq_o.value):
                return True
        except ValueError:
            pass
    return False


# ═══════════════════════════════════════════════════════════════════════
# Test 1: OUT_C=0 (zero output channels)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=1, timeout_unit="ms")
async def test_out_c_zero(dut):
    """OUT_C=0, WGT_SIZE=0, PARAM_COUNT=0 — layer completes (no actual computation).

    With wgt_size=0 and param_count=0, the controller skips all DMA phases
    that depend on those, goes to compute (which may be a no-op), then done.
    We just verify no hang occurs.
    """
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    # Activation: 4 words (minimum to have something)
    mem.populate(0x1000, [0x11] * 4)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)     # conv2d, int8
    await wb.write(0x044, 0x00010001) # IN: 1x1
    await wb.write(0x048, 16)         # IN_C = 16
    await wb.write(0x04C, 0x00010001) # OUT: 1x1
    await wb.write(0x050, 0)          # OUT_C = 0 (edge case!)
    await wb.write(0x054, 0x0101)     # kernel 1x1
    await wb.write(0x058, 0x0101)     # stride 1x1
    await wb.write(0x05C, 0)          # padding 0
    await wb.write(0x078, 0)          # sram_base
    await wb.write(0x108, 0x2000)     # WGT_ADDR
    await wb.write(0x100, 0x1000)     # IN_ADDR
    await wb.write(0x10C, 0x3000)     # PARAM_ADDR
    await wb.write(0x12C, 0)          # WGT_SIZE = 0 (no weights for 0 out_c)
    await wb.write(0x128, 16)         # IN_SIZE = 16 bytes = 4 words
    await wb.write(0x130, 0)          # OUT_SIZE = 0
    await wb.write(0x188, 0)          # PARAM_COUNT = 0
    await wb.write(0x118, 0x02)       # skip store (FUSE_START)
    await wb.write(0x008, 0x01)       # IRQ_EN

    # START
    await wb.write(0x000, 0x01)

    done = await wait_done(dut, max_cycles=50000)
    assert done, "Layer with OUT_C=0 should still complete (no hang)"

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle"

    mem_task.cancel()
    dut._log.info("PASS: OUT_C=0 completes without hang")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: OUT_C=1 (single output channel)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=2, timeout_unit="ms")
async def test_out_c_one(dut):
    """OUT_C=1 — minimum valid configuration. Must complete without error."""
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    # 1 output channel: weight = 16 in_c × 1 out_c = 16 bytes = 4 words
    mem.populate(0x1000, [0x01020304] * 4)   # 4 words weights
    mem.populate(0x2000, [0x10203040] * 4)   # 4 words activation
    mem.populate(0x3000, [0] * 4)            # 1 channel × 4 words params
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)     # conv2d, int8
    await wb.write(0x044, 0x00010001) # IN: 1x1
    await wb.write(0x048, 16)         # IN_C = 16
    await wb.write(0x04C, 0x00010001) # OUT: 1x1
    await wb.write(0x050, 1)          # OUT_C = 1 (edge case!)
    await wb.write(0x054, 0x0101)     # kernel 1x1
    await wb.write(0x058, 0x0101)     # stride 1x1
    await wb.write(0x05C, 0)          # padding 0
    await wb.write(0x078, 0)          # sram_base
    await wb.write(0x108, 0x1000)     # WGT_ADDR
    await wb.write(0x100, 0x2000)     # IN_ADDR
    await wb.write(0x10C, 0x3000)     # PARAM_ADDR
    await wb.write(0x12C, 16)         # WGT_SIZE = 16 bytes
    await wb.write(0x128, 16)         # IN_SIZE = 16 bytes
    await wb.write(0x130, 0)          # OUT_SIZE = 0 (skip store)
    await wb.write(0x188, 1)          # PARAM_COUNT = 1
    await wb.write(0x118, 0x02)       # skip store
    await wb.write(0x008, 0x01)       # IRQ_EN

    await wb.write(0x000, 0x01)  # START

    done = await wait_done(dut, max_cycles=50000)
    assert done, "OUT_C=1 layer timed out"

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle"

    mem_task.cancel()
    dut._log.info("PASS: OUT_C=1 (single channel) completes normally")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: OUT_C=256 (maximum output channels)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=20, timeout_unit="ms")
async def test_out_c_max(dut):
    """OUT_C=256 — maximum channels, large DMA + many compute passes.

    Weight: 16 in_c × 256 out_c = 4096 bytes = 1024 words
    Params: 256 channels × 4 words = 1024 words
    This exercises large transfer counts and multi-tile compute.
    """
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    # Weight: 1024 words
    mem.populate(0x10000, [0] * 1024)
    # Activation: 4 words (1×1 × 16 channels / 4 per word)
    mem.populate(0x20000, [0] * 4)
    # Params: 1024 words
    mem.populate(0x30000, [0] * 1024)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)     # conv2d, int8
    await wb.write(0x044, 0x00010001) # IN: 1x1
    await wb.write(0x048, 16)         # IN_C = 16
    await wb.write(0x04C, 0x00010001) # OUT: 1x1
    await wb.write(0x050, 256)        # OUT_C = 256 (max!)
    await wb.write(0x054, 0x0101)     # kernel 1x1
    await wb.write(0x058, 0x0101)     # stride 1x1
    await wb.write(0x05C, 0)          # padding 0
    await wb.write(0x078, 0)          # sram_base
    await wb.write(0x108, 0x10000)    # WGT_ADDR
    await wb.write(0x100, 0x20000)    # IN_ADDR
    await wb.write(0x10C, 0x30000)    # PARAM_ADDR
    await wb.write(0x12C, 4096)       # WGT_SIZE = 4096 bytes
    await wb.write(0x128, 16)         # IN_SIZE = 16 bytes
    await wb.write(0x130, 0)          # OUT_SIZE = 0 (skip store)
    await wb.write(0x188, 256)        # PARAM_COUNT = 256
    await wb.write(0x118, 0x02)       # skip store
    await wb.write(0x008, 0x01)       # IRQ_EN

    await wb.write(0x000, 0x01)  # START

    done = await wait_done(dut, max_cycles=500000)
    assert done, "OUT_C=256 layer timed out"

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle"

    # Verify all DMA loads completed
    expected_dma = 1024 + 4 + 1024  # wgt + act + param
    assert mem.txn_count >= expected_dma, \
        f"Expected ≥{expected_dma} DMA txns, got {mem.txn_count}"

    mem_task.cancel()
    dut._log.info(f"PASS: OUT_C=256, {mem.txn_count} DMA txns completed")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: 1×1 input feature map (smallest spatial)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=2, timeout_unit="ms")
async def test_1x1_input(dut):
    """1×1 input with 1×1 kernel — minimal spatial dimension."""
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    mem.populate(0x1000, [0] * 16)   # 16 words weights
    mem.populate(0x2000, [0] * 4)    # 4 words activation
    mem.populate(0x3000, [0] * 64)   # 64 words params
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)     # conv2d, int8
    await wb.write(0x044, 0x00010001) # IN: 1x1 (H=1, W=1)
    await wb.write(0x048, 16)         # IN_C = 16
    await wb.write(0x04C, 0x00010001) # OUT: 1x1
    await wb.write(0x050, 16)         # OUT_C = 16
    await wb.write(0x054, 0x0101)     # kernel 1x1
    await wb.write(0x058, 0x0101)     # stride 1x1
    await wb.write(0x05C, 0)          # padding 0
    await wb.write(0x078, 0)
    await wb.write(0x108, 0x1000)     # WGT
    await wb.write(0x100, 0x2000)     # IN
    await wb.write(0x10C, 0x3000)     # PARAM
    await wb.write(0x12C, 64)         # WGT 64 bytes
    await wb.write(0x128, 16)         # IN 16 bytes
    await wb.write(0x130, 0)
    await wb.write(0x188, 16)
    await wb.write(0x118, 0x02)       # skip store
    await wb.write(0x008, 0x01)

    await wb.write(0x000, 0x01)

    done = await wait_done(dut, max_cycles=50000)
    assert done, "1×1 input layer timed out"

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle"

    mem_task.cancel()
    dut._log.info("PASS: 1×1 input feature map completes correctly")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Large stride (stride=4, kernel=1, input=8×8)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=5, timeout_unit="ms")
async def test_large_stride(dut):
    """Stride=4 with kernel=1 on 8×8 input → 2×2 output. Tests stride > kernel."""
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    # IN: 8×8×16 = 1024 bytes = 256 words
    mem.populate(0x1000, [0] * 256)
    # WGT: 16 in_c × 16 out_c / 4 = 64 words (= 256 bytes for 1×1 kernel)
    mem.populate(0x2000, [0] * 64)
    # PARAM: 16 × 4 = 64 words
    mem.populate(0x3000, [0] * 64)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)     # conv2d, int8
    await wb.write(0x044, 0x00080008) # IN: 8x8
    await wb.write(0x048, 16)         # IN_C = 16
    await wb.write(0x04C, 0x00020002) # OUT: 2x2 (8/4 = 2)
    await wb.write(0x050, 16)         # OUT_C = 16
    await wb.write(0x054, 0x0101)     # kernel 1x1
    await wb.write(0x058, 0x0404)     # stride 4x4 (edge case: stride > kernel)
    await wb.write(0x05C, 0)          # padding 0
    await wb.write(0x078, 0)
    await wb.write(0x108, 0x2000)     # WGT
    await wb.write(0x100, 0x1000)     # IN
    await wb.write(0x10C, 0x3000)     # PARAM
    await wb.write(0x12C, 256)        # WGT 256 bytes
    await wb.write(0x128, 1024)       # IN 1024 bytes
    await wb.write(0x130, 0)
    await wb.write(0x188, 16)
    await wb.write(0x118, 0x02)       # skip store
    await wb.write(0x008, 0x01)

    await wb.write(0x000, 0x01)

    done = await wait_done(dut, max_cycles=200000)
    assert done, "Large stride layer timed out"

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle"

    mem_task.cancel()
    dut._log.info("PASS: Stride=4 with 8×8 input completes correctly")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: All DMA sizes = 0 (no data to transfer)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=1, timeout_unit="ms")
async def test_zero_dma_sizes(dut):
    """All DMA sizes = 0 — controller should skip all DMA, go to compute, then done.

    This is an extreme edge case: the controller sequences through all states
    but skips all transfers. Verifies no hang in the FSM with empty transfer requests.
    """
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)     # conv2d, int8
    await wb.write(0x044, 0x00010001) # IN: 1x1
    await wb.write(0x048, 16)         # IN_C
    await wb.write(0x04C, 0x00010001) # OUT: 1x1
    await wb.write(0x050, 16)         # OUT_C
    await wb.write(0x054, 0x0101)
    await wb.write(0x058, 0x0101)
    await wb.write(0x05C, 0)
    await wb.write(0x078, 0)
    await wb.write(0x108, 0x1000)     # WGT_ADDR (doesn't matter)
    await wb.write(0x100, 0x2000)     # IN_ADDR
    await wb.write(0x10C, 0x3000)     # PARAM_ADDR
    await wb.write(0x12C, 0)          # WGT_SIZE = 0
    await wb.write(0x128, 0)          # IN_SIZE = 0
    await wb.write(0x130, 0)          # OUT_SIZE = 0
    await wb.write(0x188, 0)          # PARAM_COUNT = 0
    await wb.write(0x118, 0x00)       # No fusion bits
    await wb.write(0x008, 0x01)       # IRQ_EN

    await wb.write(0x000, 0x01)  # START

    done = await wait_done(dut, max_cycles=50000)
    assert done, "Zero-DMA layer should still complete (no hang)"

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle"

    # No DMA transactions should have occurred
    assert mem.txn_count == 0, f"Expected 0 DMA txns, got {mem.txn_count}"

    mem_task.cancel()
    dut._log.info("PASS: All DMA sizes=0, layer completes with 0 DMA txns")
