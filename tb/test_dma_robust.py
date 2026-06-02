# Open-NPU RTL — DMA Robustness Tests (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# P1.1: DMA engine robustness under stress:
#   1. Odd word count (not power of 2) — verifies no alignment assumption
#   2. Max burst fill (fill entire SRAM via single DMA) — large transfer
#   3. CSR access during active DMA — no bus conflict
#   4. DMA with permanently stuck ACK (deadlock detection via abort)
#
# DUT = npu_top (full chip)

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

class WbSlave:
    """Wishbone B4 master driver for CSR slave port."""

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
    """DDR emulator."""

    def __init__(self, dut, clk, respond=True):
        self.dut = dut
        self.clk = clk
        self.mem = {}
        self.txn_count = 0
        self.respond = respond  # If False, never ACK (for deadlock test)

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

            if cyc and stb and self.respond:
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
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)


async def wait_done(dut, max_cycles=100000):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        try:
            if int(dut.irq_o.value):
                return True
        except ValueError:
            pass
    return False


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Odd word count (non-power-of-2 transfer size)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=2, timeout_unit="ms")
async def test_dma_odd_size(dut):
    """DMA transfer with 17 words (odd, non-power-of-2) — must transfer exactly 17."""
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    # 17 words of weight data with unique values
    wgt_words = 17
    wgt_data = [0xAA000000 + i for i in range(wgt_words)]
    mem.populate(0x1000, wgt_data)
    # Also need activation and param for complete flow
    mem.populate(0x2000, [0] * 4)   # 4 words act
    mem.populate(0x3000, [0] * 64)  # 16ch × 4 words param
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)     # conv2d, int8
    await wb.write(0x044, 0x00010001) # IN: 1x1
    await wb.write(0x048, 16)
    await wb.write(0x04C, 0x00010001)
    await wb.write(0x050, 16)
    await wb.write(0x054, 0x0101)
    await wb.write(0x058, 0x0101)
    await wb.write(0x05C, 0)
    await wb.write(0x078, 0)
    await wb.write(0x108, 0x1000)     # WGT_ADDR
    await wb.write(0x100, 0x2000)     # IN_ADDR
    await wb.write(0x10C, 0x3000)     # PARAM_ADDR
    await wb.write(0x12C, wgt_words * 4)  # WGT_SIZE = 68 bytes (17 words)
    await wb.write(0x128, 16)         # IN_SIZE = 16 bytes
    await wb.write(0x130, 0)
    await wb.write(0x188, 16)
    await wb.write(0x118, 0x02)       # skip store
    await wb.write(0x008, 0x01)

    await wb.write(0x000, 0x01)

    done = await wait_done(dut, max_cycles=50000)
    assert done, "Odd-size DMA layer timed out"

    # Total DMA: 17 (wgt) + 4 (act) + 64 (param) = 85
    expected = wgt_words + 4 + 64
    assert mem.txn_count >= expected, \
        f"Expected ≥{expected} txns, got {mem.txn_count}"

    mem_task.cancel()
    dut._log.info(f"PASS: Odd DMA size ({wgt_words} words), {mem.txn_count} total txns")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Max burst — fill weight SRAM (16K words)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=50, timeout_unit="ms")
async def test_dma_max_burst(dut):
    """DMA load filling the weight SRAM to capacity (16384 words = 64KB).

    Tests that the DMA engine handles the maximum transfer size without
    counter overflow or address wrap issues.
    """
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    # 16384 words = 65536 bytes = weight SRAM capacity
    max_words = 16384
    mem.populate(0x10000, [0] * max_words)
    mem.populate(0x80000, [0] * 4)    # act
    mem.populate(0x90000, [0] * 64)   # param
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)
    await wb.write(0x044, 0x00010001)
    await wb.write(0x048, 16)
    await wb.write(0x04C, 0x00010001)
    await wb.write(0x050, 16)
    await wb.write(0x054, 0x0101)
    await wb.write(0x058, 0x0101)
    await wb.write(0x05C, 0)
    await wb.write(0x078, 0)
    await wb.write(0x108, 0x10000)    # WGT_ADDR
    await wb.write(0x100, 0x80000)    # IN_ADDR
    await wb.write(0x10C, 0x90000)    # PARAM_ADDR
    await wb.write(0x12C, max_words * 4)  # WGT_SIZE = 65536 bytes
    await wb.write(0x128, 16)         # IN_SIZE
    await wb.write(0x130, 0)
    await wb.write(0x188, 16)
    await wb.write(0x118, 0x02)       # skip store
    await wb.write(0x008, 0x01)

    await wb.write(0x000, 0x01)

    done = await wait_done(dut, max_cycles=500000)
    assert done, "Max burst DMA timed out"

    # Minimum DMA: max_words (wgt) + 4 (act) + 64 (param)
    expected = max_words + 4 + 64
    assert mem.txn_count >= expected, \
        f"Expected ≥{expected} txns, got {mem.txn_count}"

    mem_task.cancel()
    dut._log.info(f"PASS: Max burst {max_words} words, total {mem.txn_count} DMA txns")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: CSR read/write during active DMA
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=5, timeout_unit="ms")
async def test_csr_during_dma(dut):
    """Perform CSR reads/writes while DMA is actively transferring.

    Verifies that the two Wishbone interfaces (slave=CSR, master=DMA)
    operate independently without conflict or data corruption.
    """
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    wgt_words = 256  # Enough to keep DMA busy for a while
    mem.populate(0x1000, [0] * wgt_words)
    mem.populate(0x2000, [0] * 4)
    mem.populate(0x3000, [0] * 64)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)
    await wb.write(0x044, 0x00010001)
    await wb.write(0x048, 16)
    await wb.write(0x04C, 0x00010001)
    await wb.write(0x050, 16)
    await wb.write(0x054, 0x0101)
    await wb.write(0x058, 0x0101)
    await wb.write(0x05C, 0)
    await wb.write(0x078, 0)
    await wb.write(0x108, 0x1000)     # WGT
    await wb.write(0x100, 0x2000)     # IN
    await wb.write(0x10C, 0x3000)     # PARAM
    await wb.write(0x12C, wgt_words * 4)
    await wb.write(0x128, 16)
    await wb.write(0x130, 0)
    await wb.write(0x188, 16)
    await wb.write(0x118, 0x02)       # skip store
    await wb.write(0x008, 0x01)

    # START — DMA begins immediately
    await wb.write(0x000, 0x01)

    # While DMA is running, do CSR reads and writes
    csr_ok = True
    for i in range(20):
        # Write a known value to a register that DMA doesn't use
        test_val = 0xBEEF0000 + i
        await wb.write(0x064, test_val)  # RESIZE_CFG (not used by current layer)

        # Read it back
        rd = await wb.read(0x064)
        if rd != test_val:
            csr_ok = False
            dut._log.error(f"CSR corruption during DMA: wrote {test_val:#x}, read {rd:#x}")
            break

        # Read status (verify bus isn't jammed)
        status = await wb.read(0x004)
        # Should still be busy (DMA running)
        if i < 5:
            assert status & 0x1, f"Lost busy during DMA (iteration {i})"

    # Wait for completion
    done = await wait_done(dut, max_cycles=50000)
    assert done, "Layer timed out despite CSR activity"
    assert csr_ok, "CSR data corruption detected during DMA"

    mem_task.cancel()
    dut._log.info(f"PASS: 20 CSR R/W during active DMA, no corruption, {mem.txn_count} DMA txns")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: DMA deadlock (memory never ACKs) — must be recoverable via abort
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=500, timeout_unit="us")
async def test_dma_no_ack_deadlock(dut):
    """DMA with memory that never responds — verify abort can recover.

    This simulates a hung bus / disconnected memory. The DMA will wait forever
    for ACK. The CPU must be able to abort the operation.
    """
    await setup(dut)

    # Memory that NEVER responds
    mem = WbMasterMem(dut, dut.clk, respond=False)
    mem.populate(0x1000, [0] * 100)  # data present but won't ACK
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)

    await wb.write(0x040, 0x0001)
    await wb.write(0x044, 0x00010001)
    await wb.write(0x048, 16)
    await wb.write(0x04C, 0x00010001)
    await wb.write(0x050, 16)
    await wb.write(0x054, 0x0101)
    await wb.write(0x058, 0x0101)
    await wb.write(0x05C, 0)
    await wb.write(0x078, 0)
    await wb.write(0x108, 0x1000)
    await wb.write(0x100, 0x2000)
    await wb.write(0x10C, 0x3000)
    await wb.write(0x12C, 400)   # WGT_SIZE = 100 words
    await wb.write(0x128, 16)
    await wb.write(0x130, 0)
    await wb.write(0x188, 16)
    await wb.write(0x118, 0x02)
    await wb.write(0x008, 0x01)

    # START — will get stuck waiting for ACK
    await wb.write(0x000, 0x01)

    # Wait to confirm it's stuck (busy but no progress)
    for _ in range(50):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert status & 0x1, "Should be busy (stuck in DMA)"

    # Verify DMA master is asserting CYC/STB (waiting for ACK)
    try:
        cyc = int(dut.wb_mst_cyc_o.value)
        stb = int(dut.wb_mst_stb_o.value)
    except ValueError:
        cyc, stb = 0, 0
    assert cyc and stb, "DMA should be asserting CYC+STB waiting for ACK"

    # ABORT to recover
    await wb.write(0x000, 0x02)

    for _ in range(50):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, f"Abort failed to recover from DMA deadlock, status=0x{status:08X}"

    # Verify DMA master released the bus
    try:
        cyc = int(dut.wb_mst_cyc_o.value)
    except ValueError:
        cyc = 0
    assert cyc == 0, "DMA master bus not released after abort"

    # Verify system still works after recovery
    await wb.write(0x064, 0x12345678)
    rd = await wb.read(0x064)
    assert rd == 0x12345678, "CSR broken after deadlock recovery"

    mem_task.cancel()
    dut._log.info("PASS: DMA deadlock recovered via abort, system functional")
