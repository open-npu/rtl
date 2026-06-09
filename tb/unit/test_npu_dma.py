# Open-NPU RTL — cocotb Tests for npu_dma (DMA Engine)
# SPDX-License-Identifier: Apache-2.0
#
# Tests use a simulated Wishbone slave (memory) to verify DMA transfers.

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer
import random


class WbSlaveMemory:
    """Simple Wishbone slave memory model for DMA testing.
    Responds with ack after 1 cycle (single-cycle slave).
    """

    def __init__(self, dut, size_words=256):
        self.dut = dut
        self.mem = {}
        self.size_words = size_words

    async def run(self):
        """Background task: respond to Wishbone transactions."""
        while True:
            await RisingEdge(self.dut.clk)
            if int(self.dut.wb_cyc_o.value) and int(self.dut.wb_stb_o.value):
                addr = int(self.dut.wb_adr_o.value)
                if int(self.dut.wb_we_o.value):
                    # Write
                    self.mem[addr] = int(self.dut.wb_dat_o.value)
                else:
                    # Read
                    self.dut.wb_dat_i.value = self.mem.get(addr, 0)
                self.dut.wb_ack_i.value = 1
            else:
                self.dut.wb_ack_i.value = 0


async def init_dut(dut):
    """Initialize DUT signals and start clock."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    dut.start.value = 0
    dut.abort.value = 0
    dut.dir.value = 0
    dut.ext_addr.value = 0
    dut.sram_addr.value = 0
    dut.xfer_len.value = 0
    dut.burst_cfg.value = 0
    dut.cfg_in_stride.value = 0
    dut.cfg_out_stride.value = 0
    dut.wb_ack_i.value = 0
    dut.wb_dat_i.value = 0
    dut.sram_rdata.value = 0

    # Reset
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Start WB slave
    wb_mem = WbSlaveMemory(dut)
    cocotb.start_soon(wb_mem.run())
    return wb_mem


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Single word LOAD (MEM → SRAM)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_single_load(dut):
    """DMA loads a single word from external memory to SRAM."""
    wb_mem = await init_dut(dut)
    wb_mem.mem[0x1000] = 0xDEADBEEF

    # Configure transfer
    dut.dir.value = 0           # Load
    dut.ext_addr.value = 0x1000
    dut.sram_addr.value = 0
    dut.xfer_len.value = 1

    # Start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    for _ in range(20):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")
    else:
        assert False, "DMA did not complete within timeout"

    await Timer(1, unit="step")
    assert int(dut.busy.value) == 0, "DMA still busy after done"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Multi-word LOAD
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_multi_load(dut):
    """DMA loads multiple words from external memory to SRAM."""
    wb_mem = await init_dut(dut)
    N = 8
    base_addr = 0x2000
    for i in range(N):
        wb_mem.mem[base_addr + i * 4] = 0x1000_0000 + i

    # Configure
    dut.dir.value = 0
    dut.ext_addr.value = base_addr
    dut.sram_addr.value = 0
    dut.xfer_len.value = N

    # Start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    for _ in range(100):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")
    else:
        assert False, "DMA multi-load did not complete"

    await Timer(1, unit="step")
    # Verify transfer count
    cnt = int(dut.xfer_count.value)
    assert cnt == N, f"xfer_count: got {cnt}, expected {N}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Single word STORE (SRAM → MEM)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_single_store(dut):
    """DMA stores a single word from SRAM to external memory."""
    wb_mem = await init_dut(dut)

    # Pre-load SRAM read data (will be driven when DMA reads SRAM)
    sram_data = 0xCAFEBABE

    # We need to respond to SRAM reads. Since the DMA reads sram_rdata,
    # we'll drive it based on sram_en/sram_addr_o in a background task.
    async def sram_responder():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.sram_en.value) and not int(dut.sram_we.value):
                await Timer(1, unit="step")
                dut.sram_rdata.value = sram_data

    cocotb.start_soon(sram_responder())

    # Configure store
    dut.dir.value = 1             # Store
    dut.ext_addr.value = 0x3000
    dut.sram_addr.value = 0
    dut.xfer_len.value = 1

    # Start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    for _ in range(20):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")
    else:
        assert False, "DMA store did not complete"

    await Timer(1, unit="step")
    # Verify data appeared in WB memory
    assert 0x3000 in wb_mem.mem, "DMA did not write to WB memory"
    assert wb_mem.mem[0x3000] == sram_data, \
        f"WB mem[0x3000]: got {wb_mem.mem[0x3000]:#010x}, expected {sram_data:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: LOAD with SRAM write verification
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_load_sram_write_data(dut):
    """Verify the data written to SRAM during a LOAD operation."""
    wb_mem = await init_dut(dut)
    wb_mem.mem[0x4000] = 0xABCD_1234
    wb_mem.mem[0x4004] = 0x5678_9ABC

    # Capture SRAM writes
    sram_writes = []

    async def sram_write_monitor():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.sram_en.value) and int(dut.sram_we.value):
                addr = int(dut.sram_addr_o.value)
                data = int(dut.sram_wdata.value)
                sram_writes.append((addr, data))
            await Timer(1, unit="step")

    cocotb.start_soon(sram_write_monitor())

    # Configure
    dut.dir.value = 0
    dut.ext_addr.value = 0x4000
    dut.sram_addr.value = 10  # Start writing at SRAM addr 10
    dut.xfer_len.value = 2

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    for _ in range(30):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")
    else:
        assert False, "DMA load did not complete"

    await Timer(1, unit="step")
    # Verify SRAM writes
    assert len(sram_writes) == 2, f"Expected 2 SRAM writes, got {len(sram_writes)}"
    assert sram_writes[0] == (10, 0xABCD_1234), f"Write 0: {sram_writes[0]}"
    assert sram_writes[1] == (11, 0x5678_9ABC), f"Write 1: {sram_writes[1]}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Busy signal behavior
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_busy_signal(dut):
    """Busy should be asserted during transfer and deasserted after done."""
    wb_mem = await init_dut(dut)
    wb_mem.mem[0x5000] = 0x1111_1111

    assert int(dut.busy.value) == 0, "Busy set before start"

    dut.dir.value = 0
    dut.ext_addr.value = 0x5000
    dut.sram_addr.value = 0
    dut.xfer_len.value = 1
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Next cycle should be busy
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.busy.value) == 1, "Not busy during transfer"
    await Timer(1, unit="step")

    # Wait for done
    for _ in range(20):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")

    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.busy.value) == 0, "Still busy after done"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Abort during transfer
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_abort(dut):
    """Abort should stop the transfer and return to idle."""
    wb_mem = await init_dut(dut)
    # Load 100 words
    for i in range(100):
        wb_mem.mem[0x6000 + i * 4] = i

    dut.dir.value = 0
    dut.ext_addr.value = 0x6000
    dut.sram_addr.value = 0
    dut.xfer_len.value = 100
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Let it run a few cycles then abort
    for _ in range(10):
        await RisingEdge(dut.clk)

    dut.abort.value = 1
    await RisingEdge(dut.clk)
    dut.abort.value = 0

    # Wait for done pulse
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")

    await RisingEdge(dut.clk)
    await ReadOnly()
    # Should be idle and not busy
    assert int(dut.busy.value) == 0, "Still busy after abort"
    # Transfer should not be complete
    cnt = int(dut.xfer_count.value)
    assert cnt < 100, f"Abort didn't stop early: xfer_count={cnt}"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Zero-length transfer (no-op)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_zero_length(dut):
    """Zero-length transfer should not assert busy or produce done."""
    await init_dut(dut)

    dut.dir.value = 0
    dut.ext_addr.value = 0x7000
    dut.sram_addr.value = 0
    dut.xfer_len.value = 0  # Zero length
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait a few cycles — should remain idle
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert int(dut.busy.value) == 0, "Busy asserted for zero-length"
        await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Address auto-increment
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_address_increment(dut):
    """Verify external address increments by 4 and SRAM addr by 1 per word."""
    wb_mem = await init_dut(dut)
    N = 4
    base = 0x8000
    for i in range(N):
        wb_mem.mem[base + i * 4] = 0xA0 + i

    # Monitor WB addresses
    wb_addrs = []

    async def wb_addr_monitor():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.wb_cyc_o.value) and int(dut.wb_stb_o.value):
                wb_addrs.append(int(dut.wb_adr_o.value))
            await Timer(1, unit="step")

    cocotb.start_soon(wb_addr_monitor())

    dut.dir.value = 0
    dut.ext_addr.value = base
    dut.sram_addr.value = 20
    dut.xfer_len.value = N
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    for _ in range(50):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")

    await Timer(1, unit="step")
    # Check addresses were incrementing by 4
    unique_addrs = sorted(set(wb_addrs))
    expected = [base + i * 4 for i in range(N)]
    assert unique_addrs == expected, f"WB addrs: {[hex(a) for a in unique_addrs]}, expected {[hex(a) for a in expected]}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: LOAD with non-zero stride
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_load_stride(dut):
    """DMA loads words with cfg_in_stride=8 (every other word in external mem)."""
    wb_mem = await init_dut(dut)
    N = 4
    base = 0xA000
    for i in range(N):
        wb_mem.mem[base + i * 8] = 0xF000 + i  # stride=8: data at +0, +8, +16, +24

    # Monitor WB addresses
    wb_addrs = []

    async def wb_addr_monitor():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.wb_cyc_o.value) and int(dut.wb_stb_o.value) and int(dut.wb_ack_i.value):
                wb_addrs.append(int(dut.wb_adr_o.value))
            await Timer(1, unit="step")

    cocotb.start_soon(wb_addr_monitor())

    dut.dir.value = 0
    dut.ext_addr.value = base
    dut.sram_addr.value = 0
    dut.xfer_len.value = N
    dut.cfg_in_stride.value = 8  # stride=8 bytes

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(50):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")

    await Timer(1, unit="step")
    # Verify addresses: should be base, base+8, base+16, base+24
    expected_addrs = [base + i * 8 for i in range(N)]
    assert wb_addrs == expected_addrs, \
        f"WB addrs with stride=8: {[hex(a) for a in wb_addrs]}, expected {[hex(a) for a in expected_addrs]}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: STORE with non-zero stride
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_store_stride(dut):
    """DMA stores words with cfg_out_stride=16 (every 16th byte in external mem)."""
    wb_mem = await init_dut(dut)
    N = 3
    sram_data = 0xBEEF_0000

    async def sram_responder():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.sram_en.value) and not int(dut.sram_we.value):
                await Timer(1, unit="step")
                addr = int(dut.sram_addr_o.value)
                dut.sram_rdata.value = sram_data + addr

    cocotb.start_soon(sram_responder())

    # Monitor WB addresses + data
    wb_writes = []

    async def wb_write_monitor():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.wb_cyc_o.value) and int(dut.wb_stb_o.value) and int(dut.wb_we_o.value) and int(dut.wb_ack_i.value):
                wb_writes.append((int(dut.wb_adr_o.value), int(dut.wb_dat_o.value)))
            await Timer(1, unit="step")

    cocotb.start_soon(wb_write_monitor())

    dut.dir.value = 1
    dut.ext_addr.value = 0xB000
    dut.sram_addr.value = 0
    dut.xfer_len.value = N
    dut.cfg_out_stride.value = 16  # stride=16 bytes

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(50):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done_pulse.value):
            break
        await Timer(1, unit="step")

    await Timer(1, unit="step")
    # Verify addresses: B000, B010, B020
    expected = [(0xB000 + i * 16, sram_data + i) for i in range(N)]
    assert wb_writes == expected, \
        f"WB writes with stride=16: {wb_writes}, expected {expected}"
