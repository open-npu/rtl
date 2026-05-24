# Open-NPU RTL — cocotb Tests for npu_sram (Dual-Port SRAM)
# SPDX-License-Identifier: Apache-2.0
#
# npu_sram has synchronous reads with 1-cycle latency.
# Data appears on the output register one clock AFTER the address is presented.

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer
import random


async def init_dut(dut):
    """Start clock and initialize all inputs to 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.a_en.value = 0
    dut.a_we.value = 0
    dut.a_addr.value = 0
    dut.a_wdata.value = 0
    dut.b_en.value = 0
    dut.b_we.value = 0
    dut.b_addr.value = 0
    dut.b_wdata.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)


async def sram_write_a(dut, addr, data):
    """Write one word via Port A (takes 1 cycle)."""
    dut.a_en.value = 1
    dut.a_we.value = 1
    dut.a_addr.value = addr
    dut.a_wdata.value = data
    await RisingEdge(dut.clk)
    dut.a_we.value = 0


async def sram_read_a(dut, addr):
    """Read one word via Port A. Returns data (takes 1 cycle for address, 1 to capture)."""
    dut.a_en.value = 1
    dut.a_we.value = 0
    dut.a_addr.value = addr
    await RisingEdge(dut.clk)
    await ReadOnly()  # Wait for NBA region to complete
    val = int(dut.a_rdata.value)
    await Timer(1, unit="step")  # Exit ReadOnly phase
    return val


async def sram_write_b(dut, addr, data):
    """Write one word via Port B (takes 1 cycle)."""
    dut.b_en.value = 1
    dut.b_we.value = 1
    dut.b_addr.value = addr
    dut.b_wdata.value = data
    await RisingEdge(dut.clk)
    dut.b_we.value = 0


async def sram_read_b(dut, addr):
    """Read one word via Port B. Returns data."""
    dut.b_en.value = 1
    dut.b_we.value = 0
    dut.b_addr.value = addr
    await RisingEdge(dut.clk)
    await ReadOnly()
    val = int(dut.b_rdata.value)
    await Timer(1, unit="step")
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Basic Port A write then read
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_port_a_write_read(dut):
    """Write data via Port A, then read it back via Port A."""
    await init_dut(dut)

    await sram_write_a(dut, 5, 0xDEADBEEF)
    val = await sram_read_a(dut, 5)
    assert val == 0xDEADBEEF, f"Port A read: got {val:#010x}, expected 0xDEADBEEF"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Basic Port B write then read
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_port_b_write_read(dut):
    """Write data via Port B, then read it back via Port B."""
    await init_dut(dut)

    await sram_write_b(dut, 10, 0xCAFEBABE)
    val = await sram_read_b(dut, 10)
    assert val == 0xCAFEBABE, f"Port B read: got {val:#010x}, expected 0xCAFEBABE"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Cross-port access (A writes, B reads)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_cross_port_a_write_b_read(dut):
    """Write via Port A, read back via Port B."""
    await init_dut(dut)

    await sram_write_a(dut, 7, 0x12345678)
    dut.a_en.value = 0
    val = await sram_read_b(dut, 7)
    assert val == 0x12345678, f"Cross-port B read: got {val:#010x}, expected 0x12345678"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Back-to-back writes and reads
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_back_to_back(dut):
    """Consecutive writes followed by consecutive reads."""
    await init_dut(dut)

    N = 16
    random.seed(100)
    data = [random.randint(0, 0xFFFFFFFF) for _ in range(N)]

    # Write N words via Port A
    dut.a_en.value = 1
    dut.a_we.value = 1
    for i in range(N):
        dut.a_addr.value = i
        dut.a_wdata.value = data[i]
        await RisingEdge(dut.clk)

    # Read them back via Port A
    # Synchronous read: set addr before edge, data available after edge
    dut.a_we.value = 0
    for i in range(N):
        dut.a_addr.value = i
        await RisingEdge(dut.clk)
        await ReadOnly()
        val = int(dut.a_rdata.value)
        assert val == data[i], \
            f"Back-to-back[{i}]: got {val:#010x}, expected {data[i]:#010x}"
        await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Enable gating — no access when en=0
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_enable_gating(dut):
    """When enable is low, no write should occur."""
    await init_dut(dut)

    # Write some data first
    await sram_write_a(dut, 3, 0xAAAA_AAAA)

    # Try to write with en=0
    dut.a_en.value = 0
    dut.a_we.value = 1
    dut.a_addr.value = 3
    dut.a_wdata.value = 0xBBBB_BBBB
    await RisingEdge(dut.clk)

    # Read back — should still be original value
    val = await sram_read_a(dut, 3)
    assert val == 0xAAAA_AAAA, f"Enable gating failed: got {val:#010x}, expected 0xAAAAAAAA"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Simultaneous dual-port operation
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_simultaneous_dual_port(dut):
    """Port A and Port B write simultaneously to different addresses, then cross-read."""
    await init_dut(dut)

    # Cycle 1: Port A writes addr 0, Port B writes addr 100
    dut.a_en.value = 1
    dut.a_we.value = 1
    dut.a_addr.value = 0
    dut.a_wdata.value = 0x1111_1111
    dut.b_en.value = 1
    dut.b_we.value = 1
    dut.b_addr.value = 100
    dut.b_wdata.value = 0x2222_2222
    await RisingEdge(dut.clk)

    # Cycle 2: Port A reads addr 100, Port B reads addr 0
    dut.a_we.value = 0
    dut.a_addr.value = 100
    dut.b_we.value = 0
    dut.b_addr.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()

    val_a = int(dut.a_rdata.value)
    val_b = int(dut.b_rdata.value)
    await Timer(1, unit="step")
    assert val_a == 0x2222_2222, f"Port A read addr 100: got {val_a:#010x}"
    assert val_b == 0x1111_1111, f"Port B read addr 0: got {val_b:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Boundary addresses (first and last)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_boundary_addresses(dut):
    """Write/read at address 0 and max address (DEPTH-1=1023)."""
    await init_dut(dut)
    MAX_ADDR = 1023  # DEPTH=1024 default

    await sram_write_a(dut, 0, 0xFFFF_0000)
    await sram_write_a(dut, MAX_ADDR, 0x0000_FFFF)

    val0 = await sram_read_a(dut, 0)
    assert val0 == 0xFFFF_0000, f"Addr 0: got {val0:#010x}"

    val_max = await sram_read_a(dut, MAX_ADDR)
    assert val_max == 0x0000_FFFF, f"Addr {MAX_ADDR}: got {val_max:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Random stress test
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_random_stress(dut):
    """Random reads and writes from both ports to verify data integrity."""
    await init_dut(dut)
    random.seed(42)

    # Reference model
    ref = {}
    N_OPS = 64

    # Fill with random writes via Port A
    dut.a_en.value = 1
    dut.a_we.value = 1
    for _ in range(N_OPS):
        addr = random.randint(0, 255)
        data = random.randint(0, 0xFFFFFFFF)
        dut.a_addr.value = addr
        dut.a_wdata.value = data
        ref[addr] = data
        await RisingEdge(dut.clk)

    dut.a_en.value = 0

    # Verify a subset via Port B reads
    addrs = list(ref.keys())
    random.shuffle(addrs)
    for addr in addrs[:32]:
        val = await sram_read_b(dut, addr)
        assert val == ref[addr], \
            f"Stress addr {addr}: got {val:#010x}, expected {ref[addr]:#010x}"
