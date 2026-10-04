# SPDX-License-Identifier: Apache-2.0
"""Byte-enable and narrow-write regression for npu_sram_wide."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def write_a(dut, addr, value):
    dut.a_en.value = 1
    dut.a_we.value = 1
    dut.a_addr.value = addr
    dut.a_wdata.value = value
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def read_a(dut, addr):
    dut.a_en.value = 1
    dut.a_we.value = 0
    dut.a_addr.value = addr
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    return int(dut.a_rdata.value)


@cocotb.test()
async def test_byte_mask_preserves_unselected_bytes(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.a_en.value = 0
    dut.a_we.value = 0
    dut.b_en.value = 0
    dut.b_we.value = 0
    dut.b_wmask.value = 0
    dut.b_wdata.value = 0
    dut.b_wdata_wide.value = 0

    base = 16
    original = [(0x40302010 + 0x04040404 * word) & 0xFFFFFFFF
                for word in range(16)]
    for word, value in enumerate(original):
        await write_a(dut, base + word, value)

    wide = sum(((0x80 + byte) & 0xFF) << (8 * byte)
               for byte in range(64))
    selected = (1 << 3) | (1 << 4) | (1 << 31) | (1 << 63)
    dut.a_en.value = 0
    dut.b_en.value = 1
    dut.b_we.value = 0
    dut.b_addr.value = base
    dut.b_wmask.value = selected
    dut.b_wdata_wide.value = wide
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.b_en.value = 0

    original_bytes = b"".join(value.to_bytes(4, "little")
                              for value in original)
    expected = bytearray(original_bytes)
    for byte in (3, 4, 31, 63):
        expected[byte] = (0x80 + byte) & 0xFF
    for word in range(16):
        actual = await read_a(dut, base + word)
        want = int.from_bytes(expected[4 * word:4 * word + 4], "little")
        assert actual == want, (
            f"word {word}: got 0x{actual:08x}, expected 0x{want:08x}")

    # A zero mask must retain the legacy single-word B-port write path.
    dut.a_en.value = 0
    dut.b_en.value = 1
    dut.b_we.value = 1
    dut.b_addr.value = base + 5
    dut.b_wmask.value = 0
    dut.b_wdata.value = 0xDEADBEEF
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.b_en.value = 0
    assert await read_a(dut, base + 5) == 0xDEADBEEF
    assert await read_a(dut, base + 6) == int.from_bytes(
        expected[24:28], "little")
