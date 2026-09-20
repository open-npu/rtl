# Open-NPU RTL — cocotb Tests for npu_pe
# SPDX-License-Identifier: Apache-2.0
#
# The PE is a link in a partial-sum chain, not a self-contained accumulator:
#
#     psum_out = psum_in + act_in * weight_reg
#
# Accumulation over the k dimension is done by chaining PEs down a column
# inside npu_systolic, so there is no per-PE accumulator to drain and no
# left-to-right activation passthrough (activations are broadcast within a row).

import random

import cocotb
from cocotb.triggers import ReadOnly, RisingEdge, Timer

from utils.clock_reset import clock_reset
from utils.csim_ref import pe_mac_reference

# Mode encoding (matches npu_pe.v)
MODE_IDLE     = 0b00
MODE_WGT_LOAD = 0b01
MODE_COMPUTE  = 0b10

ACC_W = 44


def int8_to_unsigned(val):
    """Sign-extend an INT8 value onto the PE's 16-bit data port."""
    return val & 0xFFFF


def to_unsigned_acc(val):
    """Two's-complement encode a value onto the ACC_W-wide psum port."""
    return val & ((1 << ACC_W) - 1)


def signed_acc(val, bits=ACC_W):
    """Interpret an unsigned port value as signed with the given width."""
    val = int(val) & ((1 << bits) - 1)
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


async def load_weight(dut, weight):
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(weight)
    dut.act_in.value = 0
    dut.psum_in.value = 0
    await RisingEdge(dut.clk)
    dut.mode.value = MODE_IDLE
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)


async def mac(dut, act, psum_in=0):
    """One COMPUTE cycle; returns the registered psum_out."""
    dut.mode.value = MODE_COMPUTE
    dut.valid_in.value = 1
    dut.act_in.value = int8_to_unsigned(act)
    dut.psum_in.value = to_unsigned_acc(psum_in)
    await RisingEdge(dut.clk)
    dut.mode.value = MODE_IDLE
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)
    return signed_acc(dut.psum_out.value)


@cocotb.test()
async def test_weight_load(dut):
    """A loaded weight is used by the next COMPUTE."""
    await clock_reset(dut)
    await load_weight(dut, 42)
    got = await mac(dut, act=1, psum_in=0)
    assert got == 42, f"Expected 42, got {got}"


@cocotb.test()
async def test_single_mac(dut):
    """psum_out = act * weight when psum_in is zero."""
    await clock_reset(dut)
    await load_weight(dut, -3)
    got = await mac(dut, act=7, psum_in=0)
    assert got == -21, f"Expected -21, got {got}"


@cocotb.test()
async def test_psum_passthrough_add(dut):
    """psum_in is added to the product, not ignored."""
    await clock_reset(dut)
    await load_weight(dut, 5)
    got = await mac(dut, act=3, psum_in=1000)
    assert got == 1015, f"Expected 1000 + 3*5 = 1015, got {got}"


@cocotb.test()
async def test_negative_values(dut):
    """Sign handling across the INT8 extremes and a negative psum_in."""
    await clock_reset(dut)
    await load_weight(dut, -128)

    for act, psum_in in [(-1, 0), (127, -5000), (-128, 1 << 20)]:
        expect = psum_in + act * (-128)
        got = await mac(dut, act=act, psum_in=psum_in)
        assert got == expect, \
            f"act={act} psum_in={psum_in}: expected {expect}, got {got}"


@cocotb.test()
async def test_no_accumulation_across_cycles(dut):
    """The PE holds no running sum: each COMPUTE depends only on psum_in.

    This is what distinguishes the chained PE from the previous
    output-stationary one, which accumulated internally and needed a drain.
    """
    await clock_reset(dut)
    await load_weight(dut, 10)

    first = await mac(dut, act=5, psum_in=0)
    second = await mac(dut, act=5, psum_in=0)
    assert first == 50, f"Expected 50, got {first}"
    assert second == 50, \
        f"PE accumulated across cycles: expected 50, got {second}"


@cocotb.test()
async def test_idle_holds_psum(dut):
    """IDLE must not disturb the registered psum_out."""
    await clock_reset(dut)
    await load_weight(dut, 7)
    held = await mac(dut, act=6, psum_in=0)
    assert held == 42

    dut.mode.value = MODE_IDLE
    dut.valid_in.value = 0
    dut.act_in.value = int8_to_unsigned(100)
    dut.psum_in.value = to_unsigned_acc(999999)
    for _ in range(5):
        await RisingEdge(dut.clk)
    assert signed_acc(dut.psum_out.value) == 42, \
        f"psum_out changed while idle: {signed_acc(dut.psum_out.value)}"


@cocotb.test()
async def test_psum_valid(dut):
    """psum_valid_out tracks COMPUTE with valid_in, one cycle delayed."""
    await clock_reset(dut)
    await load_weight(dut, 3)

    dut.mode.value = MODE_COMPUTE
    dut.valid_in.value = 1
    dut.act_in.value = int8_to_unsigned(4)
    dut.psum_in.value = 0
    await RisingEdge(dut.clk)
    # Sample after the nonblocking updates have settled, not at the edge.
    await ReadOnly()
    assert dut.psum_valid_out.value == 1, "psum_valid_out not asserted"

    await Timer(1, unit="step")
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.psum_valid_out.value == 0, \
        "psum_valid_out stuck high after valid_in dropped"


@cocotb.test()
async def test_chain_dot_product(dut):
    """Feeding psum_out back as psum_in reproduces a full dot product.

    This mimics what a column of chained PEs does spatially: PE r consumes the
    partial sum from PE r-1. Doing it in time on a single PE lets us check the
    arithmetic against the same reference the array uses.
    """
    await clock_reset(dut)

    random.seed(42)
    weight = random.randint(-128, 127)
    activations = [random.randint(-128, 127) for _ in range(64)]
    expected = pe_mac_reference(activations, weight)

    await load_weight(dut, weight)

    psum = 0
    for act in activations:
        psum = await mac(dut, act=act, psum_in=psum)

    assert psum == expected, f"Expected {expected}, got {psum}"


@cocotb.test()
async def test_wide_psum(dut):
    """A psum_in near the ACC_W limit survives the add without truncation."""
    await clock_reset(dut)
    await load_weight(dut, 127)

    # Leave headroom for one 127*127 product below the 44-bit signed max.
    big = (1 << (ACC_W - 1)) - 1 - (127 * 127)
    got = await mac(dut, act=127, psum_in=big)
    assert got == big + 127 * 127, \
        f"Expected {big + 127*127}, got {got}"
