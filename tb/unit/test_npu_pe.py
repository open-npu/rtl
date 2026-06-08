# Open-NPU RTL — cocotb Tests for npu_pe
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random

from utils.clock_reset import clock_reset
from utils.csim_ref import pe_mac_reference

# Mode encoding (matches npu_pe.v)
MODE_IDLE     = 0b00
MODE_WGT_LOAD = 0b01
MODE_COMPUTE  = 0b10
MODE_DRAIN    = 0b11


def int8_to_unsigned(val):
    """Convert signed int8 to unsigned 8-bit for driving DUT."""
    return val & 0xFF


def signed_acc(val, bits=40):
    """Interpret unsigned value as signed with given bit width."""
    val = int(val) & ((1 << bits) - 1)
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


@cocotb.test()
async def test_weight_load(dut):
    """Test: Load a weight value into the PE weight register."""
    await clock_reset(dut)

    # Load weight = 42
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(42)
    dut.act_in.value = 0
    await RisingEdge(dut.clk)

    # Back to idle
    dut.mode.value = MODE_IDLE
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)

    # Verify: compute with act=1 should give acc=42
    dut.mode.value = MODE_COMPUTE
    dut.valid_in.value = 1
    dut.act_in.value = int8_to_unsigned(1)
    await RisingEdge(dut.clk)

    dut.mode.value = MODE_IDLE
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)

    # Drain
    dut.mode.value = MODE_DRAIN
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)

    dut.valid_in.value = 0
    dut.mode.value = MODE_IDLE
    await RisingEdge(dut.clk)

    acc_val = signed_acc(dut.acc_out.value)
    assert acc_val == 42, f"Expected acc=42, got {acc_val}"
    dut._log.info(f"PASS: weight_load — acc={acc_val}")


@cocotb.test()
async def test_single_mac(dut):
    """Test: Single MAC operation (act * weight)."""
    await clock_reset(dut)

    weight = -3
    act = 7

    # Load weight
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(weight)
    dut.act_in.value = 0
    await RisingEdge(dut.clk)

    # Compute
    dut.mode.value = MODE_COMPUTE
    dut.act_in.value = int8_to_unsigned(act)
    await RisingEdge(dut.clk)

    # Drain
    dut.mode.value = MODE_DRAIN
    await RisingEdge(dut.clk)

    dut.valid_in.value = 0
    dut.mode.value = MODE_IDLE
    await RisingEdge(dut.clk)

    acc_val = signed_acc(dut.acc_out.value)
    expected = act * weight  # 7 * (-3) = -21
    assert acc_val == expected, f"Expected {expected}, got {acc_val}"
    dut._log.info(f"PASS: single_mac — {act}*{weight}={acc_val}")


@cocotb.test()
async def test_multi_mac_accumulate(dut):
    """Test: Multiple MACs accumulate correctly (dot product)."""
    await clock_reset(dut)

    weight = 5
    activations = [1, 2, 3, 4, 5, 6, 7, 8]
    expected = pe_mac_reference(activations, weight)

    # Load weight
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(weight)
    dut.act_in.value = 0
    await RisingEdge(dut.clk)

    # Stream activations
    dut.mode.value = MODE_COMPUTE
    for act in activations:
        dut.act_in.value = int8_to_unsigned(act)
        await RisingEdge(dut.clk)

    # Drain
    dut.mode.value = MODE_DRAIN
    await RisingEdge(dut.clk)

    dut.valid_in.value = 0
    dut.mode.value = MODE_IDLE
    await RisingEdge(dut.clk)

    acc_val = signed_acc(dut.acc_out.value)
    assert acc_val == expected, f"Expected {expected}, got {acc_val}"
    dut._log.info(f"PASS: multi_mac — dot([1..8], 5) = {acc_val}")


@cocotb.test()
async def test_negative_values(dut):
    """Test: MAC with negative activations and weights."""
    await clock_reset(dut)

    weight = -128  # min int8
    activations = [-1, -2, 127, -128]
    expected = pe_mac_reference(activations, weight)

    # Load weight
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(weight)
    dut.act_in.value = 0
    await RisingEdge(dut.clk)

    # Compute
    dut.mode.value = MODE_COMPUTE
    for act in activations:
        dut.act_in.value = int8_to_unsigned(act)
        await RisingEdge(dut.clk)

    # Drain
    dut.mode.value = MODE_DRAIN
    await RisingEdge(dut.clk)

    dut.valid_in.value = 0
    dut.mode.value = MODE_IDLE
    await RisingEdge(dut.clk)

    acc_val = signed_acc(dut.acc_out.value)
    assert acc_val == expected, f"Expected {expected}, got {acc_val}"
    dut._log.info(f"PASS: negative_values — acc={acc_val}")


@cocotb.test()
async def test_act_passthrough(dut):
    """Test: Activation passes through to act_out during COMPUTE."""
    await clock_reset(dut)

    weight = 1
    test_acts = [10, -20, 127, -128, 0]

    # Load weight
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(weight)
    dut.act_in.value = 0
    await RisingEdge(dut.clk)

    # Compute — check act_out follows act_in (1 cycle delayed)
    dut.mode.value = MODE_COMPUTE
    received = []
    for act in test_acts:
        dut.act_in.value = int8_to_unsigned(act)
        await RisingEdge(dut.clk)
        # Read act_out from previous cycle
        if dut.act_valid_out.value:
            out_raw = int(dut.act_out.value)
            out_signed = out_raw if out_raw < 128 else out_raw - 256
            received.append(out_signed)

    # One more cycle to capture last value
    dut.mode.value = MODE_IDLE
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)
    if dut.act_valid_out.value:
        out_raw = int(dut.act_out.value)
        out_signed = out_raw if out_raw < 128 else out_raw - 256
        received.append(out_signed)

    assert received == test_acts, f"act_out mismatch: expected {test_acts}, got {received}"
    dut._log.info(f"PASS: act_passthrough — {received}")


@cocotb.test()
async def test_drain_clears_acc(dut):
    """Test: Drain resets accumulator to zero."""
    await clock_reset(dut)

    weight = 10

    # Load weight
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(weight)
    dut.act_in.value = 0
    await RisingEdge(dut.clk)

    # Compute a few MACs
    dut.mode.value = MODE_COMPUTE
    for act in [5, 5, 5]:
        dut.act_in.value = int8_to_unsigned(act)
        await RisingEdge(dut.clk)

    # Drain
    dut.mode.value = MODE_DRAIN
    await RisingEdge(dut.clk)
    dut.mode.value = MODE_IDLE
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)

    first_drain = signed_acc(dut.acc_out.value)
    assert first_drain == 150, f"First drain: expected 150, got {first_drain}"

    # Now compute again — accumulator should start from 0
    dut.mode.value = MODE_COMPUTE
    dut.valid_in.value = 1
    dut.act_in.value = int8_to_unsigned(1)
    await RisingEdge(dut.clk)

    # Drain again
    dut.mode.value = MODE_DRAIN
    await RisingEdge(dut.clk)
    dut.mode.value = MODE_IDLE
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)

    second_drain = signed_acc(dut.acc_out.value)
    assert second_drain == 10, f"Second drain: expected 10, got {second_drain}"
    dut._log.info(f"PASS: drain_clears — first={first_drain}, second={second_drain}")


@cocotb.test()
async def test_random_dot_product(dut):
    """Test: Random dot product — compare RTL vs Python golden."""
    await clock_reset(dut)

    random.seed(42)
    weight = random.randint(-128, 127)
    length = 64  # Simulate C_in=64
    activations = [random.randint(-128, 127) for _ in range(length)]
    expected = pe_mac_reference(activations, weight)

    # Load weight
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(weight)
    dut.act_in.value = 0
    await RisingEdge(dut.clk)

    # Stream activations
    dut.mode.value = MODE_COMPUTE
    for act in activations:
        dut.act_in.value = int8_to_unsigned(act)
        await RisingEdge(dut.clk)

    # Drain
    dut.mode.value = MODE_DRAIN
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    dut.mode.value = MODE_IDLE
    await RisingEdge(dut.clk)

    acc_val = signed_acc(dut.acc_out.value)
    assert acc_val == expected, f"Random dot product: expected {expected}, got {acc_val}"
    dut._log.info(f"PASS: random_dot_product — weight={weight}, len={length}, acc={acc_val}")


@cocotb.test()
async def test_large_accumulation(dut):
    """Test: Large accumulation to verify no overflow in 40-bit acc."""
    await clock_reset(dut)

    # Worst case: 512 multiplications of 127*127 = 8,258,048
    # Fits easily in 40 bits (max ~549B)
    weight = 127
    activations = [127] * 512
    expected = pe_mac_reference(activations, weight)

    # Load weight
    dut.mode.value = MODE_WGT_LOAD
    dut.valid_in.value = 1
    dut.weight_in.value = int8_to_unsigned(weight)
    dut.act_in.value = 0
    await RisingEdge(dut.clk)

    # Stream 512 activations
    dut.mode.value = MODE_COMPUTE
    for act in activations:
        dut.act_in.value = int8_to_unsigned(act)
        await RisingEdge(dut.clk)

    # Drain
    dut.mode.value = MODE_DRAIN
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    dut.mode.value = MODE_IDLE
    await RisingEdge(dut.clk)

    acc_val = signed_acc(dut.acc_out.value)
    assert acc_val == expected, f"Large acc: expected {expected}, got {acc_val}"
    dut._log.info(f"PASS: large_accumulation — 512×(127×127)={acc_val}")
