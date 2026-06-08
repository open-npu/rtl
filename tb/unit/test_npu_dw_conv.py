# Open-NPU RTL — cocotb Tests for npu_dw_conv (Depthwise Convolution)
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer
import random


def int8_to_bits(val):
    """Convert signed int8 to unsigned 8-bit representation."""
    return val & 0xFF


def signed_acc(val, bits=40):
    """Convert unsigned accumulator reading to signed Python int."""
    val = int(val) & ((1 << bits) - 1)
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


async def init_dut(dut, kh=3, kw=3):
    """Initialize DUT: clock, reset, configure kernel size."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    dut.kernel_h.value = kh
    dut.kernel_w.value = kw
    dut.wgt_load.value = 0
    dut.wgt_valid.value = 0
    dut.wgt_data.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.acc_clear.value = 0

    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, weights):
    """Load kernel weights into the DW conv module."""
    dut.wgt_load.value = 1
    dut.acc_clear.value = 1  # Reset weight index
    await RisingEdge(dut.clk)
    dut.acc_clear.value = 0

    for w in weights:
        dut.wgt_valid.value = 1
        dut.wgt_data.value = int8_to_bits(w)
        await RisingEdge(dut.clk)

    dut.wgt_valid.value = 0
    dut.wgt_load.value = 0
    await RisingEdge(dut.clk)


async def compute_pixel(dut, inputs, clear=True):
    """Feed one pixel's worth of input data and return accumulator output."""
    if clear:
        dut.acc_clear.value = 1

    dut.in_valid.value = 1
    dut.in_data.value = int8_to_bits(inputs[0])
    await RisingEdge(dut.clk)
    dut.acc_clear.value = 0

    for i in range(1, len(inputs)):
        dut.in_data.value = int8_to_bits(inputs[i])
        await RisingEdge(dut.clk)

    dut.in_valid.value = 0
    # Output should be valid on the last input cycle
    await ReadOnly()
    valid = int(dut.out_valid.value)
    result = signed_acc(dut.acc_out.value)
    await Timer(1, unit="step")
    return result, valid


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: 1x1 kernel (trivial multiply)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_1x1_kernel(dut):
    """1x1 kernel: output = weight * input."""
    await init_dut(dut, kh=1, kw=1)
    await load_weights(dut, [3])

    result, valid = await compute_pixel(dut, [7])
    assert valid == 1, "out_valid not asserted"
    assert result == 3 * 7, f"1x1: got {result}, expected {3*7}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: 3x3 kernel with known values
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_3x3_kernel(dut):
    """3x3 kernel: verify dot product of 9 elements."""
    await init_dut(dut, kh=3, kw=3)
    weights = [1, 2, 1, 0, 0, 0, -1, -2, -1]  # Sobel-like
    inputs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    expected = sum(w * x for w, x in zip(weights, inputs))

    await load_weights(dut, weights)
    result, valid = await compute_pixel(dut, inputs)
    assert valid == 1, "out_valid not asserted for 3x3"
    assert result == expected, f"3x3: got {result}, expected {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Negative weights and inputs
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_negative_values(dut):
    """Verify correct signed arithmetic with negative values."""
    await init_dut(dut, kh=1, kw=3)
    weights = [-5, 10, -3]
    inputs = [4, -6, 8]
    expected = (-5) * 4 + 10 * (-6) + (-3) * 8  # = -20 -60 -24 = -104

    await load_weights(dut, weights)
    result, valid = await compute_pixel(dut, inputs)
    assert valid == 1, "out_valid not asserted"
    assert result == expected, f"Negative: got {result}, expected {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Multiple output pixels (streaming)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_streaming_pixels(dut):
    """Compute multiple output pixels back-to-back."""
    await init_dut(dut, kh=1, kw=3)
    weights = [1, 1, 1]
    await load_weights(dut, weights)

    # 3 output pixels
    test_inputs = [
        [10, 20, 30],   # expected: 60
        [-5, 5, -5],    # expected: -5
        [100, -100, 50] # expected: 50
    ]

    for i, inp in enumerate(test_inputs):
        result, valid = await compute_pixel(dut, inp, clear=True)
        expected = sum(w * x for w, x in zip(weights, inp))
        assert valid == 1, f"Pixel {i}: out_valid not asserted"
        assert result == expected, f"Pixel {i}: got {result}, expected {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: 5x5 kernel
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_5x5_kernel(dut):
    """5x5 kernel: verify larger MAC accumulation."""
    await init_dut(dut, kh=5, kw=5)
    random.seed(55)
    weights = [random.randint(-5, 5) for _ in range(25)]
    inputs = [random.randint(-10, 10) for _ in range(25)]
    expected = sum(w * x for w, x in zip(weights, inputs))

    await load_weights(dut, weights)
    result, valid = await compute_pixel(dut, inputs)
    assert valid == 1, "5x5: out_valid not asserted"
    assert result == expected, f"5x5: got {result}, expected {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: 7x7 kernel (max size)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_7x7_kernel(dut):
    """7x7 kernel (max supported): verify 49-element MAC."""
    await init_dut(dut, kh=7, kw=7)
    random.seed(77)
    weights = [random.randint(-3, 3) for _ in range(49)]
    inputs = [random.randint(-20, 20) for _ in range(49)]
    expected = sum(w * x for w, x in zip(weights, inputs))

    await load_weights(dut, weights)
    result, valid = await compute_pixel(dut, inputs)
    assert valid == 1, "7x7: out_valid not asserted"
    assert result == expected, f"7x7: got {result}, expected {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Accumulator clear between pixels
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_acc_clear(dut):
    """Verify accumulator is properly cleared between pixels."""
    await init_dut(dut, kh=1, kw=2)
    weights = [1, 1]
    await load_weights(dut, weights)

    # First pixel
    result1, _ = await compute_pixel(dut, [50, 50], clear=True)
    assert result1 == 100, f"Pixel 1: got {result1}, expected 100"

    # Second pixel (accumulator should be fresh)
    result2, _ = await compute_pixel(dut, [10, 20], clear=True)
    assert result2 == 30, f"Pixel 2: got {result2}, expected 30 (acc not cleared?)"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Random stress test
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_random_stress(dut):
    """Random 3x3 computations to verify correctness."""
    await init_dut(dut, kh=3, kw=3)
    random.seed(88)

    weights = [random.randint(-128, 127) for _ in range(9)]
    await load_weights(dut, weights)

    for trial in range(10):
        inputs = [random.randint(-128, 127) for _ in range(9)]
        expected = sum(w * x for w, x in zip(weights, inputs))
        result, valid = await compute_pixel(dut, inputs, clear=True)
        assert valid == 1, f"Trial {trial}: out_valid not asserted"
        assert result == expected, f"Trial {trial}: got {result}, expected {expected}"
