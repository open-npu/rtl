# Open-NPU RTL — cocotb Tests for npu_ppu (Post-Processing Unit)
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

from utils.clock_reset import clock_reset
from utils.csim_ref import ppu_reference

# ─── Constants ───
MODE_CONV_REQ    = 0b00
MODE_RELU_ONLY   = 0b10
MODE_PASSTHROUGH = 0b11

PIPELINE_LATENCY = 4  # 4 register stages (s1, s2, s3, out)


async def reset_dut(dut):
    """Apply clock + reset."""
    await clock_reset(dut)
    dut.mode.value = MODE_CONV_REQ
    dut.relu_en.value = 0
    dut.bias_en.value = 1
    dut.zp_en.value = 1
    dut.acc_in.value = 0
    dut.in_valid.value = 0
    dut.bias.value = 0
    dut.mult_m.value = 1
    dut.shift_s.value = 0
    dut.zero_point.value = 0
    await ClockCycles(dut.clk, 2)


def to_signed(val, bits):
    """Convert Python int to 2's complement unsigned for cocotb."""
    if val < 0:
        return val + (1 << bits)
    return val & ((1 << bits) - 1)


def from_signed(val, bits):
    """Convert unsigned cocotb value to Python signed int."""
    val = int(val) & ((1 << bits) - 1)
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


async def push_and_wait(dut, acc, bias, M, S, zp, mode=MODE_CONV_REQ,
                        relu_en=False, bias_en=True, zp_en=True):
    """Push one value into the pipeline and wait for output."""
    dut.mode.value = mode
    dut.relu_en.value = int(relu_en)
    dut.bias_en.value = int(bias_en)
    dut.zp_en.value = int(zp_en)
    dut.acc_in.value = to_signed(acc, 40)
    dut.bias.value = to_signed(bias, 40)
    dut.mult_m.value = int(M) & 0x7FFF
    dut.shift_s.value = int(S) & 0x3F
    dut.zero_point.value = to_signed(zp, 16)
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    # Wait for pipeline to flush
    await ClockCycles(dut.clk, PIPELINE_LATENCY)
    return from_signed(dut.out_data.value, 8)


# ═══════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_passthrough(dut):
    """PASSTHROUGH mode: acc truncated to INT8 directly."""
    await reset_dut(dut)

    # Small positive value
    result = await push_and_wait(dut, acc=42, bias=0, M=1, S=0, zp=0,
                                 mode=MODE_PASSTHROUGH)
    assert result == 42, f"Expected 42, got {result}"

    # Negative value
    result = await push_and_wait(dut, acc=-10, bias=0, M=1, S=0, zp=0,
                                 mode=MODE_PASSTHROUGH)
    assert result == -10, f"Expected -10, got {result}"


@cocotb.test()
async def test_bias_only(dut):
    """CONV_REQ with M=1, S=0: just bias addition + clamp."""
    await reset_dut(dut)

    # acc=100, bias=-50 → 50, M=1, S=0, zp=0 → clamp(50) = 50
    expected = ppu_reference(100, -50, 1, 0, 0, bias_en=True, zp_en=False)
    result = await push_and_wait(dut, acc=100, bias=-50, M=1, S=0, zp=0, zp_en=False)
    assert result == expected, f"Expected {expected}, got {result}"


@cocotb.test()
async def test_multiply_and_shift(dut):
    """CONV_REQ: acc × M >> S produces correct scaled result."""
    await reset_dut(dut)

    # acc=1000, bias=0, M=16384, S=14 → 1000*16384=16384000, >>14 = 1000 → clamp=127
    # Actually: (16384000 + 8192) >> 14 = 1001 → clamp=127
    expected = ppu_reference(1000, 0, 16384, 14, 0, bias_en=False, zp_en=False)
    result = await push_and_wait(dut, acc=1000, bias=0, M=16384, S=14, zp=0,
                                 bias_en=False, zp_en=False)
    assert result == expected, f"Expected {expected}, got {result}"

    # acc=256, bias=0, M=8192, S=14 → 256*8192=2097152, +8192>>14 = 128 → clamp=127
    expected = ppu_reference(256, 0, 8192, 14, 0, bias_en=False, zp_en=False)
    result = await push_and_wait(dut, acc=256, bias=0, M=8192, S=14, zp=0,
                                 bias_en=False, zp_en=False)
    assert result == expected, f"Expected {expected}, got {result}"


@cocotb.test()
async def test_full_pipeline_conv_req(dut):
    """CONV_REQ: full pipeline with realistic per-channel params."""
    await reset_dut(dut)

    # Typical conv output: acc=5000, bias=-200, M=16022, S=22, zp=0
    # (5000 + (-200)) * 16022 = 4800 * 16022 = 76905600
    # (76905600 + 2097152) >> 22 = 79002752 >> 22 = 18 → +0 → clamp(18) = 18
    expected = ppu_reference(5000, -200, 16022, 22, 0)
    result = await push_and_wait(dut, acc=5000, bias=-200, M=16022, S=22, zp=0)
    assert result == expected, f"Expected {expected}, got {result}"


@cocotb.test()
async def test_zero_point(dut):
    """CONV_REQ: zero_point shifts the output."""
    await reset_dut(dut)

    # acc=0, bias=0, M=1, S=0, zp=50 → 0+0=0, *1=0, >>0=0, +50=50, clamp=50
    expected = ppu_reference(0, 0, 1, 0, 50)
    result = await push_and_wait(dut, acc=0, bias=0, M=1, S=0, zp=50)
    assert result == expected, f"Expected {expected}, got {result}"

    # Negative zp
    expected = ppu_reference(100, 0, 1, 0, -120, bias_en=False)
    result = await push_and_wait(dut, acc=100, bias=0, M=1, S=0, zp=-120, bias_en=False)
    assert result == expected, f"Expected {expected}, got {result}"


@cocotb.test()
async def test_clamp_boundaries(dut):
    """CONV_REQ: results clamped to [-128, 127]."""
    await reset_dut(dut)

    # Large positive → clamp to 127
    expected = ppu_reference(10000, 0, 1, 0, 0, bias_en=False, zp_en=False)
    result = await push_and_wait(dut, acc=10000, bias=0, M=1, S=0, zp=0,
                                 bias_en=False, zp_en=False)
    assert result == 127, f"Expected 127, got {result}"

    # Large negative → clamp to -128
    expected = ppu_reference(-5000, 0, 1, 0, 0, bias_en=False, zp_en=False)
    result = await push_and_wait(dut, acc=-5000, bias=0, M=1, S=0, zp=0,
                                 bias_en=False, zp_en=False)
    assert result == -128, f"Expected -128, got {result}"


@cocotb.test()
async def test_relu(dut):
    """CONV_REQ with ReLU: negative values clamped to 0."""
    await reset_dut(dut)

    # Positive stays positive
    expected = ppu_reference(50, 0, 1, 0, 0, bias_en=False, zp_en=False, relu_en=True)
    result = await push_and_wait(dut, acc=50, bias=0, M=1, S=0, zp=0,
                                 bias_en=False, zp_en=False, relu_en=True)
    assert result == expected, f"Expected {expected}, got {result}"

    # Negative becomes 0
    expected = ppu_reference(-50, 0, 1, 0, 0, bias_en=False, zp_en=False, relu_en=True)
    result = await push_and_wait(dut, acc=-50, bias=0, M=1, S=0, zp=0,
                                 bias_en=False, zp_en=False, relu_en=True)
    assert result == 0, f"Expected 0, got {result}"


@cocotb.test()
async def test_relu_only_mode(dut):
    """RELU_ONLY mode: bypass bias/mul/shift/zp, apply clamp+relu."""
    await reset_dut(dut)

    # Positive value passes through
    result = await push_and_wait(dut, acc=42, bias=999, M=9999, S=30, zp=100,
                                 mode=MODE_RELU_ONLY, relu_en=True)
    assert result == 42, f"Expected 42, got {result}"

    # Negative value becomes 0
    result = await push_and_wait(dut, acc=-42, bias=999, M=9999, S=30, zp=100,
                                 mode=MODE_RELU_ONLY, relu_en=True)
    assert result == 0, f"Expected 0, got {result}"

    # Large value clamps to 127
    result = await push_and_wait(dut, acc=200, bias=0, M=1, S=0, zp=0,
                                 mode=MODE_RELU_ONLY, relu_en=False)
    assert result == 127, f"Expected 127, got {result}"


@cocotb.test()
async def test_streaming_pipeline(dut):
    """Push multiple values back-to-back, verify pipelined throughput."""
    await reset_dut(dut)

    # Generate test vectors
    random.seed(42)
    NUM_ELEMENTS = 16
    test_vectors = []
    for _ in range(NUM_ELEMENTS):
        acc = random.randint(-50000, 50000)
        bias = random.randint(-1000, 1000)
        M = random.randint(8000, 20000)
        S = random.randint(18, 25)
        zp = random.randint(-5, 5)
        expected = ppu_reference(acc, bias, M, S, zp)
        test_vectors.append((acc, bias, M, S, zp, expected))

    # Push all values back-to-back (pipelined) and collect results
    dut.mode.value = MODE_CONV_REQ
    dut.relu_en.value = 0
    dut.bias_en.value = 1
    dut.zp_en.value = 1

    results = []
    total_cycles = NUM_ELEMENTS + PIPELINE_LATENCY + 2
    result_idx = 0

    for cycle in range(total_cycles):
        # Drive input
        if cycle < NUM_ELEMENTS:
            acc, bias, M, S, zp, _ = test_vectors[cycle]
            dut.acc_in.value = to_signed(acc, 40)
            dut.bias.value = to_signed(bias, 40)
            dut.mult_m.value = M & 0x7FFF
            dut.shift_s.value = S & 0x3F
            dut.zero_point.value = to_signed(zp, 16)
            dut.in_valid.value = 1
        else:
            dut.in_valid.value = 0

        await RisingEdge(dut.clk)

        # Sample output after rising edge (registered outputs are now updated)
        if int(dut.out_valid.value) == 1:
            val = from_signed(dut.out_data.value, 8)
            results.append(val)
            if result_idx < NUM_ELEMENTS:
                _, _, _, _, _, exp = test_vectors[result_idx]
                if val != exp:
                    # Debug: read internal pipeline state
                    s3_sh = from_signed(dut.s3_shifted.value, 17)
                    s3_m = int(dut.s3_mode.value)
                    s2_pr = from_signed(dut.s2_product.value, 55)
                    dut._log.warning(
                        f"cycle={cycle} result[{result_idx}]: got={val} expected={exp} "
                        f"s3_shifted={s3_sh} s3_mode={s3_m:02b} s2_product={s2_pr}")
                else:
                    dut._log.info(f"cycle={cycle} result[{result_idx}]: got={val} OK")
            result_idx += 1

    # Verify
    dut._log.info(f"Collected {len(results)} results")
    for i, (acc, bias, M, S, zp, expected) in enumerate(test_vectors):
        if i < len(results):
            dut._log.info(f"  [{i}] expected={expected:4d} got={results[i]:4d} {'OK' if results[i]==expected else 'MISMATCH'}")
    assert len(results) == NUM_ELEMENTS, \
        f"Expected {NUM_ELEMENTS} results, got {len(results)}"
    for i, (acc, bias, M, S, zp, expected) in enumerate(test_vectors):
        assert results[i] == expected, \
            f"Element {i}: expected {expected}, got {results[i]}"


@cocotb.test()
async def test_rounding(dut):
    """Verify rounding behavior: (product + (1<<(S-1))) >> S."""
    await reset_dut(dut)

    # acc=3, M=1, S=1 → product=3, round=(3+1)>>1 = 2
    expected = ppu_reference(3, 0, 1, 1, 0, bias_en=False, zp_en=False)
    result = await push_and_wait(dut, acc=3, bias=0, M=1, S=1, zp=0,
                                 bias_en=False, zp_en=False)
    assert result == expected, f"Expected {expected}, got {result}"

    # acc=5, M=1, S=2 → product=5, round=(5+2)>>2 = 1
    expected = ppu_reference(5, 0, 1, 2, 0, bias_en=False, zp_en=False)
    result = await push_and_wait(dut, acc=5, bias=0, M=1, S=2, zp=0,
                                 bias_en=False, zp_en=False)
    assert result == expected, f"Expected {expected}, got {result}"

    # acc=7, M=1, S=2 → product=7, round=(7+2)>>2 = 2
    expected = ppu_reference(7, 0, 1, 2, 0, bias_en=False, zp_en=False)
    result = await push_and_wait(dut, acc=7, bias=0, M=1, S=2, zp=0,
                                 bias_en=False, zp_en=False)
    assert result == expected, f"Expected {expected}, got {result}"
