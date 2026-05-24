# Open-NPU RTL — cocotb Tests for npu_systolic
# SPDX-License-Identifier: Apache-2.0
#
# Tests use ARRAY_SIZE=4 for fast simulation.

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import random

from utils.clock_reset import clock_reset
from utils.csim_ref import pe_mac_reference

# Mode encoding
MODE_IDLE     = 0b00
MODE_WGT_LOAD = 0b01
MODE_COMPUTE  = 0b10
MODE_DRAIN    = 0b11

# Array size (must match -DARRAY_SIZE compile flag)
N = 4


def int8_to_unsigned(val):
    return val & 0xFF


def signed_acc(val, bits=40):
    val = int(val) & ((1 << bits) - 1)
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


async def reset_dut(dut):
    """Initialize and reset the DUT."""
    await clock_reset(dut)
    dut.cmd.value = MODE_IDLE
    dut.cmd_valid.value = 0
    dut.wgt_valid.value = 0
    dut.act_valid.value = 0
    dut.drain_col_sel.value = 0
    for i in range(N):
        dut.wgt_data[i].value = 0
        dut.act_data[i].value = 0
    await RisingEdge(dut.clk)


async def load_weights(dut, weight_matrix):
    """Load weight matrix into the systolic array.

    weight_matrix[col][row] = weight for PE[row][col].
    Loading is column-by-column: each cycle loads one column.
    """
    # Issue WGT_LOAD command
    dut.cmd.value = MODE_WGT_LOAD
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0
    await RisingEdge(dut.clk)  # Wait for FSM to enter WGT_LOAD

    # Load one column per cycle
    for col in range(N):
        for row in range(N):
            dut.wgt_data[row].value = int8_to_unsigned(weight_matrix[col][row])
        dut.wgt_valid.value = 1
        await RisingEdge(dut.clk)

    dut.wgt_valid.value = 0
    # Wait for wgt_load_done flag + FSM transition to READY
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def start_compute(dut):
    """Transition FSM to COMPUTE state."""
    dut.cmd.value = MODE_COMPUTE
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0
    await RisingEdge(dut.clk)  # Wait for FSM transition


async def stream_activations(dut, act_matrix):
    """Stream activations through the array.

    act_matrix[k][row] = activation for row `row` at timestep k.
    After K values, wait COLS-1 cycles for pipeline completion.
    """
    K = len(act_matrix)

    # Stream K activation vectors
    for k in range(K):
        for row in range(N):
            dut.act_data[row].value = int8_to_unsigned(act_matrix[k][row])
        dut.act_valid.value = 1
        await RisingEdge(dut.clk)

    dut.act_valid.value = 0

    # Wait for systolic pipeline to complete (COLS - 1 extra cycles)
    for _ in range(N - 1):
        await RisingEdge(dut.clk)


async def drain_column(dut, col):
    """Drain one column and return ROWS accumulator values.

    Timing:
      Cycle 0: Issue DRAIN cmd (FSM transitions from COMPUTE/READY to S_DRAIN)
      Cycle 1: S_DRAIN — PE processes drain (output not yet valid)
      Cycle 2: S_DRAIN_OUT — acc_out_valid=1, read results here
      Cycle 3: S_READY — done
    """
    dut.drain_col_sel.value = col
    dut.cmd.value = MODE_DRAIN
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)  # Cycle 0: cmd accepted
    dut.cmd_valid.value = 0

    await RisingEdge(dut.clk)  # Cycle 1: S_DRAIN (PE draining)
    await RisingEdge(dut.clk)  # Cycle 2: S_DRAIN_OUT (valid=1)

    results = []
    for row in range(N):
        results.append(signed_acc(dut.acc_out[row].value))

    await RisingEdge(dut.clk)  # Cycle 3: back to READY
    return results


async def drain_all_columns(dut):
    """Drain all columns, return results[row][col]."""
    results = [[0] * N for _ in range(N)]
    for col in range(N):
        col_results = await drain_column(dut, col)
        for row in range(N):
            results[row][col] = col_results[row]
    return results


@cocotb.test()
async def test_weight_load_basic(dut):
    """Test: Load weights and verify via single-activation compute."""
    await reset_dut(dut)

    # weights[col][row] = col + 1 (same for all rows in a column)
    weights = [[col + 1] * N for col in range(N)]
    await load_weights(dut, weights)

    assert dut.ready.value == 1, "Array should be ready after weight load"

    # Compute with act = 1 for all rows, K=1
    await start_compute(dut)
    act_matrix = [[1] * N]
    await stream_activations(dut, act_matrix)

    # Drain all columns
    results = await drain_all_columns(dut)

    # PE[r][c].acc = act * weight[c][r] = 1 * (c+1) = c+1
    for row in range(N):
        for col in range(N):
            expected = col + 1
            assert results[row][col] == expected, \
                f"PE[{row}][{col}]: expected {expected}, got {results[row][col]}"

    dut._log.info("PASS: weight_load_basic")


@cocotb.test()
async def test_single_mac_per_pe(dut):
    """Test: Each PE computes one act*weight product with unique values."""
    await reset_dut(dut)

    # Unique weight per PE: weight[col][row] = (row+1) * (col+1)
    weights = [[(row + 1) * (col + 1) for row in range(N)] for col in range(N)]
    await load_weights(dut, weights)

    # Single activation: act[row] = row + 1
    await start_compute(dut)
    act_matrix = [[row + 1 for row in range(N)]]  # K=1
    await stream_activations(dut, act_matrix)

    results = await drain_all_columns(dut)

    for row in range(N):
        for col in range(N):
            # PE[r][c].acc = act[r] * weight[c][r] = (r+1) * (r+1)*(c+1)
            expected = (row + 1) * (row + 1) * (col + 1)
            assert results[row][col] == expected, \
                f"PE[{row}][{col}]: expected {expected}, got {results[row][col]}"

    dut._log.info("PASS: single_mac_per_pe")


@cocotb.test()
async def test_dot_product_k8(dut):
    """Test: K=8 dot product, verify accumulation over multiple cycles."""
    await reset_dut(dut)

    # All PEs have weight = 3
    weights = [[3] * N for _ in range(N)]
    await load_weights(dut, weights)

    # K=8: act[k][row] = k + 1 for all rows
    K = 8
    await start_compute(dut)
    act_matrix = [[k + 1] * N for k in range(K)]
    await stream_activations(dut, act_matrix)

    results = await drain_all_columns(dut)

    # Each PE: sum_{k=0}^{7}((k+1) * 3) = 3 * 36 = 108
    expected = 3 * sum(range(1, K + 1))
    for row in range(N):
        for col in range(N):
            assert results[row][col] == expected, \
                f"PE[{row}][{col}]: expected {expected}, got {results[row][col]}"

    dut._log.info(f"PASS: dot_product_k8 — all PEs = {expected}")


@cocotb.test()
async def test_full_matmul(dut):
    """Test: Full random data, compare RTL vs golden."""
    await reset_dut(dut)

    random.seed(123)
    K = 6

    # Random weights[col][row]
    weights = [[random.randint(-128, 127) for _ in range(N)] for _ in range(N)]
    # Random activations[k][row]
    act_matrix = [[random.randint(-128, 127) for _ in range(N)] for _ in range(K)]

    await load_weights(dut, weights)
    await start_compute(dut)
    await stream_activations(dut, act_matrix)

    results = await drain_all_columns(dut)

    # Golden: PE[r][c].acc = sum_k(act[k][r] * weight[c][r])
    for row in range(N):
        for col in range(N):
            expected = sum(act_matrix[k][row] * weights[col][row] for k in range(K))
            mask = (1 << 40) - 1
            exp_masked = expected & mask
            if exp_masked >= (1 << 39):
                exp_masked -= (1 << 40)
            assert results[row][col] == exp_masked, \
                f"PE[{row}][{col}]: expected {exp_masked}, got {results[row][col]}"

    dut._log.info("PASS: full_matmul — all 4×4 results match golden")


@cocotb.test()
async def test_drain_clears_and_reload(dut):
    """Test: After drain, accumulators clear. Reload and compute again."""
    await reset_dut(dut)

    # First pass: weight=2, act=5, K=1
    weights = [[2] * N for _ in range(N)]
    await load_weights(dut, weights)
    await start_compute(dut)
    await stream_activations(dut, [[5] * N])

    results1 = await drain_all_columns(dut)
    for r in range(N):
        for c in range(N):
            assert results1[r][c] == 10, f"Pass 1 PE[{r}][{c}]: expected 10, got {results1[r][c]}"

    # Second pass: reload weight=7, act=3, K=1
    weights2 = [[7] * N for _ in range(N)]
    await load_weights(dut, weights2)
    await start_compute(dut)
    await stream_activations(dut, [[3] * N])

    results2 = await drain_all_columns(dut)
    for r in range(N):
        for c in range(N):
            # Should be 7*3=21, NOT 10+21=31 (acc was cleared by drain)
            assert results2[r][c] == 21, f"Pass 2 PE[{r}][{c}]: expected 21, got {results2[r][c]}"

    dut._log.info("PASS: drain_clears_and_reload")


@cocotb.test()
async def test_systolic_delay(dut):
    """Test: All columns eventually see all activations (systolic propagation)."""
    await reset_dut(dut)

    # All weights = 1
    weights = [[1] * N for _ in range(N)]
    await load_weights(dut, weights)

    # Feed K=4 unique activations, wait for full pipeline drain (K + N - 1)
    K = 4
    await start_compute(dut)
    act_matrix = [[k + 1] * N for k in range(K)]
    # stream_activations waits N-1 extra cycles for pipeline completion
    await stream_activations(dut, act_matrix)

    results = await drain_all_columns(dut)

    # With weight=1: PE[r][c].acc = sum of all K activations that arrived at col c.
    # Due to systolic propagation, col c receives activations from col c-1 (1 cycle delay).
    # col 0 receives all K acts directly (valid for K cycles).
    # col 1 receives K acts starting 1 cycle later (also K acts, from act_valid_out).
    # col 2 receives K acts starting 2 cycles later.
    # col 3 receives K acts starting 3 cycles later.
    # All columns receive ALL K activations (just delayed).
    # So all PEs should have sum(1..K) = 10.
    expected = sum(range(1, K + 1))  # 1+2+3+4 = 10

    for r in range(N):
        for c in range(N):
            assert results[r][c] == expected, \
                f"PE[{r}][{c}]: expected {expected}, got {results[r][c]}"

    dut._log.info(f"PASS: systolic_delay — all PEs = {expected}")
