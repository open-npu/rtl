# Open-NPU RTL — C Simulator Reference Functions
# SPDX-License-Identifier: Apache-2.0
#
# Pure-Python golden reference implementations matching csim behavior.
# Used for bit-exact comparison with RTL simulation outputs.

import numpy as np


def pe_mac_reference(activations, weight, acc_bits=40):
    """Compute MAC reference for a single PE.

    Matches npu_pe.v COMPUTE mode: acc += act_in * weight for each activation.

    Args:
        activations: list/array of int8 activation values
        weight: int8 weight value
        acc_bits: accumulator bit width (default 40)

    Returns:
        int: final accumulator value (Python int, unlimited precision)
    """
    acc = 0
    for act in activations:
        acc += int(act) * int(weight)
    # Mask to acc_bits (signed)
    mask = (1 << acc_bits) - 1
    acc_masked = acc & mask
    # Sign extend
    if acc_masked >= (1 << (acc_bits - 1)):
        acc_masked -= (1 << acc_bits)
    return acc_masked


def matmul_reference(act_matrix, weight_matrix, acc_bits=40):
    """Compute systolic array matmul reference: C = A × W.

    Models the full N×N systolic array output for a tile.
    act_matrix[k][row] provides activation for row `row` at timestep `k`.
    weight_matrix[col][k] is the weight loaded into column `col` for reduction dim `k`.

    Result[row][col] = sum_k(act_matrix[k][row] * weight_matrix[col][k])

    This matches the hardware behavior:
      - Weights are pre-loaded (weight_matrix[col] = weight for all PEs in that column)
      - Activations stream in per-row, with column-delay handled by systolic propagation
      - We compute the "logical" result ignoring systolic timing (which only affects latency)

    Args:
        act_matrix: shape [K][ROWS] — activation values per timestep per row
        weight_matrix: shape [COLS][K] — weight per column per reduction step
        acc_bits: accumulator bit width (default 40)

    Returns:
        2D list [ROWS][COLS] of signed accumulator values
    """
    K = len(act_matrix)
    rows = len(act_matrix[0]) if K > 0 else 0
    cols = len(weight_matrix)

    mask = (1 << acc_bits) - 1
    results = []
    for r in range(rows):
        row_results = []
        for c in range(cols):
            acc = 0
            for k in range(K):
                acc += int(act_matrix[k][r]) * int(weight_matrix[c][k])
            # Mask and sign-extend
            acc_masked = acc & mask
            if acc_masked >= (1 << (acc_bits - 1)):
                acc_masked -= (1 << acc_bits)
            row_results.append(acc_masked)
        results.append(row_results)
    return results


def ppu_reference(acc, bias, M, S, zp, bias_en=True, zp_en=True, relu_en=False,
                  acc_bits=40):
    """Compute PPU per-channel requantization reference.

    Matches npu_ppu.v CONV_REQ pipeline:
      acc → +bias → ×M → >>S(+round) → +zp → clamp[-128,127] → ReLU → out

    Args:
        acc: signed accumulator value (up to acc_bits wide)
        bias: signed bias value (up to acc_bits wide)
        M: unsigned 15-bit multiplier
        S: unsigned 6-bit shift amount
        zp: signed 16-bit zero point
        bias_en: whether bias addition is enabled
        zp_en: whether zero_point addition is enabled
        relu_en: whether ReLU is applied
        acc_bits: accumulator bit width (default 40)

    Returns:
        int: INT8 output value [-128, 127]
    """
    val = int(acc)

    # Stage 1: Bias
    if bias_en:
        val = val + int(bias)

    # Stage 2: Multiply by M (unsigned 15-bit)
    M_val = int(M) & 0x7FFF
    product = val * M_val

    # Stage 3: Rounding right shift
    S_val = int(S) & 0x3F
    if S_val > 0:
        result = (product + (1 << (S_val - 1))) >> S_val
    else:
        result = product

    # Stage 4: Add zero point
    if zp_en:
        result = result + int(zp)

    # Stage 5: Clamp to INT8
    if result < -128:
        result = -128
    elif result > 127:
        result = 127

    # Stage 6: ReLU
    if relu_en and result < 0:
        result = 0

    return result


def systolic_row_reference(activations_2d, weights, acc_bits=40):
    """Compute MAC reference for a full systolic row (N PEs).

    Args:
        activations_2d: 2D array [timestep][pe_index] of int8 values
        weights: array of int8 weights, one per PE
        acc_bits: accumulator bit width

    Returns:
        list of int: accumulator values for each PE
    """
    n_pes = len(weights)
    accs = [0] * n_pes
    for t, acts in enumerate(activations_2d):
        for pe_idx in range(min(len(acts), n_pes)):
            accs[pe_idx] += int(acts[pe_idx]) * int(weights[pe_idx])

    # Mask and sign-extend
    results = []
    mask = (1 << acc_bits) - 1
    for acc in accs:
        acc_masked = acc & mask
        if acc_masked >= (1 << (acc_bits - 1)):
            acc_masked -= (1 << acc_bits)
        results.append(acc_masked)
    return results
