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
