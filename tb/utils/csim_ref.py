# Open-NPU RTL — C Simulator Reference Functions
# SPDX-License-Identifier: Apache-2.0
#
# Pure-Python golden reference implementations matching csim behavior.
# Used for bit-exact comparison with RTL simulation outputs.

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
