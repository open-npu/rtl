#!/usr/bin/env python3
"""
Generate DMA E2E golden data for MobileNetV2-Tiny (INT8 + INT16).

This golden data exercises the FULL chip data path:
  CPU(CSR) → DMA(Wishbone) → DDR → SRAM → Compute → SRAM → DMA → DDR

Output structure:
  golden_dma_e2e/int8/   — 10 layers, each with wgt/param/input/output .npy
  golden_dma_e2e/int16/  — same architecture, INT16 data path

Each .npy file contains uint32 word arrays ready to be placed in DDR memory
by the cocotb testbench.

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import os
import json


# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════


def compute_ms(eff_scale):
    """Compute 15-bit multiplier M and 6-bit shift S: M / 2^S ~ eff_scale."""
    best_s, best_m = 0, max(1, int(np.round(eff_scale)))
    for s in range(64):
        m = eff_scale * (2.0 ** s)
        if 1.0 <= m <= 32767.0:
            best_s = s
            best_m = int(np.round(m))
            if best_m >= 16384:
                break
    return max(1, min(32767, best_m)), best_s


# ═══════════════════════════════════════════════════════════════════════
# Reference convolution functions
# ═══════════════════════════════════════════════════════════════════════


def ref_conv2d(input_nhwc, weight_ohwi, cfg):
    """Conv2D → int64 accumulator [out_h][out_w][out_c]."""
    in_h, in_w = cfg['in_h'], cfg['in_w']
    out_h, out_w, out_c = cfg['out_h'], cfg['out_w'], cfg['out_c']
    kh, kw = cfg['kernel_h'], cfg['kernel_w']
    sh, sw = cfg['stride_h'], cfg['stride_w']
    pad_t, pad_l = cfg['pad_top'], cfg['pad_left']
    in_c = cfg['in_c']

    acc = np.zeros((out_h, out_w, out_c), dtype=np.int64)
    for oh in range(out_h):
        for ow in range(out_w):
            for oc in range(out_c):
                s = np.int64(0)
                for fh in range(kh):
                    for fw in range(kw):
                        ih = oh * sh - pad_t + fh
                        iw = ow * sw - pad_l + fw
                        if 0 <= ih < in_h and 0 <= iw < in_w:
                            for ic in range(in_c):
                                s += np.int64(input_nhwc[ih, iw, ic]) * \
                                     np.int64(weight_ohwi[oc, fh, fw, ic])
                acc[oh, ow, oc] = s
    return acc


def ref_dwconv(input_nhwc, weight_chw, cfg):
    """DW Conv → int64 accumulator [out_h][out_w][ch]."""
    in_h, in_w, ch = cfg['in_h'], cfg['in_w'], cfg['in_c']
    out_h, out_w = cfg['out_h'], cfg['out_w']
    kh, kw = cfg['kernel_h'], cfg['kernel_w']
    sh, sw = cfg['stride_h'], cfg['stride_w']
    pad_t, pad_l = cfg['pad_top'], cfg['pad_left']

    acc = np.zeros((out_h, out_w, ch), dtype=np.int64)
    for oh in range(out_h):
        for ow in range(out_w):
            for c in range(ch):
                s = np.int64(0)
                for fh in range(kh):
                    for fw in range(kw):
                        ih = oh * sh - pad_t + fh
                        iw = ow * sw - pad_l + fw
                        if 0 <= ih < in_h and 0 <= iw < in_w:
                            s += np.int64(input_nhwc[ih, iw, c]) * \
                                 np.int64(weight_chw[c, fh, fw])
                acc[oh, ow, c] = s
    return acc


def ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr, relu6=False,
                 clamp_min=-128, clamp_max=127):
    """Per-channel requantize matching RTL PPU CONV_REQ mode.

    Pipeline: acc + bias -> * M -> >>> S (with rounding) -> + zp -> clamp -> ReLU6
    """
    out_shape = acc.shape
    n_ch = out_shape[-1]
    dtype_out = np.int8 if clamp_max <= 127 else np.int16
    out = np.zeros(out_shape, dtype=dtype_out)
    for c in range(n_ch):
        ch_acc = acc[..., c].astype(np.int64)
        biased = ch_acc + np.int64(bias_arr[c])
        m = int(M_arr[c])
        product = biased * np.int64(m)
        s = int(S_arr[c])
        if s > 0:
            rounded = product + (np.int64(1) << (s - 1))
            shifted = rounded >> s
        else:
            shifted = product
        zp_result = shifted + np.int64(zp_arr[c])
        ch_out = np.clip(zp_result, clamp_min, clamp_max)
        if relu6:
            ch_out = np.maximum(ch_out, 0)
        out[..., c] = ch_out.astype(dtype_out)
    return out


def ref_pooling(input_nhwc, cfg, int16_mode=False):
    """Reference Pooling (Max or Avg) matching CSIM npu_pooling().

    Returns int64 accumulator [out_h][out_w][ch].
    """
    in_h, in_w, ch = cfg['in_h'], cfg['in_w'], cfg['in_c']
    out_h, out_w = cfg['out_h'], cfg['out_w']
    pool_h, pool_w = cfg['pool_h'], cfg['pool_w']
    pool_sh, pool_sw = cfg['pool_stride_h'], cfg['pool_stride_w']
    pad_t, pad_l = cfg['pad_top'], cfg['pad_left']
    is_avg = cfg['pool_mode'] == 1

    if cfg.get('global_pool', False):
        pool_h, pool_w = in_h, in_w
        pool_sh, pool_sw = in_h, in_w

    acc = np.zeros((out_h, out_w, ch), dtype=np.int64)
    for oh in range(out_h):
        for ow in range(out_w):
            for c in range(ch):
                if is_avg:
                    result = np.int64(0)
                else:
                    result = np.int64(-2**63)
                count = 0
                for ph in range(pool_h):
                    ih = oh * pool_sh - pad_t + ph
                    if ih < 0 or ih >= in_h:
                        continue
                    for pw in range(pool_w):
                        iw = ow * pool_sw - pad_l + pw
                        if iw < 0 or iw >= in_w:
                            continue
                        val = np.int64(input_nhwc[ih, iw, c])
                        if is_avg:
                            result += val
                            count += 1
                        else:
                            if val > result:
                                result = val
                            count += 1
                # AvgPool: symmetric rounding integer division
                if is_avg and count > 0:
                    if result >= 0:
                        result = (result + count // 2) // count
                    else:
                        result = -(-result + count // 2) // count
                acc[oh, ow, c] = result
    return acc


def ref_eltwise_add(input_a, input_b, M_A, S_A, M_B, S_B, relu=True,
                    clamp_min=-128, clamp_max=127):
    """Reference Eltwise Add with dual rescale + clamp + relu.

    Matches RTL: rescale_A + rescale_B -> clamp -> relu (PPU MODE_RELU_ONLY).
    """
    flat_a = input_a.flatten().astype(np.int64)
    flat_b = input_b.flatten().astype(np.int64)
    n = len(flat_a)
    dtype_out = np.int8 if clamp_max <= 127 else np.int16

    out_flat = np.zeros(n, dtype=dtype_out)
    for i in range(n):
        prod_a = flat_a[i] * np.int64(M_A)
        prod_b = flat_b[i] * np.int64(M_B)
        if S_A > 0:
            rescaled_a = (prod_a + (np.int64(1) << (S_A - 1))) >> S_A
        else:
            rescaled_a = prod_a
        if S_B > 0:
            rescaled_b = (prod_b + (np.int64(1) << (S_B - 1))) >> S_B
        else:
            rescaled_b = prod_b
        s = rescaled_a + rescaled_b
        s = max(clamp_min, min(clamp_max, int(s)))
        if relu:
            s = max(0, s)
        out_flat[i] = dtype_out(s)
    return out_flat.reshape(input_a.shape)


def ref_resize(input_nhwc, cfg, int16_mode=False):
    """Reference Resize matching csim/src/resize.c.

    Returns int64 accumulator [out_h][out_w][ch].
    """
    in_h, in_w, ch = cfg['in_h'], cfg['in_w'], cfg['in_c']
    out_h, out_w = cfg['out_h'], cfg['out_w']
    acc = np.zeros((out_h, out_w, ch), dtype=np.int64)

    if cfg['resize_mode'] == 0:
        for oh in range(out_h):
            ih = (oh * in_h) // out_h
            ih = min(ih, in_h - 1)
            for ow in range(out_w):
                iw = (ow * in_w) // out_w
                iw = min(iw, in_w - 1)
                acc[oh, ow, :] = input_nhwc[ih, iw, :].astype(np.int64)
    else:
        for oh in range(out_h):
            if out_h > 1:
                src_h_q8 = oh * ((in_h - 1) << 8) // (out_h - 1)
            else:
                src_h_q8 = 0
            ih0 = src_h_q8 >> 8
            ih1 = min(ih0 + 1, in_h - 1)
            frac_h = src_h_q8 & 0xFF

            for ow in range(out_w):
                if out_w > 1:
                    src_w_q8 = ow * ((in_w - 1) << 8) // (out_w - 1)
                else:
                    src_w_q8 = 0
                iw0 = src_w_q8 >> 8
                iw1 = min(iw0 + 1, in_w - 1)
                frac_w = src_w_q8 & 0xFF

                v00 = input_nhwc[ih0, iw0, :].astype(np.int64)
                v01 = input_nhwc[ih0, iw1, :].astype(np.int64)
                v10 = input_nhwc[ih1, iw0, :].astype(np.int64)
                v11 = input_nhwc[ih1, iw1, :].astype(np.int64)

                top = v00 * (256 - frac_w) + v01 * frac_w
                bot = v10 * (256 - frac_w) + v11 * frac_w
                val = top * (256 - frac_h) + bot * frac_h
                acc[oh, ow, :] = (val + (1 << 15)) >> 16

    return acc


def pack_add_params(M_A, S_A, M_B, S_B):
    """Pack Add rescale params into 2 uint32 words.

    Word 0: M_A[14:0] at [15:0], S_A[5:0] at [21:16]
    Word 1: M_B[14:0] at [15:0], S_B[5:0] at [21:16]
    """
    w0 = (int(M_A) & 0x7FFF) | ((int(S_A) & 0x3F) << 16)
    w1 = (int(M_B) & 0x7FFF) | ((int(S_B) & 0x3F) << 16)
    return [w0, w1]


# ═══════════════════════════════════════════════════════════════════════
# Packing functions (INT8)
# ═══════════════════════════════════════════════════════════════════════


def pack_i8_to_words(data):
    """Pack int8 array into uint32 words (4 bytes/word, little-endian)."""
    byte_arr = data.astype(np.int8).tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


# ═══════════════════════════════════════════════════════════════════════
# Packing functions (INT16)
# ═══════════════════════════════════════════════════════════════════════


def pack_i16_to_words(data):
    """Pack int16 array into uint32 words (2 elements/word, little-endian)."""
    flat = data.flatten().astype(np.int16)
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.int16(0))
    words = []
    for i in range(0, len(flat), 2):
        lo = int(flat[i]) & 0xFFFF
        hi = int(flat[i + 1]) & 0xFFFF
        words.append(lo | (hi << 16))
    return words


# ═══════════════════════════════════════════════════════════════════════
# Parameter packing (shared for INT8/INT16)
# ═══════════════════════════════════════════════════════════════════════


def pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, n_ch):
    """Pack per-channel params: 4 words per channel.

    Word 0: M[14:0] | (S[5:0] << 16)
    Word 1: ZP[15:0] | (bias[15:0] << 16)
    Word 2: bias[47:16]
    Word 3: bias[63:48] in [15:0]
    """
    words = []
    for c in range(n_ch):
        m = int(M_arr[c]) & 0x7FFF
        s = int(S_arr[c]) & 0x3F
        w0 = m | (s << 16)

        zp = int(zp_arr[c]) & 0xFFFF
        bias = int(bias_arr[c])
        if bias < 0:
            bias_u64 = bias + (1 << 64)
        else:
            bias_u64 = bias
        bias_u64 &= 0xFFFFFFFFFFFFFFFF

        w1 = zp | (((bias_u64 >> 0) & 0xFFFF) << 16)
        w2 = (bias_u64 >> 16) & 0xFFFFFFFF
        w3 = (bias_u64 >> 48) & 0xFFFF

        words.extend([w0, w1, w2, w3])
    return words


# ═══════════════════════════════════════════════════════════════════════
# Weight packing
# ═══════════════════════════════════════════════════════════════════════


def pack_conv_weights_i8(weight_ohwi, out_c, k_depth):
    """Pack Conv2D INT8 weights: [out_c][kh][kw][in_c] -> uint32 words."""
    byte_arr = weight_ohwi.astype(np.int8).tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


def pack_dw_weights_i8(weight_chw, n_ch):
    """Pack DW Conv INT8 weights: [ch][3][3] -> uint32 words."""
    byte_arr = weight_chw.astype(np.int8).tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


def pack_conv_weights_i16(weight_ohwi, out_c, k_depth):
    """Pack Conv2D INT16 weights: [out_c][kh][kw][in_c] -> uint32 words."""
    flat = weight_ohwi.reshape(out_c * k_depth).astype(np.int16)
    return pack_i16_to_words(flat)


def pack_dw_weights_i16(weight_chw, n_ch):
    """Pack DW Conv INT16 weights: [ch][3][3] -> uint32 words."""
    flat = weight_chw.flatten().astype(np.int16)
    return pack_i16_to_words(flat)


# ═══════════════════════════════════════════════════════════════════════
# Model Builder
# ═══════════════════════════════════════════════════════════════════════


def build_mobilenetv2_tiny(int16_mode=False):
    """Build MobileNetV2-Tiny model and compute golden per-layer data.

    Args:
        int16_mode: If True, use INT16 weights/activations, else INT8.

    Returns:
        layers: list of layer config dicts (with DMA metadata)
        layer_data: list of dicts with packed uint32 word arrays
    """
    np.random.seed(2024)

    layers = []
    layer_data = []
    scale_in = 1.0 / 64.0

    if int16_mode:
        wgt_range = (-64, 65)
        act_range = (-500, 500)
        clamp_min, clamp_max = -32768, 32767
        dtype_act = np.int16
        dtype_wgt = np.int16
        elems_per_word = 2
    else:
        wgt_range = (-8, 9)
        act_range = (-64, 64)
        clamp_min, clamp_max = -128, 127
        dtype_act = np.int8
        dtype_wgt = np.int8
        elems_per_word = 4

    # Generate input: 16x16x3
    input_nhwc = np.random.randint(act_range[0], act_range[1],
                                   (16, 16, 3), dtype=dtype_act)
    current = input_nhwc.copy()

    def add_conv(in_act, in_h, in_w, in_c, out_c, kernel=3, stride=1, relu6=True):
        nonlocal scale_in, current
        pad = (kernel - 1) // 2
        out_h = (in_h + 2 * pad - kernel) // stride + 1
        out_w = (in_w + 2 * pad - kernel) // stride + 1
        k_depth = kernel * kernel * in_c

        cfg = {
            'op_type': 0,  # Conv2D
            'data_type': 1 if int16_mode else 0,
            'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
            'out_h': out_h, 'out_w': out_w, 'out_c': out_c,
            'kernel_h': kernel, 'kernel_w': kernel,
            'stride_h': stride, 'stride_w': stride,
            'pad_top': pad, 'pad_left': pad,
            'k_depth': k_depth,
            'relu6': relu6,
        }

        # Weights
        w = np.random.randint(wgt_range[0], wgt_range[1],
                              (out_c, kernel, kernel, in_c), dtype=dtype_wgt)

        # Per-channel params
        s_w = 1.0 / 64.0
        acc_scale = scale_in * s_w
        s_out = scale_in
        eff_scale = acc_scale / s_out

        M_arr = np.zeros(out_c, dtype=np.uint16)
        S_arr = np.zeros(out_c, dtype=np.uint8)
        bias_arr = np.random.randint(-1000 if int16_mode else -100,
                                     1000 if int16_mode else 100,
                                     (out_c,), dtype=np.int64)
        zp_arr = np.zeros(out_c, dtype=np.int16)
        for c in range(out_c):
            es = eff_scale * (0.8 + 0.4 * np.random.rand())
            M_arr[c], S_arr[c] = compute_ms(es)

        # Compute golden
        acc = ref_conv2d(in_act, w, cfg)
        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=relu6, clamp_min=clamp_min, clamp_max=clamp_max)

        # Pack data
        if int16_mode:
            wgt_words = pack_conv_weights_i16(w, out_c, k_depth)
            input_words = pack_i16_to_words(in_act)
            output_words = pack_i16_to_words(out)
        else:
            wgt_words = pack_conv_weights_i8(w, out_c, k_depth)
            input_words = pack_i8_to_words(in_act)
            output_words = pack_i8_to_words(out)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, out_c)

        # DMA sizes in bytes (= word_count * 4)
        cfg['dma_wgt_size'] = len(wgt_words) * 4
        cfg['dma_in_size'] = len(input_words) * 4
        cfg['dma_out_size'] = len(output_words) * 4
        cfg['dma_param_count'] = out_c  # channels, not words

        # POST_CTRL register encoding
        # bits[1:0]=00(CONV_REQ), bit[2]=relu_en, bit[5]=zp_en, bit[6]=bias_en
        post_ctrl = 0x60  # bias_en + zp_en
        if relu6:
            post_ctrl |= 0x04  # bit[2] = relu_en (clamps negatives to 0)
        if int16_mode:
            post_ctrl |= 0x80  # bit[7] = int16_mode
        cfg['post_ctrl'] = post_ctrl

        layers.append(cfg)
        layer_data.append({
            'wgt_words': wgt_words,
            'param_words': param_words,
            'input_words': input_words,
            'output_words': output_words,
        })

        scale_in = s_out
        current = out
        return out_h, out_w, out_c

    def add_dw(in_act, in_h, in_w, ch, stride=1, relu6=True):
        nonlocal scale_in, current
        pad = 1
        out_h = (in_h + 2 * pad - 3) // stride + 1
        out_w = (in_w + 2 * pad - 3) // stride + 1
        k_depth = 9

        cfg = {
            'op_type': 1,  # DW Conv
            'data_type': 1 if int16_mode else 0,
            'in_h': in_h, 'in_w': in_w, 'in_c': ch,
            'out_h': out_h, 'out_w': out_w, 'out_c': ch,
            'kernel_h': 3, 'kernel_w': 3,
            'stride_h': stride, 'stride_w': stride,
            'pad_top': pad, 'pad_left': pad,
            'k_depth': k_depth,
            'relu6': relu6,
        }

        # DW weights
        w = np.random.randint(wgt_range[0], wgt_range[1],
                              (ch, 3, 3), dtype=dtype_wgt)

        # Per-channel params
        s_w = 1.0 / 64.0
        acc_scale = scale_in * s_w
        s_out = scale_in
        eff_scale = acc_scale / s_out

        M_arr = np.zeros(ch, dtype=np.uint16)
        S_arr = np.zeros(ch, dtype=np.uint8)
        bias_arr = np.random.randint(-1000 if int16_mode else -100,
                                     1000 if int16_mode else 100,
                                     (ch,), dtype=np.int64)
        zp_arr = np.zeros(ch, dtype=np.int16)
        for c in range(ch):
            es = eff_scale * (0.8 + 0.4 * np.random.rand())
            M_arr[c], S_arr[c] = compute_ms(es)

        # Compute golden
        acc = ref_dwconv(in_act, w, cfg)
        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=relu6, clamp_min=clamp_min, clamp_max=clamp_max)

        # Pack data
        if int16_mode:
            wgt_words = pack_dw_weights_i16(w, ch)
            input_words = pack_i16_to_words(in_act)
            output_words = pack_i16_to_words(out)
        else:
            wgt_words = pack_dw_weights_i8(w, ch)
            input_words = pack_i8_to_words(in_act)
            output_words = pack_i8_to_words(out)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, ch)

        # DMA sizes
        cfg['dma_wgt_size'] = len(wgt_words) * 4
        cfg['dma_in_size'] = len(input_words) * 4
        cfg['dma_out_size'] = len(output_words) * 4
        cfg['dma_param_count'] = ch

        # POST_CTRL
        post_ctrl = 0x60
        if relu6:
            post_ctrl |= 0x04  # bit[2] = relu_en
        if int16_mode:
            post_ctrl |= 0x80
        cfg['post_ctrl'] = post_ctrl

        layers.append(cfg)
        layer_data.append({
            'wgt_words': wgt_words,
            'param_words': param_words,
            'input_words': input_words,
            'output_words': output_words,
        })

        scale_in = s_out
        current = out
        return out_h, out_w, ch

    # ─── Build architecture (MobileNetV2-Tiny, 10 layers) ───
    h, w, c = 16, 16, 3

    h, w, c = add_conv(current, h, w, c, 8, kernel=3, stride=2, relu6=True)    # L0
    h, w, c = add_dw(current, h, w, c, stride=1, relu6=True)                   # L1
    h, w, c = add_conv(current, h, w, c, 4, kernel=1, stride=1, relu6=False)   # L2
    h, w, c = add_conv(current, h, w, c, 24, kernel=1, stride=1, relu6=True)   # L3
    h, w, c = add_dw(current, h, w, c, stride=2, relu6=True)                   # L4
    h, w, c = add_conv(current, h, w, c, 8, kernel=1, stride=1, relu6=False)   # L5
    h, w, c = add_conv(current, h, w, c, 48, kernel=1, stride=1, relu6=True)   # L6
    h, w, c = add_dw(current, h, w, c, stride=1, relu6=True)                   # L7
    h, w, c = add_conv(current, h, w, c, 8, kernel=1, stride=1, relu6=False)   # L8
    h, w, c = add_conv(current, h, w, c, 32, kernel=1, stride=1, relu6=True)   # L9

    return layers, layer_data, input_nhwc


# ═══════════════════════════════════════════════════════════════════════
# DDR Address Layout
# ═══════════════════════════════════════════════════════════════════════

WGT_BASE   = 0x1000_0000
PARAM_BASE = 0x2000_0000
ACT_BASE   = 0x3000_0000
LAYER_OFFSET = 0x0001_0000  # 64KB per layer


def get_ddr_addrs(layer_idx, n_layers):
    """Get DDR addresses for a layer."""
    return {
        'wgt_addr':   WGT_BASE + layer_idx * LAYER_OFFSET,
        'param_addr': PARAM_BASE + layer_idx * LAYER_OFFSET,
        'in_addr':    ACT_BASE + layer_idx * LAYER_OFFSET,
        'out_addr':   ACT_BASE + (layer_idx + 1) * LAYER_OFFSET,
    }


# ═══════════════════════════════════════════════════════════════════════
# Save golden data
# ═══════════════════════════════════════════════════════════════════════


def save_golden(output_dir, int16_mode=False):
    """Generate and save golden data for DMA E2E test."""
    mode_str = "INT16" if int16_mode else "INT8"
    print(f"\n{'='*60}")
    print(f"Generating DMA E2E golden data — {mode_str}")
    print(f"{'='*60}")

    layers, layer_data, input_nhwc = build_mobilenetv2_tiny(int16_mode=int16_mode)

    os.makedirs(output_dir, exist_ok=True)

    # Print architecture
    for i, cfg in enumerate(layers):
        op_name = 'Conv2D' if cfg['op_type'] == 0 else 'DWConv'
        print(f"  L{i:2d}: {op_name:6s} [{cfg['in_h']}x{cfg['in_w']}x{cfg['in_c']}]"
              f" -> [{cfg['out_h']}x{cfg['out_w']}x{cfg['out_c']}]"
              f" k={cfg['kernel_h']}x{cfg['kernel_w']} s={cfg['stride_h']}"
              f" {'ReLU6' if cfg['relu6'] else 'lin'}"
              f" post_ctrl=0x{cfg['post_ctrl']:02X}")

    # Save per-layer .npy files
    metadata = []
    for i, (cfg, data) in enumerate(zip(layers, layer_data)):
        prefix = f'layer_{i:02d}'
        np.save(os.path.join(output_dir, f'{prefix}_wgt.npy'),
                np.array(data['wgt_words'], dtype=np.uint32))
        np.save(os.path.join(output_dir, f'{prefix}_param.npy'),
                np.array(data['param_words'], dtype=np.uint32))
        np.save(os.path.join(output_dir, f'{prefix}_input.npy'),
                np.array(data['input_words'], dtype=np.uint32))
        np.save(os.path.join(output_dir, f'{prefix}_output.npy'),
                np.array(data['output_words'], dtype=np.uint32))

        addrs = get_ddr_addrs(i, len(layers))
        metadata.append({
            'index': i,
            'op_type': cfg['op_type'],
            'data_type': cfg['data_type'],
            'in_h': cfg['in_h'], 'in_w': cfg['in_w'], 'in_c': cfg['in_c'],
            'out_h': cfg['out_h'], 'out_w': cfg['out_w'], 'out_c': cfg['out_c'],
            'kernel_h': cfg['kernel_h'], 'kernel_w': cfg['kernel_w'],
            'stride_h': cfg['stride_h'], 'stride_w': cfg['stride_w'],
            'pad_top': cfg['pad_top'], 'pad_left': cfg['pad_left'],
            'k_depth': cfg['k_depth'],
            'relu6': cfg['relu6'],
            'post_ctrl': cfg['post_ctrl'],
            'dma_wgt_size': cfg['dma_wgt_size'],
            'dma_in_size': cfg['dma_in_size'],
            'dma_out_size': cfg['dma_out_size'],
            'dma_param_count': cfg['dma_param_count'],
            'ddr_wgt_addr': addrs['wgt_addr'],
            'ddr_param_addr': addrs['param_addr'],
            'ddr_in_addr': addrs['in_addr'],
            'ddr_out_addr': addrs['out_addr'],
            'n_wgt_words': len(data['wgt_words']),
            'n_param_words': len(data['param_words']),
            'n_input_words': len(data['input_words']),
            'n_output_words': len(data['output_words']),
        })

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    total_wgt = sum(m['n_wgt_words'] for m in metadata) * 4
    total_act = sum(m['n_input_words'] for m in metadata) * 4
    total_out = sum(m['n_output_words'] for m in metadata) * 4
    print(f"\n  Saved {len(layers)} layers to {output_dir}/")
    print(f"  Total: wgt={total_wgt}B act={total_act}B out={total_out}B")
    for i, m in enumerate(metadata):
        print(f"  L{i:2d}: wgt={m['n_wgt_words']:5d} param={m['n_param_words']:4d}"
              f" in={m['n_input_words']:4d} out={m['n_output_words']:4d} words"
              f"  DDR: in=0x{m['ddr_in_addr']:08X} out=0x{m['ddr_out_addr']:08X}")

    return metadata


# ═══════════════════════════════════════════════════════════════════════
# Conv1x1 partial-OC test generator (OUT_C not multiple of 4/2)
# ═══════════════════════════════════════════════════════════════════════


def gen_conv1x1_partial_oc_test(in_h=2, in_w=2, in_c=4, out_c=2,
                                 int16_mode=False, seed=42):
    """Generate Conv2D 1x1 test with arbitrary OUT_C (for partial word flush testing).

    Uses identity requantize (M=16384, S=14, bias=0, zp=0) so output = clamp(acc).
    Returns (meta, data) suitable for program_layer() + verify.
    """
    np.random.seed(seed)

    if int16_mode:
        dtype_act = np.int16
        dtype_wgt = np.int16
        clamp_min, clamp_max = -32768, 32767
        elems_per_word = 2
    else:
        dtype_act = np.int8
        dtype_wgt = np.int8
        clamp_min, clamp_max = -128, 127
        elems_per_word = 4

    # Small random input and weights
    input_nhwc = np.random.randint(-10, 11, (in_h, in_w, in_c), dtype=dtype_act)
    weight_ohwi = np.random.randint(-5, 6, (out_c, 1, 1, in_c), dtype=dtype_wgt)

    cfg = {
        'op_type': 0,  # Conv2D
        'data_type': 1 if int16_mode else 0,
        'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
        'out_h': in_h, 'out_w': in_w, 'out_c': out_c,
        'kernel_h': 1, 'kernel_w': 1,
        'stride_h': 1, 'stride_w': 1,
        'pad_top': 0, 'pad_left': 0,
        'k_depth': in_c,
    }

    # Compute golden: identity requant (M=16384, S=14 => scale=1.0)
    M_arr = np.full(out_c, 16384, dtype=np.uint16)
    S_arr = np.full(out_c, 14, dtype=np.uint8)
    bias_arr = np.zeros(out_c, dtype=np.int64)
    zp_arr = np.zeros(out_c, dtype=np.int16)

    acc = ref_conv2d(input_nhwc, weight_ohwi, cfg)
    out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                       relu6=False, clamp_min=clamp_min, clamp_max=clamp_max)

    # Pack data
    if int16_mode:
        wgt_words = pack_conv_weights_i16(weight_ohwi, out_c, in_c)
        input_words = pack_i16_to_words(input_nhwc)
        output_words = pack_i16_to_words(out)
    else:
        wgt_words = pack_conv_weights_i8(weight_ohwi, out_c, in_c)
        input_words = pack_i8_to_words(input_nhwc)
        output_words = pack_i8_to_words(out)
    param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, out_c)

    # Metadata (same format as program_layer expects)
    meta = dict(cfg)
    meta['n_input_words'] = len(input_words)
    meta['n_wgt_words'] = len(wgt_words)
    meta['n_param_words'] = len(param_words)
    meta['n_output_words'] = len(output_words)
    meta['dma_in_size'] = len(input_words) * 4
    meta['dma_wgt_size'] = len(wgt_words) * 4
    meta['dma_out_size'] = len(output_words) * 4
    meta['dma_param_count'] = out_c
    # POST_CTRL: CONV_REQ mode, no relu, bias_en + zp_en
    meta['post_ctrl'] = 0x60 | (0x80 if int16_mode else 0)

    data = {
        'wgt_words': wgt_words,
        'param_words': param_words,
        'input_words': input_words,
        'output_words': output_words,
    }
    return meta, data


# ═══════════════════════════════════════════════════════════════════════
# Pooling + Add standalone test generators
# ═══════════════════════════════════════════════════════════════════════


def gen_pooling_test(mode='max', pool_h=2, pool_w=2, pool_sh=2, pool_sw=2,
                     in_h=4, in_w=4, in_c=8, global_pool=False,
                     int16_mode=False, seed=42):
    """Generate one pooling test layer (input + params + expected output).

    Returns dict with packed uint32 word arrays and metadata.
    """
    np.random.seed(seed)

    if int16_mode:
        act_range = (-200, 200)
        clamp_min, clamp_max = -32768, 32767
        dtype_act = np.int16
    else:
        act_range = (-64, 64)
        clamp_min, clamp_max = -128, 127
        dtype_act = np.int8

    if global_pool:
        out_h, out_w = 1, 1
    else:
        out_h = (in_h + 0 - pool_h) // pool_sh + 1  # no padding for pool tests
        out_w = (in_w + 0 - pool_w) // pool_sw + 1

    cfg = {
        'op_type': 3,
        'data_type': 1 if int16_mode else 0,
        'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
        'out_h': out_h, 'out_w': out_w, 'out_c': in_c,
        'kernel_h': pool_h, 'kernel_w': pool_w,
        'stride_h': pool_sh, 'stride_w': pool_sw,
        'pad_top': 0, 'pad_left': 0,
        'pool_h': pool_h, 'pool_w': pool_w,
        'pool_stride_h': pool_sh, 'pool_stride_w': pool_sw,
        'pool_mode': 1 if mode == 'avg' else 0,
        'global_pool': global_pool,
        'relu6': True,
    }

    # Input activation
    input_nhwc = np.random.randint(act_range[0], act_range[1],
                                   (in_h, in_w, in_c), dtype=dtype_act)

    # Compute pooling acc
    acc = ref_pooling(input_nhwc, cfg, int16_mode=int16_mode)

    # Per-channel post-processing params (same as conv)
    M_arr = np.zeros(in_c, dtype=np.uint16)
    S_arr = np.zeros(in_c, dtype=np.uint8)
    bias_arr = np.zeros(in_c, dtype=np.int64)
    zp_arr = np.zeros(in_c, dtype=np.int16)
    for c in range(in_c):
        # Scale ~1.0 to keep values in range
        M_arr[c], S_arr[c] = compute_ms(0.8 + 0.4 * np.random.rand())

    out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                       relu6=True, clamp_min=clamp_min, clamp_max=clamp_max)

    # Pack
    if int16_mode:
        input_words = pack_i16_to_words(input_nhwc)
        output_words = pack_i16_to_words(out)
    else:
        input_words = pack_i8_to_words(input_nhwc)
        output_words = pack_i8_to_words(out)
    param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, in_c)

    # POOL_CFG register encoding
    pool_cfg_reg = (cfg['pool_mode']
                    | (pool_h << 4) | (pool_w << 8)
                    | (pool_sh << 12) | (pool_sw << 16)
                    | ((1 if global_pool else 0) << 20))

    # POST_CTRL: mode=CONV_REQ(00), relu_en=1, bias_en=0, zp_en=0
    # Since bias=0 and zp=0, we still enable them (PPU passes through 0)
    post_ctrl = 0x64  # bias_en + zp_en + relu_en
    if int16_mode:
        post_ctrl |= 0x80

    meta = {
        'op_type': 3,
        'data_type': cfg['data_type'],
        'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
        'out_h': out_h, 'out_w': out_w, 'out_c': in_c,
        'kernel_h': pool_h, 'kernel_w': pool_w,
        'stride_h': pool_sh, 'stride_w': pool_sw,
        'pad_top': 0, 'pad_left': 0,
        'pool_cfg': pool_cfg_reg,
        'post_ctrl': post_ctrl,
        'dma_wgt_size': 0,  # No weights for pooling
        'dma_in_size': len(input_words) * 4,
        'dma_out_size': len(output_words) * 4,
        'dma_param_count': in_c,
        'n_input_words': len(input_words),
        'n_output_words': len(output_words),
        'n_param_words': len(param_words),
    }

    return meta, {
        'input_words': input_words,
        'param_words': param_words,
        'output_words': output_words,
    }


def gen_add_test(h=4, w=4, c=8, relu=True, int16_mode=False, seed=100):
    """Generate one eltwise add test layer (2 inputs + params + expected output).

    Returns dict with packed uint32 word arrays and metadata.
    """
    np.random.seed(seed)

    if int16_mode:
        act_range = (-200, 200)
        clamp_min, clamp_max = -32768, 32767
        dtype_act = np.int16
    else:
        act_range = (-64, 64)
        clamp_min, clamp_max = -128, 127
        dtype_act = np.int8

    # Two inputs
    input_a = np.random.randint(act_range[0], act_range[1],
                                (h, w, c), dtype=dtype_act)
    input_b = np.random.randint(act_range[0], act_range[1],
                                (h, w, c), dtype=dtype_act)

    # Rescale params: scale ~0.5 for each branch (sum of two halves ~= 1.0)
    M_A, S_A = compute_ms(0.5 + 0.2 * np.random.rand())
    M_B, S_B = compute_ms(0.5 + 0.2 * np.random.rand())

    # Compute golden output
    out = ref_eltwise_add(input_a, input_b, M_A, S_A, M_B, S_B,
                          relu=relu, clamp_min=clamp_min, clamp_max=clamp_max)

    # Pack
    if int16_mode:
        input_a_words = pack_i16_to_words(input_a)
        input_b_words = pack_i16_to_words(input_b)
        output_words = pack_i16_to_words(out)
    else:
        input_a_words = pack_i8_to_words(input_a)
        input_b_words = pack_i8_to_words(input_b)
        output_words = pack_i8_to_words(out)

    add_param_words = pack_add_params(M_A, S_A, M_B, S_B)

    # POST_CTRL: mode=RELU_ONLY(10), relu_en
    post_ctrl = 0x02  # mode=10 at bits[1:0]
    if relu:
        post_ctrl |= 0x04  # relu_en
    if int16_mode:
        post_ctrl |= 0x80

    meta = {
        'op_type': 4,
        'data_type': 1 if int16_mode else 0,
        'in_h': h, 'in_w': w, 'in_c': c,
        'out_h': h, 'out_w': w, 'out_c': c,
        'kernel_h': 1, 'kernel_w': 1,
        'stride_h': 1, 'stride_w': 1,
        'pad_top': 0, 'pad_left': 0,
        'post_ctrl': post_ctrl,
        'dma_wgt_size': 0,
        'dma_in_size': len(input_a_words) * 4,
        'dma_out_size': len(output_words) * 4,
        'dma_param_count': 0,  # Add params go separately (2 words)
        'n_input_words': len(input_a_words),
        'n_input_b_words': len(input_b_words),
        'n_output_words': len(output_words),
        'n_add_param_words': len(add_param_words),
        'relu': relu,
        'M_A': M_A, 'S_A': S_A, 'M_B': M_B, 'S_B': S_B,
    }

    return meta, {
        'input_a_words': input_a_words,
        'input_b_words': input_b_words,
        'add_param_words': add_param_words,
        'output_words': output_words,
    }


def gen_resize_test(in_h=4, in_w=4, in_c=4, out_h=8, out_w=8,
                    resize_mode=0, int16_mode=False, seed=120):
    """Generate one resize test layer (input + params + expected output)."""
    np.random.seed(seed)

    if int16_mode:
        act_range = (-200, 200)
        clamp_min, clamp_max = -32768, 32767
        dtype_act = np.int16
    else:
        act_range = (-64, 64)
        clamp_min, clamp_max = -128, 127
        dtype_act = np.int8

    input_nhwc = np.random.randint(act_range[0], act_range[1],
                                   (in_h, in_w, in_c), dtype=dtype_act)

    cfg = {
        'op_type': 5,
        'data_type': 1 if int16_mode else 0,
        'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
        'out_h': out_h, 'out_w': out_w, 'out_c': in_c,
        'kernel_h': 1, 'kernel_w': 1,
        'stride_h': 1, 'stride_w': 1,
        'pad_top': 0, 'pad_left': 0,
        'resize_mode': resize_mode,
    }

    acc = ref_resize(input_nhwc, cfg, int16_mode=int16_mode)

    M_arr = np.ones(in_c, dtype=np.uint16)
    S_arr = np.zeros(in_c, dtype=np.uint8)
    bias_arr = np.zeros(in_c, dtype=np.int64)
    zp_arr = np.zeros(in_c, dtype=np.int16)

    out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                       relu6=False, clamp_min=clamp_min, clamp_max=clamp_max)

    if int16_mode:
        input_words = pack_i16_to_words(input_nhwc)
        output_words = pack_i16_to_words(out)
    else:
        input_words = pack_i8_to_words(input_nhwc)
        output_words = pack_i8_to_words(out)
    param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, in_c)

    scale_h_q44 = int(round((out_h / in_h) * 16.0)) & 0xFF
    scale_w_q44 = int(round((out_w / in_w) * 16.0)) & 0xFF
    resize_cfg_reg = resize_mode | (scale_h_q44 << 8) | (scale_w_q44 << 16)

    post_ctrl = 0x60
    if int16_mode:
        post_ctrl |= 0x80

    meta = {
        'op_type': 5,
        'data_type': cfg['data_type'],
        'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
        'out_h': out_h, 'out_w': out_w, 'out_c': in_c,
        'kernel_h': 1, 'kernel_w': 1,
        'stride_h': 1, 'stride_w': 1,
        'pad_top': 0, 'pad_left': 0,
        'resize_mode': resize_mode,
        'resize_cfg': resize_cfg_reg,
        'post_ctrl': post_ctrl,
        'dma_wgt_size': 0,
        'dma_in_size': len(input_words) * 4,
        'dma_out_size': len(output_words) * 4,
        'dma_param_count': in_c,
        'n_input_words': len(input_words),
        'n_output_words': len(output_words),
        'n_param_words': len(param_words),
    }

    return meta, {
        'input_words': input_words,
        'param_words': param_words,
        'output_words': output_words,
    }


# ═══════════════════════════════════════════════════════════════════════
# Deconv (Transposed Convolution) reference + test generator
# ═══════════════════════════════════════════════════════════════════════


def ref_deconv(input_nhwc, weight_ohwi, cfg):
    """Deconv (transposed conv) → int64 accumulator [out_h][out_w][out_c].

    Algorithm: insert zeros between input elements, then standard conv.
    Weight layout: [out_c][kh][kw][in_c] (same as Conv2D).
    """
    in_h, in_w, in_c = cfg['in_h'], cfg['in_w'], cfg['in_c']
    out_h, out_w, out_c = cfg['out_h'], cfg['out_w'], cfg['out_c']
    kh, kw = cfg['kernel_h'], cfg['kernel_w']
    ins_h, ins_w = cfg['insert_h'], cfg['insert_w']
    pad_t, pad_l = cfg['pad_top'], cfg['pad_left']

    exp_h = in_h + (in_h - 1) * ins_h
    exp_w = in_w + (in_w - 1) * ins_w

    acc = np.zeros((out_h, out_w, out_c), dtype=np.int64)
    for oh in range(out_h):
        for ow in range(out_w):
            for oc in range(out_c):
                s = np.int64(0)
                for fh in range(kh):
                    eh = oh + pad_t - fh
                    if eh < 0 or eh >= exp_h:
                        continue
                    if eh % (ins_h + 1) != 0:
                        continue
                    ih = eh // (ins_h + 1)
                    for fw in range(kw):
                        ew = ow + pad_l - fw
                        if ew < 0 or ew >= exp_w:
                            continue
                        if ew % (ins_w + 1) != 0:
                            continue
                        iw = ew // (ins_w + 1)
                        for ic in range(in_c):
                            s += np.int64(input_nhwc[ih, iw, ic]) * \
                                 np.int64(weight_ohwi[oc, fh, fw, ic])
                acc[oh, ow, oc] = s
    return acc


def gen_deconv_test(in_h=4, in_w=4, in_c=4, out_c=4,
                    kernel_h=2, kernel_w=2, insert_h=1, insert_w=1,
                    pad_top=0, pad_left=0,
                    int16_mode=False, seed=200):
    """Generate one deconv test layer (wgt + params + input + expected output)."""
    np.random.seed(seed)

    if int16_mode:
        act_range = (-100, 100)
        wgt_range = (-20, 20)
        clamp_min, clamp_max = -32768, 32767
        dtype_act = np.int16
        dtype_wgt = np.int16
    else:
        act_range = (-64, 64)
        wgt_range = (-30, 30)
        clamp_min, clamp_max = -128, 127
        dtype_act = np.int8
        dtype_wgt = np.int8

    # Compute output dimensions: out = (in-1)*(ins+1) + kernel - 2*pad
    out_h = (in_h - 1) * (insert_h + 1) + kernel_h - 2 * pad_top
    out_w = (in_w - 1) * (insert_w + 1) + kernel_w - 2 * pad_left

    input_nhwc = np.random.randint(act_range[0], act_range[1],
                                   (in_h, in_w, in_c), dtype=dtype_act)
    weight_ohwi = np.random.randint(wgt_range[0], wgt_range[1],
                                    (out_c, kernel_h, kernel_w, in_c), dtype=dtype_wgt)

    k_depth = kernel_h * kernel_w * in_c

    cfg = {
        'op_type': 6,
        'data_type': 1 if int16_mode else 0,
        'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
        'out_h': out_h, 'out_w': out_w, 'out_c': out_c,
        'kernel_h': kernel_h, 'kernel_w': kernel_w,
        'stride_h': 1, 'stride_w': 1,
        'pad_top': pad_top, 'pad_left': pad_left,
        'insert_h': insert_h, 'insert_w': insert_w,
    }

    acc = ref_deconv(input_nhwc, weight_ohwi, cfg)

    # Per-channel PPU params (identity requant for testing: M=1, S=0, bias=0, zp=0)
    M_arr = np.ones(out_c, dtype=np.uint16)
    S_arr = np.zeros(out_c, dtype=np.uint8)
    bias_arr = np.zeros(out_c, dtype=np.int64)
    zp_arr = np.zeros(out_c, dtype=np.int16)

    out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                       relu6=False, clamp_min=clamp_min, clamp_max=clamp_max)

    # Pack data
    if int16_mode:
        input_words = pack_i16_to_words(input_nhwc)
        wgt_words = pack_conv_weights_i16(weight_ohwi, out_c, k_depth)
        output_words = pack_i16_to_words(out)
    else:
        input_words = pack_i8_to_words(input_nhwc)
        wgt_words = pack_conv_weights_i8(weight_ohwi, out_c, k_depth)
        output_words = pack_i8_to_words(out)
    param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, out_c)

    # CSR register value: DECONV_CFG = INSERT_H | (INSERT_W << 8)
    deconv_cfg_reg = insert_h | (insert_w << 8)

    # post_ctrl: MODE_CONV_REQ (0x00), no relu6
    post_ctrl = 0x00
    if int16_mode:
        post_ctrl |= 0x80

    meta = {
        'op_type': 6,
        'data_type': cfg['data_type'],
        'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
        'out_h': out_h, 'out_w': out_w, 'out_c': out_c,
        'kernel_h': kernel_h, 'kernel_w': kernel_w,
        'stride_h': 1, 'stride_w': 1,
        'pad_top': pad_top, 'pad_left': pad_left,
        'insert_h': insert_h, 'insert_w': insert_w,
        'deconv_cfg': deconv_cfg_reg,
        'k_depth': k_depth,
        'post_ctrl': post_ctrl,
        'dma_wgt_size': len(wgt_words) * 4,
        'dma_in_size': len(input_words) * 4,
        'dma_out_size': len(output_words) * 4,
        'dma_param_count': out_c,
        'n_wgt_words': len(wgt_words),
        'n_input_words': len(input_words),
        'n_output_words': len(output_words),
        'n_param_words': len(param_words),
    }

    return meta, {
        'wgt_words': wgt_words,
        'input_words': input_words,
        'param_words': param_words,
        'output_words': output_words,
    }


# ═══════════════════════════════════════════════════════════════════════
# Concat (op_type=7) — Channel concatenation with per-branch rescale
# ═══════════════════════════════════════════════════════════════════════


def ref_concat(input_nhwc, M_A, S_A, concat_offset, concat_total_c,
               relu=True, clamp_min=-128, clamp_max=127):
    """Reference Concat: rescale + place at channel offset.

    Returns flat output array of shape [H, W, total_c].
    Input shape is [H, W, in_c]. Output only has channels written at
    [offset : offset+in_c]; the rest should be pre-filled (or from other branches).
    """
    h, w, in_c = input_nhwc.shape
    dtype_out = np.int8 if clamp_max <= 127 else np.int16
    out = np.zeros((h, w, concat_total_c), dtype=dtype_out)

    for ih in range(h):
        for iw in range(w):
            for ic in range(in_c):
                val = int(input_nhwc[ih, iw, ic])
                prod = val * int(M_A)
                if S_A > 0:
                    rescaled = (prod + (1 << (S_A - 1))) >> S_A
                else:
                    rescaled = prod
                rescaled = max(clamp_min, min(clamp_max, rescaled))
                if relu:
                    rescaled = max(0, rescaled)
                out[ih, iw, concat_offset + ic] = dtype_out(rescaled)
    return out


def gen_concat_test(h=4, w=4, branches=None, relu=True,
                    int16_mode=False, seed=200):
    """Generate concat test data for multiple branches.

    branches: list of dicts with {'in_c': int} per branch.
    Returns a list of (meta, data) tuples (one per branch) plus the expected
    combined output.
    """
    np.random.seed(seed)

    if branches is None:
        branches = [{'in_c': 4}, {'in_c': 4}]

    total_c = sum(b['in_c'] for b in branches)

    if int16_mode:
        act_range = (-200, 200)
        clamp_min, clamp_max = -32768, 32767
        dtype_act = np.int16
    else:
        act_range = (-64, 64)
        clamp_min, clamp_max = -128, 127
        dtype_act = np.int8

    # Generate input and params per branch
    branch_results = []
    offset = 0
    combined_out = np.zeros((h, w, total_c), dtype=dtype_act)

    for br in branches:
        in_c = br['in_c']
        inp = np.random.randint(act_range[0], act_range[1],
                                (h, w, in_c), dtype=dtype_act)
        # Rescale: scale ~0.8 (close to unity, keep values meaningful)
        M_A, S_A = compute_ms(0.6 + 0.3 * np.random.rand())

        # Compute golden for this branch
        branch_out = ref_concat(inp, M_A, S_A, offset, total_c,
                                relu=relu, clamp_min=clamp_min, clamp_max=clamp_max)

        # Merge into combined output
        combined_out[:, :, offset:offset+in_c] = branch_out[:, :, offset:offset+in_c]

        # Pack input
        if int16_mode:
            input_words = pack_i16_to_words(inp)
        else:
            input_words = pack_i8_to_words(inp)

        # Param: 1 word (M_A, S_A only)
        param_word = (int(M_A) & 0x7FFF) | ((int(S_A) & 0x3F) << 16)
        add_param_words = [param_word]

        # POST_CTRL: mode=RELU_ONLY(10), relu_en
        post_ctrl = 0x02  # mode=10 at bits[1:0]
        if relu:
            post_ctrl |= 0x04  # relu_en
        if int16_mode:
            post_ctrl |= 0x80

        # concat_cfg register value: [15:0]=offset, [31:16]=total_c
        concat_cfg = (offset & 0xFFFF) | ((total_c & 0xFFFF) << 16)

        meta = {
            'op_type': 7,
            'data_type': 1 if int16_mode else 0,
            'in_h': h, 'in_w': w, 'in_c': in_c,
            'out_h': h, 'out_w': w, 'out_c': in_c,
            'kernel_h': 1, 'kernel_w': 1,
            'stride_h': 1, 'stride_w': 1,
            'pad_top': 0, 'pad_left': 0,
            'post_ctrl': post_ctrl,
            'concat_cfg': concat_cfg,
            'concat_offset': offset,
            'concat_total_c': total_c,
            'dma_wgt_size': 0,
            'dma_in_size': len(input_words) * 4,
            'dma_out_size': (h * w * total_c * (2 if int16_mode else 1) + 3) // 4 * 4,
            'dma_param_count': 0,
            'n_input_words': len(input_words),
            'n_add_param_words': len(add_param_words),
            'M_A': M_A, 'S_A': S_A,
            'relu': relu,
        }

        branch_results.append((meta, {
            'input_words': input_words,
            'add_param_words': add_param_words,
        }))

        offset += in_c

    # Pack combined output
    if int16_mode:
        output_words = pack_i16_to_words(combined_out)
    else:
        output_words = pack_i8_to_words(combined_out)

    return branch_results, output_words, total_c


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'golden_dma_e2e')

    # Generate INT8 golden
    save_golden(os.path.join(base_dir, 'int8'), int16_mode=False)

    # Generate INT16 golden
    save_golden(os.path.join(base_dir, 'int16'), int16_mode=True)

    print(f"\n{'='*60}")
    print("DONE. Golden data ready for DMA E2E RTL verification.")
    print("  Run: make DUT=npu_top MODULE=test_npu_dma_e2e")
    print(f"{'='*60}")
