#!/usr/bin/env python3
"""
MobileNetV2-Tiny INT16 golden data generator for RTL simulation.

Same 10-layer network as INT8 version, but:
- Weights and activations use int16 range (values beyond [-128,127])
- Packing: 2 int16 elements per uint32 word (little-endian)
- Post-processing clamps to [-32768, 32767]

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import os
import json


def compute_ms(eff_scale):
    """Compute 15-bit multiplier M and 6-bit shift S: M / 2^S ≈ eff_scale."""
    best_s, best_m = 0, max(1, int(np.round(eff_scale)))
    for s in range(64):
        m = eff_scale * (2.0 ** s)
        if 1.0 <= m <= 32767.0:
            best_s = s
            best_m = int(np.round(m))
            if best_m >= 16384:
                break
    return max(1, min(32767, best_m)), best_s


def ref_conv2d(input_nhwc, weight_ohwi, cfg):
    """Python reference Conv2D → int64 accumulator output [out_h][out_w][out_c]."""
    in_h, in_w, in_c = cfg['in_h'], cfg['in_w'], cfg['in_c']
    out_h, out_w, out_c = cfg['out_h'], cfg['out_w'], cfg['out_c']
    kh, kw = cfg['kernel_h'], cfg['kernel_w']
    sh, sw = cfg['stride_h'], cfg['stride_w']
    pad_t, pad_l = cfg['pad_top'], cfg['pad_left']

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
    """Python reference DW Conv → int64 accumulator output [out_h][out_w][ch]."""
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


def ref_postproc_int16(acc, M_arr, S_arr, bias_arr, zp_arr, relu6=False):
    """Per-channel requantize matching RTL PPU INT16 mode.

    Pipeline: acc + bias → × M → >>> S (with rounding) → + zp → clamp[-32768,32767] → ReLU
    """
    out_shape = acc.shape
    out = np.zeros(out_shape, dtype=np.int16)
    n_ch = out_shape[-1]
    for c in range(n_ch):
        ch_acc = acc[..., c].astype(np.int64)
        # Stage 1: Add bias
        biased = ch_acc + np.int64(bias_arr[c])
        # Stage 2: Multiply by M
        m = int(M_arr[c])
        product = biased * np.int64(m)
        # Stage 3: Arithmetic right shift with rounding
        s = int(S_arr[c])
        if s > 0:
            rounded = product + (np.int64(1) << (s - 1))
            shifted = rounded >> s
        else:
            shifted = product
        # Stage 4: Add zero point
        zp_result = shifted + np.int64(zp_arr[c])
        # Clamp to [-32768, 32767]
        ch_out = np.clip(zp_result, -32768, 32767)
        # ReLU: clamp lower to 0
        if relu6:
            ch_out = np.maximum(ch_out, 0)
        out[..., c] = ch_out.astype(np.int16)
    return out


def pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, n_ch):
    """Pack per-channel params into 32-bit SRAM words.

    Format per channel (4 words), matching npu_compute S_PARAM_LOAD:
      Word 0: M[14:0] | (S[5:0] << 16)
      Word 1: ZP[15:0] | (bias[15:0] << 16)
      Word 2: bias[47:16]
      Word 3: bias[63:48] in [15:0]

    Returns list of uint32 values.
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


def pack_int16_for_sram(data_int16):
    """Pack int16 array as SRAM words: 2 elements per uint32 (little-endian).

    Works for weights, activations, and outputs.
    Returns list of uint32 words.
    """
    flat = data_int16.flatten().astype(np.int16)
    # Pad to even count
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.int16(0))
    # Pack pairs: low half-word = flat[2i], high half-word = flat[2i+1]
    words = []
    for i in range(0, len(flat), 2):
        lo = int(flat[i]) & 0xFFFF
        hi = int(flat[i + 1]) & 0xFFFF
        words.append(lo | (hi << 16))
    return words


def pack_conv_weights_for_sram_int16(weight_ohwi, out_c, k_depth):
    """Pack Conv2D weights in contiguous OHWI format for INT16 SRAM.

    weight_ohwi: [out_c][kh][kw][in_c] int16
    Layout: col c occupies k_depth elements starting at element c*k_depth.
    Returns list of uint32 words (2 int16 per word).
    """
    flat = weight_ohwi.reshape(out_c * k_depth).astype(np.int16)
    return pack_int16_for_sram(flat)


def pack_dw_weights_for_sram_int16(weight_chw, n_ch, kernel_size):
    """Pack DW Conv weights for INT16 SRAM.

    weight_chw: [ch][kh][kw] int16
    Returns list of uint32 words (2 int16 per word).
    """
    flat = weight_chw.flatten().astype(np.int16)
    return pack_int16_for_sram(flat)


def build_mobilenetv2_tiny_int16():
    """Build MobileNetV2-Tiny model with INT16 data and compute golden per-layer data.

    Returns:
        layers: list of layer config dicts
        layer_data: list of dicts with SRAM data
    """
    np.random.seed(2024)

    layers = []
    layer_data = []
    scale_in = 1.0 / 64.0

    # Generate random input: 16×16×3, INT16 range (beyond INT8)
    input_nhwc = np.random.randint(-500, 500, (16, 16, 3), dtype=np.int16)
    current = input_nhwc.copy()

    def add_conv(in_act, in_h, in_w, in_c, out_c, kernel=3, stride=1, relu6=True):
        nonlocal scale_in
        pad = (kernel - 1) // 2
        out_h = (in_h + 2 * pad - kernel) // stride + 1
        out_w = (in_w + 2 * pad - kernel) // stride + 1
        k_depth = kernel * kernel * in_c

        cfg = {
            'op_type': 'conv2d',
            'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
            'out_h': out_h, 'out_w': out_w, 'out_c': out_c,
            'kernel_h': kernel, 'kernel_w': kernel,
            'stride_h': stride, 'stride_w': stride,
            'pad_top': pad, 'pad_left': pad,
            'k_depth': k_depth,
            'relu6': relu6,
        }

        # Weights: [out_c][kh][kw][in_c] — INT16 range
        w = np.random.randint(-64, 65, (out_c, kernel, kernel, in_c), dtype=np.int16)

        # Per-channel quantization params
        s_w = 1.0 / 64.0
        acc_scale = scale_in * s_w
        s_out = scale_in
        eff_scale = acc_scale / s_out

        M_arr = np.zeros(out_c, dtype=np.uint16)
        S_arr = np.zeros(out_c, dtype=np.uint8)
        bias_arr = np.random.randint(-1000, 1000, (out_c,), dtype=np.int64)
        zp_arr = np.zeros(out_c, dtype=np.int16)
        for c in range(out_c):
            es = eff_scale * (0.8 + 0.4 * np.random.rand())
            M_arr[c], S_arr[c] = compute_ms(es)

        # Compute golden output
        acc = ref_conv2d(in_act, w, cfg)
        out = ref_postproc_int16(acc, M_arr, S_arr, bias_arr, zp_arr, relu6=relu6)

        # Pack for SRAM (INT16: 2 elements per word)
        wgt_words = pack_conv_weights_for_sram_int16(w, out_c, k_depth)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, out_c)
        input_words = pack_int16_for_sram(in_act)
        output_words = pack_int16_for_sram(out)

        cfg['M'] = M_arr
        cfg['S'] = S_arr
        cfg['bias'] = bias_arr
        cfg['zp'] = zp_arr

        layers.append(cfg)
        layer_data.append({
            'wgt_words': wgt_words,
            'param_words': param_words,
            'input_words': input_words,
            'output_words': output_words,
            'output_nhwc': out,
        })

        scale_in = s_out
        return out, out_h, out_w, out_c

    def add_dw(in_act, in_h, in_w, ch, stride=1, relu6=True):
        nonlocal scale_in
        pad = 1
        out_h = (in_h + 2 * pad - 3) // stride + 1
        out_w = (in_w + 2 * pad - 3) // stride + 1
        k_depth = 9  # 3*3

        cfg = {
            'op_type': 'dw_conv',
            'in_h': in_h, 'in_w': in_w, 'in_c': ch,
            'out_h': out_h, 'out_w': out_w, 'out_c': ch,
            'kernel_h': 3, 'kernel_w': 3,
            'stride_h': stride, 'stride_w': stride,
            'pad_top': pad, 'pad_left': pad,
            'k_depth': k_depth,
            'relu6': relu6,
        }

        # DW weight: [ch][3][3] — INT16 range
        w = np.random.randint(-64, 65, (ch, 3, 3), dtype=np.int16)

        # Per-channel quantization params
        s_w = 1.0 / 64.0
        acc_scale = scale_in * s_w
        s_out = scale_in
        eff_scale = acc_scale / s_out

        M_arr = np.zeros(ch, dtype=np.uint16)
        S_arr = np.zeros(ch, dtype=np.uint8)
        bias_arr = np.random.randint(-1000, 1000, (ch,), dtype=np.int64)
        zp_arr = np.zeros(ch, dtype=np.int16)
        for c in range(ch):
            es = eff_scale * (0.8 + 0.4 * np.random.rand())
            M_arr[c], S_arr[c] = compute_ms(es)

        # Compute golden output
        acc = ref_dwconv(in_act, w, cfg)
        out = ref_postproc_int16(acc, M_arr, S_arr, bias_arr, zp_arr, relu6=relu6)

        # Pack for SRAM (INT16: 2 elements per word)
        wgt_words = pack_dw_weights_for_sram_int16(w, ch, 9)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, ch)
        input_words = pack_int16_for_sram(in_act)
        output_words = pack_int16_for_sram(out)

        cfg['M'] = M_arr
        cfg['S'] = S_arr
        cfg['bias'] = bias_arr
        cfg['zp'] = zp_arr

        layers.append(cfg)
        layer_data.append({
            'wgt_words': wgt_words,
            'param_words': param_words,
            'input_words': input_words,
            'output_words': output_words,
            'output_nhwc': out,
        })

        scale_in = s_out
        return out, out_h, out_w, ch

    # ─── Build network (same structure as INT8) ───
    h, w, c = 16, 16, 3

    # Layer 0: Conv2D 3×3, stride=2, 3→8, ReLU6
    current, h, w, c = add_conv(current, h, w, c, 8, kernel=3, stride=2, relu6=True)
    # Layer 1: DWConv 3×3, stride=1, 8ch, ReLU6
    current, h, w, c = add_dw(current, h, w, c, stride=1, relu6=True)
    # Layer 2: Conv2D 1×1, 8→4, linear
    current, h, w, c = add_conv(current, h, w, c, 4, kernel=1, stride=1, relu6=False)
    # Layer 3: Conv2D 1×1, 4→24, ReLU6
    current, h, w, c = add_conv(current, h, w, c, 24, kernel=1, stride=1, relu6=True)
    # Layer 4: DWConv 3×3, stride=2, 24ch, ReLU6
    current, h, w, c = add_dw(current, h, w, c, stride=2, relu6=True)
    # Layer 5: Conv2D 1×1, 24→8, linear
    current, h, w, c = add_conv(current, h, w, c, 8, kernel=1, stride=1, relu6=False)
    # Layer 6: Conv2D 1×1, 8→48, ReLU6
    current, h, w, c = add_conv(current, h, w, c, 48, kernel=1, stride=1, relu6=True)
    # Layer 7: DWConv 3×3, stride=1, 48ch, ReLU6
    current, h, w, c = add_dw(current, h, w, c, stride=1, relu6=True)
    # Layer 8: Conv2D 1×1, 48→8, linear
    current, h, w, c = add_conv(current, h, w, c, 8, kernel=1, stride=1, relu6=False)
    # Layer 9: Conv2D 1×1, 8→32, ReLU6 (head)
    current, h, w, c = add_conv(current, h, w, c, 32, kernel=1, stride=1, relu6=True)

    return layers, layer_data, input_nhwc


def print_summary(layers):
    """Print network summary."""
    print(f"MobileNetV2-Tiny INT16 RTL Golden: {len(layers)} layers")
    for i, cfg in enumerate(layers):
        kh, kw = cfg['kernel_h'], cfg['kernel_w']
        print(f"  Layer {i:2d}: {cfg['op_type']:8s} "
              f"[{cfg['in_h']}x{cfg['in_w']}x{cfg['in_c']}] -> "
              f"[{cfg['out_h']}x{cfg['out_w']}x{cfg['out_c']}] "
              f"k={kh}x{kw} s={cfg['stride_h']} "
              f"{'ReLU6' if cfg['relu6'] else 'linear'}")


def save_golden(output_dir):
    """Generate and save INT16 golden data to files for cocotb test."""
    layers, layer_data, input_nhwc = build_mobilenetv2_tiny_int16()
    print_summary(layers)

    os.makedirs(output_dir, exist_ok=True)

    # Save input
    np.save(os.path.join(output_dir, 'input.npy'), input_nhwc)

    # Save per-layer data
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

        metadata.append({
            'index': i,
            'op_type': cfg['op_type'],
            'in_h': cfg['in_h'], 'in_w': cfg['in_w'], 'in_c': cfg['in_c'],
            'out_h': cfg['out_h'], 'out_w': cfg['out_w'], 'out_c': cfg['out_c'],
            'kernel_h': cfg['kernel_h'], 'kernel_w': cfg['kernel_w'],
            'stride_h': cfg['stride_h'], 'stride_w': cfg['stride_w'],
            'pad_top': cfg['pad_top'], 'pad_left': cfg['pad_left'],
            'k_depth': cfg['k_depth'],
            'relu6': cfg['relu6'],
            'n_wgt_words': len(data['wgt_words']),
            'n_param_words': len(data['param_words']),
            'n_input_words': len(data['input_words']),
            'n_output_words': len(data['output_words']),
        })

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(layers)} layers to {output_dir}/")
    for i, m in enumerate(metadata):
        print(f"  Layer {i:2d}: wgt={m['n_wgt_words']} param={m['n_param_words']} "
              f"in={m['n_input_words']} out={m['n_output_words']} words")


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'golden_mobilenet_int16')
    save_golden(output_dir)
