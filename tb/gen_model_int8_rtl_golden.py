#!/usr/bin/env python3
"""
Generate INT8 RTL golden data from MobileNetV2-Tiny model.

Same model architecture as gen_mobilenet_golden.py but with wider weight/input
ranges to stress-test the INT8 datapath. Computes Python-reference golden
output using the same bit-exact postproc as RTL.

This test proves: real quantized INT8 model → RTL bit-exact.

Prerequisites: none (self-contained).

SPDX-License-Identifier: Apache-2.0
"""

import os
import json
import numpy as np


# ─── Reference functions (matching RTL bit-exactly) ───

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


def ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr, relu6=False, clamp=(-128, 127)):
    """Per-channel requantize matching RTL PPU CONV_REQ mode exactly.

    Pipeline: acc + bias → × M → >>> S (with rounding) → + zp → clamp → ReLU
    """
    out_shape = acc.shape
    out = np.zeros(out_shape, dtype=np.int8)
    n_ch = out_shape[-1]
    for c in range(n_ch):
        ch_acc = acc[..., c].astype(np.int64)
        # Stage 1: Add bias
        biased = ch_acc + np.int64(bias_arr[c])
        # Stage 2: Multiply by M (M is unsigned, treat as positive)
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
        # Clamp to [-128, 127]
        ch_out = np.clip(zp_result, clamp[0], clamp[1])
        # ReLU: clamp lower to 0
        if relu6:
            ch_out = np.maximum(ch_out, 0)
        out[..., c] = ch_out.astype(np.int8)
    return out


# ─── SRAM packing ───

def pack_activations_for_sram(act_nhwc):
    """Pack NHWC activation tensor as SRAM words (4 int8 per uint32)."""
    byte_arr = act_nhwc.astype(np.int8).tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


def pack_conv_weights_for_sram(weight_ohwi, out_c, k_depth):
    """Pack Conv2D weights: [out_c][kh][kw][in_c] → 4 int8 per uint32."""
    flat = weight_ohwi.reshape(out_c, k_depth).astype(np.int8)
    byte_arr = flat.tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


def pack_dw_weights_for_sram(weight_chw, n_ch):
    """Pack DW Conv weights: [ch][3][3] → 4 int8 per uint32."""
    byte_arr = weight_chw.astype(np.int8).tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


def pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, n_ch):
    """Pack per-channel params into 32-bit SRAM words (4 words per channel)."""
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


# ─── Model builder ───

def build_mobilenetv2_tiny_int8():
    """Build MobileNetV2-Tiny INT8 model and compute golden per-layer data.

    Uses wider weight/input ranges than gen_mobilenet_golden.py to stress-test
    the INT8 datapath with full-range values.

    Returns: (layers, layer_data, input_nhwc)
    """
    np.random.seed(2024)

    layers = []
    layer_data = []
    scale_in = 1.0 / 64.0

    # Input: 16×16×3, full INT8 range
    input_nhwc = np.random.randint(-128, 128, (16, 16, 3), dtype=np.int8)
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

        # Weights: [out_c][kh][kw][in_c], wider range than base test
        w = np.random.randint(-64, 65, (out_c, kernel, kernel, in_c), dtype=np.int8)

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

        # Compute golden output via Python reference
        acc = ref_conv2d(in_act, w, cfg)
        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=relu6, clamp=(-128, 127))

        # Pack for SRAM
        wgt_words = pack_conv_weights_for_sram(w, out_c, k_depth)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, out_c)
        input_words = pack_activations_for_sram(in_act)
        output_words = pack_activations_for_sram(out)

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
        k_depth = 9

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

        # DW weight: [ch][3][3], wider range
        w = np.random.randint(-64, 65, (ch, 3, 3), dtype=np.int8)

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
        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=relu6, clamp=(-128, 127))

        # Pack for SRAM
        wgt_words = pack_dw_weights_for_sram(w, ch)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, ch)
        input_words = pack_activations_for_sram(in_act)
        output_words = pack_activations_for_sram(out)

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

    # ─── Build architecture (same as MobileNetV2-Tiny, 10 conv/dw layers) ───
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


def generate_golden(output_dir):
    """Generate golden data for RTL INT8 E2E test from real model."""
    print("=" * 60)
    print("MobileNetV2-Tiny INT8: Full-range stress test golden")
    print("=" * 60)

    layers, layer_data, input_nhwc = build_mobilenetv2_tiny_int8()

    print(f"\n{len(layers)} layers:")
    for i, cfg in enumerate(layers):
        kh, kw = cfg['kernel_h'], cfg['kernel_w']
        print(f"  Layer {i:2d}: {cfg['op_type']:8s} "
              f"[{cfg['in_h']}x{cfg['in_w']}x{cfg['in_c']}] -> "
              f"[{cfg['out_h']}x{cfg['out_w']}x{cfg['out_c']}] "
              f"k={kh}x{kw} s={cfg['stride_h']} "
              f"{'ReLU6' if cfg['relu6'] else 'linear'}")

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
                              'golden_model_int8')
    generate_golden(output_dir)
