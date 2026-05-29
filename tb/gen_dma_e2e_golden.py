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
