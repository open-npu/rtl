#!/usr/bin/env python3
"""
Generate DMA E2E golden data for 32×32 Medium-Scale Tiling Test.

Model: MobileNetV2-style, 6 layers, 32×32 input, real channel widths (32/48).
Five variants:
  - int8:              INT8, no tiling (all layers fit in SRAM)
  - int8_tiled:        INT8 with spatial tiling (same model, tiling enabled)
  - int16:             INT16 with spatial tiling (required for layers 2,3)
  - int8_tiled_db_en:  INT8 with spatial tiling + DB_EN (max 32ch, bank-constrained)
  - int16_tiled_db_en: INT16 with spatial tiling + DB_EN (max 16ch, bank-constrained)

SRAM constraints:
  - ACT SRAM: 8192 words (32KB), holds input + output
  - WGT SRAM: 16384 words (64KB)
  - PARAM SRAM: 2048 words (8KB)
  - DB_EN: ACT SRAM split into two banks of 4096 words each

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import os
import json

ACT_SRAM_WORDS = 8192

# ═══════════════════════════════════════════════════════════════════════
# Utility functions (same as gen_dma_e2e_golden.py)
# ═══════════════════════════════════════════════════════════════════════


def compute_ms(eff_scale):
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
# Packing functions
# ═══════════════════════════════════════════════════════════════════════


def pack_i8_to_words(data):
    byte_arr = data.astype(np.int8).tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


def pack_i16_to_words(data):
    flat = data.flatten().astype(np.int16)
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.int16(0))
    words = []
    for i in range(0, len(flat), 2):
        lo = int(flat[i]) & 0xFFFF
        hi = int(flat[i + 1]) & 0xFFFF
        words.append(lo | (hi << 16))
    return words


def pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, n_ch):
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


def pack_conv_weights_i8(weight_ohwi, out_c, k_depth):
    byte_arr = weight_ohwi.astype(np.int8).tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


def pack_conv_weights_i16(weight_ohwi, out_c, k_depth):
    flat = weight_ohwi.reshape(out_c * k_depth).astype(np.int16)
    return pack_i16_to_words(flat)


def pack_dw_weights_i8(weight_chw, n_ch):
    byte_arr = weight_chw.astype(np.int8).tobytes()
    pad_len = (4 - len(byte_arr) % 4) % 4
    byte_arr = byte_arr + b'\x00' * pad_len
    words = []
    for i in range(0, len(byte_arr), 4):
        w = int.from_bytes(byte_arr[i:i+4], 'little', signed=False)
        words.append(w)
    return words


def pack_dw_weights_i16(weight_chw, n_ch):
    flat = weight_chw.flatten().astype(np.int16)
    return pack_i16_to_words(flat)


# ═══════════════════════════════════════════════════════════════════════
# Tiling computation
# ═══════════════════════════════════════════════════════════════════════


def compute_tile_params(in_words, out_words, out_h, out_w, out_c,
                        elems_per_word):
    """Compute tile_h/tile_w/tile_num_h/tile_num_w to fit SRAM.

    Returns (tile_h, tile_w, tile_num_h, tile_num_w).
    tile_h=0 means no tiling needed.
    """
    total = in_words + out_words
    if total <= ACT_SRAM_WORDS:
        return 0, 0, 1, 1

    # Need tiling. Output is out_h × out_w × out_c.
    # Try halving tile_h until it fits.
    tile_w = out_w  # keep full width
    for tile_h in [out_h // 2, out_h // 4, out_h // 8]:
        if tile_h < 1:
            continue
        tile_out_elems = tile_h * tile_w * out_c
        tile_out_words = (tile_out_elems + elems_per_word - 1) // elems_per_word
        if in_words + tile_out_words <= ACT_SRAM_WORDS:
            tile_num_h = (out_h + tile_h - 1) // tile_h
            tile_num_w = 1
            return tile_h, tile_w, tile_num_h, tile_num_w

    raise ValueError(f"Cannot fit layer in SRAM even with tiling: "
                     f"in_words={in_words}, out={out_h}x{out_w}x{out_c}")


# ═══════════════════════════════════════════════════════════════════════
# Model Builder
# ═══════════════════════════════════════════════════════════════════════


def build_model_32x32(int16_mode=False, force_tiling=False, db_en=False):
    """Build 32×32 medium-scale model.

    Args:
        int16_mode: Use INT16 data path.
        force_tiling: Force tiling even when SRAM fits (for INT8 tiled test).
        db_en: Use DB_EN-compatible channel widths (max 32ch) so that
               n_input_words + n_output_words <= ACT_DEPTH/2 per layer.

    Returns:
        layers, layer_data, input_nhwc
    """
    np.random.seed(42)  # Different seed from Tiny model

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

    # 32×32×3 input
    input_nhwc = np.random.randint(act_range[0], act_range[1],
                                   (32, 32, 3), dtype=dtype_act)
    current = input_nhwc.copy()

    def add_conv(in_act, in_h, in_w, in_c, out_c, kernel=3, stride=1,
                 relu6=True):
        nonlocal scale_in, current
        pad = (kernel - 1) // 2
        out_h = (in_h + 2 * pad - kernel) // stride + 1
        out_w = (in_w + 2 * pad - kernel) // stride + 1
        k_depth = kernel * kernel * in_c

        cfg = {
            'op_type': 0,
            'data_type': 1 if int16_mode else 0,
            'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
            'out_h': out_h, 'out_w': out_w, 'out_c': out_c,
            'kernel_h': kernel, 'kernel_w': kernel,
            'stride_h': stride, 'stride_w': stride,
            'pad_top': pad, 'pad_left': pad,
            'k_depth': k_depth,
            'relu6': relu6,
        }

        w = np.random.randint(wgt_range[0], wgt_range[1],
                              (out_c, kernel, kernel, in_c), dtype=dtype_wgt)

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

        acc = ref_conv2d(in_act, w, cfg)
        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=relu6, clamp_min=clamp_min, clamp_max=clamp_max)

        if int16_mode:
            wgt_words = pack_conv_weights_i16(w, out_c, k_depth)
            input_words = pack_i16_to_words(in_act)
            output_words = pack_i16_to_words(out)
        else:
            wgt_words = pack_conv_weights_i8(w, out_c, k_depth)
            input_words = pack_i8_to_words(in_act)
            output_words = pack_i8_to_words(out)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr,
                                           out_c)

        cfg['dma_wgt_size'] = len(wgt_words) * 4
        cfg['dma_in_size'] = len(input_words) * 4
        cfg['dma_out_size'] = len(output_words) * 4
        cfg['dma_param_count'] = out_c

        # Tiling
        in_w_count = len(input_words)
        out_w_count = len(output_words)
        if force_tiling and out_h >= 4:
            # Force tiling: use tile_h = out_h // 2
            th = out_h // 2
            cfg['tile_h'] = th
            cfg['tile_w'] = out_w
            cfg['tile_num_h'] = (out_h + th - 1) // th
            cfg['tile_num_w'] = 1
        else:
            th, tw, tnh, tnw = compute_tile_params(
                in_w_count, out_w_count, out_h, out_w, out_c, elems_per_word)
            cfg['tile_h'] = th
            cfg['tile_w'] = tw
            cfg['tile_num_h'] = tnh
            cfg['tile_num_w'] = tnw

        post_ctrl = 0x60
        if relu6:
            post_ctrl |= 0x04
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
        return out_h, out_w, out_c

    def add_dw(in_act, in_h, in_w, ch, stride=1, relu6=True):
        nonlocal scale_in, current
        pad = 1
        out_h = (in_h + 2 * pad - 3) // stride + 1
        out_w = (in_w + 2 * pad - 3) // stride + 1
        k_depth = 9

        cfg = {
            'op_type': 1,
            'data_type': 1 if int16_mode else 0,
            'in_h': in_h, 'in_w': in_w, 'in_c': ch,
            'out_h': out_h, 'out_w': out_w, 'out_c': ch,
            'kernel_h': 3, 'kernel_w': 3,
            'stride_h': stride, 'stride_w': stride,
            'pad_top': pad, 'pad_left': pad,
            'k_depth': k_depth,
            'relu6': relu6,
        }

        w = np.random.randint(wgt_range[0], wgt_range[1],
                              (ch, 3, 3), dtype=dtype_wgt)

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

        acc = ref_dwconv(in_act, w, cfg)
        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=relu6, clamp_min=clamp_min, clamp_max=clamp_max)

        if int16_mode:
            wgt_words = pack_dw_weights_i16(w, ch)
            input_words = pack_i16_to_words(in_act)
            output_words = pack_i16_to_words(out)
        else:
            wgt_words = pack_dw_weights_i8(w, ch)
            input_words = pack_i8_to_words(in_act)
            output_words = pack_i8_to_words(out)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, ch)

        cfg['dma_wgt_size'] = len(wgt_words) * 4
        cfg['dma_in_size'] = len(input_words) * 4
        cfg['dma_out_size'] = len(output_words) * 4
        cfg['dma_param_count'] = ch

        # Tiling
        in_w_count = len(input_words)
        out_w_count = len(output_words)
        if force_tiling and out_h >= 4:
            th = out_h // 2
            cfg['tile_h'] = th
            cfg['tile_w'] = out_w
            cfg['tile_num_h'] = (out_h + th - 1) // th
            cfg['tile_num_w'] = 1
        else:
            th, tw, tnh, tnw = compute_tile_params(
                in_w_count, out_w_count, out_h, out_w, ch, elems_per_word)
            cfg['tile_h'] = th
            cfg['tile_w'] = tw
            cfg['tile_num_h'] = tnh
            cfg['tile_num_w'] = tnw

        post_ctrl = 0x60
        if relu6:
            post_ctrl |= 0x04
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

    # ─── 6-layer model: 32×32 input ───
    # DB_EN: ACT SRAM split into Bank[0]=[0,4096), Bank[1]=[4096,8192)
    #   n_input_words + n_output_words <= 4096 per layer
    #   INT8: 4 elems/word → max 32ch (16×16×32=8192 elem=2048 words)
    #   INT16: 2 elems/word → max 16ch (16×16×16=4096 elem=2048 words)
    h, w, c = 32, 32, 3
    if db_en:
        # DB_EN tiling requires non-overlapping tiles (no halos = no padding).
        # Use Conv1x1 stride=1 pad=0 only. 16ch so per-tile n_in+n_out <= 4096.
        # With 32×32 spatial, tile_h=16 → per tile = 16×32×16 = 2048 words.
        out_ch = 8 if int16_mode else 16
        # All layers: Conv1x1, same spatial size
        h, w, c = add_conv(current, h, w, c, out_ch, kernel=1, stride=1, relu6=True)  # L0
        h, w, c = add_conv(current, h, w, c, out_ch, kernel=1, stride=1, relu6=True)  # L1
        h, w, c = add_conv(current, h, w, c, out_ch, kernel=1, stride=1, relu6=True)  # L2
        h, w, c = add_conv(current, h, w, c, out_ch, kernel=1, stride=1, relu6=True)  # L3
        h, w, c = add_conv(current, h, w, c, out_ch, kernel=1, stride=1, relu6=False) # L4
        h, w, c = add_conv(current, h, w, c, out_ch, kernel=1, stride=1, relu6=True)  # L5
    else:
        # Original model: varied channel widths (non-DB_EN)
        h, w, c = add_conv(current, h, w, c, 32, kernel=3, stride=2, relu6=True)  # L0: [32x32x3]->[16x16x32]
        h, w, c = add_dw(current, h, w, c, stride=1, relu6=True)                  # L1: [16x16x32]->[16x16x32]
        h, w, c = add_conv(current, h, w, c, 48, kernel=1, stride=1, relu6=True)  # L2: [16x16x32]->[16x16x48]
        h, w, c = add_dw(current, h, w, c, stride=1, relu6=True)                  # L3: [16x16x48]->[16x16x48]
        h, w, c = add_conv(current, h, w, c, 16, kernel=1, stride=1, relu6=False) # L4: [16x16x48]->[16x16x16]
        h, w, c = add_conv(current, h, w, c, 32, kernel=1, stride=1, relu6=True)  # L5: [16x16x16]->[16x16x32]

    return layers, layer_data, input_nhwc


# ═══════════════════════════════════════════════════════════════════════
# DDR Address Layout
# ═══════════════════════════════════════════════════════════════════════

WGT_BASE = 0x1000_0000
PARAM_BASE = 0x2000_0000
ACT_BASE = 0x3000_0000
LAYER_OFFSET = 0x0001_0000  # 64KB per layer (sufficient for medium-scale)


def get_ddr_addrs(layer_idx):
    return {
        'wgt_addr':   WGT_BASE + layer_idx * LAYER_OFFSET,
        'param_addr': PARAM_BASE + layer_idx * LAYER_OFFSET,
        'in_addr':    ACT_BASE + layer_idx * LAYER_OFFSET,
        'out_addr':   ACT_BASE + (layer_idx + 1) * LAYER_OFFSET,
    }


# ═══════════════════════════════════════════════════════════════════════
# Save golden data
# ═══════════════════════════════════════════════════════════════════════


def save_golden(output_dir, int16_mode=False, force_tiling=False, db_en=False):
    mode_str = "INT16" if int16_mode else "INT8"
    tile_str = " (tiled)" if force_tiling else ""
    db_str = " (DB_EN)" if db_en else ""
    print(f"\n{'='*60}")
    print(f"Generating 32x32 tiling golden — {mode_str}{tile_str}{db_str}")
    print(f"{'='*60}")

    layers, layer_data, input_nhwc = build_model_32x32(
        int16_mode=int16_mode, force_tiling=force_tiling, db_en=db_en)

    os.makedirs(output_dir, exist_ok=True)

    for i, cfg in enumerate(layers):
        op_name = 'Conv2D' if cfg['op_type'] == 0 else 'DWConv'
        tile_info = (f" tile={cfg['tile_h']}x{cfg['tile_w']}"
                     f" ({cfg['tile_num_h']}x{cfg['tile_num_w']})"
                     if cfg['tile_h'] > 0 else " (no tiling)")
        print(f"  L{i}: {op_name:6s} [{cfg['in_h']}x{cfg['in_w']}x{cfg['in_c']}]"
              f" -> [{cfg['out_h']}x{cfg['out_w']}x{cfg['out_c']}]"
              f" k={cfg['kernel_h']}x{cfg['kernel_w']} s={cfg['stride_h']}"
              f" {'ReLU6' if cfg['relu6'] else 'lin'}"
              f" post=0x{cfg['post_ctrl']:02X}{tile_info}")

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

        addrs = get_ddr_addrs(i)
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
            'tile_in_size': cfg['dma_in_size'] // (cfg['tile_num_h'] * cfg['tile_num_w'])
                           if (db_en and cfg['tile_h'] > 0) else 0,
            'dma_param_count': cfg['dma_param_count'],
            'tile_h': cfg['tile_h'],
            'tile_w': cfg['tile_w'],
            'tile_num_h': cfg['tile_num_h'],
            'tile_num_w': cfg['tile_num_w'],
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

    total_wgt = sum(m['n_wgt_words'] for m in metadata) * 4
    total_act = sum(m['n_input_words'] for m in metadata) * 4
    total_out = sum(m['n_output_words'] for m in metadata) * 4
    print(f"\n  Saved {len(layers)} layers to {output_dir}/")
    print(f"  Total: wgt={total_wgt}B act={total_act}B out={total_out}B")
    for i, m in enumerate(metadata):
        tile_s = (f" tile={m['tile_h']}x{m['tile_w']}"
                  if m['tile_h'] > 0 else "")
        print(f"  L{i}: wgt={m['n_wgt_words']:5d} param={m['n_param_words']:4d}"
              f" in={m['n_input_words']:5d} out={m['n_output_words']:5d} words"
              f"  DDR: in=0x{m['ddr_in_addr']:08X}"
              f" out=0x{m['ddr_out_addr']:08X}{tile_s}")

    return metadata


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'golden_dma_e2e_tiling')

    # Variant A: INT8, no tiling
    save_golden(os.path.join(base_dir, 'int8'),
                int16_mode=False, force_tiling=False)

    # Variant B: INT8, forced tiling
    save_golden(os.path.join(base_dir, 'int8_tiled'),
                int16_mode=False, force_tiling=True)

    # Variant C: INT16, auto tiling (required for some layers)
    save_golden(os.path.join(base_dir, 'int16'),
                int16_mode=True, force_tiling=False)

    # Variant D: INT8, forced tiling + DB_EN (max 32ch, bank-constrained)
    save_golden(os.path.join(base_dir, 'int8_tiled_db_en'),
                int16_mode=False, force_tiling=True, db_en=True)

    # Variant E: INT16, forced tiling + DB_EN (max 16ch, bank-constrained)
    save_golden(os.path.join(base_dir, 'int16_tiled_db_en'),
                int16_mode=True, force_tiling=True, db_en=True)

    print(f"\n{'='*60}")
    print("DONE. Golden data ready for 32x32 tiling DMA E2E tests.")
    print(f"{'='*60}")
