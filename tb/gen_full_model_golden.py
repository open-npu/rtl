#!/usr/bin/env python3
"""
Generate golden data for AllOps-Mini full-model RTL E2E test.

18-layer model with 16×16 input, covering all 7 operator types:
  Conv2D(0), DWConv(1), Pooling(3), Add(4), Resize(5), Deconv(6), Concat(7)

Compact dimensions (max 8×8 spatial) ensure fast Icarus simulation while
exercising every operator in a realistic model topology with residual add
and concat skip connections.

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import os
import json

from gen_dma_e2e_golden import (
    compute_ms, ref_conv2d, ref_dwconv, ref_postproc, ref_pooling,
    ref_eltwise_add, ref_resize, ref_deconv, ref_concat,
    pack_i8_to_words, pack_i16_to_words,
    pack_params_for_sram, pack_conv_weights_i8, pack_conv_weights_i16,
    pack_dw_weights_i8, pack_dw_weights_i16, pack_add_params,
)

ACT_SRAM_WORDS = 8192   # 32KB
WGT_SRAM_WORDS = 16384  # 64KB


# ═══════════════════════════════════════════════════════════════════════
# Per-tile DMA scheduling for large layers
# ═══════════════════════════════════════════════════════════════════════


def needs_tiling(in_h, in_w, in_c, out_h, out_w, out_c, int16_mode):
    """Check if a layer's input+output exceeds SRAM."""
    epw = 2 if int16_mode else 4  # elements per word
    in_elems = in_h * in_w * in_c
    out_elems = out_h * out_w * out_c
    in_words = (in_elems + epw - 1) // epw
    out_words = (out_elems + epw - 1) // epw
    return (in_words + out_words) > ACT_SRAM_WORDS


def compute_tile_plan(in_h, in_w, in_c, out_h, out_w, out_c,
                      kernel_h, stride_h, pad_top, int16_mode):
    """Compute number of output-row tiles so each fits in SRAM.

    Returns tile_h (output rows per tile) and tile_num_h.
    The input crop for each tile is: (tile_h-1)*stride + kernel_h rows
    (potentially less at borders).
    """
    epw = 2 if int16_mode else 4

    # Try increasing tile divisions until it fits
    for div in [1, 2, 4, 8, 16, 32, 64]:
        tile_h = max(1, out_h // div)
        # Compute worst-case input crop height
        crop_h = (tile_h - 1) * stride_h + kernel_h
        crop_h = min(crop_h, in_h)
        in_elems = crop_h * in_w * in_c
        out_elems = tile_h * out_w * out_c
        in_words = (in_elems + epw - 1) // epw
        out_words = (out_elems + epw - 1) // epw
        if in_words + out_words <= ACT_SRAM_WORDS:
            tile_num_h = (out_h + tile_h - 1) // tile_h
            return tile_h, tile_num_h
    raise ValueError(f"Cannot fit even 1-row tile in SRAM: "
                     f"{in_w}x{in_c} -> {out_w}x{out_c}")


def crop_input_for_tile(full_input, tile_y, tile_h, out_h,
                        kernel_h, stride_h, pad_top, in_h):
    """Crop the rows of full_input needed for output tile tile_y.

    Returns (cropped_input, effective_pad_top, crop_in_h).
    """
    # Output row range for this tile
    out_row_start = tile_y * tile_h
    out_row_end = min(out_row_start + tile_h, out_h)
    actual_tile_h = out_row_end - out_row_start

    # Input row range needed (before padding)
    in_row_start = out_row_start * stride_h - pad_top
    in_row_end = (out_row_end - 1) * stride_h - pad_top + kernel_h

    # Clamp to valid input range
    actual_start = max(0, in_row_start)
    actual_end = min(in_h, in_row_end)

    # Effective padding for this tile
    eff_pad_top = max(0, -in_row_start)

    crop = full_input[actual_start:actual_end, :, :]
    return crop, eff_pad_top, actual_tile_h


# ═══════════════════════════════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════════════════════════════


def build_allops_mini(int16_mode=False):
    """Build AllOps-Mini model: 18 layers, 16×16×3 input, all 7 op types.

    Compact dimensions (max 8×8 spatial) for fast Icarus simulation.
    All layers fit in SRAM natively — no tiling needed.

    Returns list of "invocations" — each invocation is a dict with:
      - 'meta': CSR metadata for the DMA E2E test
      - 'data': dict of packed uint32 word arrays
      - 'layer_idx': original layer index
      - 'tile_idx': tile index within the layer (-1 if not tiled)
    """
    np.random.seed(2025)

    if int16_mode:
        wgt_range = (-32, 33)
        act_range = (-200, 200)
        clamp_min, clamp_max = -32768, 32767
        dtype_act = np.int16
        dtype_wgt = np.int16
        epw = 2
    else:
        wgt_range = (-8, 9)
        act_range = (-64, 64)
        clamp_min, clamp_max = -128, 127
        dtype_act = np.int8
        dtype_wgt = np.int8
        epw = 4

    invocations = []
    scale_in = 1.0 / 64.0

    # Generate input — 16×16×3 for fast Icarus simulation
    INPUT_H, INPUT_W = 16, 16
    input_img = np.random.randint(act_range[0], act_range[1],
                                   (INPUT_H, INPUT_W, 3), dtype=dtype_act)

    # Storage for layer outputs (for residual connections)
    layer_outputs = {}

    def gen_ppu_params(n_ch, eff_scale, relu6=True):
        """Generate per-channel PPU params."""
        M_arr = np.zeros(n_ch, dtype=np.uint16)
        S_arr = np.zeros(n_ch, dtype=np.uint8)
        bias_arr = np.random.randint(-50, 50, (n_ch,), dtype=np.int64)
        zp_arr = np.zeros(n_ch, dtype=np.int16)
        for c in range(n_ch):
            es = eff_scale * (0.8 + 0.4 * np.random.rand())
            M_arr[c], S_arr[c] = compute_ms(es)
        return M_arr, S_arr, bias_arr, zp_arr

    def make_post_ctrl(mode_bits, relu6, int16_mode):
        """Build POST_CTRL register value."""
        pc = mode_bits
        if relu6:
            pc |= 0x04
        if mode_bits == 0x00:  # CONV_REQ mode
            pc |= 0x60  # bias_en + zp_en
        if int16_mode:
            pc |= 0x80
        return pc

    def pack_act(arr):
        return pack_i16_to_words(arr) if int16_mode else pack_i8_to_words(arr)

    def add_conv_invocations(layer_idx, in_act, in_h, in_w, in_c,
                             out_c, kernel, stride, relu6, pad=None):
        """Add Conv2D invocations (potentially tiled)."""
        nonlocal scale_in
        if pad is None:
            pad = (kernel - 1) // 2
        out_h = (in_h + 2 * pad - kernel) // stride + 1
        out_w = (in_w + 2 * pad - kernel) // stride + 1
        k_depth = kernel * kernel * in_c

        # Generate weights
        w = np.random.randint(wgt_range[0], wgt_range[1],
                              (out_c, kernel, kernel, in_c), dtype=dtype_wgt)
        if int16_mode:
            wgt_words = pack_conv_weights_i16(w, out_c, k_depth)
        else:
            wgt_words = pack_conv_weights_i8(w, out_c, k_depth)

        # PPU params
        s_w = 1.0 / 64.0
        eff_scale = (scale_in * s_w) / scale_in
        M_arr, S_arr, bias_arr, zp_arr = gen_ppu_params(out_c, eff_scale, relu6)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, out_c)
        post_ctrl = make_post_ctrl(0x00, relu6, int16_mode)

        # Compute full golden output
        cfg_full = {
            'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
            'out_h': out_h, 'out_w': out_w, 'out_c': out_c,
            'kernel_h': kernel, 'kernel_w': kernel,
            'stride_h': stride, 'stride_w': stride,
            'pad_top': pad, 'pad_left': pad,
        }
        acc_full = ref_conv2d(in_act, w, cfg_full)
        out_full = ref_postproc(acc_full, M_arr, S_arr, bias_arr, zp_arr,
                                relu6=relu6, clamp_min=clamp_min,
                                clamp_max=clamp_max)

        # Check if tiling needed
        in_words_full = len(pack_act(in_act))
        out_words_full = len(pack_act(out_full))

        if in_words_full + out_words_full <= ACT_SRAM_WORDS:
            # No tiling — single invocation
            input_words = pack_act(in_act)
            output_words = pack_act(out_full)
            invocations.append({
                'meta': {
                    'op_type': 0, 'data_type': 1 if int16_mode else 0,
                    'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
                    'out_h': out_h, 'out_w': out_w, 'out_c': out_c,
                    'kernel_h': kernel, 'kernel_w': kernel,
                    'stride_h': stride, 'stride_w': stride,
                    'pad_top': pad, 'pad_left': pad,
                    'k_depth': k_depth, 'relu6': relu6,
                    'post_ctrl': post_ctrl,
                    'dma_param_count': out_c,
                    'tile_h': 0, 'tile_w': 0,
                    'tile_num_h': 1, 'tile_num_w': 1,
                },
                'data': {
                    'wgt_words': wgt_words,
                    'param_words': param_words,
                    'input_words': input_words,
                    'output_words': output_words,
                },
                'layer_idx': layer_idx,
                'tile_idx': -1,
            })
        else:
            # Tiled — software-level per-tile scheduling
            tile_h, tile_num_h = compute_tile_plan(
                in_h, in_w, in_c, out_h, out_w, out_c,
                kernel, stride, pad, int16_mode)

            for ty in range(tile_num_h):
                crop, eff_pad_top, actual_tile_h = crop_input_for_tile(
                    in_act, ty, tile_h, out_h, kernel, stride, pad, in_h)

                crop_h = crop.shape[0]

                # Compute golden for this tile
                cfg_tile = {
                    'in_h': crop_h, 'in_w': in_w, 'in_c': in_c,
                    'out_h': actual_tile_h, 'out_w': out_w, 'out_c': out_c,
                    'kernel_h': kernel, 'kernel_w': kernel,
                    'stride_h': stride, 'stride_w': stride,
                    'pad_top': eff_pad_top, 'pad_left': pad,
                }
                acc_tile = ref_conv2d(crop, w, cfg_tile)
                out_tile = ref_postproc(acc_tile, M_arr, S_arr, bias_arr,
                                        zp_arr, relu6=relu6,
                                        clamp_min=clamp_min,
                                        clamp_max=clamp_max)

                input_words = pack_act(crop)
                output_words = pack_act(out_tile)

                invocations.append({
                    'meta': {
                        'op_type': 0, 'data_type': 1 if int16_mode else 0,
                        'in_h': crop_h, 'in_w': in_w, 'in_c': in_c,
                        'out_h': actual_tile_h, 'out_w': out_w, 'out_c': out_c,
                        'kernel_h': kernel, 'kernel_w': kernel,
                        'stride_h': stride, 'stride_w': stride,
                        'pad_top': eff_pad_top, 'pad_left': pad,
                        'k_depth': k_depth, 'relu6': relu6,
                        'post_ctrl': post_ctrl,
                        'dma_param_count': out_c,
                        'tile_h': 0, 'tile_w': 0,  # single-tile to HW
                        'tile_num_h': 1, 'tile_num_w': 1,
                    },
                    'data': {
                        'wgt_words': wgt_words,
                        'param_words': param_words,
                        'input_words': input_words,
                        'output_words': output_words,
                    },
                    'layer_idx': layer_idx,
                    'tile_idx': ty,
                })

        layer_outputs[layer_idx] = out_full
        scale_in = scale_in  # keep scale_in unchanged for simplicity
        return out_full, out_h, out_w, out_c

    def add_dw_invocations(layer_idx, in_act, in_h, in_w, ch,
                           stride, relu6):
        """Add DWConv invocations (potentially tiled)."""
        nonlocal scale_in
        pad = 1
        out_h = (in_h + 2 * pad - 3) // stride + 1
        out_w = (in_w + 2 * pad - 3) // stride + 1
        k_depth = 9

        w = np.random.randint(wgt_range[0], wgt_range[1],
                              (ch, 3, 3), dtype=dtype_wgt)
        if int16_mode:
            wgt_words = pack_dw_weights_i16(w, ch)
        else:
            wgt_words = pack_dw_weights_i8(w, ch)

        s_w = 1.0 / 64.0
        eff_scale = (scale_in * s_w) / scale_in
        M_arr, S_arr, bias_arr, zp_arr = gen_ppu_params(ch, eff_scale, relu6)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, ch)
        post_ctrl = make_post_ctrl(0x00, relu6, int16_mode)

        cfg_full = {
            'in_h': in_h, 'in_w': in_w, 'in_c': ch,
            'out_h': out_h, 'out_w': out_w, 'out_c': ch,
            'kernel_h': 3, 'kernel_w': 3,
            'stride_h': stride, 'stride_w': stride,
            'pad_top': pad, 'pad_left': pad,
        }
        acc_full = ref_dwconv(in_act, w, cfg_full)
        out_full = ref_postproc(acc_full, M_arr, S_arr, bias_arr, zp_arr,
                                relu6=relu6, clamp_min=clamp_min,
                                clamp_max=clamp_max)

        in_words_full = len(pack_act(in_act))
        out_words_full = len(pack_act(out_full))

        if in_words_full + out_words_full <= ACT_SRAM_WORDS:
            input_words = pack_act(in_act)
            output_words = pack_act(out_full)
            invocations.append({
                'meta': {
                    'op_type': 1, 'data_type': 1 if int16_mode else 0,
                    'in_h': in_h, 'in_w': in_w, 'in_c': ch,
                    'out_h': out_h, 'out_w': out_w, 'out_c': ch,
                    'kernel_h': 3, 'kernel_w': 3,
                    'stride_h': stride, 'stride_w': stride,
                    'pad_top': pad, 'pad_left': pad,
                    'k_depth': k_depth, 'relu6': relu6,
                    'post_ctrl': post_ctrl,
                    'dma_param_count': ch,
                    'tile_h': 0, 'tile_w': 0,
                    'tile_num_h': 1, 'tile_num_w': 1,
                },
                'data': {
                    'wgt_words': wgt_words,
                    'param_words': param_words,
                    'input_words': input_words,
                    'output_words': output_words,
                },
                'layer_idx': layer_idx,
                'tile_idx': -1,
            })
        else:
            tile_h, tile_num_h = compute_tile_plan(
                in_h, in_w, ch, out_h, out_w, ch,
                3, stride, pad, int16_mode)

            for ty in range(tile_num_h):
                crop, eff_pad_top, actual_tile_h = crop_input_for_tile(
                    in_act, ty, tile_h, out_h, 3, stride, pad, in_h)
                crop_h = crop.shape[0]

                cfg_tile = {
                    'in_h': crop_h, 'in_w': in_w, 'in_c': ch,
                    'out_h': actual_tile_h, 'out_w': out_w, 'out_c': ch,
                    'kernel_h': 3, 'kernel_w': 3,
                    'stride_h': stride, 'stride_w': stride,
                    'pad_top': eff_pad_top, 'pad_left': pad,
                }
                acc_tile = ref_dwconv(crop, w, cfg_tile)
                out_tile = ref_postproc(acc_tile, M_arr, S_arr, bias_arr,
                                        zp_arr, relu6=relu6,
                                        clamp_min=clamp_min,
                                        clamp_max=clamp_max)

                input_words = pack_act(crop)
                output_words = pack_act(out_tile)

                invocations.append({
                    'meta': {
                        'op_type': 1, 'data_type': 1 if int16_mode else 0,
                        'in_h': crop_h, 'in_w': in_w, 'in_c': ch,
                        'out_h': actual_tile_h, 'out_w': out_w, 'out_c': ch,
                        'kernel_h': 3, 'kernel_w': 3,
                        'stride_h': stride, 'stride_w': stride,
                        'pad_top': eff_pad_top, 'pad_left': pad,
                        'k_depth': k_depth, 'relu6': relu6,
                        'post_ctrl': post_ctrl,
                        'dma_param_count': ch,
                        'tile_h': 0, 'tile_w': 0,
                        'tile_num_h': 1, 'tile_num_w': 1,
                    },
                    'data': {
                        'wgt_words': wgt_words,
                        'param_words': param_words,
                        'input_words': input_words,
                        'output_words': output_words,
                    },
                    'layer_idx': layer_idx,
                    'tile_idx': ty,
                })

        layer_outputs[layer_idx] = out_full
        return out_full, out_h, out_w, ch

    def add_pooling_invocation(layer_idx, in_act, in_h, in_w, in_c,
                               pool_h, pool_w, pool_sh, pool_sw,
                               mode='max', global_pool=False, relu6=True):
        """Add Pooling invocation (no tiling — pooling layers are small)."""
        if global_pool:
            o_h, o_w = 1, 1
        else:
            o_h = (in_h - pool_h) // pool_sh + 1
            o_w = (in_w - pool_w) // pool_sw + 1

        cfg = {
            'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
            'out_h': o_h, 'out_w': o_w, 'out_c': in_c,
            'pool_h': pool_h, 'pool_w': pool_w,
            'pool_stride_h': pool_sh, 'pool_stride_w': pool_sw,
            'pool_mode': 1 if mode == 'avg' else 0,
            'global_pool': global_pool,
            'pad_top': 0, 'pad_left': 0,
        }

        acc = ref_pooling(in_act, cfg, int16_mode=int16_mode)
        M_arr, S_arr, bias_arr, zp_arr = gen_ppu_params(in_c, 1.0, relu6)
        bias_arr[:] = 0
        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=relu6, clamp_min=clamp_min, clamp_max=clamp_max)

        input_words = pack_act(in_act)
        output_words = pack_act(out)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, in_c)

        pool_cfg_reg = (cfg['pool_mode']
                        | (pool_h << 4) | (pool_w << 8)
                        | (pool_sh << 12) | (pool_sw << 16)
                        | ((1 if global_pool else 0) << 20))

        post_ctrl = make_post_ctrl(0x00, relu6, int16_mode)

        invocations.append({
            'meta': {
                'op_type': 3, 'data_type': 1 if int16_mode else 0,
                'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
                'out_h': o_h, 'out_w': o_w, 'out_c': in_c,
                'kernel_h': pool_h, 'kernel_w': pool_w,
                'stride_h': pool_sh, 'stride_w': pool_sw,
                'pad_top': 0, 'pad_left': 0,
                'pool_cfg': pool_cfg_reg,
                'relu6': relu6,
                'post_ctrl': post_ctrl,
                'dma_param_count': in_c,
                'tile_h': 0, 'tile_w': 0,
                'tile_num_h': 1, 'tile_num_w': 1,
            },
            'data': {
                'wgt_words': [],
                'param_words': param_words,
                'input_words': input_words,
                'output_words': output_words,
            },
            'layer_idx': layer_idx,
            'tile_idx': -1,
        })

        layer_outputs[layer_idx] = out
        return out, o_h, o_w, in_c

    def add_add_invocation(layer_idx, in_a, in_b, h, w, c, relu=True):
        """Add Eltwise Add invocation."""
        M_A, S_A = compute_ms(0.5 + 0.2 * np.random.rand())
        M_B, S_B = compute_ms(0.5 + 0.2 * np.random.rand())

        out = ref_eltwise_add(in_a, in_b, M_A, S_A, M_B, S_B,
                              relu=relu, clamp_min=clamp_min, clamp_max=clamp_max)

        input_a_words = pack_act(in_a)
        input_b_words = pack_act(in_b)
        output_words = pack_act(out)
        add_param_words = pack_add_params(M_A, S_A, M_B, S_B)

        post_ctrl = 0x02  # RELU_ONLY mode
        if relu:
            post_ctrl |= 0x04
        if int16_mode:
            post_ctrl |= 0x80

        invocations.append({
            'meta': {
                'op_type': 4, 'data_type': 1 if int16_mode else 0,
                'in_h': h, 'in_w': w, 'in_c': c,
                'out_h': h, 'out_w': w, 'out_c': c,
                'kernel_h': 1, 'kernel_w': 1,
                'stride_h': 1, 'stride_w': 1,
                'pad_top': 0, 'pad_left': 0,
                'relu': relu,
                'M_A': int(M_A), 'S_A': int(S_A),
                'M_B': int(M_B), 'S_B': int(S_B),
                'post_ctrl': post_ctrl,
                'dma_param_count': 0,
                'tile_h': 0, 'tile_w': 0,
                'tile_num_h': 1, 'tile_num_w': 1,
            },
            'data': {
                'wgt_words': [],
                'param_words': [],
                'input_words': input_a_words,
                'input_b_words': input_b_words,
                'add_param_words': add_param_words,
                'output_words': output_words,
            },
            'layer_idx': layer_idx,
            'tile_idx': -1,
        })

        layer_outputs[layer_idx] = out
        return out, h, w, c

    def add_resize_invocation(layer_idx, in_act, in_h, in_w, in_c,
                              out_h, out_w, resize_mode=0):
        """Add Resize invocation."""
        cfg = {
            'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
            'out_h': out_h, 'out_w': out_w, 'out_c': in_c,
            'resize_mode': resize_mode,
        }
        acc = ref_resize(in_act, cfg, int16_mode=int16_mode)

        M_arr = np.ones(in_c, dtype=np.uint16)
        S_arr = np.zeros(in_c, dtype=np.uint8)
        bias_arr = np.zeros(in_c, dtype=np.int64)
        zp_arr = np.zeros(in_c, dtype=np.int16)

        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=False, clamp_min=clamp_min, clamp_max=clamp_max)

        input_words = pack_act(in_act)
        output_words = pack_act(out)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, in_c)

        scale_h_q44 = int(round((out_h / in_h) * 16.0)) & 0xFF
        scale_w_q44 = int(round((out_w / in_w) * 16.0)) & 0xFF
        resize_cfg_reg = resize_mode | (scale_h_q44 << 8) | (scale_w_q44 << 16)

        post_ctrl = 0x60  # bias_en + zp_en, no relu
        if int16_mode:
            post_ctrl |= 0x80

        invocations.append({
            'meta': {
                'op_type': 5, 'data_type': 1 if int16_mode else 0,
                'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
                'out_h': out_h, 'out_w': out_w, 'out_c': in_c,
                'kernel_h': 1, 'kernel_w': 1,
                'stride_h': 1, 'stride_w': 1,
                'pad_top': 0, 'pad_left': 0,
                'resize_mode': resize_mode,
                'resize_cfg': resize_cfg_reg,
                'post_ctrl': post_ctrl,
                'dma_param_count': in_c,
                'tile_h': 0, 'tile_w': 0,
                'tile_num_h': 1, 'tile_num_w': 1,
            },
            'data': {
                'wgt_words': [],
                'param_words': param_words,
                'input_words': input_words,
                'output_words': output_words,
            },
            'layer_idx': layer_idx,
            'tile_idx': -1,
        })

        layer_outputs[layer_idx] = out
        return out, out_h, out_w, in_c

    def add_deconv_invocation(layer_idx, in_act, in_h, in_w, in_c,
                              out_c, kernel_h, kernel_w, insert_h, insert_w,
                              pad_top=0, pad_left=0):
        """Add Deconv invocation."""
        o_h = (in_h - 1) * (insert_h + 1) + kernel_h - 2 * pad_top
        o_w = (in_w - 1) * (insert_w + 1) + kernel_w - 2 * pad_left
        k_depth = kernel_h * kernel_w * in_c

        w = np.random.randint(wgt_range[0], wgt_range[1],
                              (out_c, kernel_h, kernel_w, in_c), dtype=dtype_wgt)

        cfg = {
            'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
            'out_h': o_h, 'out_w': o_w, 'out_c': out_c,
            'kernel_h': kernel_h, 'kernel_w': kernel_w,
            'stride_h': 1, 'stride_w': 1,
            'pad_top': pad_top, 'pad_left': pad_left,
            'insert_h': insert_h, 'insert_w': insert_w,
        }
        acc = ref_deconv(in_act, w, cfg)

        M_arr = np.ones(out_c, dtype=np.uint16)
        S_arr = np.zeros(out_c, dtype=np.uint8)
        bias_arr = np.zeros(out_c, dtype=np.int64)
        zp_arr = np.zeros(out_c, dtype=np.int16)

        out = ref_postproc(acc, M_arr, S_arr, bias_arr, zp_arr,
                           relu6=False, clamp_min=clamp_min, clamp_max=clamp_max)

        input_words = pack_act(in_act)
        output_words = pack_act(out)
        param_words = pack_params_for_sram(M_arr, S_arr, bias_arr, zp_arr, out_c)

        if int16_mode:
            wgt_words = pack_conv_weights_i16(w, out_c, k_depth)
        else:
            wgt_words = pack_conv_weights_i8(w, out_c, k_depth)

        deconv_cfg_reg = insert_h | (insert_w << 8)
        post_ctrl = 0x00  # CONV_REQ, no relu
        if int16_mode:
            post_ctrl |= 0x80

        invocations.append({
            'meta': {
                'op_type': 6, 'data_type': 1 if int16_mode else 0,
                'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
                'out_h': o_h, 'out_w': o_w, 'out_c': out_c,
                'kernel_h': kernel_h, 'kernel_w': kernel_w,
                'stride_h': 1, 'stride_w': 1,
                'pad_top': pad_top, 'pad_left': pad_left,
                'insert_h': insert_h, 'insert_w': insert_w,
                'deconv_cfg': deconv_cfg_reg,
                'k_depth': k_depth,
                'post_ctrl': post_ctrl,
                'dma_param_count': out_c,
                'tile_h': 0, 'tile_w': 0,
                'tile_num_h': 1, 'tile_num_w': 1,
            },
            'data': {
                'wgt_words': wgt_words,
                'param_words': param_words,
                'input_words': input_words,
                'output_words': output_words,
            },
            'layer_idx': layer_idx,
            'tile_idx': -1,
        })

        layer_outputs[layer_idx] = out
        return out, o_h, o_w, out_c

    def add_concat_invocations(layer_idx_a, layer_idx_b,
                               in_a, in_b, h, w, c_a, c_b, relu=True):
        """Add Concat invocations (2 branches → 2 invocations)."""
        total_c = c_a + c_b

        # Branch A: offset=0
        M_A_a, S_A_a = compute_ms(0.7 + 0.3 * np.random.rand())
        out_a = ref_concat(in_a, M_A_a, S_A_a, 0, total_c,
                           relu=relu, clamp_min=clamp_min, clamp_max=clamp_max)

        # Branch B: offset=c_a
        M_A_b, S_A_b = compute_ms(0.7 + 0.3 * np.random.rand())
        out_b = ref_concat(in_b, M_A_b, S_A_b, c_a, total_c,
                           relu=relu, clamp_min=clamp_min, clamp_max=clamp_max)

        # Combined output
        combined = out_a.copy()
        combined[:, :, c_a:c_a+c_b] = out_b[:, :, c_a:c_a+c_b]

        # Pack
        combined_words = pack_act(combined)

        # Branch A invocation
        input_a_words = pack_act(in_a)
        param_a = [(int(M_A_a) & 0x7FFF) | ((int(S_A_a) & 0x3F) << 16)]
        concat_cfg_a = (0 & 0xFFFF) | ((total_c & 0xFFFF) << 16)

        post_ctrl = 0x02  # RELU_ONLY mode
        if relu:
            post_ctrl |= 0x04
        if int16_mode:
            post_ctrl |= 0x80

        # Both branches need to know total output size for DMA_OUT
        dma_out_size = len(combined_words) * 4

        invocations.append({
            'meta': {
                'op_type': 7, 'data_type': 1 if int16_mode else 0,
                'in_h': h, 'in_w': w, 'in_c': c_a,
                'out_h': h, 'out_w': w, 'out_c': c_a,
                'kernel_h': 1, 'kernel_w': 1,
                'stride_h': 1, 'stride_w': 1,
                'pad_top': 0, 'pad_left': 0,
                'concat_cfg': concat_cfg_a,
                'concat_offset': 0,
                'concat_total_c': total_c,
                'M_A': int(M_A_a), 'S_A': int(S_A_a),
                'relu': relu,
                'post_ctrl': post_ctrl,
                'dma_param_count': 0,
                'dma_out_size_override': dma_out_size,
                'tile_h': 0, 'tile_w': 0,
                'tile_num_h': 1, 'tile_num_w': 1,
            },
            'data': {
                'wgt_words': [],
                'param_words': [],
                'input_words': input_a_words,
                'add_param_words': param_a,
                'output_words': [],  # intermediate, not checked
            },
            'layer_idx': layer_idx_a,
            'tile_idx': -1,
        })

        # Branch B invocation
        input_b_words = pack_act(in_b)
        param_b = [(int(M_A_b) & 0x7FFF) | ((int(S_A_b) & 0x3F) << 16)]
        concat_cfg_b = (c_a & 0xFFFF) | ((total_c & 0xFFFF) << 16)

        invocations.append({
            'meta': {
                'op_type': 7, 'data_type': 1 if int16_mode else 0,
                'in_h': h, 'in_w': w, 'in_c': c_b,
                'out_h': h, 'out_w': w, 'out_c': c_b,
                'kernel_h': 1, 'kernel_w': 1,
                'stride_h': 1, 'stride_w': 1,
                'pad_top': 0, 'pad_left': 0,
                'concat_cfg': concat_cfg_b,
                'concat_offset': c_a,
                'concat_total_c': total_c,
                'M_A': int(M_A_b), 'S_A': int(S_A_b),
                'relu': relu,
                'post_ctrl': post_ctrl,
                'dma_param_count': 0,
                'dma_out_size_override': dma_out_size,
                'tile_h': 0, 'tile_w': 0,
                'tile_num_h': 1, 'tile_num_w': 1,
            },
            'data': {
                'wgt_words': [],
                'param_words': [],
                'input_words': input_b_words,
                'add_param_words': param_b,
                'output_words': combined_words,  # final output to check
            },
            'layer_idx': layer_idx_b,
            'tile_idx': -1,
        })

        layer_outputs[layer_idx_a] = None  # intermediate
        layer_outputs[layer_idx_b] = combined
        return combined, h, w, total_c

    # ─── Build AllOps-Mini architecture (16×16 input, all 7 ops) ───
    # Compact dims (max 8×8 spatial) — all layers fit in SRAM, no tiling.
    #
    # Layer 0: Conv2D 16×16×3 → 8×8×8, k=3 s=2
    cur = input_img
    cur, h, w, c = add_conv_invocations(0, cur, INPUT_H, INPUT_W, 3, 8, 3, 2, True)

    # Layer 1: DWConv 8×8×8 → 8×8×8, k=3 s=1
    cur, h, w, c = add_dw_invocations(1, cur, h, w, c, 1, True)

    # Layer 2: Conv2D 8×8×8 → 8×8×16, k=1 s=1
    cur, h, w, c = add_conv_invocations(2, cur, h, w, c, 16, 1, 1, True)
    save_l2 = cur.copy()  # 8×8×16 for residual

    # Layer 3: DWConv 8×8×16 → 8×8×16, k=3 s=1
    cur, h, w, c = add_dw_invocations(3, cur, h, w, c, 1, True)

    # Layer 4: Conv2D 8×8×16 → 8×8×16, k=1 s=1
    cur, h, w, c = add_conv_invocations(4, cur, h, w, c, 16, 1, 1, False)

    # Layer 5: Add (residual L2 + L4)
    # 8×8×16 INT8 = 256 words → Add needs 2×256 = 512 ≤ 8192 ✓
    cur, h, w, c = add_add_invocation(5, save_l2, cur, h, w, c, relu=True)

    # Layer 6: DWConv 8×8×16 → 4×4×16, k=3 s=2
    cur, h, w, c = add_dw_invocations(6, cur, h, w, c, 2, True)

    # Layer 7: Conv2D 4×4×16 → 4×4×32, k=1 s=1
    cur, h, w, c = add_conv_invocations(7, cur, h, w, c, 32, 1, 1, True)
    save_l7 = cur.copy()  # 4×4×32 for concat branch B

    # Layer 8: Pooling MaxPool 4×4×32 → 2×2×32, k=2 s=2
    cur, h, w, c = add_pooling_invocation(8, cur, h, w, c,
                                           2, 2, 2, 2, mode='max')

    # Layer 9: Conv2D 2×2×32 → 2×2×32, k=1 s=1
    cur, h, w, c = add_conv_invocations(9, cur, h, w, c, 32, 1, 1, True)

    # Layer 10: Resize 2×2×32 → 4×4×32, nearest 2×
    cur, h, w, c = add_resize_invocation(10, cur, h, w, c, 4, 4, 0)

    # Layer 11: Conv2D 4×4×32 → 4×4×16, k=1 s=1
    cur, h, w, c = add_conv_invocations(11, cur, h, w, c, 16, 1, 1, True)

    # Layer 12+13: Concat (L11=4×4×16, L7=4×4×32) → 4×4×48
    concat_branch_a = cur       # L11: 4×4×16
    concat_branch_b = save_l7   # L7:  4×4×32
    cur, h, w, c = add_concat_invocations(
        12, 13, concat_branch_a, concat_branch_b,
        4, 4, 16, 32, relu=True)

    # Layer 14: Conv2D 4×4×48 → 2×2×16, k=3 s=2
    cur, h, w, c = add_conv_invocations(14, cur, h, w, c, 16, 3, 2, True)

    # Layer 15: Pooling GlobalAvgPool 2×2×16 → 1×1×16
    cur, h, w, c = add_pooling_invocation(15, cur, h, w, c,
                                           2, 2, 2, 2,
                                           mode='avg', global_pool=True)

    # Layer 16: Deconv 1×1×16 → 2×2×16, k=2 s=2
    cur, h, w, c = add_deconv_invocation(16, cur, h, w, c, 16,
                                          2, 2, 1, 1, 0, 0)

    # Layer 17: Conv2D 2×2×16 → 1×1×8, k=2 s=1 pad=0
    cur, h, w, c = add_conv_invocations(17, cur, h, w, c, 8, 2, 1, False, pad=0)

    return invocations


# ═══════════════════════════════════════════════════════════════════════
# DDR Address Layout
# ═══════════════════════════════════════════════════════════════════════


WGT_BASE   = 0x1000_0000
PARAM_BASE = 0x2000_0000
ACT_BASE   = 0x3000_0000
INV_OFFSET = 0x0000_4000   # 16KB per invocation (sufficient for mini model)
ADD_B_BASE = 0x4000_0000


def assign_ddr_addrs(invocations):
    """Assign DDR addresses to each invocation.

    For layer chaining across tiles: consecutive tiles of the same layer
    write to contiguous DDR output regions. The next layer reads from
    the combined output.
    """
    # Group invocations by layer_idx to compute chained addresses
    # Each layer's output DDR address = next layer's input DDR address
    #
    # For simplicity: each invocation gets its own address space.
    # The test will handle chaining by populating DDR from the golden
    # output of each invocation.

    for i, inv in enumerate(invocations):
        meta = inv['meta']
        data = inv['data']

        meta['ddr_wgt_addr'] = WGT_BASE + i * INV_OFFSET
        meta['ddr_param_addr'] = PARAM_BASE + i * INV_OFFSET
        meta['ddr_in_addr'] = ACT_BASE + i * INV_OFFSET
        meta['ddr_out_addr'] = ACT_BASE + (i + 1) * INV_OFFSET

        if 'input_b_words' in data:
            meta['ddr_add_b_addr'] = ADD_B_BASE + i * INV_OFFSET

        # Compute DMA sizes
        meta['dma_in_size'] = len(data['input_words']) * 4
        meta['dma_wgt_size'] = len(data.get('wgt_words', [])) * 4

        if 'dma_out_size_override' in meta:
            meta['dma_out_size'] = meta['dma_out_size_override']
        else:
            meta['dma_out_size'] = len(data['output_words']) * 4

        meta['n_input_words'] = len(data['input_words'])
        meta['n_output_words'] = len(data['output_words'])
        meta['n_wgt_words'] = len(data.get('wgt_words', []))
        meta['n_param_words'] = len(data.get('param_words', []))

        if 'input_b_words' in data:
            meta['n_input_b_words'] = len(data['input_b_words'])
        if 'add_param_words' in data:
            meta['n_add_param_words'] = len(data['add_param_words'])


# ═══════════════════════════════════════════════════════════════════════
# Save golden data
# ═══════════════════════════════════════════════════════════════════════


def save_golden(output_dir, int16_mode=False):
    """Generate and save golden data for full model E2E test."""
    mode_str = "INT16" if int16_mode else "INT8"
    print(f"\n{'='*60}")
    print(f"Generating AllOps-Mini golden data — {mode_str}")
    print(f"{'='*60}")

    invocations = build_allops_mini(int16_mode=int16_mode)
    assign_ddr_addrs(invocations)

    os.makedirs(output_dir, exist_ok=True)

    op_names = {0: 'Conv2D', 1: 'DWConv', 3: 'Pooling', 4: 'Add',
                5: 'Resize', 6: 'Deconv', 7: 'Concat'}

    metadata = []
    for i, inv in enumerate(invocations):
        meta = inv['meta']
        data = inv['data']

        prefix = f'inv_{i:03d}'

        # Save word arrays
        for key in ['wgt_words', 'param_words', 'input_words', 'output_words']:
            arr = data.get(key, [])
            np.save(os.path.join(output_dir, f'{prefix}_{key}.npy'),
                    np.array(arr, dtype=np.uint32))

        # Save input_b and add_params if present
        if 'input_b_words' in data:
            np.save(os.path.join(output_dir, f'{prefix}_input_b_words.npy'),
                    np.array(data['input_b_words'], dtype=np.uint32))
        if 'add_param_words' in data:
            np.save(os.path.join(output_dir, f'{prefix}_add_param_words.npy'),
                    np.array(data['add_param_words'], dtype=np.uint32))

        # Build metadata entry
        m = dict(meta)
        m['invocation_idx'] = i
        m['layer_idx'] = inv['layer_idx']
        m['tile_idx'] = inv['tile_idx']
        metadata.append(m)

        op_name = op_names.get(meta['op_type'], f"Op{meta['op_type']}")
        tile_s = f" tile[{inv['tile_idx']}]" if inv['tile_idx'] >= 0 else ""
        print(f"  Inv{i:3d} L{inv['layer_idx']:2d}{tile_s}: {op_name:7s} "
              f"[{meta['in_h']}x{meta['in_w']}x{meta['in_c']}] -> "
              f"[{meta['out_h']}x{meta['out_w']}x{meta['out_c']}] "
              f"in={meta['n_input_words']} out={meta['n_output_words']}w")

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved {len(invocations)} invocations ({len(set(inv['layer_idx'] for inv in invocations))} layers) to {output_dir}/")
    print(f"  Op type coverage: {sorted(set(m['op_type'] for m in metadata))}")

    return metadata


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'golden_full_model')

    save_golden(os.path.join(base_dir, 'int8'), int16_mode=False)

    print(f"\n{'='*60}")
    print("DONE. Golden data ready for AllOps-Mini full-model RTL E2E.")
    print(f"{'='*60}")
