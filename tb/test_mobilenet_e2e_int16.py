"""
MobileNetV2-Tiny INT16 End-to-End RTL Test.

Runs 10 layers of MobileNetV2-Tiny through the RTL compute unit in INT16 mode,
comparing output against Python golden reference.

Prerequisites: run gen_mobilenet_golden_int16.py first to generate golden data.

SPDX-License-Identifier: Apache-2.0
"""

import os
import json
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly


GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'golden_mobilenet_int16')


def get_array_size(dut):
    """Get ARRAY_SIZE from the DUT parameter."""
    try:
        return int(dut.u_compute.ARRAY_SIZE.value)
    except AttributeError:
        return 16


async def reset_dut(dut):
    """Apply reset for 5 cycles with INT16 mode enabled."""
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.ppu_mode.value = 0
    dut.ppu_relu_en.value = 0
    dut.ppu_bias_en.value = 1
    dut.ppu_zp_en.value = 1
    dut.cfg_int16.value = 1     # INT16 mode
    dut.cfg_act_base.value = 0
    dut.cfg_out_base.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def write_sram_word(dut, bank, addr, data):
    """Write a 32-bit word to SRAM."""
    if bank == 'wgt':
        dut.u_sram_wgt.mem[addr].value = data
    elif bank == 'act':
        dut.u_sram_act.mem[addr].value = data
    elif bank == 'param':
        dut.u_sram_param.mem[addr].value = data


def read_sram_word(dut, bank, addr):
    """Read a 32-bit word from SRAM."""
    if bank == 'wgt':
        return int(dut.u_sram_wgt.mem[addr].value)
    elif bank == 'act':
        return int(dut.u_sram_act.mem[addr].value)
    elif bank == 'param':
        return int(dut.u_sram_param.mem[addr].value)


def set_cfg(dut, cfg, array_size, act_base=0, out_base=0):
    """Configure the DUT for a layer."""
    op_type = 0 if cfg['op_type'] == 'conv2d' else 1
    dut.cfg_op_type.value = op_type
    dut.cfg_in_c.value = cfg['in_c']
    dut.cfg_out_h.value = cfg['out_h']
    dut.cfg_out_w.value = cfg['out_w']
    dut.cfg_out_c.value = cfg['out_c']
    dut.cfg_kernel_h.value = cfg['kernel_h']
    dut.cfg_kernel_w.value = cfg['kernel_w']
    dut.cfg_stride_h.value = cfg['stride_h']
    dut.cfg_stride_w.value = cfg['stride_w']
    dut.cfg_pad_top.value = cfg['pad_top']
    dut.cfg_pad_left.value = cfg['pad_left']
    dut.cfg_tile_h.value = 0
    dut.cfg_tile_w.value = 0
    dut.cfg_tile_num_h.value = 1
    dut.cfg_tile_num_w.value = 1
    dut.cfg_in_w.value = cfg['in_w']
    dut.cfg_in_h.value = cfg['in_h']
    dut.cfg_act_base.value = act_base
    dut.cfg_out_base.value = out_base

    # PPU config: CONV_REQ mode with bias + zp enabled
    dut.ppu_mode.value = 0  # CONV_REQ
    dut.ppu_bias_en.value = 1
    dut.ppu_zp_en.value = 1
    dut.ppu_relu_en.value = 1 if cfg['relu6'] else 0


def unpack_i16(word, idx):
    """Unpack int16 from uint32 word at index (0=low, 1=high)."""
    val = (word >> (idx * 16)) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val


async def run_layer(dut, layer_idx, cfg, wgt_words, param_words,
                    input_words, expected_words, array_size):
    """Run a single layer through RTL in INT16 mode and verify output."""

    act_base = 0
    out_base = len(input_words)

    # Load weights into SRAM
    for i, w in enumerate(wgt_words):
        write_sram_word(dut, 'wgt', i, int(w))

    # Load params into SRAM
    for i, p in enumerate(param_words):
        write_sram_word(dut, 'param', i, int(p))

    # Load input activations at act_base
    for i, a in enumerate(input_words):
        write_sram_word(dut, 'act', act_base + i, int(a))

    # Configure
    set_cfg(dut, cfg, array_size, act_base=act_base, out_base=out_base)

    # Start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    timeout = 500000
    done_seen = False
    final_cyc = 0
    first_wr_addr = None
    for cyc in range(timeout):
        await RisingEdge(dut.clk)
        await ReadOnly()
        try:
            if first_wr_addr is None:
                wr_en = int(dut.u_compute.act_wr_en.value)
                if wr_en:
                    first_wr_addr = int(dut.u_compute.act_wr_addr.value)
            if int(dut.done.value) == 1:
                done_seen = True
                final_cyc = cyc + 1
                break
        except ValueError:
            pass

    if not done_seen:
        return False, f"Layer {layer_idx}: timeout after {timeout} cycles"

    # Read output and compare
    n_out_words = len(expected_words)
    mismatches = 0
    first_mismatch = None
    for i in range(n_out_words):
        try:
            got = read_sram_word(dut, 'act', out_base + i)
        except ValueError:
            got = 0xDEADBEEF
        exp = int(expected_words[i])
        if got != exp:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = (i, got, exp)

    if mismatches > 0:
        wi, got, exp = first_mismatch
        # Decode as two int16 half-words for readable error
        got_elems = [unpack_i16(got, 0), unpack_i16(got, 1)]
        exp_elems = [unpack_i16(exp, 0), unpack_i16(exp, 1)]
        return False, (f"Layer {layer_idx}: {mismatches}/{n_out_words} word mismatches. "
                       f"First at word {wi}: got {got_elems} exp {exp_elems}. "
                       f"out_base={out_base}, first_wr_addr={first_wr_addr}")

    return True, f"Layer {layer_idx}: PASS ({final_cyc} cycles)"


@cocotb.test()
async def test_mobilenet_int16_layer0(dut):
    """INT16 MobileNetV2-Tiny Layer 0: Conv2D 3x3, stride=2, 3->8, ReLU6."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    array_size = get_array_size(dut)

    with open(os.path.join(GOLDEN_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    layer_idx = 0
    cfg = metadata[layer_idx]
    wgt = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_wgt.npy'))
    param = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_param.npy'))
    inp = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_input.npy'))
    out = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_output.npy'))

    ok, msg = await run_layer(dut, layer_idx, cfg, wgt, param, inp, out, array_size)
    dut._log.info(msg)
    assert ok, msg


@cocotb.test()
async def test_mobilenet_int16_layer1(dut):
    """INT16 MobileNetV2-Tiny Layer 1: DWConv 3x3, stride=1, 8ch, ReLU6."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    array_size = get_array_size(dut)

    with open(os.path.join(GOLDEN_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    layer_idx = 1
    cfg = metadata[layer_idx]
    wgt = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_wgt.npy'))
    param = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_param.npy'))
    inp = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_input.npy'))
    out = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_output.npy'))

    ok, msg = await run_layer(dut, layer_idx, cfg, wgt, param, inp, out, array_size)
    dut._log.info(msg)
    assert ok, msg


@cocotb.test()
async def test_mobilenet_int16_layer2(dut):
    """INT16 MobileNetV2-Tiny Layer 2: Conv2D 1x1, 8->4, linear."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    array_size = get_array_size(dut)

    with open(os.path.join(GOLDEN_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    layer_idx = 2
    cfg = metadata[layer_idx]
    wgt = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_wgt.npy'))
    param = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_param.npy'))
    inp = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_input.npy'))
    out = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_output.npy'))

    ok, msg = await run_layer(dut, layer_idx, cfg, wgt, param, inp, out, array_size)
    dut._log.info(msg)
    assert ok, msg


@cocotb.test()
async def test_mobilenet_int16_all_layers(dut):
    """INT16 MobileNetV2-Tiny: run all 10 layers sequentially."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    array_size = get_array_size(dut)

    with open(os.path.join(GOLDEN_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    results = []
    for layer_idx in range(len(metadata)):
        await RisingEdge(dut.clk)
        await reset_dut(dut)

        cfg = metadata[layer_idx]
        wgt = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_wgt.npy'))
        param = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_param.npy'))
        inp = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_input.npy'))
        out = np.load(os.path.join(GOLDEN_DIR, f'layer_{layer_idx:02d}_output.npy'))

        ok, msg = await run_layer(dut, layer_idx, cfg, wgt, param, inp, out, array_size)
        dut._log.info(msg)
        results.append((layer_idx, ok, msg))

        if not ok:
            pass

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    dut._log.info(f"INT16 MobileNetV2-Tiny: {passed}/{total} layers passed")

    failed = [(i, msg) for i, ok, msg in results if not ok]
    assert len(failed) == 0, f"Failed layers: {failed}"
