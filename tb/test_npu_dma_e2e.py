# Open-NPU RTL — DMA E2E Tests (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# Full-chip DMA E2E tests exercising:
#   CPU(CSR) -> DMA(Wishbone) -> DDR -> SRAM -> Compute -> SRAM -> DMA -> DDR
#
# Uses DUT=npu_top with WbMasterMem as DDR emulator.
# Golden data from gen_dma_e2e_golden.py.

import os
import json
import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly


# ═══════════════════════════════════════════════════════════════════════
# Helpers (same as test_npu_top.py)
# ═══════════════════════════════════════════════════════════════════════


class WbSlave:
    """Wishbone B4 single-cycle master driver for npu_top's slave port."""

    def __init__(self, dut, clk):
        self.dut = dut
        self.clk = clk

    async def write(self, addr, data, sel=0xF):
        self.dut.wb_slv_cyc_i.value = 1
        self.dut.wb_slv_stb_i.value = 1
        self.dut.wb_slv_we_i.value = 1
        self.dut.wb_slv_adr_i.value = addr
        self.dut.wb_slv_dat_i.value = data
        self.dut.wb_slv_sel_i.value = sel
        await RisingEdge(self.clk)
        for _ in range(100):
            try:
                if self.dut.wb_slv_ack_o.value:
                    break
            except ValueError:
                pass
            await RisingEdge(self.clk)
        self.dut.wb_slv_cyc_i.value = 0
        self.dut.wb_slv_stb_i.value = 0
        await RisingEdge(self.clk)

    async def read(self, addr):
        self.dut.wb_slv_cyc_i.value = 1
        self.dut.wb_slv_stb_i.value = 1
        self.dut.wb_slv_we_i.value = 0
        self.dut.wb_slv_adr_i.value = addr
        self.dut.wb_slv_sel_i.value = 0xF
        await RisingEdge(self.clk)
        for _ in range(100):
            try:
                if self.dut.wb_slv_ack_o.value:
                    break
            except ValueError:
                pass
            await RisingEdge(self.clk)
        await ReadOnly()
        try:
            data = int(self.dut.wb_slv_dat_o.value)
        except ValueError:
            data = 0
        await Timer(1, unit="step")
        await RisingEdge(self.clk)
        self.dut.wb_slv_cyc_i.value = 0
        self.dut.wb_slv_stb_i.value = 0
        await RisingEdge(self.clk)
        return data


class WbMasterMem:
    """Emulates external memory (DDR) responding to DMA WB master requests."""

    def __init__(self, dut, clk):
        self.dut = dut
        self.clk = clk
        self.mem = {}  # addr -> 32-bit word

    def populate(self, base_addr, words):
        """Load uint32 word array into DDR at base_addr."""
        for i, w in enumerate(words):
            self.mem[base_addr + i * 4] = int(w)

    async def run(self):
        """Background task: respond to WB master transactions."""
        while True:
            await RisingEdge(self.clk)
            try:
                cyc = int(self.dut.wb_mst_cyc_o.value)
                stb = int(self.dut.wb_mst_stb_o.value)
            except ValueError:
                self.dut.wb_mst_ack_i.value = 0
                continue
            if cyc and stb:
                try:
                    addr = int(self.dut.wb_mst_adr_o.value)
                except ValueError:
                    self.dut.wb_mst_ack_i.value = 0
                    continue
                if self.dut.wb_mst_we_o.value:
                    try:
                        self.mem[addr] = int(self.dut.wb_mst_dat_o.value)
                    except ValueError:
                        self.mem[addr] = 0
                else:
                    self.dut.wb_mst_dat_i.value = self.mem.get(addr, 0)
                self.dut.wb_mst_ack_i.value = 1
            else:
                self.dut.wb_mst_ack_i.value = 0


async def reset(dut):
    """Apply reset sequence."""
    dut.rst_n.value = 0
    dut.wb_slv_cyc_i.value = 0
    dut.wb_slv_stb_i.value = 0
    dut.wb_slv_we_i.value = 0
    dut.wb_slv_adr_i.value = 0
    dut.wb_slv_dat_i.value = 0
    dut.wb_slv_sel_i.value = 0
    dut.wb_mst_ack_i.value = 0
    dut.wb_mst_dat_i.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


# ═══════════════════════════════════════════════════════════════════════
# Golden data loading
# ═══════════════════════════════════════════════════════════════════════

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'golden_dma_e2e')


def load_golden(mode='int8'):
    """Load golden data and metadata for given mode."""
    d = os.path.join(GOLDEN_DIR, mode)
    with open(os.path.join(d, 'metadata.json')) as f:
        metadata = json.load(f)

    layer_data = []
    for i in range(len(metadata)):
        prefix = f'layer_{i:02d}'
        layer_data.append({
            'wgt': np.load(os.path.join(d, f'{prefix}_wgt.npy')),
            'param': np.load(os.path.join(d, f'{prefix}_param.npy')),
            'input': np.load(os.path.join(d, f'{prefix}_input.npy')),
            'output': np.load(os.path.join(d, f'{prefix}_output.npy')),
        })
    return metadata, layer_data


# ═══════════════════════════════════════════════════════════════════════
# CSR programming helper
# ═══════════════════════════════════════════════════════════════════════


async def program_layer(wb, meta):
    """Program all CSR registers for one layer."""
    # LAYER_MODE: op_type | (data_type << 4)
    await wb.write(0x040, meta['op_type'] | (meta['data_type'] << 4))

    # Dimensions
    await wb.write(0x044, (meta['in_h'] << 16) | meta['in_w'])     # IN_DIM_HW
    await wb.write(0x048, meta['in_c'])                             # IN_DIM_C
    await wb.write(0x04C, (meta['out_h'] << 16) | meta['out_w'])   # OUT_DIM_HW
    await wb.write(0x050, meta['out_c'])                            # OUT_DIM_C

    # Kernel & stride & padding
    await wb.write(0x054, meta['kernel_h'] | (meta['kernel_w'] << 8))  # KERNEL_SIZE
    await wb.write(0x058, meta['stride_h'] | (meta['stride_w'] << 8))  # STRIDE
    await wb.write(0x05C, meta['pad_top'] | (meta['pad_left'] << 8))   # PADDING

    # Tiling config (from metadata if present, else no tiling)
    tile_h = meta.get('tile_h', 0)
    tile_w = meta.get('tile_w', 0)
    tile_num_h = meta.get('tile_num_h', 1)
    tile_num_w = meta.get('tile_num_w', 1)
    await wb.write(0x070, tile_h | (tile_w << 16))          # [15:0]=TILE_H, [31:16]=TILE_W
    await wb.write(0x074, tile_num_h | (tile_num_w << 16))  # [15:0]=NUM_H,  [31:16]=NUM_W

    # SRAM_BASE: act_base=0, out_base=after input data
    # out_base in upper 16 bits, act_base in lower 16 bits
    out_base = meta['n_input_words']  # output starts after input in SRAM
    await wb.write(0x078, (out_base << 16) | 0)

    # DMA addresses
    await wb.write(0x100, meta['ddr_in_addr'])     # DMA_IN_ADDR
    await wb.write(0x104, meta['ddr_out_addr'])    # DMA_OUT_ADDR
    await wb.write(0x108, meta['ddr_wgt_addr'])    # DMA_WGT_ADDR
    await wb.write(0x10C, meta['ddr_param_addr'])  # DMA_PARAM_ADDR

    # DMA sizes (bytes)
    await wb.write(0x128, meta['dma_in_size'])     # DMA_IN_SIZE
    await wb.write(0x12C, meta['dma_wgt_size'])    # DMA_WGT_SIZE
    await wb.write(0x130, meta['dma_out_size'])    # DMA_OUT_SIZE

    # Post-processing
    await wb.write(0x180, meta['post_ctrl'])       # POST_CTRL
    await wb.write(0x188, meta['dma_param_count']) # POST_PARAM_COUNT

    # No fusion
    await wb.write(0x118, 0)                       # DMA_CTRL


async def run_layer_and_wait(wb, dut, timeout=500000):
    """Start layer and poll until done. Returns True if completed."""
    await wb.write(0x000, 0x01)  # CTRL: START

    done = False
    for cyc in range(timeout):
        await RisingEdge(dut.clk)
        if cyc > 100 and cyc % 200 == 0:
            try:
                s = await wb.read(0x004)
                if (s & 0x1) == 0:
                    done = True
                    break
            except (ValueError, AttributeError):
                continue
    return done


def verify_output(mem, meta, golden_output, layer_idx):
    """Verify DMA output in DDR matches golden. Returns (pass, details)."""
    out_addr = meta['ddr_out_addr']
    n_words = meta['n_output_words']
    mismatches = []

    for i in range(n_words):
        addr = out_addr + i * 4
        got = mem.mem.get(addr, None)
        exp = int(golden_output[i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        detail = f"Layer {layer_idx}: {len(mismatches)}/{n_words} word mismatches"
        for idx, exp, got in mismatches[:5]:
            if got == 'NOT_WRITTEN':
                detail += f"\n  word[{idx}]: expected 0x{exp:08X}, NOT WRITTEN"
            else:
                detail += f"\n  word[{idx}]: expected 0x{exp:08X}, got 0x{got:08X}"
        if len(mismatches) > 5:
            detail += f"\n  ... and {len(mismatches)-5} more"
        return False, detail
    return True, f"Layer {layer_idx}: {n_words} words match"


# ═══════════════════════════════════════════════════════════════════════
# Test: Single layer smoke test (INT8)
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_single_layer_int8(dut):
    """Smoke test: single Conv2D 1x1 layer (L2) through DMA (INT8)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('int8')

    # Use layer 2 (Conv2D 1x1, 8->4, smallest compute)
    idx = 2
    meta = metadata[idx]
    data = layer_data[idx]

    # Populate DDR
    mem.populate(meta['ddr_wgt_addr'], data['wgt'])
    mem.populate(meta['ddr_param_addr'], data['param'])
    mem.populate(meta['ddr_in_addr'], data['input'])

    # Program CSR and run
    await program_layer(wb, meta)
    done = await run_layer_and_wait(wb, dut, timeout=50000)
    assert done, "Single layer INT8 did not complete"

    # Verify
    ok, detail = verify_output(mem, meta, data['output'], idx)
    dut._log.info(detail)
    assert ok, detail


# ═══════════════════════════════════════════════════════════════════════
# Test: Single layer smoke test (INT16)
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_single_layer_int16(dut):
    """Smoke test: single Conv2D 1x1 layer (L2) through DMA (INT16)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('int16')

    # Use layer 2 (Conv2D 1x1, 8->4)
    idx = 2
    meta = metadata[idx]
    data = layer_data[idx]

    # Populate DDR
    mem.populate(meta['ddr_wgt_addr'], data['wgt'])
    mem.populate(meta['ddr_param_addr'], data['param'])
    mem.populate(meta['ddr_in_addr'], data['input'])

    # Program CSR and run
    await program_layer(wb, meta)
    done = await run_layer_and_wait(wb, dut, timeout=50000)
    assert done, "Single layer INT16 did not complete"

    # Verify
    ok, detail = verify_output(mem, meta, data['output'], idx)
    dut._log.info(detail)
    assert ok, detail


# ═══════════════════════════════════════════════════════════════════════
# Test: Single DW Conv layer standalone (INT8) - Layer 1
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_single_dw_int8(dut):
    """Standalone DW Conv 3x3 layer (L1) through DMA (INT8, no prior layer)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('int8')

    # Use layer 1 (DW Conv 3x3, 8ch, stride=1, padding=1)
    idx = 1
    meta = metadata[idx]
    data = layer_data[idx]

    # Populate DDR
    mem.populate(meta['ddr_wgt_addr'], data['wgt'])
    mem.populate(meta['ddr_param_addr'], data['param'])
    mem.populate(meta['ddr_in_addr'], data['input'])

    # Program CSR and run
    await program_layer(wb, meta)
    done = await run_layer_and_wait(wb, dut, timeout=100000)
    assert done, "DW Conv layer did not complete"

    # Verify
    ok, detail = verify_output(mem, meta, data['output'], idx)
    dut._log.info(detail)
    assert ok, detail


# ═══════════════════════════════════════════════════════════════════════
# Test: Single DW Conv layer standalone (INT16) - Layer 1
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_single_dw_int16(dut):
    """Standalone DW Conv 3x3 layer (L1) through DMA (INT16, no prior layer)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('int16')

    # Use layer 1 (DW Conv 3x3, 8ch, stride=1, padding=1)
    idx = 1
    meta = metadata[idx]
    data = layer_data[idx]

    # Populate DDR
    mem.populate(meta['ddr_wgt_addr'], data['wgt'])
    mem.populate(meta['ddr_param_addr'], data['param'])
    mem.populate(meta['ddr_in_addr'], data['input'])

    # Program CSR and run
    await program_layer(wb, meta)
    done = await run_layer_and_wait(wb, dut, timeout=500000)
    assert done, "DW Conv INT16 layer did not complete"

    # Verify
    ok, detail = verify_output(mem, meta, data['output'], idx)
    dut._log.info(detail)
    assert ok, detail


# ═══════════════════════════════════════════════════════════════════════
# Test: Full 10-layer MobileNetV2-Tiny (INT8)
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_mobilenet_int8(dut):
    """Full 10-layer MobileNetV2-Tiny via DMA+Wishbone (INT8).

    Layer chaining: output of layer N in DDR is used as input for layer N+1.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('int8')

    # Pre-populate all weights and params in DDR
    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        mem.populate(meta['ddr_wgt_addr'], data['wgt'])
        mem.populate(meta['ddr_param_addr'], data['param'])

    # Load model input (layer 0 input) into DDR
    mem.populate(metadata[0]['ddr_in_addr'], layer_data[0]['input'])

    # Run all 10 layers sequentially
    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        dut._log.info(f"[INT8] Layer {idx}: "
                      f"{'Conv2D' if meta['op_type']==0 else 'DWConv'} "
                      f"[{meta['in_h']}x{meta['in_w']}x{meta['in_c']}] -> "
                      f"[{meta['out_h']}x{meta['out_w']}x{meta['out_c']}]")

        await program_layer(wb, meta)
        done = await run_layer_and_wait(wb, dut, timeout=500000)
        assert done, f"Layer {idx} did not complete within timeout"

        # Verify output
        ok, detail = verify_output(mem, meta, data['output'], idx)
        dut._log.info(f"  {detail}")
        assert ok, detail

    dut._log.info("[INT8] All 10 layers PASSED - DMA E2E bit-exact!")


# ═══════════════════════════════════════════════════════════════════════
# Test: Full 10-layer MobileNetV2-Tiny (INT16)
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_mobilenet_int16(dut):
    """Full 10-layer MobileNetV2-Tiny via DMA+Wishbone (INT16).

    Layer chaining: output of layer N in DDR is used as input for layer N+1.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('int16')

    # Pre-populate all weights and params in DDR
    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        mem.populate(meta['ddr_wgt_addr'], data['wgt'])
        mem.populate(meta['ddr_param_addr'], data['param'])

    # Load model input (layer 0 input) into DDR
    mem.populate(metadata[0]['ddr_in_addr'], layer_data[0]['input'])

    # Run all 10 layers sequentially
    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        dut._log.info(f"[INT16] Layer {idx}: "
                      f"{'Conv2D' if meta['op_type']==0 else 'DWConv'} "
                      f"[{meta['in_h']}x{meta['in_w']}x{meta['in_c']}] -> "
                      f"[{meta['out_h']}x{meta['out_w']}x{meta['out_c']}]")

        await program_layer(wb, meta)
        done = await run_layer_and_wait(wb, dut, timeout=500000)
        assert done, f"Layer {idx} did not complete within timeout"

        # Verify output
        ok, detail = verify_output(mem, meta, data['output'], idx)
        dut._log.info(f"  {detail}")
        assert ok, detail

    dut._log.info("[INT16] All 10 layers PASSED - DMA E2E bit-exact!")


# ═══════════════════════════════════════════════════════════════════════
# Tiling golden data loading
# ═══════════════════════════════════════════════════════════════════════

TILING_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'golden_dma_e2e_tiling')


def load_golden_tiling(mode='int8'):
    """Load golden data for 32x32 tiling tests."""
    d = os.path.join(TILING_GOLDEN_DIR, mode)
    with open(os.path.join(d, 'metadata.json')) as f:
        metadata = json.load(f)
    layer_data = []
    for i in range(len(metadata)):
        prefix = f'layer_{i:02d}'
        layer_data.append({
            'wgt': np.load(os.path.join(d, f'{prefix}_wgt.npy')),
            'param': np.load(os.path.join(d, f'{prefix}_param.npy')),
            'input': np.load(os.path.join(d, f'{prefix}_input.npy')),
            'output': np.load(os.path.join(d, f'{prefix}_output.npy')),
        })
    return metadata, layer_data


# ═══════════════════════════════════════════════════════════════════════
# Test: 32x32 Medium-Scale — INT8, No Tiling
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_tiling_int8(dut):
    """32x32 medium-scale 6-layer INT8, no tiling (validates multi-OC-group)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden_tiling('int8')

    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        mem.populate(meta['ddr_wgt_addr'], data['wgt'])
        mem.populate(meta['ddr_param_addr'], data['param'])

    mem.populate(metadata[0]['ddr_in_addr'], layer_data[0]['input'])

    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        dut._log.info(f"[INT8] Layer {idx}: "
                      f"{'Conv2D' if meta['op_type']==0 else 'DWConv'} "
                      f"[{meta['in_h']}x{meta['in_w']}x{meta['in_c']}] -> "
                      f"[{meta['out_h']}x{meta['out_w']}x{meta['out_c']}]")

        await program_layer(wb, meta)
        done = await run_layer_and_wait(wb, dut, timeout=2000000)
        assert done, f"Layer {idx} did not complete within timeout"

        ok, detail = verify_output(mem, meta, data['output'], idx)
        dut._log.info(f"  {detail}")
        assert ok, detail

    dut._log.info("[INT8 32x32] All 6 layers PASSED - no tiling!")


# ═══════════════════════════════════════════════════════════════════════
# Test: 32x32 Medium-Scale — INT8, With Tiling
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_tiling_int8_tiled(dut):
    """32x32 medium-scale 6-layer INT8 with spatial tiling (validates tiling FSM)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden_tiling('int8_tiled')

    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        mem.populate(meta['ddr_wgt_addr'], data['wgt'])
        mem.populate(meta['ddr_param_addr'], data['param'])

    mem.populate(metadata[0]['ddr_in_addr'], layer_data[0]['input'])

    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        tile_info = (f" tile={meta['tile_h']}x{meta['tile_w']}"
                     f" ({meta['tile_num_h']}x{meta['tile_num_w']})"
                     if meta['tile_h'] > 0 else "")
        dut._log.info(f"[INT8 tiled] Layer {idx}: "
                      f"{'Conv2D' if meta['op_type']==0 else 'DWConv'} "
                      f"[{meta['in_h']}x{meta['in_w']}x{meta['in_c']}] -> "
                      f"[{meta['out_h']}x{meta['out_w']}x{meta['out_c']}]"
                      f"{tile_info}")

        await program_layer(wb, meta)
        done = await run_layer_and_wait(wb, dut, timeout=2000000)
        assert done, f"Layer {idx} did not complete within timeout"

        ok, detail = verify_output(mem, meta, data['output'], idx)
        dut._log.info(f"  {detail}")
        assert ok, detail

    dut._log.info("[INT8 32x32 tiled] All 6 layers PASSED - tiling verified!")


# ═══════════════════════════════════════════════════════════════════════
# Test: 32x32 Medium-Scale — INT16, With Tiling
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_dma_e2e_tiling_int16(dut):
    """32x32 medium-scale 6-layer INT16 with spatial tiling (required for L2,L3)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden_tiling('int16')

    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        mem.populate(meta['ddr_wgt_addr'], data['wgt'])
        mem.populate(meta['ddr_param_addr'], data['param'])

    mem.populate(metadata[0]['ddr_in_addr'], layer_data[0]['input'])

    for idx, (meta, data) in enumerate(zip(metadata, layer_data)):
        tile_info = (f" tile={meta['tile_h']}x{meta['tile_w']}"
                     f" ({meta['tile_num_h']}x{meta['tile_num_w']})"
                     if meta['tile_h'] > 0 else "")
        dut._log.info(f"[INT16] Layer {idx}: "
                      f"{'Conv2D' if meta['op_type']==0 else 'DWConv'} "
                      f"[{meta['in_h']}x{meta['in_w']}x{meta['in_c']}] -> "
                      f"[{meta['out_h']}x{meta['out_w']}x{meta['out_c']}]"
                      f"{tile_info}")

        await program_layer(wb, meta)
        done = await run_layer_and_wait(wb, dut, timeout=2000000)
        assert done, f"Layer {idx} did not complete within timeout"

        ok, detail = verify_output(mem, meta, data['output'], idx)
        dut._log.info(f"  {detail}")
        assert ok, detail

    dut._log.info("[INT16 32x32] All 6 layers PASSED - tiling verified!")
