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


# ═══════════════════════════════════════════════════════════════════════
# Pooling + Add test imports
# ═══════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_dma_e2e_golden import gen_pooling_test, gen_add_test, gen_resize_test, gen_deconv_test, gen_concat_test


# DDR addresses for operator tests
POOL_WGT_ADDR   = 0x1000_0000
POOL_PARAM_ADDR = 0x2000_0000
POOL_IN_ADDR    = 0x3000_0000
POOL_OUT_ADDR   = 0x3001_0000
ADD_B_ADDR      = 0x4000_0000
ADD_PARAM_ADDR  = 0x2001_0000


async def program_pooling_layer(wb, meta, data):
    """Program CSR registers for a pooling test layer."""
    # LAYER_MODE: op_type=3 | (data_type << 4)
    await wb.write(0x040, meta['op_type'] | (meta['data_type'] << 4))

    # Dimensions
    await wb.write(0x044, (meta['in_h'] << 16) | meta['in_w'])
    await wb.write(0x048, meta['in_c'])
    await wb.write(0x04C, (meta['out_h'] << 16) | meta['out_w'])
    await wb.write(0x050, meta['out_c'])

    # Kernel & stride & padding (used by tiling / addr calc)
    await wb.write(0x054, meta['kernel_h'] | (meta['kernel_w'] << 8))
    await wb.write(0x058, meta['stride_h'] | (meta['stride_w'] << 8))
    await wb.write(0x05C, meta['pad_top'] | (meta['pad_left'] << 8))

    # POOL_CFG register (0x060)
    await wb.write(0x060, meta['pool_cfg'])

    # Tiling: no tiling
    await wb.write(0x070, 0)
    await wb.write(0x074, (1 << 16) | 1)

    # SRAM_BASE: act_base=0, out_base after input
    out_base = meta['n_input_words']
    await wb.write(0x078, (out_base << 16) | 0)

    # DMA addresses
    await wb.write(0x100, POOL_IN_ADDR)
    await wb.write(0x104, POOL_OUT_ADDR)
    await wb.write(0x108, POOL_WGT_ADDR)
    await wb.write(0x10C, POOL_PARAM_ADDR)

    # DMA sizes
    await wb.write(0x128, meta['dma_in_size'])
    await wb.write(0x12C, meta['dma_wgt_size'])  # 0 for pooling
    await wb.write(0x130, meta['dma_out_size'])

    # Post-processing
    await wb.write(0x180, meta['post_ctrl'])
    await wb.write(0x188, meta['dma_param_count'])

    # No fusion
    await wb.write(0x118, 0)


async def program_add_layer(wb, meta, data):
    """Program CSR registers for an eltwise add test layer."""
    # LAYER_MODE: op_type=4 | (data_type << 4)
    await wb.write(0x040, meta['op_type'] | (meta['data_type'] << 4))

    # Dimensions
    await wb.write(0x044, (meta['in_h'] << 16) | meta['in_w'])
    await wb.write(0x048, meta['in_c'])
    await wb.write(0x04C, (meta['out_h'] << 16) | meta['out_w'])
    await wb.write(0x050, meta['out_c'])

    # Kernel, stride, padding (1x1, stride 1, no pad)
    await wb.write(0x054, 1 | (1 << 8))
    await wb.write(0x058, 1 | (1 << 8))
    await wb.write(0x05C, 0)

    # Tiling: no tiling
    await wb.write(0x070, 0)
    await wb.write(0x074, (1 << 16) | 1)

    # SRAM_BASE: act_base=0, out_base after input A
    out_base = meta['n_input_words']
    await wb.write(0x078, (out_base << 16) | 0)

    # DMA addresses
    await wb.write(0x100, POOL_IN_ADDR)     # Input A from DDR
    await wb.write(0x104, POOL_OUT_ADDR)    # Output to DDR (reads from out_base in SRAM)
    await wb.write(0x108, POOL_WGT_ADDR)    # No weights
    await wb.write(0x10C, ADD_PARAM_ADDR)   # Add params (2 words)
    await wb.write(0x120, ADD_B_ADDR)       # DMA_ADD_B_ADDR

    # DMA sizes
    await wb.write(0x128, meta['dma_in_size'])
    await wb.write(0x12C, 0)  # No weights
    await wb.write(0x130, meta['dma_out_size'])

    # Post-processing
    await wb.write(0x180, meta['post_ctrl'])
    # Param count: for Add, the per-channel PPU params are not used.
    # The add params (2 words) are loaded to param SRAM via DMA_PARAM_ADDR.
    # We load 2 words (set param_count such that param_words=2 → count=0 special case).
    # Actually the controller loads param_count*4 words. We have 2 words.
    # Let's just set param_count=1 (loads 4 words, only first 2 matter).
    await wb.write(0x188, 1)

    # No fusion
    await wb.write(0x118, 0)


async def program_resize_layer(wb, meta, data):
    """Program CSR registers for a resize test layer."""
    await wb.write(0x040, meta['op_type'] | (meta['data_type'] << 4))

    await wb.write(0x044, (meta['in_h'] << 16) | meta['in_w'])
    await wb.write(0x048, meta['in_c'])
    await wb.write(0x04C, (meta['out_h'] << 16) | meta['out_w'])
    await wb.write(0x050, meta['out_c'])

    await wb.write(0x054, 1 | (1 << 8))
    await wb.write(0x058, 1 | (1 << 8))
    await wb.write(0x05C, 0)
    await wb.write(0x064, meta['resize_cfg'])

    await wb.write(0x070, 0)
    await wb.write(0x074, (1 << 16) | 1)

    out_base = meta['n_input_words']
    await wb.write(0x078, (out_base << 16) | 0)

    await wb.write(0x100, POOL_IN_ADDR)
    await wb.write(0x104, POOL_OUT_ADDR)
    await wb.write(0x108, POOL_WGT_ADDR)
    await wb.write(0x10C, POOL_PARAM_ADDR)

    await wb.write(0x128, meta['dma_in_size'])
    await wb.write(0x12C, 0)
    await wb.write(0x130, meta['dma_out_size'])

    await wb.write(0x180, meta['post_ctrl'])
    await wb.write(0x188, meta['dma_param_count'])
    await wb.write(0x118, 0)


# ═══════════════════════════════════════════════════════════════════════
# Test: MaxPool 2x2 stride 2, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_pool_max_2x2_s2(dut):
    """MaxPool 2x2 stride 2, 4x4x8 -> 2x2x8, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_pooling_test(mode='max', pool_h=2, pool_w=2,
                                  pool_sh=2, pool_sw=2,
                                  in_h=4, in_w=4, in_c=8, seed=42)

    # Populate DDR
    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    # Program and run
    await program_pooling_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=100000)
    assert done, "MaxPool 2x2 did not complete"

    # Verify output
    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"MaxPool: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"MaxPool 2x2 PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: AvgPool 2x2 stride 2, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_pool_avg_2x2_s2(dut):
    """AvgPool 2x2 stride 2, 4x4x8 -> 2x2x8, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_pooling_test(mode='avg', pool_h=2, pool_w=2,
                                  pool_sh=2, pool_sw=2,
                                  in_h=4, in_w=4, in_c=8, seed=43)

    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_pooling_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=100000)
    assert done, "AvgPool 2x2 did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"AvgPool: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"AvgPool 2x2 PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Global AvgPool, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_pool_avg_global(dut):
    """Global AvgPool 4x4x4 -> 1x1x4, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_pooling_test(mode='avg', pool_h=4, pool_w=4,
                                  pool_sh=4, pool_sw=4,
                                  in_h=4, in_w=4, in_c=4,
                                  global_pool=True, seed=44)

    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_pooling_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=100000)
    assert done, "Global AvgPool did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"GlobalPool: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Global AvgPool PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Eltwise Add basic, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_add_basic(dut):
    """Eltwise Add 4x4x8 with relu, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_add_test(h=4, w=4, c=8, relu=True, int16_mode=False, seed=100)

    # Populate DDR: input A, input B, add params
    mem.populate(POOL_IN_ADDR, data['input_a_words'])
    mem.populate(ADD_B_ADDR, data['input_b_words'])
    mem.populate(ADD_PARAM_ADDR, data['add_param_words'])

    await program_add_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=200000)
    assert done, "Add basic did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Add basic: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Add basic PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Eltwise Add no relu, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_add_no_relu(dut):
    """Eltwise Add 4x4x8 without relu, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_add_test(h=4, w=4, c=8, relu=False, int16_mode=False, seed=101)

    mem.populate(POOL_IN_ADDR, data['input_a_words'])
    mem.populate(ADD_B_ADDR, data['input_b_words'])
    mem.populate(ADD_PARAM_ADDR, data['add_param_words'])

    await program_add_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=200000)
    assert done, "Add no_relu did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Add no_relu: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Add no_relu PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Eltwise Add, INT16
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_add_int16(dut):
    """Eltwise Add 4x4x4 with relu, INT16."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_add_test(h=4, w=4, c=4, relu=True, int16_mode=True, seed=102)

    mem.populate(POOL_IN_ADDR, data['input_a_words'])
    mem.populate(ADD_B_ADDR, data['input_b_words'])
    mem.populate(ADD_PARAM_ADDR, data['add_param_words'])

    await program_add_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=200000)
    assert done, "Add INT16 did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Add INT16: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Add INT16 PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Resize nearest 2x, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_resize_nearest_2x(dut):
    """Resize nearest 4x4x4 -> 8x8x4, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_resize_test(in_h=4, in_w=4, in_c=4,
                                 out_h=8, out_w=8,
                                 resize_mode=0, int16_mode=False, seed=120)

    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_resize_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=200000)
    assert done, "Resize nearest 2x did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Resize nearest 2x: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Resize nearest 2x PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Resize bilinear 2x, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_resize_bilinear_2x(dut):
    """Resize bilinear 4x4x4 -> 8x8x4, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_resize_test(in_h=4, in_w=4, in_c=4,
                                 out_h=8, out_w=8,
                                 resize_mode=1, int16_mode=False, seed=121)

    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_resize_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=200000)
    assert done, "Resize bilinear 2x did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Resize bilinear 2x: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Resize bilinear 2x PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Resize nearest 2x, INT16
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_resize_nearest_int16(dut):
    """Resize nearest 4x4x4 -> 8x8x4, INT16."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_resize_test(in_h=4, in_w=4, in_c=4,
                                 out_h=8, out_w=8,
                                 resize_mode=0, int16_mode=True, seed=122)

    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_resize_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=200000)
    assert done, "Resize nearest INT16 did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Resize nearest INT16: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Resize nearest INT16 PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Resize bilinear 3x3 -> 5x5, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_resize_bilinear_3x3to5x5(dut):
    """Resize bilinear 3x3x4 -> 5x5x4, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_resize_test(in_h=3, in_w=3, in_c=4,
                                 out_h=5, out_w=5,
                                 resize_mode=1, int16_mode=False, seed=123)

    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_resize_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=200000)
    assert done, "Resize bilinear 3x3->5x5 did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Resize bilinear 3x3->5x5: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Resize bilinear 3x3->5x5 PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Deconv (Transposed Convolution) helpers and tests
# ═══════════════════════════════════════════════════════════════════════


async def program_deconv_layer(wb, meta, data):
    """Program CSR registers for a deconv test layer."""
    # LAYER_MODE: op_type=6 | (data_type << 4)
    await wb.write(0x040, meta['op_type'] | (meta['data_type'] << 4))

    # Dimensions
    await wb.write(0x044, (meta['in_h'] << 16) | meta['in_w'])
    await wb.write(0x048, meta['in_c'])
    await wb.write(0x04C, (meta['out_h'] << 16) | meta['out_w'])
    await wb.write(0x050, meta['out_c'])

    # Kernel & stride & padding
    await wb.write(0x054, meta['kernel_h'] | (meta['kernel_w'] << 8))
    await wb.write(0x058, meta['stride_h'] | (meta['stride_w'] << 8))
    await wb.write(0x05C, meta['pad_top'] | (meta['pad_left'] << 8))

    # DECONV_CFG: [7:0]=INSERT_H, [15:8]=INSERT_W
    await wb.write(0x068, meta['deconv_cfg'])

    # Tiling: no tiling (single tile covers full output)
    await wb.write(0x070, 0)
    await wb.write(0x074, (1 << 16) | 1)

    # SRAM_BASE: act_base=0, out_base after input+wgt
    out_base = meta['n_input_words']
    await wb.write(0x078, (out_base << 16) | 0)

    # DMA addresses
    await wb.write(0x100, POOL_IN_ADDR)
    await wb.write(0x104, POOL_OUT_ADDR)
    await wb.write(0x108, POOL_WGT_ADDR)
    await wb.write(0x10C, POOL_PARAM_ADDR)

    # DMA sizes
    await wb.write(0x128, meta['dma_in_size'])
    await wb.write(0x12C, meta['dma_wgt_size'])
    await wb.write(0x130, meta['dma_out_size'])

    # Post-processing (MODE_CONV_REQ)
    await wb.write(0x180, meta['post_ctrl'])
    await wb.write(0x188, meta['dma_param_count'])
    await wb.write(0x118, 0)


# ═══════════════════════════════════════════════════════════════════════
# Test: Deconv 2x2 stride 2, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_deconv_2x2_stride2(dut):
    """Deconv 2x2 stride-2: 3x3x4 -> 6x6x4, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_deconv_test(in_h=3, in_w=3, in_c=4, out_c=4,
                                  kernel_h=2, kernel_w=2, insert_h=1, insert_w=1,
                                  pad_top=0, pad_left=0, int16_mode=False, seed=200)

    mem.populate(POOL_WGT_ADDR, data['wgt_words'])
    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_deconv_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=300000)
    assert done, "Deconv 2x2 stride-2 did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Deconv 2x2 stride-2: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Deconv 2x2 stride-2 PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Deconv 3x3 stride 2, INT8
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_deconv_3x3_stride2(dut):
    """Deconv 3x3 stride-2 with padding: 4x4x4 -> 7x7x4, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_deconv_test(in_h=4, in_w=4, in_c=4, out_c=4,
                                  kernel_h=3, kernel_w=3, insert_h=1, insert_w=1,
                                  pad_top=1, pad_left=1, int16_mode=False, seed=201)

    mem.populate(POOL_WGT_ADDR, data['wgt_words'])
    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_deconv_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=300000)
    assert done, "Deconv 3x3 stride-2 did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Deconv 3x3 stride-2: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Deconv 3x3 stride-2 PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Deconv 2x2 stride 2, INT16
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_deconv_int16(dut):
    """Deconv 2x2 stride-2: 3x3x4 -> 6x6x4, INT16."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_deconv_test(in_h=3, in_w=3, in_c=4, out_c=4,
                                  kernel_h=2, kernel_w=2, insert_h=1, insert_w=1,
                                  pad_top=0, pad_left=0, int16_mode=True, seed=202)

    mem.populate(POOL_WGT_ADDR, data['wgt_words'])
    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_deconv_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=300000)
    assert done, "Deconv INT16 did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Deconv INT16: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Deconv INT16 PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Test: Deconv multichannel (out_c=16, in_c=8)
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_deconv_multichannel(dut):
    """Deconv multichannel: 4x4x8 -> 8x8x16, kernel 2x2 stride-2, INT8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    meta, data = gen_deconv_test(in_h=4, in_w=4, in_c=8, out_c=16,
                                  kernel_h=2, kernel_w=2, insert_h=1, insert_w=1,
                                  pad_top=0, pad_left=0, int16_mode=False, seed=203)

    mem.populate(POOL_WGT_ADDR, data['wgt_words'])
    mem.populate(POOL_IN_ADDR, data['input_words'])
    mem.populate(POOL_PARAM_ADDR, data['param_words'])

    await program_deconv_layer(wb, meta, data)
    done = await run_layer_and_wait(wb, dut, timeout=500000)
    assert done, "Deconv multichannel did not complete"

    out_addr = POOL_OUT_ADDR
    n_words = meta['n_output_words']
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(data['output_words'][i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"Deconv multichannel: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"Deconv multichannel PASSED: {n_words} words bit-exact")


# ═══════════════════════════════════════════════════════════════════════
# Concat tests (op_type=7)
# ═══════════════════════════════════════════════════════════════════════


async def program_concat_layer(wb, meta, data, out_base):
    """Program CSR registers for a concat branch layer."""
    # LAYER_MODE: op_type=7 | (data_type << 4)
    await wb.write(0x040, meta['op_type'] | (meta['data_type'] << 4))

    # Dimensions
    await wb.write(0x044, (meta['in_h'] << 16) | meta['in_w'])
    await wb.write(0x048, meta['in_c'])
    await wb.write(0x04C, (meta['out_h'] << 16) | meta['out_w'])
    await wb.write(0x050, meta['out_c'])

    # Kernel, stride, padding (1x1, stride 1, no pad)
    await wb.write(0x054, 1 | (1 << 8))
    await wb.write(0x058, 1 | (1 << 8))
    await wb.write(0x05C, 0)

    # CONCAT_CFG register (0x06C)
    await wb.write(0x06C, meta['concat_cfg'])

    # Tiling: no tiling
    await wb.write(0x070, 0)
    await wb.write(0x074, (1 << 16) | 1)

    # SRAM_BASE: act_base=0, out_base is shared across all branches
    await wb.write(0x078, (out_base << 16) | 0)

    # DMA addresses
    await wb.write(0x100, POOL_IN_ADDR)     # Input from DDR
    await wb.write(0x104, POOL_OUT_ADDR)    # Output to DDR
    await wb.write(0x108, POOL_WGT_ADDR)    # No weights
    await wb.write(0x10C, ADD_PARAM_ADDR)   # Concat params (1 word)

    # DMA sizes
    await wb.write(0x128, meta['dma_in_size'])
    await wb.write(0x12C, 0)  # No weights
    await wb.write(0x130, meta['dma_out_size'])

    # Post-processing
    await wb.write(0x180, meta['post_ctrl'])
    await wb.write(0x188, 1)  # param_count=1 (loads 4 words, only first 1 matters)

    # No fusion
    await wb.write(0x118, 0)


async def run_concat_test(dut, branches, int16_mode, relu, seed, test_name):
    """Generic concat test runner for N branches."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    branch_results, output_words, total_c = gen_concat_test(
        h=4, w=4, branches=branches, relu=relu,
        int16_mode=int16_mode, seed=seed
    )

    # Use a fixed out_base that won't overlap with any branch's input.
    # Max input across branches determines the safe out_base.
    max_input_words = max(meta['n_input_words'] for meta, _ in branch_results)
    out_base = max_input_words  # Output starts after the largest input

    # Run each branch as a separate layer
    for br_idx, (meta, data) in enumerate(branch_results):
        # Populate DDR: input + params
        mem.populate(POOL_IN_ADDR, data['input_words'])
        mem.populate(ADD_PARAM_ADDR, data['add_param_words'])

        await program_concat_layer(wb, meta, data, out_base)
        done = await run_layer_and_wait(wb, dut, timeout=300000)
        assert done, f"Concat branch {br_idx} did not complete"

    # Check final output (last DMA_OUT wrote the full combined output)
    out_addr = POOL_OUT_ADDR
    n_words = len(output_words)
    mismatches = []
    for i in range(n_words):
        got = mem.mem.get(out_addr + i * 4, None)
        exp = int(output_words[i])
        if got is None:
            mismatches.append((i, exp, 'NOT_WRITTEN'))
        elif got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        for idx_m, exp, got in mismatches[:10]:
            dut._log.error(f"  word[{idx_m}]: exp=0x{exp:08X} got={'NOT_WRITTEN' if got=='NOT_WRITTEN' else f'0x{got:08X}'}")
    assert len(mismatches) == 0, f"{test_name}: {len(mismatches)}/{n_words} mismatches"
    dut._log.info(f"{test_name} PASSED: {n_words} words bit-exact")


@cocotb.test()
async def test_concat_passthrough(dut):
    """Concat 2 branches (4ch + 4ch), INT8, with relu."""
    await run_concat_test(dut,
        branches=[{'in_c': 4}, {'in_c': 4}],
        int16_mode=False, relu=True, seed=200,
        test_name="Concat passthrough")


@cocotb.test()
async def test_concat_with_rescale(dut):
    """Concat 2 branches (8ch + 4ch), INT8, with relu and rescale."""
    await run_concat_test(dut,
        branches=[{'in_c': 8}, {'in_c': 4}],
        int16_mode=False, relu=True, seed=201,
        test_name="Concat with rescale")


@cocotb.test()
async def test_concat_int16(dut):
    """Concat 2 branches (4ch + 4ch), INT16, with relu."""
    await run_concat_test(dut,
        branches=[{'in_c': 4}, {'in_c': 4}],
        int16_mode=True, relu=True, seed=202,
        test_name="Concat INT16")


@cocotb.test()
async def test_concat_3branch(dut):
    """Concat 3 branches (4ch + 4ch + 4ch), INT8, with relu."""
    await run_concat_test(dut,
        branches=[{'in_c': 4}, {'in_c': 4}, {'in_c': 4}],
        int16_mode=False, relu=True, seed=203,
        test_name="Concat 3-branch")
