# Open-NPU RTL — DW Conv Tiled Regression Test
# SPDX-License-Identifier: Apache-2.0
"""DW Conv with non-zero tiles — regression for tile-local activation addressing.

Tests the fix where DW conv (S_DW_ACT_STREAM) must use tile-local
addressing instead of full-image addressing when tiled input is in SRAM.
Without the fix, non-zero tiles read wrong data → acc=0 (all padding).

Uses model_a L53: DW 14×14×192→14×14×192, k=3×3, tile=2×4, 28 tiles.
"""

import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait


@cocotb.test()
async def test_dw_tiled_regression(dut):
    """DW Conv with non-zero tiles: verify tile-local activation addressing."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('model_a_int16')

    # L53: DW 14x14x192 -> 14x14x192, k=3x3, tile=2x4, 28 tiles
    i = 53
    m = metadata[i]
    d = layer_data[i]

    mem.populate(m['ddr_in_addr'], d['input'])
    if len(d['wgt']) > 0:
        mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0:
        mem.populate(m['ddr_param_addr'], d['param'])
    if 'input_b' in d and len(d['input_b']) > 0 and 'ddr_add_b_addr' in m:
        mem.populate(m['ddr_add_b_addr'], d['input_b'])

    await program_layer(wb, m)
    dut._log.info(
        f"L{i} DW {m['in_h']}x{m['in_w']}x{m['in_c']} -> "
        f"{m['out_h']}x{m['out_w']}x{m['out_c']} "
        f"k={m['kernel_h']}x{m['kernel_w']} "
        f"tile={m.get('tile_h',0)}x{m.get('tile_w',0)} "
        f"tiles={m.get('tile_num_h',1)*m.get('tile_num_w',1)}"
    )

    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    assert done, f"Layer {i} did not complete"

    nw = m['n_output_words']
    ref = d['output']
    oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)

    if len(ref) != nw:
        dut._log.error(f"FAIL L{i}: size {len(ref)} vs {nw}")
        assert False, f"Output size mismatch: {len(ref)} vs {nw}"
    elif len(mm) == 0:
        dut._log.info(f"PASS L{i}: {nw}/{nw}")
    else:
        dut._log.error(
            f"FAIL L{i}: {len(mm)}/{nw} first w[{mm[0]}] "
            f"exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}"
        )
        assert False, f"{len(mm)}/{nw} mismatches, first at w[{mm[0]}]"
