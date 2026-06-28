# Open-NPU RTL — MODEL_B INT16 Layer-by-Layer Test (All 25 Layers)
# SPDX-License-Identifier: Apache-2.0
"""MODEL_B INT16 all 25 layers: run each layer independently, compare vs CSIM golden."""

import os, sys, cocotb
import numpy as np
from cocotb.clock import Clock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait


@cocotb.test()
async def test_model_b_int16_e2e(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('model_b_int16')
    num_layers = len(metadata)

    dut._log.info(f"MODEL_B All Layers: {num_layers} layers")

    # Populate ALL weights and params
    for i in range(num_layers):
        meta = metadata[i]
        data = layer_data[i]
        mem.populate(meta['ddr_wgt_addr'], data['wgt'])
        mem.populate(meta['ddr_param_addr'], data['param'])

    passed = 0
    failed = 0
    op_names = {0:'Conv2D',1:'DW',2:'FC',3:'Pool',4:'Add',5:'Resize',7:'Concat'}

    for i in range(num_layers):
        meta = metadata[i]
        data = layer_data[i]
        op = op_names.get(meta['op_type'], f"op{meta['op_type']}")

        # Populate this layer's input
        mem.populate(meta['ddr_in_addr'], data['input'])

        # Populate residual input for Add layers
        if 'input_b' in data and 'ddr_add_b_addr' in meta:
            mem.populate(meta['ddr_add_b_addr'], data['input_b'])

        await program_layer(wb, meta)

        tile_h = meta.get('tile_h', 0)
        tile_w = meta.get('tile_w', 0)
        tiles_h = meta.get('tile_num_h', 1)
        tiles_w = meta.get('tile_num_w', 1)
        dut._log.info(f"  L{i:2d} {op:6s} {meta['in_h']}x{meta['in_w']}x{meta['in_c']:3d} → "
                      f"{meta['out_h']}x{meta['out_w']}x{meta['out_c']:3d}  "
                      f"k={meta['kernel_h']}x{meta['kernel_w']} "
                      f"tile={tile_h}x{tile_w}@{tiles_h}x{tiles_w} "
                      f"DB_EN={meta.get('sched_ctrl',0)&1}")

        done = await run_layer_and_wait(wb, dut, timeout=120000000)
        assert done, f"L{i} timed out"

        # Verify
        out_addr = meta['ddr_out_addr']
        nw = meta['n_output_words']
        ref = data['output']
        got = np.array([mem.mem.get(out_addr + j * 4, 0) for j in range(nw)], dtype=np.uint32)

        if len(ref) != nw:
            ok, detail = False, f"golden size mismatch ({len(ref)} vs {nw})"
        else:
            mismatches = np.where(got != ref)[0]
            if len(mismatches) == 0:
                ok, detail = True, f"{nw}/{nw} words PASS"
            else:
                ok, detail = False, f"{len(mismatches)}/{nw} FAIL, first: w[{mismatches[0]}] exp={ref[mismatches[0]]:08X} got={got[mismatches[0]]:08X}"

        if ok:
            passed += 1
            dut._log.info(f"    ✅ {detail}")
        else:
            failed += 1
            dut._log.error(f"    ❌ {detail}")
            # Continue to next layer to find all failures

    dut._log.info(f"[MODEL_B] {passed}/{num_layers} PASS, {failed} FAIL")
    assert failed == 0, f"{failed} layers failed"
    dut._log.info("[MODEL_B INT16] All 25 layers bit-exact!")
