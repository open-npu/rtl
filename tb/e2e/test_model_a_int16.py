# Open-NPU RTL — MODEL_A INT16 Full Model E2E Test
# SPDX-License-Identifier: Apache-2.0
"""MODEL_A INT16 full model E2E RTL validation (L0: Conv 224x224).

Validates the full pipeline: ONNX → converter → CSIM → bin2golden → RTL.
"""

import os, sys, json, cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait, verify_output, GOLDEN_DIR


@cocotb.test()
async def test_model_a_l0_conv_int16(dut):
    """MODEL_A INT16 Layer 0: Conv 224x224x3 → 112x112x56 (tiled)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('model_a_int16')

    # Pre-populate weights and params for Layer 0
    meta = metadata[0]
    data = layer_data[0]
    mem.populate(meta['ddr_wgt_addr'], data['wgt'])
    mem.populate(meta['ddr_param_addr'], data['param'])
    mem.populate(meta['ddr_in_addr'], data['input'])

    # Run Layer 0
    dut._log.info(f"L0: Conv {meta['in_h']}x{meta['in_w']}x{meta['in_c']} → "
                  f"{meta['out_h']}x{meta['out_w']}x{meta['out_c']} "
                  f"tile={meta['tile_h']}x{meta['tile_w']} "
                  f"tiles={meta['tile_num_h']}x{meta['tile_num_w']}")

    await program_layer(wb, meta)
    done = await run_layer_and_wait(wb, dut, timeout=20000000)
    assert done, "Layer 0 did not complete"

    ok, detail = verify_output(mem, meta, data['output'], 0)
    dut._log.info(f"  {detail}")
    assert ok, detail

    dut._log.info("[MODEL_A INT16] Layer 0 RTL PASSED — full pipeline bit-exact!")
