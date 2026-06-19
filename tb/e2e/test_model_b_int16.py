# Open-NPU RTL — MODEL_B INT16 Layer 0 E2E Test
# SPDX-License-Identifier: Apache-2.0
"""MODEL_B INT16 Layer 0 Conv RTL validation."""

import os, sys, cocotb
import numpy as np
from cocotb.clock import Clock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait, verify_output


@cocotb.test()
async def test_model_b_l0_conv_int16(dut):
    """MODEL_B INT16 Layer 0: Conv 112x112x1 → 56x56x64 (28 tiles)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    metadata, layer_data = load_golden('model_b_int16')

    meta = metadata[0]
    data = layer_data[0]
    mem.populate(meta['ddr_wgt_addr'], data['wgt'])
    mem.populate(meta['ddr_param_addr'], data['param'])
    mem.populate(meta['ddr_in_addr'], data['input'])

    dut._log.info(f"MODEL_B L0: Conv {meta['in_h']}x{meta['in_w']}x{meta['in_c']} → "
                  f"{meta['out_h']}x{meta['out_w']}x{meta['out_c']} "
                  f"tile={meta['tile_h']}x{meta['tile_w']} "
                  f"tiles={meta['tile_num_h']}x{meta['tile_num_w']}")

    await program_layer(wb, meta)

    # Debug: check irq_o before start
    try:
        irq_before = int(dut.irq_o.value)
        dut._log.info(f"  irq_o before START: {irq_before}")
    except:
        dut._log.warning("  cannot read irq_o")

    done = await run_layer_and_wait(wb, dut, timeout=60000000)

    # Debug: check irq_o after wait
    try:
        irq_after = int(dut.irq_o.value)
        s_val = int((await wb.read(0x004)))
        dut._log.info(f"  done={done}, irq_o={irq_after}, STATUS=0x{s_val:08X}")
    except:
        pass
    assert done, "Layer 0 did not complete"

    # Debug: dump output DDR + golden for comparison
    out_addr = meta['ddr_out_addr']
    nw = meta['n_output_words']
    got = np.array([mem.mem.get(out_addr + i * 4, 0) for i in range(nw)], dtype=np.uint32)
    ref = data['output']
    dut._log.info(f"  DDR[0:4]:    {' '.join(f'{got[i]:08X}' for i in range(min(4, nw)))}")
    dut._log.info(f"  GOLDEN[0:4]: {' '.join(f'{ref[i]:08X}' for i in range(min(4, len(ref))))}")

    # Debug: read last PPU accumulator (captured during compute)
    try:
        lpa = int(dut.u_compute.last_ppu_acc.value)
        ldc = int(dut.u_compute.last_drain_col.value)
        ltx = int(dut.u_compute.last_tile_x.value)
        lty = int(dut.u_compute.last_tile_y.value)
        lpa_s = lpa & 0xFFFFFFFFFF
        if lpa_s & 0x8000000000: lpa_s = lpa_s | ~0xFFFFFFFFFF
        ch = ldc + 16  # approximate - need oc_group too
        dut._log.info(f"  LAST_PPU: tile({lty},{ltx}) ch~{ldc} acc={lpa_s} (0x{lpa_s & 0xFFFFFFFFFF:010x})")
    except Exception as e:
        dut._log.info(f"  Cannot read RTL internals: {e}")

    # Show non-zero words in DDR
    nz_ddr = np.where(got != 0)[0]
    nz_ref = np.where(ref != 0)[0] if len(ref) == nw else []
    dut._log.info(f"  DDR non-zero: {len(nz_ddr)}/{nw}, GOLDEN non-zero: {len(nz_ref)}/{nw}")
    if len(nz_ddr) > 0 and len(nz_ref) > 0:
        overlap = len(np.intersect1d(nz_ddr, nz_ref))
        dut._log.info(f"  Overlap (both non-zero): {overlap}")

    # Verify output: read from DDR and compare with CSIM reference
    ref = data['output']
    if len(ref) == nw:
        mismatches = np.where(got != ref)[0]
        if len(mismatches) == 0:
            ok = True
            detail = f"Layer 0: {nw}/{nw} words match (all PASS)"
        else:
            ok = False
            detail = f"Layer 0: {len(mismatches)}/{nw} mismatches, first: word[{mismatches[0]}] exp={ref[mismatches[0]]:08X} got={got[mismatches[0]]:08X}"
    else:
        ok = False
        detail = f"Layer 0: golden output empty ({len(ref)} words, expected {nw})"

    dut._log.info(f"  {detail}")
    assert ok, detail

    dut._log.info("[MODEL_B INT16] Layer 0 RTL PASSED — full pipeline bit-exact!")
