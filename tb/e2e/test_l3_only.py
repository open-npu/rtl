import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

@cocotb.test()
async def test_l3_only(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    metadata, layer_data = load_golden('model_b_int16')
    i = 3
    meta = metadata[i]
    data = layer_data[i]
    mem.populate(meta['ddr_wgt_addr'], data['wgt'])
    mem.populate(meta['ddr_param_addr'], data['param'])
    mem.populate(meta['ddr_in_addr'], data['input'])
    if 'input_b' in data and 'ddr_add_b_addr' in meta:
        mem.populate(meta['ddr_add_b_addr'], data['input_b'])
    await program_layer(wb, meta)
    dut._log.info("L3: %dx%dx%d -> %dx%dx%d k=%dx%d tile=%dx%d" % (
        meta['in_h'], meta['in_w'], meta['in_c'], meta['out_h'], meta['out_w'], meta['out_c'],
        meta['kernel_h'], meta['kernel_w'], meta.get('tile_h',0), meta.get('tile_w',0)))
    done = await run_layer_and_wait(wb, dut, timeout=120000000)
    out_addr = meta['ddr_out_addr']
    nw = meta['n_output_words']
    ref = data['output']
    got = np.array([mem.mem.get(out_addr + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    mism = np.where(got != ref)[0]
    if len(mism) == 0:
        dut._log.info("L3: %d/%d PASS" % (nw, nw))
    else:
        dut._log.error("L3: %d/%d FAIL, first w[%d] exp=%08X got=%08X" % (len(mism), nw, mism[0], ref[mism[0]], got[mism[0]]))
        # Show first 10 mismatches
        for j in mism[:10]:
            dut._log.error("  w[%d] exp=%08X got=%08X" % (j, ref[j], got[j]))
