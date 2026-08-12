import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, '/data/sam/open-npu/rtl/tb/e2e')
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait
@cocotb.test()
async def test_mc_l13_full(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_c_int16')
    i = 13; m = md[i]; d = ld[i]
    mem.populate(m['ddr_in_addr'], d['input'])
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    if 'input_b' in d and len(d['input_b']) > 0 and 'ddr_add_b_addr' in m:
        mem.populate(m['ddr_add_b_addr'], d['input_b'])
    await program_layer(wb, m)
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa+j*4,0) for j in range(nw)], dtype=np.uint32)
    np.save('/tmp/mc_l13_got.npy', got); np.save('/tmp/mc_l13_ref.npy', ref)
    mm = np.where(got != ref)[0]
    if len(mm) == 0:
        dut._log.info(f"PASS L13-FULL: {nw}/{nw}")
    else:
        for k in mm[:6]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL L13-FULL: {len(mm)}/{nw} first w[{mm[0]}]")
