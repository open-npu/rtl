import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait
@cocotb.test()
async def test_mb_l0_only(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_b_int16')
    i = 0; m = md[i]; d = ld[i]
    mem.populate(m['ddr_in_addr'], d['input'])
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    await program_layer(wb, m)
    dut._log.info(f"L24 {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    out_addr = m['ddr_out_addr']; nw = len(d['output'])
    mm = 0
    for j in range(nw):
        got = mem.mem.get(out_addr + j*4, 0)
        exp = int(d['output'][j])
        if got != exp:
            if mm < 3: dut._log.error(f"w[{j}] exp=0x{exp:08X} got=0x{got:08X}")
            mm += 1
    if mm == 0: dut._log.info(f"PASS L24: {nw}/{nw}")
    else: dut._log.error(f"FAIL L24: {mm}/{nw}")
