import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

@cocotb.test()
async def test_conv_rest(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_b_int16')
    # Test remaining layers one by one with reset between
    test_layers = [14, 16, 17, 18, 20, 21, 23, 24]
    for i in test_layers:
        m = md[i]; d = ld[i]
        if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
        if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
        mem.populate(m['ddr_in_addr'], d['input'])
        await program_layer(wb, m)
        ops = {0:'Conv2D', 2:'FC', 3:'Pool'}
        dut._log.info(f"L{i} {ops.get(m['op_type'],'?')} {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)}")
        done = await run_layer_and_wait(wb, dut, timeout=600000000)
        nw=m['n_output_words']; ref=d['output']; oa=m['ddr_out_addr']
        got=np.array([mem.mem.get(oa+j*4,0) for j in range(nw)],dtype=np.uint32)
        mm=np.where(got!=ref)[0] if len(ref)==nw else np.arange(nw)
        if len(ref)!=nw: dut._log.error(f"FAIL L{i}: size {len(ref)} vs {nw}")
        elif len(mm)==0: dut._log.info(f"PASS L{i}: {nw}/{nw}")
        else: dut._log.error(f"FAIL L{i}: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
        # Reset between layers to avoid state issues
        await reset(dut)
