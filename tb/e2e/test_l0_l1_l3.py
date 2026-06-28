import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

@cocotb.test()
async def test_l0_l1_l3(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    metadata, layer_data = load_golden('model_b_int16')

    for i in [0, 1, 3]:
        m = metadata[i]; d = layer_data[i]
        mem.populate(m['ddr_wgt_addr'], d['wgt'])
        mem.populate(m['ddr_param_addr'], d['param'])

    ops = {0:'Conv2D',3:'Pool'}
    for i in [0, 1, 3]:
        m = metadata[i]; d = layer_data[i]
        mem.populate(m['ddr_in_addr'], d['input'])
        if 'input_b' in d and 'ddr_add_b_addr' in m:
            mem.populate(m['ddr_add_b_addr'], d['input_b'])
        await program_layer(wb, m)
        op = ops.get(m['op_type'], '?')
        dut._log.info(f"L{i} {op} {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)}")
        done = await run_layer_and_wait(wb, dut, timeout=300000000)
        assert done, f"L{i} timeout"
        out_addr = m['ddr_out_addr']; nw = m['n_output_words']; ref = d['output']
        got = np.array([mem.mem.get(out_addr + j * 4, 0) for j in range(nw)], dtype=np.uint32)
        mism = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
        if len(ref) != nw:
            dut._log.error(f"L{i}: SIZE MISMATCH {len(ref)} vs {nw}")
        elif len(mism) == 0:
            dut._log.info(f"PASS L{i}: {nw}/{nw} words match")
        else:
            dut._log.error(f"FAIL L{i}: {len(mism)}/{nw} mismatches, first w[{mism[0]}] exp={ref[mism[0]]:08X} got={got[mism[0]]:08X}")
