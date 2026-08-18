import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Regression: model_b L2 is a standard Conv2D 3x3 pad=1 in chained 2D-load mode.
# The Conv systolic path (S_ACT_CMD) already compensates for border tiles, so
# this must PASS both before and after the DW-path fix (which must not touch Conv).
@cocotb.test()
async def test_mb_l2_chain2d(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_b_int8')
    i = 2; m = md[i]; d = ld[i]
    # L2 input = L1 golden output (NHWC full tensor), the SoC chained path.
    nhwc_input = ld[1]['output']
    mem.populate(m['ddr_in_addr'], nhwc_input)
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    if 'input_b' in d and len(d['input_b']) > 0 and 'ddr_add_b_addr' in m:
        mem.populate(m['ddr_add_b_addr'], d['input_b'])
    await program_layer(wb, m)
    eb = 2 if m.get('data_type', 0) == 1 else 1
    await wb.write(0x110, m['in_w'] * m['in_c'] * eb)
    dut._log.info(f"L2-CHAIN2D-CONV {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} "
                  f"k={m['kernel_h']}x{m['kernel_w']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)} in_stride={m['in_w']*m['in_c']*eb}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL L2-CHAIN2D-CONV: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS L2-CHAIN2D-CONV: {nw}/{nw}")
    else:
        for k in mm[:10]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL L2-CHAIN2D-CONV: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
