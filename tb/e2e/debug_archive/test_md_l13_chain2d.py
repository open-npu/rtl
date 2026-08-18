import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Reproduce model_d L13 (Conv 1x1, 14x14x128->512, out_c=512) SoC chained failure.
# Feed L13 the NHWC L12 golden output + in_stride (2D load), compare to golden.
@cocotb.test()
async def test_md_l13_chain2d(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_d_int8')
    i = 13; m = md[i]; d = ld[i]
    nhwc_input = ld[12]['output']   # L13 in = L12 (Add) NHWC golden output
    mem.populate(m['ddr_in_addr'], nhwc_input)
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    if 'input_b' in d and len(d['input_b']) > 0 and 'ddr_add_b_addr' in m:
        mem.populate(m['ddr_add_b_addr'], d['input_b'])
    await program_layer(wb, m)
    eb = 2 if m.get('data_type', 0) == 1 else 1
    await wb.write(0x110, m['in_w'] * m['in_c'] * eb)
    dut._log.info(f"L13-CHAIN2D {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} "
                  f"k={m['kernel_h']}x{m['kernel_w']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)} "
                  f"wgt_per_oc={m.get('wgt_per_oc_words',0)} in_stride={m['in_w']*m['in_c']*eb}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    np.save('/tmp/md_l13_chain2d_got.npy', got); np.save('/tmp/md_l13_chain2d_ref.npy', ref)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL L13-CHAIN2D: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS L13-CHAIN2D: {nw}/{nw}")
    else:
        for k in mm[:10]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL L13-CHAIN2D: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
