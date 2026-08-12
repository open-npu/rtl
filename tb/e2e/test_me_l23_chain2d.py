import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Reproduce model_e int16 L23 (Conv 3x3 7x7x512 pad=1, tile 1x1 grid 7x7,
# per-OC reload wgt_per_oc=36864 words, DB_EN+PTS) SoC chained failure
# (6236/12544). Input = L22 golden output (NHWC), 2D chain load.
@cocotb.test()
async def test_me_l23_chain2d(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_e_int16')
    i = 23; m = md[i]; d = ld[i]
    mem.populate(m['ddr_in_addr'], ld[22]['output'])
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    await program_layer(wb, m)
    await wb.write(0x110, m['in_w'] * m['in_c'] * 2)  # 2D chain stride (int16)
    dut._log.info(f"ME-L23 {m['in_h']}x{m['in_w']}x{m['in_c']} k=3x3 pad=1 tile=1x1 grid=7x7 per_oc={m['wgt_per_oc_words']}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    np.save('/tmp/me_l23_got.npy', got); np.save('/tmp/me_l23_ref.npy', ref)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL ME-L23: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS ME-L23: {nw}/{nw}")
    else:
        for k in mm[:10]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL ME-L23: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
