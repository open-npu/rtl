import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Reproduce model_e int8 L30 (FC classifier, 1x1x512 -> 1x1x1000, per-OC reload
# wgt_per_oc=2048). SoC E2E fails: output words 192+ (= channel 768+ = oc_group 48+)
# read as zero. Non-tiled (tile=0), so 1D load (in_stride=0).
@cocotb.test()
async def test_me_l30_chain2d(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_e_int8')
    i = 30; m = md[i]; d = ld[i]
    nhwc_input = ld[29]['output']   # L30 in = L29 (global pool) output, 1x1x512
    mem.populate(m['ddr_in_addr'], nhwc_input)
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    await program_layer(wb, m)
    # L30 is non-tiled (tile_h=0) -> 1D load, in_stride=0. But it's chained (input
    # from L29's DDR output), so the firmware would NOT set 2D stride for it.
    dut._log.info(f"ME-L30-FC {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_c']} "
                  f"wgt_per_oc={m.get('wgt_per_oc_words',0)} tile={m.get('tile_h',0)}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    np.save('/tmp/me_l30_got.npy', got); np.save('/tmp/me_l30_ref.npy', ref)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL ME-L30: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS ME-L30: {nw}/{nw}")
    else:
        for k in mm[:8]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL ME-L30: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
