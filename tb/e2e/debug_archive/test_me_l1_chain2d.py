import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Reproduce model_e int8 L1 (Pool 3x3 stride 2, 112x112x64 -> 56x56x64,
# tile 4x8 grid 14x7 = 98 tiles) SoC failure. model_b L1 Pool (28 tiles, same
# config) PASSes, so this is a tile-count / large-input dependent Pool bug.
@cocotb.test()
async def test_me_l1_chain2d(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_e_int8')
    i = 1; m = md[i]; d = ld[i]
    nhwc_input = ld[0]['output']   # L1 in = L0 NHWC golden output
    mem.populate(m['ddr_in_addr'], nhwc_input)
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    await program_layer(wb, m)
    eb = 2 if m.get('data_type', 0) == 1 else 1
    await wb.write(0x110, m['in_w'] * m['in_c'] * eb)
    dut._log.info(f"ME-L1-POOL {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} "
                  f"tile={m.get('tile_h',0)}x{m.get('tile_w',0)} grid={m.get('tile_num_h',0)}x{m.get('tile_num_w',0)} "
                  f"pool_cfg={m.get('pool_cfg',0)} in_stride={m['in_w']*m['in_c']*eb}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    np.save('/tmp/me_l1_got.npy', got); np.save('/tmp/me_l1_ref.npy', ref)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL ME-L1: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS ME-L1: {nw}/{nw}")
    else:
        for k in mm[:8]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL ME-L1: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
