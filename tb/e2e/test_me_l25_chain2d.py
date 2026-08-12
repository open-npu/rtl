import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Reproduce model_e int16 L25 (Eltwise Add residual, 7x7x512, tile 2x3 grid 4x3,
# DB_EN+PTS) SoC chained failure (3302/12544). Feed L24 golden output as input
# (full NHWC + in_stride 2D load) and L23 golden output as residual B (the
# firmware resolves add_b = layer_out_addr[residual_src=23]).
@cocotb.test()
async def test_me_l25_chain2d(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_e_int16')
    i = 25; m = md[i]; d = ld[i]
    mem.populate(m['ddr_in_addr'], ld[24]['output'])          # main input = L24 out (NHWC)
    mem.populate(m['ddr_add_b_addr'], ld[23]['output'])       # residual B = L23 out (NHWC)
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    await program_layer(wb, m)
    eb = 2 if m.get('data_type', 0) == 1 else 1
    await wb.write(0x110, m['in_w'] * m['in_c'] * eb)         # 2D chain stride
    dut._log.info(f"ME-L25-ADD {m['in_h']}x{m['in_w']}x{m['in_c']} tile={m['tile_h']}x{m['tile_w']} "
                  f"grid={m['tile_num_h']}x{m['tile_num_w']} stride={m['in_w']*m['in_c']*eb}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    np.save('/tmp/me_l25_got.npy', got); np.save('/tmp/me_l25_ref.npy', ref)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL ME-L25: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS ME-L25: {nw}/{nw}")
    else:
        for k in mm[:10]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL ME-L25: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
