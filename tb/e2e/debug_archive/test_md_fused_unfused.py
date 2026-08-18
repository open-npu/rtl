import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Candidate fix for the fused super-layer L9-11 SoC failure: run the block
# UNFUSED — each layer does a full DDR round-trip (2D load + per-tile store),
# clearing the FUSE_START/MID/END bits so no layer skips act_load/store.
# Fusion is mathematically identical to independent execution (it's an SRAM
# optimization), so L11's golden output must match. If this PASSES, the fix is
# to unfuse the block (in firmware or packer) rather than debug tiled fusion.
@cocotb.test()
async def test_md_fused_unfused(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_d_int8')

    # Populate weights/params for L9,L10,L11, and L9's NHWC input (= L8 golden output)
    for i in [9, 10, 11]:
        m = md[i]; d = ld[i]
        if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
        if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    mem.populate(md[9]['ddr_in_addr'], ld[8]['output'])  # L9 in = L8 NHWC output

    async def run_indep(i, in_addr):
        m = md[i]
        await program_layer(wb, m)
        eb = 2 if m.get('data_type', 0) == 1 else 1
        await wb.write(0x110, m['in_w'] * m['in_c'] * eb)  # 2D load (NHWC chained)
        # Clear FUSE bits (1-3), keep DB_EN(bit0)+PTS(bit4) -> independent layer
        await wb.write(0x118, m.get('sched_ctrl', 0) & ~0x0E)
        # Override input addr to the previous layer's DDR output (chain)
        await wb.write(0x100, in_addr)
        dut._log.info(f"L{i} INDEP: {m['in_h']}x{m['in_w']}x{m['in_c']}->{m['out_h']}x{m['out_w']}x{m['out_c']} "
                      f"tile={m.get('tile_h',0)}x{m.get('tile_w',0)} sched={(m.get('sched_ctrl',0)&~0x0E):#x}")
        done = await run_layer_and_wait(wb, dut, timeout=20000000000)
        assert done, f"L{i} did not complete"
        # Return this layer's DDR output (NHWC) to feed the next layer
        oa = m['ddr_out_addr']; nw = m['n_output_words']
        return oa, np.array([mem.mem.get(oa + j*4, 0) for j in range(nw)], dtype=np.uint32)

    oa9, out9 = await run_indep(9,  md[9]['ddr_in_addr'])
    oa10, out10 = await run_indep(10, oa9)    # L10 in = L9 out (NHWC)
    oa11, out11 = await run_indep(11, oa10)   # L11 in = L10 out (NHWC)

    m11 = md[11]; nw = m11['n_output_words']; ref = ld[11]['output']
    got = out11
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL L11 unfused: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS L11 unfused: {nw}/{nw}")
    else:
        for k in mm[:8]:
            dut._log.error(f"  w[{k}] exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL L11 unfused: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
