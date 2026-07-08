import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait
@cocotb.test()
async def test_l5_only(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_b_int16')
    i = 5; m = md[i]; d = ld[i]
    mem.populate(m['ddr_in_addr'], d['input'])
    if 'input_b' in d: mem.populate(m['ddr_add_b_addr'], d['input_b'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    await program_layer(wb, m)
    dut._log.info(f"L5 Add {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)}")
    done = await run_layer_and_wait(wb, dut, timeout=300000000)
    nw=m['n_output_words']; ref=d['output']; oa=m['ddr_out_addr']
    got=np.array([mem.mem.get(oa+j*4,0) for j in range(nw)],dtype=np.uint32)
    mm=np.where(got!=ref)[0] if len(ref)==nw else np.arange(nw)
    if len(ref)!=nw: dut._log.error(f"FAIL L5: size {len(ref)} vs {nw}")
    elif len(mm)==0: dut._log.info(f"PASS L5: {nw}/{nw}")
    else:
        dut._log.error(f"FAIL L5: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
        # Check if row0 (w[0..191]) is correct, row1 (w[448..639]) is zero
        r0_ok = sum(1 for j in range(192) if got[j]==ref[j])
        r1_nonzero = sum(1 for j in range(448,640) if got[j]!=0)
        dut._log.error(f"  row0: {r0_ok}/192 match, row1: {r1_nonzero}/192 non-zero")
        # Show pattern of non-zero in row1
        nz_pos = [j-448 for j in range(448,640) if got[j]!=0]
        dut._log.error(f"  row1 non-zero at offsets: {nz_pos[:20]}")
        for j in [0, 1, 446, 447, 448, 449]:
            dut._log.error(f"  w[{j}] addr=0x{oa+j*4:08X} mem={mem.mem.get(oa+j*4,0):08X} exp={ref[j]:08X}")
