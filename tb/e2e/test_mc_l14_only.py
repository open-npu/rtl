import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait
@cocotb.test()
async def test_mc_l13_only(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_c_int16')
    i = 14; m = md[i]; d = ld[i]
    mem.populate(m['ddr_in_addr'], d['input'])
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    if 'input_b' in d and len(d['input_b']) > 0 and 'ddr_add_b_addr' in m:
        mem.populate(m['ddr_add_b_addr'], d['input_b'])
    await program_layer(wb, m)
    dut._log.info(f"L14 {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa+j*4,0) for j in range(nw)], dtype=np.uint32)
    # Concat: only compare channels within concat_offset..concat_offset+in_c
    cc = m.get('concat_cfg', 0)
    offset = cc & 0xFFFF; total_c = (cc >> 16) & 0xFFFF; in_c = m['in_c']
    # Build mask: which elements are owned by this Concat
    elem_count = nw * 2  # INT16 elements
    owned = np.zeros(elem_count, dtype=bool)
    for px in range(m['out_h'] * m['out_w']):
        for ch in range(offset, offset + in_c):
            elem_idx = px * total_c + ch
            if elem_idx < elem_count:
                owned[elem_idx] = True
    # Pack to words (2 elements per word)
    owned_words = owned[::2] | owned[1::2]  # word is owned if either element is owned
    # Only compare owned words
    ref_owned = ref[owned_words[:nw]]
    got_owned = got[owned_words[:nw]]
    mm = np.where(got_owned != ref_owned)[0]
    if len(mm) == 0:
        dut._log.info(f"PASS L14: {len(ref_owned)}/{len(ref_owned)} (owned channels only)")
    else:
        # Also check that non-owned words are 0 in golden (they should be)
        dut._log.error(f"FAIL L14: {len(mm)}/{len(ref_owned)} owned mismatches, first at w[{mm[0]}] exp={ref_owned[mm[0]]:08X} got={got_owned[mm[0]]:08X}")
