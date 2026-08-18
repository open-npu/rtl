import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Fast layers only: L1(Pool), L5(Add), L9(Conv), L12(Add), L16(Conv), L23(Pool), L24(FC)
TEST_LAYERS = [1, 5, 9, 12, 16, 23, 24]

@cocotb.test()
async def test_mb_fast_layers(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_b_int16')

    # Populate ALL weights and params
    for i in range(len(md)):
        m = md[i]; d = ld[i]
        mem.populate(m['ddr_wgt_addr'], d['wgt'])
        mem.populate(m['ddr_param_addr'], d['param'])

    op_names = {0:'Conv2D',1:'DW',2:'FC',3:'Pool',4:'Add',5:'Resize',7:'Concat'}
    passed = 0; failed = 0

    for i in TEST_LAYERS:
        m = md[i]; d = ld[i]
        op = op_names.get(m['op_type'], f"op{m['op_type']}")
        mem.populate(m['ddr_in_addr'], d['input'])
        if 'input_b' in d and 'ddr_add_b_addr' in m:
            mem.populate(m['ddr_add_b_addr'], d['input_b'])
        if i > TEST_LAYERS[0]:
            await reset(dut)
        await program_layer(wb, m)
        dut._log.info(f"  L{i:2d} {op:6s} {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']}")
        done = await run_layer_and_wait(wb, dut, timeout=600000000)
        assert done, f"L{i} timed out"
        out_addr = m['ddr_out_addr']; nw = m['n_output_words']
        ref = d['output']
        got = np.array([mem.mem.get(out_addr + j*4, 0) for j in range(nw)], dtype=np.uint32)
        if len(ref) != nw:
            ok, detail = False, f"size mismatch ({len(ref)} vs {nw})"
        else:
            mm = np.where(got != ref)[0]
            if len(mm) == 0:
                ok, detail = True, f"{nw}/{nw} PASS"
            else:
                ok, detail = False, f"{len(mm)}/{nw} FAIL, first: w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}"
        if ok:
            passed += 1; dut._log.info(f"    PASS {detail}")
        else:
            failed += 1; dut._log.error(f"    FAIL {detail}")
    dut._log.info(f"[MODEL_B fast] {passed}/{len(TEST_LAYERS)} PASS, {failed} FAIL")
    assert failed == 0, f"{failed} layers failed"
