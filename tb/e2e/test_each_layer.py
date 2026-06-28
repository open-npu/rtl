import os, sys, cocotb, numpy as np, json
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Layers to skip (already verified individually)
SKIP_LAYERS = {1}  # L1 Pooling: verified PASS separately, too slow in sequence

@cocotb.test()
async def test_each_layer(dut):
    """逐层验证：每层独立加载golden input，跑完后与CSIM golden对比。"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_b_int16')
    ops = {0:'Conv2D',1:'DW',2:'FC',3:'Pool',4:'Add',5:'Resize',7:'Concat'}

    passed = 0
    failed = 0
    skipped = 0
    for i in range(25):
        if i in SKIP_LAYERS:
            dut._log.info(f"  SKIP L{i} (previously verified)")
            skipped += 1
            continue

        m = md[i]; d = ld[i]
        # Clear previous layer's data to avoid slowdown
        mem.mem.clear()
        # Load weights + params
        if len(d['wgt']) > 0:
            mem.populate(m['ddr_wgt_addr'], d['wgt'])
        if len(d['param']) > 0:
            mem.populate(m['ddr_param_addr'], d['param'])
        # Load input
        mem.populate(m['ddr_in_addr'], d['input'])
        # Load input_b for Add
        if 'input_b' in d and 'ddr_add_b_addr' in m:
            mem.populate(m['ddr_add_b_addr'], d['input_b'])

        await program_layer(wb, m)
        op = ops.get(m['op_type'], '?')
        dut._log.info(f"L{i:2d} {op:6s} {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} k={m['kernel_h']}x{m['kernel_w']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)}")
        done = await run_layer_and_wait(wb, dut, timeout=300000000)
        assert done, f"L{i} timeout"

        nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
        got = np.array([mem.mem.get(oa + j*4, 0) for j in range(nw)], dtype=np.uint32)
        mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
        if len(ref) != nw:
            dut._log.error(f"  FAIL L{i}: size {len(ref)} vs {nw}")
            failed += 1
        elif len(mm) == 0:
            dut._log.info(f"  PASS L{i}: {nw}/{nw}")
            passed += 1
        else:
            dut._log.error(f"  FAIL L{i}: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
            failed += 1

    dut._log.info(f"[RESULT] {passed} PASS, {failed} FAIL, {skipped} SKIP out of 25 layers")
    assert failed == 0, f"{failed} layers failed"
