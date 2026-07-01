import os, sys, cocotb, numpy as np, json
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Layers already verified PASS
VERIFIED_PASS = {0, 2, 3}
# Layer to skip (verified separately)
SKIP_LAYERS = {1}

@cocotb.test()
async def test_remaining(dut):
    """从L4开始逐层验证，遇到FAIL立刻停。"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_b_int16')
    ops = {0:'Conv2D',1:'DW',2:'FC',3:'Pool',4:'Add',5:'Resize',7:'Concat'}

    for i in range(25):
        if i in VERIFIED_PASS:
            continue
        if i in SKIP_LAYERS:
            dut._log.info(f"  SKIP L{i} (verified separately)")
            continue

        m = md[i]; d = ld[i]
        mem.mem.clear()
        # Load input first, then wgt (DDR may have input overlapping wgt region)
        mem.populate(m['ddr_in_addr'], d['input'])
        if len(d['wgt']) > 0:
            mem.populate(m['ddr_wgt_addr'], d['wgt'])
        if len(d['param']) > 0:
            mem.populate(m['ddr_param_addr'], d['param'])
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
            dut._log.error(f"FAIL L{i}: size {len(ref)} vs {nw}")
            assert False, f"L{i} failed"
        elif len(mm) == 0:
            dut._log.info(f"  PASS L{i}: {nw}/{nw}")
        else:
            dut._log.error(f"FAIL L{i}: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
            assert False, f"L{i} failed"

    dut._log.info("[RESULT] All remaining layers PASS")
