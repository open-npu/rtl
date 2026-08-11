import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Reproduce model_a int8 L61 (DW Conv global pool: 14x14x512 -> 1x1x512,
# kernel=14x14, non-tiled). Input = 25088 words > 12288-word act SRAM,
# weights = 25088 words > 24576-word wgt SRAM. Exercises the DW streaming
# path: 32 groups x 16 channels, per-group act slice (2D load, 4w x 196 rows,
# stride 512B) + weight block (784w) reload, output from DW_STREAM_OUT_BASE.
@cocotb.test()
async def test_ma_l61_dwstream(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_a_int8')
    i = 61; m = md[i]; d = ld[i]
    nhwc_input = ld[60]['output']   # L61 in = L60 output, 14x14x512 NHWC words
    mem.populate(m['ddr_in_addr'], nhwc_input)
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    await program_layer(wb, m)
    dut._log.info(f"MA-L61-DWS {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_c']} "
                  f"k={m['kernel_h']}x{m['kernel_w']} tile={m.get('tile_h',0)} "
                  f"in_words={m['n_input_words']}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m['n_output_words']; ref = d['output']; oa = m['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    np.save('/tmp/ma_l61_got.npy', got); np.save('/tmp/ma_l61_ref.npy', ref)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL MA-L61: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS MA-L61: {nw}/{nw}")
    else:
        for k in mm[:8]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        k0 = int(mm[0])
        dut._log.error(f"FAIL MA-L61: {len(mm)}/{nw} first w[{k0}] exp={ref[k0]:08X} got={got[k0]:08X}")
