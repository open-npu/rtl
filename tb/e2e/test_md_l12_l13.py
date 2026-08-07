import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# Reproduce the model_d SoC L12(Add)→L13(Conv) interaction. L13 passes standalone
# (chain-2D from golden L12 output) but fails in SoC after L12 runs. Run L12 then
# L13 back-to-back (chained via DDR), verifying BOTH outputs, to isolate whether
# L12's RTL output or a stale state breaks L13.
@cocotb.test()
async def test_md_l12_l13(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_d_int8')

    # Populate L12 inputs: main input = L11 golden output (NHWC), add_b = L8 golden output.
    m12 = md[12]
    mem.populate(m12['ddr_in_addr'], ld[11]['output'])
    mem.populate(m12['ddr_add_b_addr'], ld[8]['output'])
    # L13 weights/params
    m13 = md[13]
    if len(ld[13]['wgt']) > 0: mem.populate(m13['ddr_wgt_addr'], ld[13]['wgt'])
    if len(ld[13]['param']) > 0: mem.populate(m13['ddr_param_addr'], ld[13]['param'])

    # --- Run L12 (Add) ---
    await program_layer(wb, m12)
    eb = 1
    await wb.write(0x110, m12['in_w'] * m12['in_c'] * eb)  # 2D load input
    dut._log.info(f"L12 ADD {m12['in_h']}x{m12['in_w']}x{m12['in_c']} residual_src={m12.get('residual_src')}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000); assert done, "L12 timeout"
    # Verify L12 output
    nw12 = m12['n_output_words']; ref12 = ld[12]['output']; oa12 = m12['ddr_out_addr']
    got12 = np.array([mem.mem.get(oa12 + j*4, 0) for j in range(nw12)], dtype=np.uint32)
    mm12 = np.where(got12 != ref12)[0]
    dut._log.info(f"L12 output: {'PASS' if len(mm12)==0 else f'FAIL {len(mm12)}/{nw12}'}")

    # --- Run L13 (Conv) chained from L12's RTL output ---
    await program_layer(wb, m13)
    await wb.write(0x110, m13['in_w'] * m13['in_c'] * eb)
    await wb.write(0x100, m12['ddr_out_addr'])  # L13 input = L12 DDR output
    dut._log.info(f"L13 CONV {m13['in_h']}x{m13['in_w']}x{m13['in_c']}->{m13['out_c']}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000); assert done, "L13 timeout"
    nw13 = m13['n_output_words']; ref13 = ld[13]['output']; oa13 = m13['ddr_out_addr']
    got13 = np.array([mem.mem.get(oa13 + j*4, 0) for j in range(nw13)], dtype=np.uint32)
    mm13 = np.where(got13 != ref13)[0]
    if len(mm13) == 0:
        dut._log.info(f"PASS L13 after L12: {nw13}/{nw13}")
    else:
        for k in mm13[:10]:
            dut._log.error(f"  w[{k}]: exp={ref13[k]:08X} got={got13[k]:08X}")
        dut._log.error(f"FAIL L13 after L12: {len(mm13)}/{nw13} first w[{mm13[0]}] exp={ref13[mm13[0]]:08X} got={got13[mm13[0]]:08X}")
