import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait

# model_e int16 L24->L25 back-to-back chained repro.
# L24: Conv 1x1 s2, 14x14x256->7x7x512, per-OC wgt reload (wgt_per_oc=2048).
# L25: Add residual (B = L23 out). L25 passes standalone (chain-2d) but fails
# in SoC right after L24 — test state carryover from per-OC reload.
@cocotb.test()
async def test_me_l24_l25_chain(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_e_int16')

    # ─── L24: input = L21 golden output (input_src=21; 14x14x256 NHWC) ───
    m24 = md[24]; d24 = ld[24]
    mem.populate(m24['ddr_in_addr'], ld[21]['output'])
    if len(d24['wgt']) > 0: mem.populate(m24['ddr_wgt_addr'], d24['wgt'])
    if len(d24['param']) > 0: mem.populate(m24['ddr_param_addr'], d24['param'])
    await program_layer(wb, m24)
    await wb.write(0x110, m24['in_w'] * m24['in_c'] * 2)  # 2D chain stride (int16)
    dut._log.info(f"L24 {m24['in_h']}x{m24['in_w']}x{m24['in_c']}->{m24['out_c']} per_oc={m24['wgt_per_oc_words']}")
    await run_layer_and_wait(wb, dut, timeout=20000000000)
    # verify L24
    nw24 = m24['n_output_words']; ref24 = d24['output']; oa24 = m24['ddr_out_addr']
    got24 = np.array([mem.mem.get(oa24 + j * 4, 0) for j in range(nw24)], dtype=np.uint32)
    mm24 = int((got24 != ref24).sum())
    dut._log.info(f"L24 result: {nw24 - mm24}/{nw24} match")

    # soft reset between layers (as firmware does)
    await wb.write(0x000, 4)  # CTRL_SOFT_RST
    await wb.write(0x00C, 7)  # IRQ_STATUS clear

    # ─── L25: input = L24's ACTUAL RTL output, B = L23 golden output ───
    m25 = md[25]; d25 = ld[25]
    mem.populate(m25['ddr_add_b_addr'], ld[23]['output'])
    if len(d25['param']) > 0: mem.populate(m25['ddr_param_addr'], d25['param'])
    # L25 input addr := L24's out addr (chained)
    m25_in = dict(m25)
    await program_layer(wb, m25)
    await wb.write(0x100, m24['ddr_out_addr'])              # DMA_IN_ADDR = L24 out
    await wb.write(0x110, m25['in_w'] * m25['in_c'] * 2)    # 2D stride
    dut._log.info(f"L25 {m25['in_h']}x{m25['in_w']}x{m25['in_c']} tile={m25['tile_h']}x{m25['tile_w']}")
    await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw = m25['n_output_words']; ref = d25['output']; oa = m25['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j * 4, 0) for j in range(nw)], dtype=np.uint32)
    np.save('/tmp/me_l25_chain_got.npy', got); np.save('/tmp/me_l25_chain_ref.npy', ref)
    mm = np.where(got != ref)[0]
    if len(mm) == 0:
        dut._log.info(f"PASS L24+L25-CHAIN: {nw}/{nw}")
    else:
        for k in mm[:10]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL L24+L25-CHAIN: {len(mm)}/{nw} first w[{mm[0]}]")
