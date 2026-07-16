import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, run_layer_and_wait

@cocotb.test()
async def test_md_fused_l16_18(dut):
    """Fused block L16-18: Conv1x1→DW3x3→Conv1x1 (7x7x128→7x7x128)"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_d_int16')

    # Load weights and params for L16, L17, L18
    for i in [16, 17, 18]:
        m = md[i]; d = ld[i]
        if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
        if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])

    # Load block input (L16 input = L15 output)
    mem.populate(md[16]['ddr_in_addr'], ld[16]['input'])

    async def prog_fused(meta, sched_ctrl, act_base, out_base):
        await wb.write(0x040, meta['op_type'] | (meta['data_type'] << 4) | ((meta.get('in_zp',0)&0xFFFF)<<8))
        await wb.write(0x044, (meta['in_h'] << 16) | meta['in_w'])
        await wb.write(0x048, meta['in_c'])
        await wb.write(0x04C, (meta['out_h'] << 16) | meta['out_w'])
        await wb.write(0x050, meta['out_c'])
        await wb.write(0x054, meta['kernel_h'] | (meta['kernel_w'] << 8))
        await wb.write(0x058, meta['stride_h'] | (meta['stride_w'] << 8))
        await wb.write(0x05C, meta.get('pad_top',0) | (meta.get('pad_left',0) << 8))
        await wb.write(0x070, 0)  # no tiling for fused
        await wb.write(0x074, (1 << 16) | 1)
        await wb.write(0x078, (out_base << 16) | act_base)
        await wb.write(0x100, meta['ddr_in_addr'])
        await wb.write(0x104, meta['ddr_out_addr'])
        await wb.write(0x108, meta['ddr_wgt_addr'])
        await wb.write(0x10C, meta['ddr_param_addr'])
        await wb.write(0x128, meta['dma_in_size'])
        await wb.write(0x12C, meta['dma_wgt_size'])
        await wb.write(0x130, meta['dma_out_size'])
        await wb.write(0x180, meta['post_ctrl'])
        await wb.write(0x18C, meta.get('clamp_max', 32767))
        await wb.write(0x188, meta['dma_param_count'])
        await wb.write(0x118, sched_ctrl)

    # L16: FUSE_START + DB_EN
    m16 = md[16]
    in_words_16 = len(ld[16]['input'])
    dut._log.info(f"L16 FUSE_START: {m16['in_h']}x{m16['in_w']}x{m16['in_c']}->{m16['out_h']}x{m16['out_w']}x{m16['out_c']}")
    await prog_fused(m16, 0x03, 0, in_words_16)  # DB_EN | FUSE_START
    done = await run_layer_and_wait(wb, dut, timeout=50000000)
    assert done, "L16 FUSE_START did not complete"
    dut._log.info("  L16 FUSE_START done")

    # L17: FUSE_MID (no DMA load, input already in SRAM)
    m17 = md[17]
    dut._log.info(f"L17 FUSE_MID: {m17['in_h']}x{m17['in_w']}x{m17['in_c']}->{m17['out_h']}x{m17['out_w']}x{m17['out_c']}")
    await prog_fused(m17, 0x04, in_words_16, 0)  # FUSE_MID
    done = await run_layer_and_wait(wb, dut, timeout=50000000)
    assert done, "L17 FUSE_MID did not complete"
    dut._log.info("  L17 FUSE_MID done")

    # L18: FUSE_END (store output to DDR)
    m18 = md[18]
    dut._log.info(f"L18 FUSE_END: {m18['in_h']}x{m18['in_w']}x{m18['in_c']}->{m18['out_h']}x{m18['out_w']}x{m18['out_c']}")
    await prog_fused(m18, 0x08, 0, m18.get('n_input_words', in_words_16))  # FUSE_END
    done = await run_layer_and_wait(wb, dut, timeout=50000000)
    assert done, "L18 FUSE_END did not complete"
    dut._log.info("  L18 FUSE_END done")

    # Verify output
    nw = len(ld[18]['output'])
    ref = ld[18]['output']
    oa = m18['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j*4, 0) for j in range(nw)], dtype=np.uint32)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL L18 fused: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS L18 fused: {nw}/{nw}")
    else:
        dut._log.error(f"FAIL L18 fused: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
