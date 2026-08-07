import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, run_layer_and_wait

# Mimic the SoC firmware's fused-block programming to isolate the L11 bug.
# The SoC npu_program_layer passes act_base=0 for ALL layers and computes
# out_base=tile_in_size/4 (per-tile), breaking the fused ping-pong:
#   correct:  L9(act=0,out=B) -> L10(act=B,out=0) -> L11(act=0,out=DDR)
#   SoC bug:  L10 reads act=0 (L9 *input*) instead of L9 output region B.
# This test runs the fused block NON-TILED (like the passing standalone) but
# with the SoC's act_base=0 for L10/L11. If it FAILS, the act_base ping-pong
# bug is confirmed as a root cause.
@cocotb.test()
async def test_md_fused_soc_actbase(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_d_int16')
    for i in [9, 10, 11]:
        m = md[i]; d = ld[i]
        if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
        if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    mem.populate(md[9]['ddr_in_addr'], ld[9]['input'])

    async def prog_fused(meta, sched_ctrl, act_base, out_base):
        await wb.write(0x040, meta['op_type'] | (meta['data_type'] << 4) | ((meta.get('in_zp',0)&0xFFFF)<<8))
        await wb.write(0x044, (meta['in_h'] << 16) | meta['in_w'])
        await wb.write(0x048, meta['in_c'])
        await wb.write(0x04C, (meta['out_h'] << 16) | meta['out_w'])
        await wb.write(0x050, meta['out_c'])
        await wb.write(0x054, meta['kernel_h'] | (meta['kernel_w'] << 8))
        await wb.write(0x058, meta['stride_h'] | (meta['stride_w'] << 8))
        await wb.write(0x05C, meta.get('pad_top',0) | (meta.get('pad_left',0) << 8))
        await wb.write(0x070, 0)  # no tiling
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

    m9, m10, m11 = md[9], md[10], md[11]
    in9 = len(ld[9]['input'])
    # SoC-style: act_base=0 for all (no ping-pong). L10/L11 read region 0.
    await prog_fused(m9,  0x03, 0, in9)   # FUSE_START: read 0, write in9
    done = await run_layer_and_wait(wb, dut, timeout=50000000); assert done, "L9"
    await prog_fused(m10, 0x04, 0, 0)     # FUSE_MID (SoC bug): read 0 (should be in9)
    done = await run_layer_and_wait(wb, dut, timeout=50000000); assert done, "L10"
    await prog_fused(m11, 0x08, 0, m11['n_output_words'])  # FUSE_END (SoC bug): read 0
    done = await run_layer_and_wait(wb, dut, timeout=50000000); assert done, "L11"

    nw = m11['n_output_words']; ref = ld[11]['output']; oa = m11['ddr_out_addr']
    got = np.array([mem.mem.get(oa + j*4, 0) for j in range(nw)], dtype=np.uint32)
    mm = np.where(got != ref)[0] if len(ref) == nw else np.arange(nw)
    if len(ref) != nw:
        dut._log.error(f"FAIL L11: size {len(ref)} vs {nw}")
    elif len(mm) == 0:
        dut._log.info(f"PASS L11 fused (SoC act_base=0): {nw}/{nw}")
    else:
        for k in mm[:5]:
            dut._log.error(f"  w[{k}] exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL L11 fused (SoC act_base=0): {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
