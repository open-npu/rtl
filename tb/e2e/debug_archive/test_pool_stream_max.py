import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, run_layer_and_wait, POOL_IN_ADDR, POOL_OUT_ADDR, ADD_PARAM_ADDR

# Synthetic oversized global MaxPool to cover the pool_stream MAX branch
# (model_e int16 L29 only exercises AvgPool). 16x16x512 int16 input =
# 65536 words > 12288-word act SRAM -> pool_stream slice mode must kick in.
IN_DIM = 16
IC = 448
NW_IN = IN_DIM * IN_DIM * IC * 2 // 4   # 65536 words
NW_OUT = IC * 2 // 4                    # 256 words

@cocotb.test()
async def test_pool_stream_max(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    rng = np.random.default_rng(42)
    x = rng.integers(-30000, 30000, size=(IN_DIM, IN_DIM, IC), dtype=np.int16)
    words = x.astype(np.uint16).reshape(-1, 2)
    in_words = (words[:, 0].astype(np.uint32) | (words[:, 1].astype(np.uint32) << 16))
    mem.populate(POOL_IN_ADDR, in_words)

    # Identity PPU params: M=16384, S=14, zp=0, bias=0  -> out = acc (max)
    prm = np.zeros(IC * 4, dtype=np.uint32)
    prm[0::4] = 16384 | (14 << 16)
    mem.populate(ADD_PARAM_ADDR, prm)

    # CSR programming (op 3 Pool, int16, global, Max, non-tiled)
    await wb.write(0x040, 3 | (1 << 4))                 # LAYER_MODE: op=3, int16
    await wb.write(0x044, (IN_DIM << 16) | IN_DIM)      # in_h/in_w
    await wb.write(0x048, IC)
    await wb.write(0x04C, (1 << 16) | 1)                # out 1x1
    await wb.write(0x050, IC)
    await wb.write(0x054, IN_DIM | (IN_DIM << 8))       # kernel (= in, global)
    await wb.write(0x058, 1 | (1 << 8))                 # stride
    await wb.write(0x05C, 0)                            # pad
    await wb.write(0x060, 1 << 20)                      # POOL_CFG: global, mode=Max
    await wb.write(0x070, 0)                            # no tiling
    await wb.write(0x074, (1 << 16) | 1)
    await wb.write(0x078, 0)                            # SRAM bases (stream overrides)
    await wb.write(0x100, POOL_IN_ADDR)
    await wb.write(0x104, POOL_OUT_ADDR)
    await wb.write(0x108, 0)
    await wb.write(0x10C, ADD_PARAM_ADDR)
    await wb.write(0x110, 0)                            # in_stride
    await wb.write(0x114, 0)                            # out_stride
    await wb.write(0x118, 0)                            # no auto-next/db_en
    await wb.write(0x120, 0)                            # add_b
    await wb.write(0x128, NW_IN * 4)                    # dma_in_size (bytes)
    await wb.write(0x12C, 0)                            # no weights
    await wb.write(0x130, NW_OUT * 4)                   # dma_out_size
    await wb.write(0x180, 0x80)                         # post_ctrl: int16, full pipe
    await wb.write(0x188, IC)                           # param count (x4 words)
    await wb.write(0x18C, 32767)                        # clamp_max

    dut._log.info(f"POOL-STREAM-MAX {IN_DIM}x{IN_DIM}x{IC} int16 global MaxPool")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)

    exp = x.reshape(-1, IC).max(axis=0)                 # per-channel max
    ew = exp.astype(np.uint16)
    ref = (ew[0::2].astype(np.uint32) | (ew[1::2].astype(np.uint32) << 16))
    got = np.array([mem.mem.get(POOL_OUT_ADDR + j * 4, 0) for j in range(NW_OUT)], dtype=np.uint32)
    np.save('/tmp/pool_stream_max_got.npy', got); np.save('/tmp/pool_stream_max_ref.npy', ref)
    mm = np.where(got != ref)[0]
    if len(mm) == 0:
        dut._log.info(f"PASS POOL-STREAM-MAX: {NW_OUT}/{NW_OUT}")
    else:
        for k in mm[:8]:
            dut._log.error(f"  w[{k}]: exp={ref[k]:08X} got={got[k]:08X}")
        dut._log.error(f"FAIL POOL-STREAM-MAX: {len(mm)}/{NW_OUT}")
