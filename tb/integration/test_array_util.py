"""Measure real systolic-array utilization on a Conv layer.

Counts, per cycle, how many of the ARRAY_SIZE activation lanes carry a nonzero
value while sa_act_valid is asserted. Each such lane drives COLS PEs, so
    useful_macs = sum(nonzero_lanes) * COLS
while the array's structural peak over the same window is
    peak_macs = cycles * ROWS * COLS

Run:
    make DUT=npu_compute_tb MODULE=integration.test_array_util
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

DATA_W = 16

# npu_compute.v FSM encoding (Conv path only).
STATE_NAMES = {
    0: "IDLE", 1: "TILE_SETUP", 2: "OC_SETUP", 3: "WGT_CMD", 4: "WGT_LOAD",
    5: "WGT_EMIT", 6: "ACT_CMD", 7: "ACT_LOAD", 8: "ACT_EMIT", 9: "ACT_FLUSH",
        10: "DRAIN_CMD_RETIRED", 11: "PSUM_COLLECT", 12: "PARAM_LOAD", 13: "PPU_FEED",
        14: "PPU_WAIT", 15: "WRITEBACK", 16: "OC_NEXT", 17: "TILE_NEXT",
        18: "DONE", 27: "SPATIAL_SETUP", 28: "REDUCE_RETIRED", 29: "PIXEL_NEXT",
        66: "PARAM_CACHE", 67: "PPU_STREAM",
}


def wr_act(dut, addr, data):
    dut.u_sram_act.mem[addr].value = data


def wr_wgt(dut, addr, data):
    dut.u_sram_wgt.mem[addr].value = data


def wr_param(dut, addr, data):
    dut.u_sram_param.mem[addr].value = data


async def run_conv(dut, in_h, in_w, in_c, out_c, k=3, stride=1):
    """Program one untiled Conv and return (cycles, useful_mac_slots)."""
    pad = k // 2
    out_h = (in_h + 2 * pad - k) // stride + 1
    out_w = (in_w + 2 * pad - k) // stride + 1

    dut.rst_n.value = 0
    dut.start.value = 0
    dut.db_prefetch_done.value = 1
    dut.ppu_mode.value = 3          # PASSTHROUGH: isolate the MAC datapath
    dut.ppu_relu_en.value = 0
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0
    dut.cfg_int16.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Activations: NHWC INT8, 4 elements per 32-bit word.
    act_bytes = in_h * in_w * in_c
    for w in range((act_bytes + 3) // 4):
        wr_act(dut, w, 0x01010101)
    # Weights: k*k*in_c per output channel, INT8.
    wgt_bytes = k * k * in_c * out_c
    for w in range((wgt_bytes + 3) // 4):
        wr_wgt(dut, w, 0x01010101)
    for w in range(256):
        wr_param(dut, w, 0)

    dut.cfg_op_type.value = 0
    dut.cfg_in_h.value = in_h
    dut.cfg_in_w.value = in_w
    dut.cfg_in_c.value = in_c
    dut.cfg_out_h.value = out_h
    dut.cfg_out_w.value = out_w
    dut.cfg_out_c.value = out_c
    dut.cfg_kernel_h.value = k
    dut.cfg_kernel_w.value = k
    dut.cfg_stride_h.value = stride
    dut.cfg_stride_w.value = stride
    dut.cfg_pad_top.value = pad
    dut.cfg_pad_left.value = pad
    dut.cfg_tile_h.value = 0
    dut.cfg_tile_w.value = 0
    dut.cfg_tile_num_h.value = 1
    dut.cfg_tile_num_w.value = 1
    dut.cfg_act_base.value = 0
    dut.cfg_out_base.value = (act_bytes + 3) // 4
    await RisingEdge(dut.clk)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    cycles = 0
    lane_slots = 0        # sum of nonzero activation lanes over all cycles
    feed_cycles = 0       # cycles with sa_act_valid asserted
    hist = {}
    limit = 4_000_000
    while cycles < limit:
        await RisingEdge(dut.clk)
        cycles += 1
        st = int(dut.u_compute.state.value)
        hist[st] = hist.get(st, 0) + 1
        if dut.sa_act_valid.value == 1:
            feed_cycles += 1
            flat = int(dut.sa_act_data_flat.value)
            array_n = int(dut.ARRAY_SIZE.value)
            for lane in range(array_n):
                if (flat >> (DATA_W * lane)) & ((1 << DATA_W) - 1):
                    lane_slots += 1
        if dut.done.value == 1:
            break

    assert cycles < limit, "compute never asserted done"
    macs = out_h * out_w * out_c * in_c * k * k
    array_n = int(dut.ARRAY_SIZE.value)
    return cycles, lane_slots, feed_cycles, macs, hist, array_n


@cocotb.test()
async def test_array_utilization(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # The tb wrapper gives 1024-word (4KB) weight and activation SRAMs, so
    # keep k*k*in_c*out_c <= 4096 bytes.
    shapes = [
        # (in_h, in_w, in_c, out_c, k)
        (8, 8, 16, 16, 3),
        (8, 8, 16, 16, 1),
        (8, 8, 16, 32, 1),
    ]

    dut._log.info("shape            cycles  feed_cyc  lane_slots  "
                  "useful_MAC  MAC/cyc  util%")
    for (ih, iw, ic, oc, k) in shapes:
        cycles, lane_slots, feed_cycles, macs, hist, array_n = await run_conv(
            dut, ih, iw, ic, oc, k)
        useful = lane_slots * array_n
        peak = cycles * array_n * array_n
        dut._log.info(
            f"{ih}x{iw}x{ic}->{oc} k{k}  {cycles:7d}  {feed_cycles:8d}  "
            f"{lane_slots:10d}  {useful:10d}  {useful/cycles:7.2f}  "
            f"{100.0*useful/peak:5.2f}")
        # 256-bit beat packs up to 16 INT8 / 16 INT16 lanes per sa_act_valid
        # cycle, so lane_slots/feed_cycles is the achieved pack width.
        dut._log.info(f"    lane_slots/feed_cycles = "
                      f"{lane_slots/max(1,feed_cycles):.2f}"
                      f"  layer MACs = {macs}")
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
        parts = [f"{STATE_NAMES.get(s, s)}={n}({100.0*n/cycles:.0f}%)"
                 for s, n in top]
        dut._log.info("    cycles by state: " + "  ".join(parts))
