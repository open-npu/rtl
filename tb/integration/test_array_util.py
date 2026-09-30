"""MODEL_B utilization on the 64-MAC row or the 8x8 mesh.

Every conv and depthwise layer is included. The full feature map does not
fit in SRAM, so each distinct tile geometry is simulated once and counted
once per spatial tile of that layer (the scheduler reloads each tile).
The 12 Add layers are not MAC layers.

    useful = mac1d_lanes on the row
           = nonzero K-lanes x live OC columns on mesh sa_act_valid
           = nonzero DW lanes on dw_in_valid_w
    peak   = cycles x 64

    make DUT=npu_compute_tb MODULE=integration.test_array_util SIM_BUILD=sim_build_row
    make DUT=npu_compute_tb MODULE=integration.test_array_util SIM_BUILD=sim_build_mesh FORCE_MESH=1
"""

import json
import os
from collections import OrderedDict

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

MAC_N = 64
MESH_N = 8
META = os.environ.get(
    "UTIL_META",
    "/data/sam/open-npu/rtl/tb/golden/golden_dma_e2e/model_b_int8/metadata.json",
)


def model_a_shapes():
    """Unique tile geometries. Value is (tile_count, layer_ids)."""
    layers = json.load(open(META))
    groups = OrderedDict()
    for i, m in enumerate(layers):
        if m["op_type"] not in (0, 1):
            continue
        k, s, p = m["kernel_h"], m["stride_h"], m["pad_top"]
        if m["tile_h"] == 0:
            ih, iw = m["in_h"], m["in_w"]
            oh, ow = m["out_h"], m["out_w"]
            nt = 1
        else:
            oh, ow = m["tile_h"], m["tile_w"]
            ih = (oh - 1) * s + k - 2 * p
            iw = (ow - 1) * s + k - 2 * p
            nt = m["tile_num_h"] * m["tile_num_w"]
        key = (m["op_type"], ih, iw, m["in_c"], oh, ow, m["out_c"], k, s, p)
        if key not in groups:
            groups[key] = [0, []]
        groups[key][0] += nt
        groups[key][1].append(i)
    return groups


def wr_act(dut, addr, data):
    dut.u_sram_act.mem[addr].value = data


def wr_wgt(dut, addr, data):
    dut.u_sram_wgt.mem[addr].value = data


def wr_param(dut, addr, data):
    dut.u_sram_param.mem[addr].value = data


def mesh_conv_useful(dut):
    flat = int(dut.sa_act_data_flat.value)
    nz = 0
    for i in range(MESH_N):
        lane = (flat >> (16 * i)) & 0xFFFF
        if lane & 0x8000:
            lane -= 0x10000
        if lane != 0:
            nz += 1
    cols = int(dut.u_compute.col_last.value) + 1
    return nz * cols


def mesh_dw_useful(dut):
    valid = int(dut.u_compute.dw_in_valid_w.value)
    if valid == 0:
        return 0
    data = int(dut.u_compute.dw_in_data_w.value)
    n = 0
    for i in range(MESH_N):
        if (valid >> i) & 1:
            lane = (data >> (16 * i)) & 0xFFFF
            if lane & 0x8000:
                lane -= 0x10000
            if lane != 0:
                n += 1
    return n


async def run_layer(dut, op, in_h, in_w, in_c, out_h, out_w, out_c,
                    k, stride, pad):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.db_prefetch_done.value = 1
    dut.ppu_mode.value = 3
    dut.ppu_relu_en.value = 0
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0
    dut.cfg_int16.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    act_bytes = in_h * in_w * in_c
    for w in range((act_bytes + 3) // 4):
        wr_act(dut, w, 0x01010101)
    if op == 1:
        wgt_bytes = k * k * out_c
    else:
        wgt_bytes = k * k * in_c * out_c
    for w in range((wgt_bytes + 3) // 4):
        wr_wgt(dut, w, 0x01010101)
    for w in range(4096):
        wr_param(dut, w, 0)

    dut.cfg_op_type.value = op
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
    dut.cfg_out_base.value = 0
    # The layout bit is what routes a layer to the 64-lane row, so a
    # FORCE_MESH build must also present OC-major descriptors.
    dut.cfg_wgt_layout.value = 0 if os.environ.get("FORCE_MESH") else 1
    await RisingEdge(dut.clk)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    cycles = 0
    useful = 0
    limit = 2_000_000
    while cycles < limit:
        await RisingEdge(dut.clk)
        cycles += 1
        if int(dut.u_compute.mac1d_fire.value) == 1:
            useful += int(dut.u_compute.mac1d_lanes.value)
        if int(dut.u_compute.sa_act_valid.value) == 1:
            useful += mesh_conv_useful(dut)
        if int(dut.u_compute.dw_in_valid_w.value) != 0:
            useful += mesh_dw_useful(dut)
        if dut.done.value == 1:
            break
    assert cycles < limit, "compute never asserted done"
    return cycles, useful


@cocotb.test()
async def test_array_utilization(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    shapes = model_a_shapes()
    total_cyc = 0
    total_useful = 0
    engine = None
    dut._log.info(f"{os.path.basename(os.path.dirname(META))} tile x count   cyc_1tile  useful_1tile  util")
    for key, (nt, ids) in shapes.items():
        op, ih, iw, ic, oh, ow, oc, k, stride, pad = key
        cycles, useful = await run_layer(
            dut, op, ih, iw, ic, oh, ow, oc, k, stride, pad)
        if engine is None and op == 0:
            engine = "row" if int(dut.u_compute.use_row.value) else "mesh"
        peak = cycles * MAC_N
        kind = "dw" if op == 1 else "conv"
        dut._log.info(
            f"RESULT {engine} {kind} L{ids[0]} {ih}x{iw}x{ic}->{oh}x{ow}x{oc} "
            f"k{k} tiles {nt:4d}  {cycles:8d}  {useful:10d}  "
            f"{useful / peak:.3f}")
        total_cyc += cycles * nt
        total_useful += useful * nt
    peak = total_cyc * MAC_N
    dut._log.info(
        f"SUMMARY {engine} cycles={total_cyc} useful={total_useful} "
        f"util={total_useful / peak:.4f}")
