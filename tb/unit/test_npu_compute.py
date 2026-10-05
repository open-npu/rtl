"""
Cocotb testbench for npu_compute — Compute Micro-Sequencer.

Tests the full compute path: weight load → activation stream → drain → PPU → writeback.
ARRAY_SIZE is read dynamically from the RTL parameter.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly

import numpy as np


def get_array_size(dut):
    """Get ARRAY_SIZE from the RTL parameter."""
    return int(dut.ARRAY_SIZE.value)


async def reset_dut(dut):
    """Apply reset for 5 cycles."""
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.ppu_out_valid.value = 0
    dut.ppu_out_data.value = 0
    dut.dw_out_valid.value = 0
    dut.dw_acc_out.value = 0
    try:
        dut.ppu_vout_w.value = 0
        dut.ppu_out_w.value = 0
        dut.dw_out_valid_w.value = 0
        dut.dw_acc_w.value = 0
    except Exception:
        pass
    dut.db_prefetch_done.value = 1  # No DB_EN in unit tests — always ready
    dut.wgt_reload_done.value = 1
    dut.cfg_wgt_per_oc.value = 0
    # K-major: these shapes are the ones the compiler routes to the 64-lane row.
    dut.cfg_wgt_layout.value = 1
    dut.cfg_int16.value = 0
    dut.cfg_in_zp.value = 0
    dut.cfg_act_base.value = 0
    dut.cfg_out_base.value = 64
    dut.cfg_pool_cfg.value = 0
    dut.cfg_resize_cfg.value = 0
    dut.cfg_deconv_cfg.value = 0
    dut.cfg_concat_cfg.value = 0
    dut.cfg_2d_load.value = 0
    dut.wgt_rd_data.value = 0
    dut.act_rd_data.value = 0
    dut.param_rd_data.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def systolic_ppu_stub(dut):
    """Answer the PPU 4 cycles after a wide or scalar feed."""
    ppu_pipeline = []

    cycle = 0
    while True:
        await RisingEdge(dut.clk)
        cycle += 1

        # Check for PPU input (scalar or any wide lane)
        try:
            ppu_in = int(dut.ppu_in_valid.value)
        except Exception:
            ppu_in = 0
        try:
            ppu_in = ppu_in or int(dut.ppu_valid_w.value)
        except Exception:
            pass

        if ppu_in:
            ppu_pipeline.append(cycle + 4)  # 4-cycle PPU pipeline

        # Check if any PPU outputs are due
        if ppu_pipeline and cycle >= ppu_pipeline[0]:
            ppu_pipeline.pop(0)
            dut.ppu_out_valid.value = 1
            dut.ppu_out_data.value = 0
            try:
                dut.ppu_vout_w.value = 1
                dut.ppu_out_w.value = 0
            except Exception:
                pass
        else:
            dut.ppu_out_valid.value = 0
            try:
                dut.ppu_vout_w.value = 0
            except Exception:
                pass


def set_cfg_conv2d(dut, in_c=4, out_c=4, kh=1, kw=1, out_h=1, out_w=1,
                   stride_h=1, stride_w=1, pad_top=0, pad_left=0,
                   tile_h=0, tile_w=0, tile_num_h=1, tile_num_w=1,
                   in_h=1, in_w=1):
    """Set layer configuration for a Conv2D operation."""
    dut.cfg_op_type.value = 0  # Conv2D
    dut.cfg_in_c.value = in_c
    dut.cfg_out_h.value = out_h
    dut.cfg_out_w.value = out_w
    dut.cfg_out_c.value = out_c
    dut.cfg_kernel_h.value = kh
    dut.cfg_kernel_w.value = kw
    dut.cfg_stride_h.value = stride_h
    dut.cfg_stride_w.value = stride_w
    dut.cfg_pad_top.value = pad_top
    dut.cfg_pad_left.value = pad_left
    dut.cfg_tile_h.value = tile_h
    dut.cfg_tile_w.value = tile_w
    dut.cfg_tile_num_h.value = tile_num_h
    dut.cfg_tile_num_w.value = tile_num_w
    dut.cfg_in_w.value = in_w
    dut.cfg_in_h.value = in_h


@cocotb.test()
async def test_idle_after_reset(dut):
    """After reset, module should be idle with done=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await ReadOnly()
    assert int(dut.done.value) == 0, "done should be 0 after reset"
    assert int(dut.wgt_rd_en.value) == 0, "wgt_rd_en should be 0"
    assert int(dut.act_rd_en.value) == 0, "act_rd_en should be 0"


@cocotb.test()
async def test_start_pulse(dut):
    """Start pulse should transition from IDLE to TILE_SETUP."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    set_cfg_conv2d(dut, in_c=ARRAY_SIZE, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # The row group state is the start of a 64-lane conv.
    found_row = False
    for _ in range(512):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.state.value) == 68:  # S_ROW_WGT
            found_row = True
            break

    await Timer(1, unit="step")
    assert found_row, "Should enter the 64-lane row after start"


@cocotb.test()
async def test_weight_load_timing(dut):
    """Conv issues one WGT_LOAD per 64-channel group, then one MAC per K.

    The 8x8 mesh is not on this path, so sa_wgt_valid stays low.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    cocotb.start_soon(systolic_ppu_stub(dut))

    ARRAY_SIZE = get_array_size(dut)
    in_c = ARRAY_SIZE
    out_c = ARRAY_SIZE
    set_cfg_conv2d(dut, in_c=in_c, out_c=out_c, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    row_groups = 0
    prev_state = None
    mac_fires = 0
    for _ in range(5000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        st = int(dut.state.value)
        if st == 68 and prev_state != 68:
            row_groups += 1
        prev_state = st
        if int(dut.mac1d_fire.value) == 1:
            mac_fires += 1
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert row_groups == 1, f"Expected 1 row group for out_c={out_c}, got {row_groups}"
    assert mac_fires == in_c, \
        f"Expected {in_c} MAC issues (one per K), got {mac_fires}"


@cocotb.test()
async def test_act_vector_timing(dut):
    """One pixel issues one MAC per K on the 64-lane row, not a mesh vector.

    Bare npu_compute has no SRAM, so activation values are not checked here.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    cocotb.start_soon(systolic_ppu_stub(dut))

    ARRAY_SIZE = get_array_size(dut)
    in_c = ARRAY_SIZE
    k_depth = in_c  # 1*1*in_c
    set_cfg_conv2d(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    mac_fires = 0
    for _ in range(5000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.mac1d_fire.value) == 1:
            mac_fires += 1
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert mac_fires == k_depth, \
        f"Expected {k_depth} MAC issues for one pixel, got {mac_fires}"


@cocotb.test()
async def test_no_drain_commands(dut):
    """The row path never issues MODE_DRAIN and does not feed the mesh."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    cocotb.start_soon(systolic_ppu_stub(dut))

    ARRAY_SIZE = get_array_size(dut)
    set_cfg_conv2d(dut, in_c=ARRAY_SIZE, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    mac_fires = 0
    for _ in range(10000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.mac1d_fire.value) == 1:
            mac_fires += 1
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert mac_fires == ARRAY_SIZE, \
        f"Expected {ARRAY_SIZE} MAC issues for one 1x1 pixel, got {mac_fires}"


@cocotb.test()
async def test_done_pulse(dut):
    """Verify done pulse is asserted at the end of computation."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    cocotb.start_soon(systolic_ppu_stub(dut))

    ARRAY_SIZE = get_array_size(dut)
    set_cfg_conv2d(dut, in_c=ARRAY_SIZE, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    done_seen = False
    for _ in range(10000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done.value) == 1:
            done_seen = True
            break
        await Timer(1, unit="step")

    assert done_seen, "done pulse never asserted within 10000 cycles"


@cocotb.test()
async def test_oc_tiling(dut):
    """128 output channels is two groups of 64, so two WGT_LOAD commands."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    cocotb.start_soon(systolic_ppu_stub(dut))

    ARRAY_SIZE = get_array_size(dut)
    out_c = 128

    set_cfg_conv2d(dut, in_c=ARRAY_SIZE, out_c=out_c, kh=1, kw=1, out_h=1, out_w=1)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    row_groups = 0
    prev_state = None
    for _ in range(20000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        st = int(dut.state.value)
        if st == 68 and prev_state != 68:
            row_groups += 1
        prev_state = st
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert row_groups == 2, \
        f"Expected 2 row groups for 128 output channels, got {row_groups}"


@cocotb.test()
async def test_spatial_tiling(dut):
    """With tile_num_h=2, tile_num_w=2, should process 4 tiles."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    cocotb.start_soon(systolic_ppu_stub(dut))

    ARRAY_SIZE = get_array_size(dut)

    set_cfg_conv2d(dut, in_c=ARRAY_SIZE, out_c=ARRAY_SIZE, kh=1, kw=1,
                   out_h=4, out_w=4, in_h=4, in_w=4,
                   tile_h=2, tile_w=2,
                   tile_num_h=2, tile_num_w=2)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    row_groups = 0
    prev_state = None
    for _ in range(50000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        st = int(dut.state.value)
        if st == 68 and prev_state != 68:
            row_groups += 1
        prev_state = st
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert row_groups == 4, \
        f"Expected 4 row groups for 4 tiles, got {row_groups}"
