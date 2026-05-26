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
    # Drive feedback inputs that would normally come from systolic/PPU
    dut.sa_ready.value = 1
    dut.sa_busy.value = 0
    dut.sa_acc_out_valid.value = 0
    dut.ppu_out_valid.value = 0
    dut.ppu_out_data.value = 0
    dut.dw_out_valid.value = 0
    dut.dw_acc_out.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def systolic_ppu_stub(dut):
    """Stub coroutine that mimics systolic drain + PPU pipeline responses.

    Monitors sa_cmd_valid/sa_cmd and responds:
    - On MODE_DRAIN (3): after 2 cycles, pulse sa_acc_out_valid with zero accumulators
    - On ppu_in_valid: after 4-cycle pipeline delay, pulse ppu_out_valid with ppu_out_data=0
    """
    ARRAY_SIZE = get_array_size(dut)
    ppu_pipeline = []  # Queue of (cycle_due, data) for PPU outputs

    cycle = 0
    while True:
        await RisingEdge(dut.clk)
        cycle += 1

        # Check for drain command
        try:
            cmd_valid = int(dut.sa_cmd_valid.value)
            cmd = int(dut.sa_cmd.value)
        except Exception:
            cmd_valid = 0
            cmd = 0

        if cmd_valid == 1 and cmd == 3:  # MODE_DRAIN
            # Wait 2 cycles then pulse acc_out_valid (simulating drain timing)
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
            dut.sa_acc_out_valid.value = 1
            for i in range(ARRAY_SIZE):
                dut.sa_acc_out[i].value = 0
            await RisingEdge(dut.clk)
            dut.sa_acc_out_valid.value = 0
            cycle += 3
            continue

        # Check for PPU input
        try:
            ppu_in = int(dut.ppu_in_valid.value)
        except Exception:
            ppu_in = 0

        if ppu_in == 1:
            ppu_pipeline.append(cycle + 4)  # 4-cycle PPU pipeline

        # Check if any PPU outputs are due
        if ppu_pipeline and cycle >= ppu_pipeline[0]:
            ppu_pipeline.pop(0)
            dut.ppu_out_valid.value = 1
            dut.ppu_out_data.value = 0
        else:
            dut.ppu_out_valid.value = 0


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
    assert int(dut.sa_cmd_valid.value) == 0, "sa_cmd_valid should be 0"
    assert int(dut.sa_wgt_valid.value) == 0, "sa_wgt_valid should be 0"
    assert int(dut.sa_act_valid.value) == 0, "sa_act_valid should be 0"
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

    # Wait a few cycles and check state changed (wgt_cmd issued)
    found_wgt_cmd = False
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.sa_cmd_valid.value) == 1 and int(dut.sa_cmd.value) == 1:  # MODE_WGT_LOAD
            found_wgt_cmd = True
            break

    await Timer(1, unit="step")
    assert found_wgt_cmd, "Should issue WGT_LOAD command after start"


@cocotb.test()
async def test_weight_load_timing(dut):
    """Verify weight load produces ARRAY_SIZE wgt_valid pulses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    in_c = ARRAY_SIZE
    out_c = ARRAY_SIZE
    set_cfg_conv2d(dut, in_c=in_c, out_c=out_c, kh=1, kw=1, out_h=1, out_w=1)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Count wgt_valid pulses
    wgt_valid_count = 0
    for _ in range(500):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.sa_wgt_valid.value) == 1:
            wgt_valid_count += 1
        # Stop after seeing compute command
        if int(dut.sa_cmd_valid.value) == 1 and int(dut.sa_cmd.value) == 2:  # MODE_COMPUTE
            break
        await Timer(1, unit="step")

    assert wgt_valid_count == ARRAY_SIZE, \
        f"Expected {ARRAY_SIZE} wgt_valid pulses, got {wgt_valid_count}"


@cocotb.test()
async def test_act_stream_timing(dut):
    """Verify activation streaming produces k_depth act_valid pulses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    in_c = ARRAY_SIZE
    k_depth = in_c  # 1*1*in_c
    set_cfg_conv2d(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for activation streaming phase (after weight load)
    act_valid_count = 0
    in_act_phase = False
    for _ in range(500):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.sa_cmd_valid.value) == 1 and int(dut.sa_cmd.value) == 2:  # MODE_COMPUTE
            in_act_phase = True
        if in_act_phase and int(dut.sa_act_valid.value) == 1:
            act_valid_count += 1
        # Stop when drain starts
        if in_act_phase and int(dut.sa_cmd_valid.value) == 1 and int(dut.sa_cmd.value) == 3:
            break
        await Timer(1, unit="step")

    assert act_valid_count == k_depth, \
        f"Expected {k_depth} act_valid pulses, got {act_valid_count}"


@cocotb.test()
async def test_drain_sequence(dut):
    """Verify drain issues ARRAY_SIZE DRAIN commands (one per column)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    cocotb.start_soon(systolic_ppu_stub(dut))

    ARRAY_SIZE = get_array_size(dut)
    set_cfg_conv2d(dut, in_c=ARRAY_SIZE, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Count drain commands
    drain_count = 0
    drain_cols_seen = set()
    for _ in range(10000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.sa_cmd_valid.value) == 1 and int(dut.sa_cmd.value) == 3:  # MODE_DRAIN
            drain_count += 1
            drain_cols_seen.add(int(dut.sa_drain_col_sel.value))
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert drain_count == ARRAY_SIZE, \
        f"Expected {ARRAY_SIZE} drain commands, got {drain_count}"
    assert drain_cols_seen == set(range(ARRAY_SIZE)), \
        f"Expected drain cols {{0..{ARRAY_SIZE-1}}}, got {drain_cols_seen}"


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
    """With 2*ARRAY_SIZE output channels, should process 2 OC groups."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    cocotb.start_soon(systolic_ppu_stub(dut))

    ARRAY_SIZE = get_array_size(dut)
    out_c = ARRAY_SIZE * 2  # 2 OC groups

    set_cfg_conv2d(dut, in_c=ARRAY_SIZE, out_c=out_c, kh=1, kw=1, out_h=1, out_w=1)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Count how many WGT_LOAD commands are issued (should be 2: one per OC group)
    wgt_load_cmds = 0
    for _ in range(20000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.sa_cmd_valid.value) == 1 and int(dut.sa_cmd.value) == 1:  # MODE_WGT_LOAD
            wgt_load_cmds += 1
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert wgt_load_cmds == 2, \
        f"Expected 2 WGT_LOAD commands for 2 OC groups, got {wgt_load_cmds}"


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

    # Count WGT_LOAD commands: should be 4 tiles × 1 OC group = 4
    wgt_load_cmds = 0
    for _ in range(50000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.sa_cmd_valid.value) == 1 and int(dut.sa_cmd.value) == 1:
            wgt_load_cmds += 1
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert wgt_load_cmds == 4, \
        f"Expected 4 WGT_LOAD commands for 4 tiles, got {wgt_load_cmds}"
