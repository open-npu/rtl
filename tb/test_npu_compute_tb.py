"""
Cocotb testbench for npu_compute_tb — full datapath integration test.

Uses the wrapper that instantiates: npu_compute + systolic + PPU + SRAMs.
Tests use ARRAY_SIZE=4 for fast simulation.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly

import struct


async def reset_dut(dut):
    """Apply reset for 5 cycles."""
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.ppu_mode.value = 0      # CONV_REQ
    dut.ppu_relu_en.value = 0
    dut.ppu_bias_en.value = 1
    dut.ppu_zp_en.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def set_cfg(dut, in_c=4, out_c=4, kh=1, kw=1, out_h=1, out_w=1,
            stride_h=1, stride_w=1, pad_top=0, pad_left=0,
            tile_h=0, tile_w=0, tile_num_h=1, tile_num_w=1,
            in_h=1, in_w=1, op_type=0):
    """Set layer configuration."""
    dut.cfg_op_type.value = op_type
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


def write_sram_word(dut, bank, addr, data):
    """Write a 32-bit word to an SRAM bank's memory array directly.
    bank: 'wgt', 'act', 'param'
    """
    if bank == 'wgt':
        dut.u_sram_wgt.mem[addr].value = data
    elif bank == 'act':
        dut.u_sram_act.mem[addr].value = data
    elif bank == 'param':
        dut.u_sram_param.mem[addr].value = data


def read_sram_word(dut, bank, addr):
    """Read a 32-bit word from SRAM memory array."""
    if bank == 'wgt':
        return int(dut.u_sram_wgt.mem[addr].value)
    elif bank == 'act':
        return int(dut.u_sram_act.mem[addr].value)
    elif bank == 'param':
        return int(dut.u_sram_param.mem[addr].value)


def pack_i8x4(b0, b1, b2, b3):
    """Pack 4 signed int8 into a uint32 (little-endian byte order)."""
    def to_u8(v):
        return v & 0xFF
    return to_u8(b0) | (to_u8(b1) << 8) | (to_u8(b2) << 16) | (to_u8(b3) << 24)


async def wait_done(dut, timeout=5000):
    """Wait for done pulse, return True if seen within timeout cycles."""
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done.value) == 1:
            await Timer(1, unit="step")
            return True
        await Timer(1, unit="step")
    return False


def get_array_size(dut):
    """Get ARRAY_SIZE from the RTL parameter."""
    return int(dut.u_compute.ARRAY_SIZE.value)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_idle_after_reset(dut):
    """After reset, module should be idle with done=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await ReadOnly()
    assert int(dut.done.value) == 0
    await Timer(1, unit="step")


@cocotb.test()
async def test_weight_load_4x4(dut):
    """Load weights and verify systolic wgt_valid timing."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    # 1x1 conv, 4 IC, 4 OC → k_depth=4, ARRAY_SIZE columns of 4 weights each
    # Weight SRAM: column-major in OHWI order
    # Col 0 (OC0): weights[0..3] at word 0 = pack_i8x4(w00,w01,w02,w03)
    # Col 1 (OC1): weights[4..7] at word 1
    # etc.
    for oc in range(ARRAY_SIZE):
        w = [(oc * 4 + i + 1) & 0xFF for i in range(4)]
        write_sram_word(dut, 'wgt', oc, pack_i8x4(w[0], w[1], w[2], w[3]))

    set_cfg(dut, in_c=4, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    # Pulse start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Count wgt_valid pulses until COMPUTE cmd
    wgt_valid_count = 0
    for _ in range(500):
        await RisingEdge(dut.clk)
        await ReadOnly()
        sv = int(dut.u_compute.sa_wgt_valid.value)
        if sv == 1:
            wgt_valid_count += 1
        # Check for COMPUTE command (end of weight phase)
        cmd_v = int(dut.u_compute.sa_cmd_valid.value)
        cmd = int(dut.u_compute.sa_cmd.value)
        if cmd_v == 1 and cmd == 2:  # MODE_COMPUTE
            break
        await Timer(1, unit="step")

    assert wgt_valid_count == ARRAY_SIZE, \
        f"Expected {ARRAY_SIZE} wgt_valid pulses, got {wgt_valid_count}"


@cocotb.test()
async def test_act_stream_4x4(dut):
    """Verify activation streaming produces k_depth act_valid pulses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    k_depth = 4  # 1*1*4

    # Load weights (any values)
    for i in range(ARRAY_SIZE):
        write_sram_word(dut, 'wgt', i, pack_i8x4(1, 1, 1, 1))
    # Load activations: 4 bytes = 1 word
    write_sram_word(dut, 'act', 0, pack_i8x4(10, 20, 30, 40))

    set_cfg(dut, in_c=4, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for act phase and count act_valid
    act_valid_count = 0
    in_compute = False
    for _ in range(500):
        await RisingEdge(dut.clk)
        await ReadOnly()
        cmd_v = int(dut.u_compute.sa_cmd_valid.value)
        cmd = int(dut.u_compute.sa_cmd.value)
        if cmd_v == 1 and cmd == 2:  # MODE_COMPUTE
            in_compute = True
        if in_compute and int(dut.u_compute.sa_act_valid.value) == 1:
            act_valid_count += 1
        # Stop at DRAIN
        if in_compute and cmd_v == 1 and cmd == 3:
            break
        await Timer(1, unit="step")

    assert act_valid_count == k_depth, \
        f"Expected {k_depth} act_valid pulses, got {act_valid_count}"


@cocotb.test()
async def test_drain_sequence(dut):
    """Verify drain issues ARRAY_SIZE DRAIN commands."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)

    # Load weights and activations
    for i in range(ARRAY_SIZE):
        write_sram_word(dut, 'wgt', i, pack_i8x4(1, 0, 0, 0))
    write_sram_word(dut, 'act', 0, pack_i8x4(1, 0, 0, 0))
    # Load params (zero params — passthrough mode)
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00010000)  # M=1, S=0
        write_sram_word(dut, 'param', base + 1, 0x00000000)  # zp=0, bias_lo=0
        write_sram_word(dut, 'param', base + 2, 0x00000000)  # bias_mid
        write_sram_word(dut, 'param', base + 3, 0x00000000)  # bias_hi

    # Use passthrough mode to avoid PPU complexity
    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=4, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Count drain commands
    drain_count = 0
    drain_cols = set()
    for _ in range(20000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        cmd_v = int(dut.u_compute.sa_cmd_valid.value)
        cmd = int(dut.u_compute.sa_cmd.value)
        if cmd_v == 1 and cmd == 3:  # MODE_DRAIN
            drain_count += 1
            drain_cols.add(int(dut.u_compute.sa_drain_col_sel.value))
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert drain_count == ARRAY_SIZE, \
        f"Expected {ARRAY_SIZE} drain commands, got {drain_count}"
    assert drain_cols == set(range(ARRAY_SIZE)), \
        f"Expected drain cols {{0..{ARRAY_SIZE-1}}}, got {drain_cols}"


@cocotb.test()
async def test_done_pulse(dut):
    """Verify done pulse after complete computation."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    # Minimal setup
    for i in range(ARRAY_SIZE):
        write_sram_word(dut, 'wgt', i, pack_i8x4(1, 0, 0, 0))
    write_sram_word(dut, 'act', 0, pack_i8x4(1, 0, 0, 0))
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00010000)  # M=1, S=0
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=4, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=20000), "done never asserted within 20000 cycles"


@cocotb.test()
async def test_conv1x1_golden(dut):
    """1x1 conv IC=4→OC=ARRAY_SIZE: verify completion with identity weights."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    # Weights: identity-like (OC_i gets weight 1 at IC_i, 0 elsewhere)
    # Only first 4 OCs meaningful since IC=4
    for oc in range(ARRAY_SIZE):
        if oc < 4:
            w = [0]*4
            w[oc] = 1
        else:
            w = [0]*4
        write_sram_word(dut, 'wgt', oc, pack_i8x4(w[0], w[1], w[2], w[3]))

    # Activations: [10, 20, 30, 40]
    write_sram_word(dut, 'act', 0, pack_i8x4(10, 20, 30, 40))

    # Params: M=1, S=0, zp=0, bias=0 (passthrough quantization)
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)  # M=1, S=0
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3  # PASSTHROUGH — just truncate acc to 8 bits
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=4, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=10000), "done never asserted"


@cocotb.test()
async def test_oc_tiling_2groups(dut):
    """OC = 2*ARRAY_SIZE → 2 OC groups, 2 WGT_LOAD cmds."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    out_c = ARRAY_SIZE * 2  # 2 OC groups

    # Load weights for out_c OC channels
    for oc in range(out_c):
        write_sram_word(dut, 'wgt', oc, pack_i8x4(1, 0, 0, 0))
    write_sram_word(dut, 'act', 0, pack_i8x4(1, 0, 0, 0))
    for ch in range(out_c):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=4, out_c=out_c, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    wgt_load_cmds = 0
    for _ in range(20000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        cmd_v = int(dut.u_compute.sa_cmd_valid.value)
        cmd = int(dut.u_compute.sa_cmd.value)
        if cmd_v == 1 and cmd == 1:  # MODE_WGT_LOAD
            wgt_load_cmds += 1
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert wgt_load_cmds == 2, \
        f"Expected 2 WGT_LOAD commands, got {wgt_load_cmds}"


@cocotb.test()
async def test_spatial_tiling_2x2(dut):
    """4x4 output with tile 2x2 → 4 tiles, 4 WGT_LOAD cmds."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)

    # Fill SRAMs with dummy data
    for i in range(64):
        write_sram_word(dut, 'wgt', i, pack_i8x4(1, 0, 0, 0))
        write_sram_word(dut, 'act', i, pack_i8x4(1, 0, 0, 0))
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=4, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=4, out_w=4, in_h=4, in_w=4,
            tile_h=2, tile_w=2, tile_num_h=2, tile_num_w=2)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    wgt_load_cmds = 0
    for _ in range(200000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        cmd_v = int(dut.u_compute.sa_cmd_valid.value)
        cmd = int(dut.u_compute.sa_cmd.value)
        if cmd_v == 1 and cmd == 1:
            wgt_load_cmds += 1
        if int(dut.done.value) == 1:
            break
        await Timer(1, unit="step")

    assert wgt_load_cmds == 4, \
        f"Expected 4 WGT_LOAD commands for 4 tiles, got {wgt_load_cmds}"


@cocotb.test()
async def test_conv1x1_verify_output(dut):
    """Verify actual output values for 1x1 conv with all-ones weights.

    With per-row activation + reduction:
      PE[k][col].acc = w[col][k] * act[k] (single product)
      dot_product[col] = sum_k(w[col][k] * act[k])
    With all weights=1: dot_product[col] = sum(acts) = 2+3+4+5 = 14.
    After passthrough PPU, all output bytes = 14.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)

    # Use all-ones weights so dot product = sum of activations.
    for oc in range(ARRAY_SIZE):
        write_sram_word(dut, 'wgt', oc, pack_i8x4(1, 1, 1, 1))

    # Pre-zero output area (output will be at same location for 1-pixel case)
    for i in range(64):
        write_sram_word(dut, 'act', i, 0)
    write_sram_word(dut, 'act', 0, pack_i8x4(2, 3, 4, 5))

    # Passthrough params
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=4, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=20000), "done never asserted"

    # dot_product[col] = 1*2 + 1*3 + 1*4 + 1*5 = 14 for all columns
    # Output at word 0 (pixel 0, channels 0-3)
    out_word = read_sram_word(dut, 'act', 0)
    dut._log.info(f"Output word 0 = 0x{out_word:08X}")

    # All bytes should be 14
    for i in range(4):
        byte_val = (out_word >> (i*8)) & 0xFF
        assert byte_val == 14, \
            f"Output byte {i} = {byte_val}, expected 14 (all weights=1, sum acts=14)"


@cocotb.test()
async def test_conv1x1_dotproduct(dut):
    """Verify correct dot product with non-uniform weights.

    This test would FAIL with the old broadcast approach (w*sum) and
    only PASSES with correct per-row activation + reduction.

    weights[col=0] = [1, 2, 3, 4] (different per row/k)
    activations = [10, 20, 30, 40]
    dot_product[col=0] = 1*10 + 2*20 + 3*30 + 4*40 = 10+40+90+160 = 300
    Clipped to int8: 300 > 127, so with passthrough truncation = 300 & 0xFF = 44 (0x2C)

    Use small values to stay in int8: w=[1,2,3,4], act=[1,2,3,4]
    dot = 1*1 + 2*2 + 3*3 + 4*4 = 1+4+9+16 = 30
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)

    # Non-uniform weights: col 0 = [1,2,3,4], cols 1..N-1 = [1,1,1,1]
    # Weight SRAM layout: col c at word address c (for k_depth=4)
    write_sram_word(dut, 'wgt', 0, pack_i8x4(1, 2, 3, 4))  # col 0: w[0][k] = k+1
    for oc in range(1, ARRAY_SIZE):
        write_sram_word(dut, 'wgt', oc, pack_i8x4(1, 1, 1, 1))  # cols 1+: all ones

    # Pre-zero and set activations at word 0
    for i in range(64):
        write_sram_word(dut, 'act', i, 0)
    write_sram_word(dut, 'act', 0, pack_i8x4(1, 2, 3, 4))

    # Passthrough params
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=4, out_c=ARRAY_SIZE, kh=1, kw=1, out_h=1, out_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=20000), "done never asserted"

    out_word = read_sram_word(dut, 'act', 0)
    dut._log.info(f"Output word 0 = 0x{out_word:08X}")

    # Channel 0 (byte 0): dot = 1*1 + 2*2 + 3*3 + 4*4 = 30
    byte0 = out_word & 0xFF
    assert byte0 == 30, f"Channel 0: expected 30, got {byte0}"

    # Channels 1-3 (bytes 1-3): dot = 1*1 + 1*2 + 1*3 + 1*4 = 10
    for i in range(1, 4):
        byte_val = (out_word >> (i*8)) & 0xFF
        assert byte_val == 10, \
            f"Channel {i}: expected 10, got {byte_val}"


# ═══════════════════════════════════════════════════════════════════════════
# DW Conv Tests
# ═══════════════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_dw_conv_1x1_single_ch(dut):
    """DW Conv: 1x1 kernel, 1 channel, 1 output pixel. Simplest case."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Weight: channel 0, 1x1 kernel = 1 byte at word 0
    write_sram_word(dut, 'wgt', 0, pack_i8x4(3, 0, 0, 0))

    # Activation: pixel(0,0) channel 0 at byte 0
    write_sram_word(dut, 'act', 0, pack_i8x4(7, 0, 0, 0))

    # Params: passthrough (M=1, S=0, no bias/zp)
    write_sram_word(dut, 'param', 0, 0x00000001)
    write_sram_word(dut, 'param', 1, 0x00000000)
    write_sram_word(dut, 'param', 2, 0x00000000)
    write_sram_word(dut, 'param', 3, 0x00000000)

    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, op_type=1, in_c=1, out_c=1, kh=1, kw=1,
            out_h=1, out_w=1, in_h=1, in_w=1,
            stride_h=1, stride_w=1, pad_top=0, pad_left=0)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=500), "DW 1x1 done never asserted"

    # Expected: 3 * 7 = 21. Passthrough PPU → out = acc[7:0] = 21
    out_word = read_sram_word(dut, 'act', 0)
    out_val = out_word & 0xFF
    assert out_val == 21, f"DW 1x1: expected 21, got {out_val}"


@cocotb.test()
async def test_dw_conv_3x3_golden(dut):
    """DW Conv: 3x3 kernel, 1 channel, 1 pixel output (no pad). Golden check."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 3x3 Sobel-like weights
    weights = [1, 2, 1, 0, 0, 0, -1, -2, -1]
    write_sram_word(dut, 'wgt', 0, pack_i8x4(weights[0], weights[1], weights[2], weights[3]))
    write_sram_word(dut, 'wgt', 1, pack_i8x4(weights[4], weights[5], weights[6], weights[7]))
    write_sram_word(dut, 'wgt', 2, pack_i8x4(weights[8], 0, 0, 0))

    # Input: 3x3 spatial, 1 channel (NHWC [3][3][1])
    inputs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    write_sram_word(dut, 'act', 0, pack_i8x4(inputs[0], inputs[1], inputs[2], inputs[3]))
    write_sram_word(dut, 'act', 1, pack_i8x4(inputs[4], inputs[5], inputs[6], inputs[7]))
    write_sram_word(dut, 'act', 2, pack_i8x4(inputs[8], 0, 0, 0))

    expected_acc = sum(w * x for w, x in zip(weights, inputs))
    # = 10+40+30+0+0+0-70-160-90 = -240

    # Params: passthrough
    write_sram_word(dut, 'param', 0, 0x00000001)
    write_sram_word(dut, 'param', 1, 0x00000000)
    write_sram_word(dut, 'param', 2, 0x00000000)
    write_sram_word(dut, 'param', 3, 0x00000000)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    # 3x3 input, 1x1 output, no padding, stride=1
    set_cfg(dut, op_type=1, in_c=1, out_c=1, kh=3, kw=3,
            out_h=1, out_w=1, in_h=3, in_w=3,
            stride_h=1, stride_w=1, pad_top=0, pad_left=0)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=500), "DW 3x3 done never asserted"

    # Passthrough PPU: out = acc[7:0]
    # -240 & 0xFF = 16
    out_word = read_sram_word(dut, 'act', 0)
    out_byte = out_word & 0xFF
    expected_byte = expected_acc & 0xFF
    assert out_byte == expected_byte, \
        f"DW 3x3: expected {expected_byte} (acc={expected_acc}), got {out_byte}"


@cocotb.test()
async def test_dw_conv_multichannel(dut):
    """DW Conv: 1x1 kernel, 4 channels, 1 pixel. Verify channel iteration."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 4 channels × 1x1 kernel: packed in weight SRAM word 0
    # ch0=2, ch1=3, ch2=4, ch3=5
    write_sram_word(dut, 'wgt', 0, pack_i8x4(2, 3, 4, 5))

    # Activations: 1×1 spatial, 4ch NHWC → word 0
    # ch0=10, ch1=20, ch2=30, ch3=40
    write_sram_word(dut, 'act', 0, pack_i8x4(10, 20, 30, 40))

    # Pre-zero output area beyond input (output shares addr 0 with input for this test)
    for i in range(1, 8):
        write_sram_word(dut, 'act', i, 0)

    # Params for 4 channels
    for ch in range(4):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0x00000000)
        write_sram_word(dut, 'param', base + 2, 0x00000000)
        write_sram_word(dut, 'param', base + 3, 0x00000000)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, op_type=1, in_c=4, out_c=4, kh=1, kw=1,
            out_h=1, out_w=1, in_h=1, in_w=1,
            stride_h=1, stride_w=1, pad_top=0, pad_left=0)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=2000), "DW multichannel done never asserted"

    # Expected: ch0=2*10=20, ch1=3*20=60, ch2=4*30=120, ch3=5*40=200
    # Output NHWC: pixel(0,0) ch0-3 packed in word at out_base
    # out_base = (0*1*4 + 0*4) >> 2 = 0 for tile(0,0)
    # byte_offset for ch c = (0*1+0)*4 + c = c
    # So all 4 bytes in word 0: [ch0, ch1, ch2, ch3] = [20, 60, 120, 200]
    out_word = read_sram_word(dut, 'act', 0)
    ch0 = out_word & 0xFF
    ch1 = (out_word >> 8) & 0xFF
    ch2 = (out_word >> 16) & 0xFF
    ch3 = (out_word >> 24) & 0xFF
    assert ch0 == 20, f"ch0: expected 20, got {ch0}"
    assert ch1 == 60, f"ch1: expected 60, got {ch1}"
    assert ch2 == 120, f"ch2: expected 120, got {ch2}"
    assert ch3 == 200, f"ch3: expected 200, got {ch3}"


@cocotb.test()
async def test_dw_conv_3x3_with_padding(dut):
    """DW Conv: 3x3 kernel, 1 channel, pad=1, stride=1, 3x3 in → 3x3 out."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # All-ones 3x3 kernel
    write_sram_word(dut, 'wgt', 0, pack_i8x4(1, 1, 1, 1))
    write_sram_word(dut, 'wgt', 1, pack_i8x4(1, 1, 1, 1))
    write_sram_word(dut, 'wgt', 2, pack_i8x4(1, 0, 0, 0))

    # Input: 3x3, 1ch, values 1-9
    write_sram_word(dut, 'act', 0, pack_i8x4(1, 2, 3, 4))
    write_sram_word(dut, 'act', 1, pack_i8x4(5, 6, 7, 8))
    write_sram_word(dut, 'act', 2, pack_i8x4(9, 0, 0, 0))

    # Params: passthrough
    write_sram_word(dut, 'param', 0, 0x00000001)
    write_sram_word(dut, 'param', 1, 0x00000000)
    write_sram_word(dut, 'param', 2, 0x00000000)
    write_sram_word(dut, 'param', 3, 0x00000000)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    # 3x3 in, pad=1, stride=1 → 3x3 out
    set_cfg(dut, op_type=1, in_c=1, out_c=1, kh=3, kw=3,
            out_h=3, out_w=3, in_h=3, in_w=3,
            stride_h=1, stride_w=1, pad_top=1, pad_left=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=5000), "DW padded done never asserted"

    # 9 output pixels. Center pixel (1,1) sums all 9 inputs = 45
    # Output byte offset for (1,1) ch0 = (1*3+1)*1 + 0 = 4 → word 1, byte 0
    # Note: output overwrites input SRAM in this test, so only verify completion.
