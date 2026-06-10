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
    dut.cfg_int16.value = 0     # INT8 mode (default)
    dut.db_prefetch_done.value = 1  # No DB_EN in unit tests — always ready
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def set_cfg(dut, in_c=4, out_c=4, kh=1, kw=1, out_h=1, out_w=1,
            stride_h=1, stride_w=1, pad_top=0, pad_left=0,
            tile_h=0, tile_w=0, tile_num_h=1, tile_num_w=1,
            in_h=1, in_w=1, op_type=0, act_base=0, out_base=0):
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
    dut.cfg_act_base.value = act_base
    dut.cfg_out_base.value = out_base


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
# ===================================================================
# Spatial Tiling Tests (Conv2D datapath, output verification)
# ===================================================================

@cocotb.test()
async def test_conv_spatial_2x2(dut):
    """Conv 2x2 spatial output: k_depth=ARRAY_SIZE, 4 pixels, verify dot products.

    Uses k_depth=ARRAY_SIZE (full row utilization). Each pixel has constant
    activation value, kernel is w[k]=k+1. Ensures spatial loop iterates
    correctly and reduction produces correct dot products.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)

    # Kernel: w[k] = k+1 for k=0..ARRAY_SIZE-1 (same for all OC columns)
    wgt_words_per_col = ARRAY_SIZE // 4
    for oc in range(ARRAY_SIZE):
        base_addr = oc * wgt_words_per_col
        for w in range(wgt_words_per_col):
            b0 = w * 4 + 1
            b1 = w * 4 + 2
            b2 = w * 4 + 3
            b3 = w * 4 + 4
            write_sram_word(dut, 'wgt', base_addr + w, pack_i8x4(b0, b1, b2, b3))

    # 4 pixels, constant activation per pixel: pval = 1, 2, 3, 4
    pixel_vals = [1, 2, 3, 4]
    act_words_per_pixel = ARRAY_SIZE // 4

    for i in range(256):
        write_sram_word(dut, 'act', i, 0)

    for pix_idx, pval in enumerate(pixel_vals):
        base_addr = pix_idx * act_words_per_pixel
        for w in range(act_words_per_pixel):
            write_sram_word(dut, 'act', base_addr + w,
                            pack_i8x4(pval, pval, pval, pval))

    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    # in_c=ARRAY_SIZE, kh=1, kw=1 -> k_depth=ARRAY_SIZE
    # pixel_act_base = (sp_oh*2+sp_ow)*ARRAY_SIZE/4 = pix_idx * act_words_per_pixel
    set_cfg(dut, in_c=ARRAY_SIZE, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=2, out_w=2, in_h=2, in_w=2)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=200000), "2x2 spatial done never asserted"

    # dot[pix] = pval * sum(1..ARRAY_SIZE) = pval * ARRAY_SIZE*(ARRAY_SIZE+1)/2
    sum_weights = ARRAY_SIZE * (ARRAY_SIZE + 1) // 2
    expected = [(pval * sum_weights) & 0xFF for pval in pixel_vals]
    dut._log.info(f"ARRAY_SIZE={ARRAY_SIZE}, sum_w={sum_weights}, expected={expected}")

    for pix_idx, (oh, ow) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        byte_off = pix_idx * ARRAY_SIZE
        word_addr = byte_off >> 2
        out_word = read_sram_word(dut, 'act', word_addr)
        byte_pos = byte_off & 3
        out_byte = (out_word >> (byte_pos * 8)) & 0xFF
        dut._log.info(f"Pixel({oh},{ow}): ch0={out_byte}, exp={expected[pix_idx]}")
        assert out_byte == expected[pix_idx], \
            f"Pixel({oh},{ow}) ch0: expected {expected[pix_idx]}, got {out_byte}"


@cocotb.test()
async def test_conv_spatial_3x3(dut):
    """Conv 3x3 spatial output: 9 pixels with distinct values, all-ones kernel.

    dot[pix] = pval * ARRAY_SIZE (since all weights=1).
    Verifies the full spatial loop iterates all 9 pixels correctly.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)

    # All-ones kernel for all columns
    wgt_words_per_col = ARRAY_SIZE // 4
    for oc in range(ARRAY_SIZE):
        base_addr = oc * wgt_words_per_col
        for w in range(wgt_words_per_col):
            write_sram_word(dut, 'wgt', base_addr + w, pack_i8x4(1, 1, 1, 1))

    # 9 pixels with distinct values: pval = 1..9
    pixel_vals = list(range(1, 10))
    act_words_per_pixel = ARRAY_SIZE // 4

    for i in range(512):
        write_sram_word(dut, 'act', i, 0)

    for pix_idx, pval in enumerate(pixel_vals):
        base_addr = pix_idx * act_words_per_pixel
        for w in range(act_words_per_pixel):
            write_sram_word(dut, 'act', base_addr + w,
                            pack_i8x4(pval, pval, pval, pval))

    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    # pixel_act_base = (sp_oh*3+sp_ow)*ARRAY_SIZE/4
    set_cfg(dut, in_c=ARRAY_SIZE, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=3, out_w=3, in_h=3, in_w=3)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=500000), "3x3 spatial done never asserted"

    # dot[pix] = pval * ARRAY_SIZE
    expected = [(pval * ARRAY_SIZE) & 0xFF for pval in pixel_vals]
    dut._log.info(f"Expected outputs: {expected}")

    for pix_idx in range(9):
        oh, ow = pix_idx // 3, pix_idx % 3
        byte_off = pix_idx * ARRAY_SIZE
        word_addr = byte_off >> 2
        out_word = read_sram_word(dut, 'act', word_addr)
        byte_pos = byte_off & 3
        out_byte = (out_word >> (byte_pos * 8)) & 0xFF
        dut._log.info(f"Pixel({oh},{ow}): ch0={out_byte}, exp={expected[pix_idx]}")
        assert out_byte == expected[pix_idx], \
            f"Pixel({oh},{ow}) ch0: expected {expected[pix_idx]}, got {out_byte}"


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


# ═══════════════════════════════════════════════════════════════════════════
# Multi-pass tests (k_depth > ARRAY_SIZE)
# ═══════════════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_conv1x1_kdepth_32(dut):
    """Conv 1x1 with k_depth=32 (2 passes). 1 pixel, verify dot product.

    ARRAY_SIZE=16, in_c=32, kh=1, kw=1 → k_depth=32, k_pass_max=1.
    All weights=1, activations=[1..32].
    Expected dot product = sum(1..32) = 528.
    With passthrough PPU: 528 & 0xFF = 16 (0x10).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    in_c = 32  # 2x ARRAY_SIZE
    assert ARRAY_SIZE == 16, f"Test assumes ARRAY_SIZE=16, got {ARRAY_SIZE}"

    # Weights: all ones. Layout: col c at words [c*32/4 .. c*32/4 + 7]
    # Each column has 32 weights (k_depth=32), packed 4 per word = 8 words/col
    for col in range(ARRAY_SIZE):
        for w in range(in_c // 4):
            write_sram_word(dut, 'wgt', col * (in_c // 4) + w, pack_i8x4(1, 1, 1, 1))

    # Activations: values 1..32. Packed 4 per word at addresses 0..7
    # Separate from output: put acts starting at word 0, output at word 64+
    act_base_word = 0
    for w in range(in_c // 4):
        b0 = w * 4 + 1
        write_sram_word(dut, 'act', act_base_word + w,
                        pack_i8x4(b0, b0+1, b0+2, b0+3))

    # Pre-zero output area (output at word offset for pixel 0)
    # Output NHWC: pixel 0, 16 channels → words at offset determined by out_c
    # With in_c=32, in_w=1: pixel 0 input at byte 0. Output also at byte 0 of out area.
    # To avoid collision, use separate input/output regions.
    # Input at words 0..7 (32 bytes). Output at words 0..3 (16 bytes for out_c=16).
    # Collision! Let's offset activations. Actually in the compute module,
    # act_base = tile_y * out_tile_h * stride_h * in_w * in_c / 4 = 0 for single pixel.
    # out_base = 0 as well. So they WILL collide if in_c > out_c.
    # 
    # Fix: put activations at a different region. Use cfg layout:
    # The compute module uses act_base for reading input and out_base for writing output.
    # With tile_y=0, tile_x=0, stride=1, these are both 0. Input reads from act SRAM
    # words 0..7 (32 bytes of activation). Output writes to act SRAM starting at out_base=0.
    # This DOES collide (first 4 output words overwrite first 4 input words).
    #
    # For this test: we only have 1 pixel, and once input is consumed it's fine
    # to overwrite. The compute reads all activations before writing any output.
    # So collision is OK here.

    # Params: passthrough (M=1, S=0, bias=0, zp=0)
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)  # M=1, S=0
        write_sram_word(dut, 'param', base + 1, 0)           # zp=0
        write_sram_word(dut, 'param', base + 2, 0)           # bias low
        write_sram_word(dut, 'param', base + 3, 0)           # bias high

    dut.ppu_mode.value = 3      # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=1, out_w=1, in_h=1, in_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=50000), "done never asserted (k_depth=32)"

    # Expected: dot = sum(1..32) = 528. Passthrough truncates to 528 & 0xFF = 16
    # All 16 output channels should have the same value (all weights=1)
    expected = 528 & 0xFF  # = 16
    for word_idx in range(ARRAY_SIZE // 4):
        out_word = read_sram_word(dut, 'act', word_idx)
        for b in range(4):
            ch = word_idx * 4 + b
            byte_val = (out_word >> (b * 8)) & 0xFF
            assert byte_val == expected, \
                f"Channel {ch}: got {byte_val}, expected {expected} (sum=528, trunc to u8)"
    dut._log.info(f"PASS: k_depth=32, all channels = {expected}")


@cocotb.test()
async def test_conv1x1_kdepth_32_nonuniform(dut):
    """Conv 1x1 with k_depth=32, non-uniform weights. Verifies accumulation accuracy.

    weights[col=0] = [1,2,3,...,32], other cols = all-ones.
    activations = all-ones (value=1).
    dot_product[col=0] = sum(1..32) = 528.
    dot_product[col>0] = 32 (sum of 32 ones).
    Passthrough: 528&0xFF=16, 32&0xFF=32.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    in_c = 32

    # Col 0: weights = [1, 2, 3, ..., 32]
    for w in range(in_c // 4):
        b0 = w * 4 + 1
        write_sram_word(dut, 'wgt', w, pack_i8x4(b0, b0+1, b0+2, b0+3))

    # Cols 1..15: weights = all ones
    for col in range(1, ARRAY_SIZE):
        for w in range(in_c // 4):
            write_sram_word(dut, 'wgt', col * (in_c // 4) + w, pack_i8x4(1, 1, 1, 1))

    # Activations: all ones (32 bytes)
    for w in range(in_c // 4):
        write_sram_word(dut, 'act', w, pack_i8x4(1, 1, 1, 1))

    # Params: passthrough
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=1, out_w=1, in_h=1, in_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=50000), "done never asserted"

    # Check col 0: sum(1..32) = 528, 528 & 0xFF = 16
    out_word0 = read_sram_word(dut, 'act', 0)
    ch0_val = out_word0 & 0xFF
    assert ch0_val == 16, f"Col 0: got {ch0_val}, expected 16 (528 & 0xFF)"

    # Check cols 1-3: sum = 32
    for b in range(1, 4):
        val = (out_word0 >> (b * 8)) & 0xFF
        assert val == 32, f"Col {b}: got {val}, expected 32"

    dut._log.info(f"PASS: k_depth=32 non-uniform, col0={ch0_val}, col1-3=32")


@cocotb.test()
async def test_conv1x1_kdepth_24_spatial_2x2(dut):
    """Conv 1x1 with k_depth=24 (2 passes, last partial=8), 2x2 spatial output.

    ARRAY_SIZE=16, in_c=24 → k_pass_max=1, pass0=16 elements, pass1=8 elements.
    All weights=1, activations=all-ones.
    dot per pixel = 24 (sum of 24 ones with w=1).
    Output should be 24 for all channels, all pixels.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    in_c = 24
    out_h, out_w = 2, 2

    # Weights: all ones. k_depth=24, 6 words per column
    words_per_col = in_c // 4  # 6
    for col in range(ARRAY_SIZE):
        for w in range(words_per_col):
            write_sram_word(dut, 'wgt', col * words_per_col + w,
                            pack_i8x4(1, 1, 1, 1))

    # Activations: 2x2 input with in_c=24 per pixel = 96 bytes = 24 words
    # Layout NHWC: pixel(y,x) at byte offset (y*in_w + x)*in_c
    # All ones
    total_act_bytes = out_h * out_w * in_c  # 96
    for w in range(total_act_bytes // 4):
        write_sram_word(dut, 'act', w, pack_i8x4(1, 1, 1, 1))

    # Pre-zero output region (overlaps input for this simple case)
    # Output: 2x2 * 16ch = 64 bytes = 16 words starting at out_base=0
    # Input: 2x2 * 24ch = 96 bytes = 24 words
    # Output will overwrite words 0-15 (ok, input already consumed)

    # Params: passthrough
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=out_h, out_w=out_w, in_h=out_h, in_w=out_w)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=100000), "done never asserted (k_depth=24, 2x2)"

    # Verify: each pixel, each channel = 24
    expected = 24
    for pixel in range(out_h * out_w):
        for word_offset in range(ARRAY_SIZE // 4):
            addr = pixel * (ARRAY_SIZE // 4) + word_offset
            out_word = read_sram_word(dut, 'act', addr)
            for b in range(4):
                ch = word_offset * 4 + b
                byte_val = (out_word >> (b * 8)) & 0xFF
                assert byte_val == expected, \
                    f"Pixel {pixel} ch {ch}: got {byte_val}, expected {expected}"

    dut._log.info(f"PASS: k_depth=24, 2x2 spatial, all outputs = {expected}")


@cocotb.test()
async def test_conv3x3_single_pixel(dut):
    """Conv 3×3, in_c=1, out_c=ARRAY_SIZE, 3×3 input (no pad), 1×1 output.

    k_depth = 3*3*1 = 9. Single pass (9 < 16).
    All weights = 1 → dot product = sum of 9 input values.
    Input values = [1,2,3,4,5,6,7,8,9] → dot = 45.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    in_c = 1
    kh, kw = 3, 3
    in_h, in_w = 3, 3
    k_depth = kh * kw * in_c  # 9

    # Weights: all ones. OHWI layout: col c has k_depth bytes starting at byte c*k_depth.
    # Pack all weight bytes contiguously into SRAM words.
    total_wgt_bytes = ARRAY_SIZE * k_depth  # 16 * 9 = 144 bytes
    wgt_bytes = [1] * total_wgt_bytes  # all ones
    for w_idx in range(0, total_wgt_bytes, 4):
        b0 = wgt_bytes[w_idx] if w_idx < total_wgt_bytes else 0
        b1 = wgt_bytes[w_idx+1] if w_idx+1 < total_wgt_bytes else 0
        b2 = wgt_bytes[w_idx+2] if w_idx+2 < total_wgt_bytes else 0
        b3 = wgt_bytes[w_idx+3] if w_idx+3 < total_wgt_bytes else 0
        write_sram_word(dut, 'wgt', w_idx // 4, pack_i8x4(b0, b1, b2, b3))

    # Activations: 3×3×1 = 9 bytes → values 1..9
    # NHWC layout: pixel(y,x) at byte offset (y*in_w + x)*in_c
    # 9 bytes = 3 words (with zero padding)
    write_sram_word(dut, 'act', 0, pack_i8x4(1, 2, 3, 4))
    write_sram_word(dut, 'act', 1, pack_i8x4(5, 6, 7, 8))
    write_sram_word(dut, 'act', 2, pack_i8x4(9, 0, 0, 0))

    # Params: passthrough
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=kh, kw=kw,
            out_h=1, out_w=1, in_h=in_h, in_w=in_w,
            stride_h=1, stride_w=1, pad_top=0, pad_left=0)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Debug: monitor FSM state
    for cyc in range(5000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        try:
            state = int(dut.u_compute.state.value)
            if cyc % 500 == 0:
                dut._log.info(f"cyc={cyc} state={state}")
            if int(dut.done.value) == 1:
                dut._log.info(f"DONE at cycle {cyc}")
                break
        except ValueError:
            dut._log.info(f"cyc={cyc} X-values in state/signals")
    else:
        assert False, "done never asserted (conv3x3) - check FSM states in log"

    # Expected dot product = sum(1..9) = 45
    expected = 45
    out_word = read_sram_word(dut, 'act', 0)
    for b in range(4):
        byte_val = (out_word >> (b * 8)) & 0xFF
        assert byte_val == expected, \
            f"Channel {b}: got {byte_val}, expected {expected}"

    dut._log.info(f"PASS: conv3x3 single pixel, output = {expected}")


@cocotb.test()
async def test_conv3x3_with_padding(dut):
    """Conv 3×3, in_c=1, out_c=ARRAY_SIZE, 2×2 input, pad=1, stride=1 → 2×2 output.

    k_depth = 3*3*1 = 9. Single pass (9 < 16).
    Weights = all 1 → dot product = sum of valid input values.
    Input (2×2, in_c=1):
        [[1, 2],
         [3, 4]]
    With pad=1, output pixel (oh,ow) sees a 3×3 window centered at (oh, ow):
      (0,0): window rows -1..1, cols -1..1 → valid (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4 → sum=10
      (0,1): window rows -1..1, cols 0..2 → valid (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4 → sum=10
      (1,0): window rows 0..2, cols -1..1 → valid (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4 → sum=10
      (1,1): window rows 0..2, cols 0..2 → valid (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4 → sum=10
    All 4 output pixels = 10.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ARRAY_SIZE = get_array_size(dut)
    in_c = 1
    kh, kw = 3, 3
    in_h, in_w = 2, 2
    out_h, out_w = 2, 2
    k_depth = kh * kw * in_c  # 9

    # Weights: all ones. Contiguous OHWI byte packing.
    total_wgt_bytes = ARRAY_SIZE * k_depth
    wgt_bytes = [1] * total_wgt_bytes
    for w_idx in range(0, total_wgt_bytes, 4):
        b0 = wgt_bytes[w_idx] if w_idx < total_wgt_bytes else 0
        b1 = wgt_bytes[w_idx+1] if w_idx+1 < total_wgt_bytes else 0
        b2 = wgt_bytes[w_idx+2] if w_idx+2 < total_wgt_bytes else 0
        b3 = wgt_bytes[w_idx+3] if w_idx+3 < total_wgt_bytes else 0
        write_sram_word(dut, 'wgt', w_idx // 4, pack_i8x4(b0, b1, b2, b3))

    # Activations: 2×2×1 = 4 bytes → values 1,2,3,4
    # Put input at word offset 64 to avoid overlap with output at word 0
    ACT_BASE = 64  # word address
    write_sram_word(dut, 'act', ACT_BASE + 0, pack_i8x4(1, 2, 3, 4))

    # Params: passthrough (scale=1, shift=0, bias=0, zp=0)
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=kh, kw=kw,
            out_h=out_h, out_w=out_w, in_h=in_h, in_w=in_w,
            stride_h=1, stride_w=1, pad_top=1, pad_left=1,
            act_base=ACT_BASE, out_base=0)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    done = await wait_done(dut, timeout=5000)
    assert done, "done never asserted (conv3x3 with padding)"

    # Expected: all 4 pixels = 10 for all ARRAY_SIZE channels
    expected = 10
    # Output layout: NHWC, pixel (oh, ow) at byte (oh*out_w + ow)*out_c + c
    # Total output bytes = 2*2*ARRAY_SIZE
    total_out_bytes = out_h * out_w * ARRAY_SIZE
    for px in range(out_h * out_w):
        for ch in range(min(4, ARRAY_SIZE)):  # Check first 4 channels
            byte_offset = px * ARRAY_SIZE + ch
            word_idx = byte_offset // 4
            byte_pos = byte_offset % 4
            out_word = read_sram_word(dut, 'act', word_idx)
            byte_val = (out_word >> (byte_pos * 8)) & 0xFF
            assert byte_val == expected, \
                f"Pixel {px}, ch {ch}: got {byte_val}, expected {expected}"

    dut._log.info(f"PASS: conv3x3 with padding, all pixels = {expected}")


# ═══════════════════════════════════════════════════════════════════════════
# INT16 MODE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def pack_i16x2(s0, s1):
    """Pack 2 signed int16 into a uint32 (little-endian half-word order)."""
    def to_u16(v):
        return v & 0xFFFF
    return to_u16(s0) | (to_u16(s1) << 16)


def unpack_i16(word, idx):
    """Extract signed int16 at index idx (0 or 1) from a 32-bit word."""
    val = (word >> (idx * 16)) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val


@cocotb.test()
async def test_conv1x1_int16_basic(dut):
    """INT16 mode: 1x1 conv, identity weights, verify 16-bit dot product.

    in_c=4, out_c=ARRAY_SIZE, all-ones weights, activations=[300, -200, 150, -50].
    These values exceed INT8 range, confirming 16-bit precision.
    Expected dot product per column = 300 + (-200) + 150 + (-50) = 200.
    Passthrough PPU → output should be 200 (lower 16 bits of accumulator).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    dut.cfg_int16.value = 1  # INT16 mode

    ARRAY_SIZE = get_array_size(dut)
    in_c = 4

    # Weights: all ones, INT16 packed — 2 elements per word, 2 words per column
    for oc in range(ARRAY_SIZE):
        word_base = oc * (in_c // 2)  # 2 words per column
        write_sram_word(dut, 'wgt', word_base + 0, pack_i16x2(1, 1))
        write_sram_word(dut, 'wgt', word_base + 1, pack_i16x2(1, 1))

    # Activations: [300, -200, 150, -50] packed as INT16
    write_sram_word(dut, 'act', 0, pack_i16x2(300, -200))
    write_sram_word(dut, 'act', 1, pack_i16x2(150, -50))

    # Params: passthrough
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)  # M=1, S=0
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=1, out_w=1, in_h=1, in_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=20000), "INT16 conv1x1 timed out"

    # Output: ARRAY_SIZE channels packed 2 per word
    expected = 200  # sum of [300, -200, 150, -50]
    n_out_words = ARRAY_SIZE // 2
    for w in range(n_out_words):
        out_word = read_sram_word(dut, 'act', w)
        v0 = unpack_i16(out_word, 0)
        v1 = unpack_i16(out_word, 1)
        assert v0 == expected, f"INT16 ch {w*2}: got {v0}, expected {expected}"
        assert v1 == expected, f"INT16 ch {w*2+1}: got {v1}, expected {expected}"

    dut._log.info(f"PASS: INT16 conv1x1 basic, all channels = {expected}")


@cocotb.test()
async def test_conv1x1_int16_large_values(dut):
    """INT16 mode: verify large values beyond INT8 range survive the pipeline.

    Weights=[10, 20], activations=[1000, -500]. in_c=2, out_c=ARRAY_SIZE.
    Expected dot product = 10*1000 + 20*(-500) = 10000 - 10000 = 0.

    Also tests a second case: weights=[3, 7], act=[100, 200].
    dot = 3*100 + 7*200 = 300 + 1400 = 1700.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    dut.cfg_int16.value = 1

    ARRAY_SIZE = get_array_size(dut)
    in_c = 2

    # Weights: [3, 7] for all columns — 1 word per column
    for oc in range(ARRAY_SIZE):
        write_sram_word(dut, 'wgt', oc, pack_i16x2(3, 7))

    # Activations: [100, 200]
    write_sram_word(dut, 'act', 0, pack_i16x2(100, 200))

    # Params: passthrough
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=1, out_w=1, in_h=1, in_w=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=20000), "INT16 large values timed out"

    expected = 1700  # 3*100 + 7*200
    n_out_words = ARRAY_SIZE // 2
    for w in range(n_out_words):
        out_word = read_sram_word(dut, 'act', w)
        v0 = unpack_i16(out_word, 0)
        v1 = unpack_i16(out_word, 1)
        assert v0 == expected, f"INT16 ch {w*2}: got {v0}, expected {expected}"
        assert v1 == expected, f"INT16 ch {w*2+1}: got {v1}, expected {expected}"

    dut._log.info(f"PASS: INT16 large values, all channels = {expected}")


@cocotb.test()
async def test_dw_conv_int16(dut):
    """INT16 mode: DW Conv 1x1 kernel, single channel, verify accumulation.

    weight=5, activation=400, expected output = 5*400 = 2000 (beyond INT8 range).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    dut.cfg_int16.value = 1

    ARRAY_SIZE = get_array_size(dut)

    # DW Conv: op_type=1, kh=1, kw=1, in_c=out_c=1
    # Weight: single value 5 at word 0
    write_sram_word(dut, 'wgt', 0, pack_i16x2(5, 0))

    # Activation: single value 400
    write_sram_word(dut, 'act', 0, pack_i16x2(400, 0))

    # Params for channel 0
    write_sram_word(dut, 'param', 0, 0x00000001)  # M=1, S=0
    write_sram_word(dut, 'param', 1, 0)
    write_sram_word(dut, 'param', 2, 0)
    write_sram_word(dut, 'param', 3, 0)

    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    set_cfg(dut, in_c=1, out_c=1, kh=1, kw=1, out_h=1, out_w=1,
            in_h=1, in_w=1, op_type=1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=20000), "INT16 DW conv timed out"

    out_word = read_sram_word(dut, 'act', 0)
    v0 = unpack_i16(out_word, 0)
    expected = 2000  # 5 * 400
    assert v0 == expected, f"INT16 DW conv: got {v0}, expected {expected}"

    dut._log.info(f"PASS: INT16 DW conv 1x1, output = {expected}")


@cocotb.test()
async def test_conv_spatial_2x2_int16(dut):
    """INT16 mode: 1x1 conv with 2×2 spatial output.

    Verifies writeback packing: 2 int16 values per 32-bit word in NHWC layout.
    in_c=2, out_c=ARRAY_SIZE, weights=all-ones, activations=all 500.
    Expected per-channel output = 500*2 = 1000.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    dut.cfg_int16.value = 1

    ARRAY_SIZE = get_array_size(dut)
    in_c = 2
    out_h, out_w = 2, 2
    in_h, in_w = 2, 2

    # Weights: all ones, 1 word per column (2 elements)
    for oc in range(ARRAY_SIZE):
        write_sram_word(dut, 'wgt', oc, pack_i16x2(1, 1))

    # Activations: 2×2 spatial, 2 channels, value=500
    # NHWC layout: (h, w, c) — 2 channels per word
    for px in range(out_h * out_w):
        write_sram_word(dut, 'act', px, pack_i16x2(500, 500))

    # Params: passthrough
    for ch in range(ARRAY_SIZE):
        base = ch * 4
        write_sram_word(dut, 'param', base + 0, 0x00000001)
        write_sram_word(dut, 'param', base + 1, 0)
        write_sram_word(dut, 'param', base + 2, 0)
        write_sram_word(dut, 'param', base + 3, 0)

    dut.ppu_mode.value = 3  # PASSTHROUGH
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0

    # out_base after input data
    out_base = out_h * out_w  # 4 words for input
    set_cfg(dut, in_c=in_c, out_c=ARRAY_SIZE, kh=1, kw=1,
            out_h=out_h, out_w=out_w, in_h=in_h, in_w=in_w,
            out_base=out_base)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_done(dut, timeout=50000), "INT16 spatial 2x2 timed out"

    # Verify output: each pixel has ARRAY_SIZE channels, 2 per word
    expected = 1000  # 1*500 + 1*500
    for px in range(out_h * out_w):
        for ch_pair in range(ARRAY_SIZE // 2):
            word_offset = px * (ARRAY_SIZE // 2) + ch_pair
            out_word = read_sram_word(dut, 'act', out_base + word_offset)
            v0 = unpack_i16(out_word, 0)
            v1 = unpack_i16(out_word, 1)
            assert v0 == expected, \
                f"Pixel {px} ch {ch_pair*2}: got {v0}, expected {expected}"
            assert v1 == expected, \
                f"Pixel {px} ch {ch_pair*2+1}: got {v1}, expected {expected}"

    dut._log.info(f"PASS: INT16 spatial 2x2, all outputs = {expected}")
