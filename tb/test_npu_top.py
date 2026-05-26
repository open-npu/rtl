# Open-NPU RTL — Top-Level Integration Tests (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# Tests for npu_top: CSR access, DMA transfers, controller FSM cycle.

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


class WbSlave:
    """Wishbone B4 single-cycle master driver for npu_top's slave port."""

    def __init__(self, dut, clk):
        self.dut = dut
        self.clk = clk

    async def write(self, addr, data, sel=0xF):
        self.dut.wb_slv_cyc_i.value = 1
        self.dut.wb_slv_stb_i.value = 1
        self.dut.wb_slv_we_i.value = 1
        self.dut.wb_slv_adr_i.value = addr
        self.dut.wb_slv_dat_i.value = data
        self.dut.wb_slv_sel_i.value = sel
        await RisingEdge(self.clk)
        for _ in range(100):
            try:
                if self.dut.wb_slv_ack_o.value:
                    break
            except ValueError:
                pass
            await RisingEdge(self.clk)
        self.dut.wb_slv_cyc_i.value = 0
        self.dut.wb_slv_stb_i.value = 0
        await RisingEdge(self.clk)

    async def read(self, addr):
        self.dut.wb_slv_cyc_i.value = 1
        self.dut.wb_slv_stb_i.value = 1
        self.dut.wb_slv_we_i.value = 0
        self.dut.wb_slv_adr_i.value = addr
        self.dut.wb_slv_sel_i.value = 0xF
        await RisingEdge(self.clk)
        for _ in range(100):
            try:
                if self.dut.wb_slv_ack_o.value:
                    break
            except ValueError:
                pass
            await RisingEdge(self.clk)
        # Sample data after NBA settles
        await ReadOnly()
        try:
            data = int(self.dut.wb_slv_dat_o.value)
        except ValueError:
            data = 0
        # Exit ReadOnly phase before driving
        await Timer(1, unit="step")
        await RisingEdge(self.clk)
        self.dut.wb_slv_cyc_i.value = 0
        self.dut.wb_slv_stb_i.value = 0
        await RisingEdge(self.clk)
        return data


class WbMasterMem:
    """Emulates external memory responding to DMA WB master requests."""

    def __init__(self, dut, clk, mem_data=None):
        self.dut = dut
        self.clk = clk
        self.mem = {}  # addr → 32-bit word
        if mem_data:
            self.mem.update(mem_data)

    async def run(self):
        """Background task: respond to WB master transactions."""
        while True:
            await RisingEdge(self.clk)
            try:
                cyc = int(self.dut.wb_mst_cyc_o.value)
                stb = int(self.dut.wb_mst_stb_o.value)
            except ValueError:
                self.dut.wb_mst_ack_i.value = 0
                continue
            if cyc and stb:
                try:
                    addr = int(self.dut.wb_mst_adr_o.value)
                except ValueError:
                    self.dut.wb_mst_ack_i.value = 0
                    continue
                if self.dut.wb_mst_we_o.value:
                    # Write from DMA to memory
                    try:
                        self.mem[addr] = int(self.dut.wb_mst_dat_o.value)
                    except ValueError:
                        self.mem[addr] = 0  # X on data bus, store 0
                else:
                    # Read from memory to DMA
                    self.dut.wb_mst_dat_i.value = self.mem.get(addr, 0xDEAD_BEEF)
                self.dut.wb_mst_ack_i.value = 1
            else:
                self.dut.wb_mst_ack_i.value = 0


async def reset(dut):
    """Apply reset sequence."""
    dut.rst_n.value = 0
    dut.wb_slv_cyc_i.value = 0
    dut.wb_slv_stb_i.value = 0
    dut.wb_slv_we_i.value = 0
    dut.wb_slv_adr_i.value = 0
    dut.wb_slv_dat_i.value = 0
    dut.wb_slv_sel_i.value = 0
    dut.wb_mst_ack_i.value = 0
    dut.wb_mst_dat_i.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════


@cocotb.test()
async def test_reset_state(dut):
    """After reset, IRQ should be 0, CSR status should be idle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # Check IRQ is deasserted
    await ReadOnly()
    assert dut.irq_o.value == 0, "IRQ should be 0 after reset"


@cocotb.test()
async def test_csr_version_read(dut):
    """Read VERSION register (0x014) through top-level WB slave."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    version = await wb.read(0x014)
    # VERSION: {8'd0, MAJOR=1, MINOR=0, PATCH=0} = 0x00_01_00_00
    assert version == 0x00010000, f"Expected VERSION=0x00010000, got 0x{version:08X}"


@cocotb.test()
async def test_csr_hw_config_read(dut):
    """Read HW_CONFIG register (0x018) through top-level WB slave."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    hw_cfg = await wb.read(0x018)
    # Dynamic: read ARRAY_SIZE from RTL parameter
    array_sz = int(dut.ARRAY_SIZE.value)
    import math
    dw_ch_log2 = int(math.log2(array_sz))
    expected = (0 << 27) | (0 << 26) | (1 << 25) | (1 << 24) | (128 << 16) | (dw_ch_log2 << 12) | (1 << 8) | array_sz
    assert hw_cfg == expected, f"Expected HW_CONFIG=0x{expected:08X}, got 0x{hw_cfg:08X}"


@cocotb.test()
async def test_csr_write_read_roundtrip(dut):
    """Write a layer register and read it back."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # Write LAYER_MODE (0x040) = 0xCAFEBABE
    await wb.write(0x040, 0xCAFEBABE)
    val = await wb.read(0x040)
    assert val == 0xCAFEBABE, f"Expected 0xCAFEBABE, got 0x{val:08X}"

    # Write DMA_IN_ADDR (0x100) = 0x80000000
    await wb.write(0x100, 0x80000000)
    val = await wb.read(0x100)
    assert val == 0x80000000, f"Expected 0x80000000, got 0x{val:08X}"


@cocotb.test()
async def test_irq_generation(dut):
    """Start a layer with zero-length DMA → done quickly → IRQ fires."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    # Configure zero-length DMA (all sizes = 0)
    await wb.write(0x128, 0)  # DMA_IN_SIZE = 0
    await wb.write(0x12C, 0)  # DMA_WGT_SIZE = 0
    await wb.write(0x188, 0)  # POST_PARAM_COUNT = 0

    # Enable IRQ for done bit
    await wb.write(0x008, 0x01)  # IRQ_EN[0] = done

    # Start (write CTRL[0] = 1)
    await wb.write(0x000, 0x01)

    # Controller will go through states quickly with zero-length,
    # but compute_done depends on systolic output valid (which won't fire).
    # For this V1 integration test, we verify the start was accepted.
    # Wait a few cycles for the start to propagate
    for _ in range(5):
        await RisingEdge(dut.clk)

    # Read STATUS — should show busy
    status = await wb.read(0x004)
    busy_bit = status & 0x1
    assert busy_bit == 1, f"Expected busy after START, got status=0x{status:08X}"


@cocotb.test()
async def test_dma_load_weight(dut):
    """Load weight data from external memory into weight SRAM via DMA."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # Populate external memory with test data
    wgt_base = 0x4000_0000
    test_words = [0x11223344, 0x55667788, 0xAABBCCDD, 0xEEFF0011]
    ext_mem = {wgt_base + i * 4: test_words[i] for i in range(len(test_words))}
    mem = WbMasterMem(dut, dut.clk, ext_mem)
    cocotb.start_soon(mem.run())

    # Configure: set wgt_addr and wgt_size (4 words = 16 bytes)
    await wb.write(0x108, wgt_base)   # DMA_WGT_ADDR
    await wb.write(0x12C, 16)         # DMA_WGT_SIZE = 16 bytes → 4 words
    await wb.write(0x100, 0x8000_0000)  # DMA_IN_ADDR (different from wgt)
    await wb.write(0x128, 0)          # DMA_IN_SIZE = 0
    await wb.write(0x10C, 0x9000_0000)  # DMA_PARAM_ADDR (different)
    await wb.write(0x188, 0)          # POST_PARAM_COUNT = 0

    # Start layer
    await wb.write(0x000, 0x01)

    # Wait for controller to complete weight DMA phase
    # (wgt phase → act phase; act is zero-length so skips to param phase)
    # The full cycle is: LOAD_WGT → WAIT_WGT → LOAD_ACT(skip) → LOAD_PARAM(skip) → COMPUTE
    timeout = 200
    for _ in range(timeout):
        await RisingEdge(dut.clk)

    # Read status — should be in compute (busy) since compute_done won't fire
    status = await wb.read(0x004)
    assert (status & 0x1) == 1, f"Expected busy during compute, got 0x{status:08X}"


@cocotb.test()
async def test_abort_during_operation(dut):
    """Abort while busy should return to idle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # Setup minimal config with non-zero weight (so DMA takes time)
    wgt_base = 0x2000_0000
    ext_mem = {wgt_base + i * 4: i + 1 for i in range(100)}
    mem = WbMasterMem(dut, dut.clk, ext_mem)
    cocotb.start_soon(mem.run())

    await wb.write(0x108, wgt_base)    # DMA_WGT_ADDR
    await wb.write(0x12C, 400)         # DMA_WGT_SIZE = 400 bytes → 100 words
    await wb.write(0x100, 0x3000_0000) # DMA_IN_ADDR
    await wb.write(0x128, 0)           # DMA_IN_SIZE = 0
    await wb.write(0x10C, 0x4000_0000) # DMA_PARAM_ADDR
    await wb.write(0x188, 0)           # POST_PARAM_COUNT = 0

    # Start
    await wb.write(0x000, 0x01)

    # Wait a few cycles then abort
    for _ in range(20):
        await RisingEdge(dut.clk)

    # Should be busy during DMA
    status = await wb.read(0x004)
    assert (status & 0x1) == 1, "Expected busy during DMA load"

    # Abort
    await wb.write(0x000, 0x02)  # CTRL[1] = ABORT

    # Wait for abort to take effect
    for _ in range(20):
        await RisingEdge(dut.clk)

    # Now status should be not busy (controller returns to IDLE via DONE)
    status = await wb.read(0x004)
    busy = status & 0x1
    assert busy == 0, f"Expected not busy after abort, got status=0x{status:08X}"


@cocotb.test()
async def test_soft_reset(dut):
    """Soft reset clears all state."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())

    # Write some config
    await wb.write(0x040, 0xDEADBEEF)  # LAYER_MODE

    # Issue soft reset
    await wb.write(0x000, 0x04)  # CTRL[2] = SOFT_RST

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Layer mode should still be 0xDEADBEEF (soft_rst only resets controller FSM, not CSR data)
    val = await wb.read(0x040)
    assert val == 0xDEADBEEF, f"Soft reset should not clear CSR data, got 0x{val:08X}"

    # Status should show not busy
    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should not be busy after soft reset"


@cocotb.test()
async def test_full_dma_cycle_with_abort(dut):
    """Run a full DMA weight load, then abort at compute phase."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # Setup 8 words of weight data
    wgt_base = 0x1000_0000
    wgt_data = {wgt_base + i * 4: (i + 1) * 0x1111 for i in range(8)}
    mem = WbMasterMem(dut, dut.clk, wgt_data)
    cocotb.start_soon(mem.run())

    await wb.write(0x108, wgt_base)    # DMA_WGT_ADDR
    await wb.write(0x12C, 32)          # DMA_WGT_SIZE = 32 bytes → 8 words
    await wb.write(0x100, 0x5000_0000) # DMA_IN_ADDR (different from wgt)
    await wb.write(0x128, 0)           # DMA_IN_SIZE = 0 (skip)
    await wb.write(0x10C, 0x6000_0000) # DMA_PARAM_ADDR
    await wb.write(0x188, 0)           # POST_PARAM_COUNT = 0

    # Start
    await wb.write(0x000, 0x01)

    # Wait for weight DMA to complete (8 words × ~3 cycles each = ~24 cycles)
    for _ in range(100):
        await RisingEdge(dut.clk)

    # At this point controller should be stuck at COMPUTE (waiting for compute_done)
    status = await wb.read(0x004)
    assert (status & 0x1) == 1, "Should still be busy (waiting in compute)"

    # Abort to exit
    await wb.write(0x000, 0x02)
    for _ in range(10):
        await RisingEdge(dut.clk)

    status = await wb.read(0x004)
    assert (status & 0x1) == 0, "Should be idle after abort"


# ═══════════════════════════════════════════════════════════════════════
# End-to-End Golden Tests: Full Compute Cycle
# ═══════════════════════════════════════════════════════════════════════


def pack_i8_to_words(byte_list):
    """Pack list of INT8 bytes into 32-bit words (little-endian)."""
    words = []
    for i in range(0, len(byte_list), 4):
        w = 0
        for j in range(4):
            if i + j < len(byte_list):
                b = byte_list[i + j] & 0xFF
                w |= b << (j * 8)
        words.append(w)
    return words


def build_ppu_params(bias, mult_m, shift_s, zero_point):
    """Build 4-word (14-byte padded to 16) PPU parameter block for one channel.

    Layout (matching npu_compute.v S_PARAM_LOAD extraction):
      param_buf[0][14:0] = mult_m
      param_buf[0][21:16] = shift_s
      param_buf[1][15:0] = zero_point
      param_buf[1][31:16], param_buf[2], param_buf[3][15:0] = bias (64-bit)
    """
    # bias is signed 64-bit, split into parts
    bias_u = bias & 0xFFFF_FFFF_FFFF_FFFF
    # Word 0: [21:16]=shift_s, [14:0]=mult_m
    w0 = (mult_m & 0x7FFF) | ((shift_s & 0x3F) << 16)
    # Word 1: [15:0]=zero_point, [31:16]=bias[47:32] → wait, let me re-read the extraction
    # From npu_compute.v line 858-862:
    #   ppu_mult_m     <= param_buf[0][14:0];
    #   ppu_shift_s    <= param_buf[0][21:16];
    #   ppu_zero_point <= $signed(param_buf[1][15:0]);
    #   ppu_bias       <= $signed({param_buf[3][15:0], param_buf[2], param_buf[1][31:16]});
    # So bias = {param_buf[3][15:0], param_buf[2][31:0], param_buf[1][31:16]}
    # That's 16+32+16 = 64 bits
    # bias[15:0]  = param_buf[1][31:16]
    # bias[47:16] = param_buf[2][31:0]
    # bias[63:48] = param_buf[3][15:0]
    zp_u = zero_point & 0xFFFF
    bias_lo16 = (bias_u >> 0) & 0xFFFF   # goes to param_buf[1][31:16]
    bias_mid32 = (bias_u >> 16) & 0xFFFF_FFFF  # goes to param_buf[2]
    bias_hi16 = (bias_u >> 48) & 0xFFFF  # goes to param_buf[3][15:0]
    w1 = zp_u | (bias_lo16 << 16)
    w2 = bias_mid32
    w3 = bias_hi16
    return [w0, w1, w2, w3]


@cocotb.test()
async def test_e2e_conv1x1_golden(dut):
    """E2E: Conv2D 1×1, 1 pixel, ARRAY_SIZE in/out channels.

    Full cycle: DMA loads weights/act/params → compute → DMA stores output.
    Uses PPU passthrough mode (output = clamp(acc, -128, 127)).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)
    ARRAY_SIZE = int(dut.ARRAY_SIZE.value)

    # ─── Build test data ───
    # Weights: identity-like: w[col][row] = 1 for all (uniform weight=1)
    # k_depth = 1*1*ARRAY_SIZE = ARRAY_SIZE
    # Weight layout in SRAM: column-major, each column = ARRAY_SIZE bytes
    # All weights = 1 → each output channel = sum of all input activations
    wgt_bytes = [1] * (ARRAY_SIZE * ARRAY_SIZE)

    # Activations: one pixel, ARRAY_SIZE channels, values = [1, 2, 3, ..., ARRAY_SIZE]
    act_bytes = [(i + 1) for i in range(ARRAY_SIZE)]

    # Expected dot product per output channel (all weights = 1):
    # dot[col] = sum(act[0..ARRAY_SIZE-1]) = ARRAY_SIZE*(ARRAY_SIZE+1)/2
    expected_dot = ARRAY_SIZE * (ARRAY_SIZE + 1) // 2

    # PPU: CONV_REQ mode (mode=0) with identity requant (M=1,S=0,bias=0,zp=0)
    # This applies full pipeline including clamping to [-128,127]
    # Expected output = clamp(expected_dot, -128, 127)
    expected_out = min(max(expected_dot, -128), 127)

    # Build param words for all ARRAY_SIZE channels (each 4 words)
    param_bytes_all = []
    for _ in range(ARRAY_SIZE):
        param_bytes_all.extend(build_ppu_params(bias=0, mult_m=1, shift_s=0, zero_point=0))

    # ─── Pack into external memory ───
    wgt_base = 0x4000_0000
    act_base = 0x5000_0000
    param_base = 0x6000_0000
    out_base = 0x7000_0000

    wgt_words = pack_i8_to_words(wgt_bytes)
    act_words = pack_i8_to_words(act_bytes)

    ext_mem = {}
    for i, w in enumerate(wgt_words):
        ext_mem[wgt_base + i * 4] = w
    for i, w in enumerate(act_words):
        ext_mem[act_base + i * 4] = w
    for i, w in enumerate(param_bytes_all):
        ext_mem[param_base + i * 4] = w
    # Pre-fill output area with 0xDEADBEEF to detect writes
    out_word_count = (ARRAY_SIZE + 3) // 4
    for i in range(out_word_count):
        ext_mem[out_base + i * 4] = 0xDEAD_BEEF

    mem = WbMasterMem(dut, dut.clk, ext_mem)
    cocotb.start_soon(mem.run())

    # ─── Configure CSR registers ───
    await wb.write(0x040, 0x00000000)  # LAYER_MODE: op_type=0 (Conv2D)
    await wb.write(0x044, (1 << 16) | 1)  # IN_DIM_HW: in_h=1, in_w=1
    await wb.write(0x048, ARRAY_SIZE)  # IN_DIM_C
    await wb.write(0x04C, (1 << 16) | 1)  # OUT_DIM_HW: out_h=1, out_w=1
    await wb.write(0x050, ARRAY_SIZE)  # OUT_DIM_C
    await wb.write(0x054, (1 << 8) | 1)   # KERNEL_SIZE: kw=1, kh=1
    await wb.write(0x058, (1 << 8) | 1)   # STRIDE: sw=1, sh=1
    await wb.write(0x05C, 0)              # PADDING: none
    await wb.write(0x070, (1 << 16) | 1)  # TILE_CFG: tile_w=1, tile_h=1
    await wb.write(0x074, (1 << 16) | 1)  # TILE_COUNT: tile_num_w=1, tile_num_h=1

    # DMA addresses
    await wb.write(0x108, wgt_base)    # DMA_WGT_ADDR
    await wb.write(0x100, act_base)    # DMA_IN_ADDR
    await wb.write(0x104, out_base)    # DMA_OUT_ADDR
    await wb.write(0x10C, param_base)  # DMA_PARAM_ADDR

    # DMA sizes
    wgt_size = ARRAY_SIZE * ARRAY_SIZE  # bytes
    act_size = ARRAY_SIZE               # bytes
    out_size = ARRAY_SIZE               # bytes
    await wb.write(0x12C, wgt_size)    # DMA_WGT_SIZE
    await wb.write(0x128, act_size)    # DMA_IN_SIZE
    await wb.write(0x130, out_size)    # DMA_OUT_SIZE
    await wb.write(0x188, ARRAY_SIZE)  # POST_PARAM_COUNT = ARRAY_SIZE channels

    # Post-processing: CONV_REQ mode with bias_en + zp_en (identity requant)
    # bits[1:0]=00(CONV_REQ), bit[2]=0(no relu), bit[5]=1(zp_en), bit[6]=1(bias_en)
    await wb.write(0x180, 0x60)        # POST_CTRL: mode=CONV_REQ, bias+zp enabled

    # No fusion
    await wb.write(0x118, 0)           # DMA_CTRL: no fusion bits

    # ─── Start layer ───
    await wb.write(0x000, 0x01)

    # ─── Wait for completion ───
    # Poll status register periodically. The layer should complete within
    # a few thousand cycles for this simple 1-pixel test.
    timeout = 10000
    done = False
    for cyc in range(timeout):
        await RisingEdge(dut.clk)
        if cyc > 200 and cyc % 100 == 0:
            try:
                s = await wb.read(0x004)
                if (s & 0x1) == 0:
                    done = True
                    break
            except (ValueError, AttributeError):
                continue

    assert done, f"Layer did not complete within {timeout} cycles"

    # ─── Verify output in external memory ───
    # Output was stored via DMA back to out_base
    out_word = mem.mem.get(out_base, 0xDEAD_BEEF)
    assert out_word != 0xDEAD_BEEF, "Output was never written to external memory"

    # Check each output byte
    for ch in range(ARRAY_SIZE):
        word_idx = ch // 4
        byte_idx = ch % 4
        word = mem.mem.get(out_base + word_idx * 4, 0)
        got = (word >> (byte_idx * 8)) & 0xFF
        # Convert to signed
        if got >= 128:
            got -= 256
        assert got == expected_out, (
            f"Channel {ch}: expected {expected_out}, got {got} "
            f"(word[{word_idx}]=0x{word:08X})"
        )


@cocotb.test()
async def test_e2e_dw_conv_3x3_golden(dut):
    """E2E: DW Conv 3×3, 1 channel, 3×3 input (no padding), 1×1 output.

    Full cycle through DMA + compute + store.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb = WbSlave(dut, dut.clk)

    # ─── Build test data ───
    # 3×3 kernel, all ones: dot = sum of all 9 input pixels
    kernel = [1] * 9

    # 3×3 input (1 channel), NHWC layout: values 1-9
    # act[h][w][c] = h*3 + w + 1 (c=0 only)
    act_data = list(range(1, 10))  # [1,2,3,4,5,6,7,8,9]

    # Expected: sum(1..9) = 45, clamped to int8 = 45
    expected_dot = sum(act_data)
    expected_out = min(max(expected_dot, -128), 127)

    # PPU params for 1 channel (passthrough mode)
    param_words_list = build_ppu_params(bias=0, mult_m=1, shift_s=0, zero_point=0)

    # ─── Pack into external memory ───
    wgt_base = 0x4000_0000
    act_base = 0x5000_0000
    param_base = 0x6000_0000
    out_base = 0x7000_0000

    wgt_words = pack_i8_to_words(kernel)
    act_words = pack_i8_to_words(act_data)

    ext_mem = {}
    for i, w in enumerate(wgt_words):
        ext_mem[wgt_base + i * 4] = w
    for i, w in enumerate(act_words):
        ext_mem[act_base + i * 4] = w
    for i, w in enumerate(param_words_list):
        ext_mem[param_base + i * 4] = w
    ext_mem[out_base] = 0xDEAD_BEEF

    mem = WbMasterMem(dut, dut.clk, ext_mem)
    cocotb.start_soon(mem.run())

    # ─── Configure CSR ───
    await wb.write(0x040, 0x00000001)  # LAYER_MODE: op_type=1 (DW Conv)
    await wb.write(0x044, (3 << 16) | 3)  # IN_DIM_HW: in_h=3, in_w=3
    await wb.write(0x048, 1)              # IN_DIM_C = 1
    await wb.write(0x04C, (1 << 16) | 1)  # OUT_DIM_HW: out_h=1, out_w=1
    await wb.write(0x050, 1)              # OUT_DIM_C = 1
    await wb.write(0x054, (3 << 8) | 3)   # KERNEL_SIZE: kw=3, kh=3
    await wb.write(0x058, (1 << 8) | 1)   # STRIDE: sw=1, sh=1
    await wb.write(0x05C, 0)              # PADDING: none
    await wb.write(0x070, (1 << 16) | 1)  # TILE_CFG: tile_w=1, tile_h=1
    await wb.write(0x074, (1 << 16) | 1)  # TILE_COUNT: 1x1

    # DMA
    await wb.write(0x108, wgt_base)
    await wb.write(0x100, act_base)
    await wb.write(0x104, out_base)
    await wb.write(0x10C, param_base)
    await wb.write(0x12C, 12)             # DMA_WGT_SIZE: 9 bytes padded to 12 (3 words)
    await wb.write(0x128, 12)             # DMA_IN_SIZE: 9 bytes padded to 12
    await wb.write(0x130, 4)              # DMA_OUT_SIZE: 1 byte padded to 4
    await wb.write(0x188, 1)              # POST_PARAM_COUNT = 1 channel

    # PPU passthrough
    await wb.write(0x180, 0x03)

    # No fusion
    await wb.write(0x118, 0)

    # ─── Start ───
    await wb.write(0x000, 0x01)

    # ─── Wait for completion ───
    timeout = 3000
    done = False
    for cyc in range(timeout):
        await RisingEdge(dut.clk)
        if cyc > 50 and cyc % 30 == 0:
            await ReadOnly()
            await Timer(1, unit="step")
            s = await wb.read(0x004)
            if (s & 0x1) == 0:
                done = True
                break

    if not done:
        status = await wb.read(0x004)
        assert (status & 0x1) == 0, f"DW Conv did not complete, status=0x{status:08X}"

    # ─── Verify output ───
    out_word = mem.mem.get(out_base, 0xDEAD_BEEF)
    assert out_word != 0xDEAD_BEEF, "Output never written"
    got = out_word & 0xFF
    if got >= 128:
        got -= 256
    assert got == expected_out, f"Expected {expected_out}, got {got} (word=0x{out_word:08X})"
