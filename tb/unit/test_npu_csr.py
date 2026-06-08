# Open-NPU RTL — cocotb Tests for npu_csr (CSR Register File)
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock

from utils.wishbone import WishboneMaster
from utils.clock_reset import clock_reset


async def init_dut(dut):
    """Initialize DUT: start clock, reset, zero hw inputs, return WB master."""
    # Zero all hardware status inputs
    dut.hw_busy.value = 0
    dut.hw_dma_busy.value = 0
    dut.hw_done.value = 0
    dut.hw_error.value = 0
    dut.hw_dma_done.value = 0
    dut.hw_error_code.value = 0
    dut.hw_curr_layer.value = 0
    dut.hw_perf_cnt.value = 0
    dut.hw_mac_cnt.value = 0

    # Zero WB signals
    dut.wb_cyc_i.value = 0
    dut.wb_stb_i.value = 0
    dut.wb_we_i.value = 0
    dut.wb_adr_i.value = 0
    dut.wb_dat_i.value = 0
    dut.wb_sel_i.value = 0xF

    await clock_reset(dut)
    wb = WishboneMaster(dut, dut.clk, prefix="wb")
    return wb


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Read-only VERSION register
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_version_register(dut):
    """Read VERSION register — should return V1.0.0 (0x00010000)."""
    wb = await init_dut(dut)
    val = await wb.read(0x014)
    expected = 0x00_01_00_00  # MAJOR=1, MINOR=0, PATCH=0
    assert val == expected, f"VERSION: got {val:#010x}, expected {expected:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Read-only HW_CONFIG register
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_hw_config_register(dut):
    """Read HW_CONFIG — verify array size, INT16 support, etc."""
    wb = await init_dut(dut)
    val = await wb.read(0x018)
    # ARRAY_SIZE is parameterized — read expected from compile-time define
    expected_array_sz = int(dut.ARRAY_SZ.value)
    array_sz = val & 0xFF
    assert array_sz == expected_array_sz, \
        f"ARRAY_SIZE: got {array_sz}, expected {expected_array_sz}"
    # NUM_ARRAYS=1 → [11:8]=1
    num_arr = (val >> 8) & 0xF
    assert num_arr == 1, f"NUM_ARRAYS: got {num_arr}, expected 1"
    # HAS_INT16=1 → bit 24
    has_int16 = (val >> 24) & 1
    assert has_int16 == 1, f"HAS_INT16: got {has_int16}, expected 1"
    # HAS_LUT=1 → bit 25
    has_lut = (val >> 25) & 1
    assert has_lut == 1, f"HAS_LUT: got {has_lut}, expected 1"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Read-only registers ignore writes
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_ro_ignore_write(dut):
    """Writing to RO registers (STATUS, VERSION, HW_CONFIG) should have no effect."""
    wb = await init_dut(dut)
    # Try to write VERSION
    await wb.write(0x014, 0xDEADBEEF)
    val = await wb.read(0x014)
    expected = 0x00_01_00_00
    assert val == expected, f"VERSION changed after write: {val:#010x}"

    # Try to write HW_CONFIG
    await wb.write(0x018, 0xFFFFFFFF)
    val = await wb.read(0x018)
    expected_array_sz = int(dut.ARRAY_SZ.value)
    assert (val & 0xFF) == expected_array_sz, "HW_CONFIG changed after write"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: RW register write/readback
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_rw_register_basic(dut):
    """Write and readback various RW registers."""
    wb = await init_dut(dut)
    test_cases = [
        (0x040, 0x0000_0013, "LAYER_MODE"),
        (0x044, 0x0038_0038, "IN_DIM_H_W (56x56)"),
        (0x048, 0x0000_0040, "IN_DIM_C (64)"),
        (0x054, 0x0101_0303, "KERNEL_SIZE (3x3, dil=1)"),
        (0x100, 0x2000_0000, "DMA_IN_ADDR"),
        (0x108, 0x0800_0000, "DMA_WEIGHT_ADDR"),
        (0x180, 0x0000_0045, "POST_CTRL"),
    ]
    for addr, data, name in test_cases:
        await wb.write(addr, data)
        val = await wb.read(addr)
        assert val == data, f"{name} @ {addr:#05x}: wrote {data:#010x}, read {val:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Self-clearing CTRL bits (START, ABORT, SOFT_RST)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_ctrl_self_clear(dut):
    """CTRL[0:2] should be self-clearing: pulse for 1 cycle then go to 0."""
    wb = await init_dut(dut)
    # Write START
    await wb.write(0x000, 0x0000_0001)
    # After ack, the START bit should have pulsed and cleared
    # ctrl_start output should be 0 by now (it was high for 1 cycle)
    await RisingEdge(dut.clk)
    assert dut.ctrl_start.value == 0, "ctrl_start not self-cleared"

    # Read back CTRL — START reads as 0
    val = await wb.read(0x000)
    assert (val & 0x7) == 0, f"CTRL self-clear bits read back non-zero: {val:#010x}"

    # AUTO_NEXT (bit 3) is NOT self-clearing — should persist
    await wb.write(0x000, 0x0000_0008)
    val = await wb.read(0x000)
    assert (val >> 3) & 1 == 1, f"AUTO_NEXT should persist: {val:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: W1C IRQ_STATUS behavior
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_irq_w1c(dut):
    """IRQ_STATUS should set on hw pulse and clear on W1C write."""
    wb = await init_dut(dut)

    # Trigger hw_done pulse
    dut.hw_done.value = 1
    await RisingEdge(dut.clk)
    dut.hw_done.value = 0
    await RisingEdge(dut.clk)

    # Read IRQ_STATUS — bit 0 should be set
    val = await wb.read(0x00C)
    assert val & 1, f"DONE_IRQ not set: {val:#010x}"

    # Write 1 to clear bit 0
    await wb.write(0x00C, 0x0000_0001)
    val = await wb.read(0x00C)
    assert (val & 1) == 0, f"DONE_IRQ not cleared by W1C: {val:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: IRQ output assertion
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_irq_output(dut):
    """irq_o should assert when enabled IRQ is pending."""
    wb = await init_dut(dut)

    # Enable DONE interrupt
    await wb.write(0x008, 0x0000_0001)

    # No IRQ yet
    await RisingEdge(dut.clk)
    assert dut.irq_o.value == 0, "IRQ asserted without pending"

    # Trigger done
    dut.hw_done.value = 1
    await RisingEdge(dut.clk)
    dut.hw_done.value = 0
    await RisingEdge(dut.clk)

    # IRQ should be asserted
    assert dut.irq_o.value == 1, "IRQ not asserted after hw_done"

    # Clear it
    await wb.write(0x00C, 0x0000_0001)
    await RisingEdge(dut.clk)
    assert dut.irq_o.value == 0, "IRQ not de-asserted after W1C clear"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Dual-mapped registers (DMA_PARAM_ADDR == POST_PARAM_ADDR)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_dual_map_param_addr(dut):
    """Writing DMA_PARAM_ADDR (0x10C) should be readable from POST_PARAM_ADDR (0x184)."""
    wb = await init_dut(dut)
    test_val = 0x0802_0000
    await wb.write(0x10C, test_val)
    val = await wb.read(0x184)
    assert val == test_val, f"Dual-map DMA→POST: got {val:#010x}, expected {test_val:#010x}"

    # And reverse: write POST, read DMA
    test_val2 = 0x0900_0000
    await wb.write(0x184, test_val2)
    val2 = await wb.read(0x10C)
    assert val2 == test_val2, f"Dual-map POST→DMA: got {val2:#010x}, expected {test_val2:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: LUT read/write
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_lut_readwrite(dut):
    """Write and readback LUT entries at various offsets."""
    wb = await init_dut(dut)
    # Write first LUT word (0x200)
    await wb.write(0x200, 0x04030201)
    val = await wb.read(0x200)
    assert val == 0x04030201, f"LUT[0]: got {val:#010x}"

    # Write last INT8 LUT word (0x2FC = entry 63)
    await wb.write(0x2FC, 0xAABBCCDD)
    val = await wb.read(0x2FC)
    assert val == 0xAABBCCDD, f"LUT[63]: got {val:#010x}"

    # Write INT16 range (0x300+)
    await wb.write(0x300, 0x11223344)
    val = await wb.read(0x300)
    assert val == 0x11223344, f"LUT[64]: got {val:#010x}"

    # Verify first write still intact
    val = await wb.read(0x200)
    assert val == 0x04030201, f"LUT[0] corrupted: got {val:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: VENDOR_ID and SERIAL registers
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_vendor_serial(dut):
    """Read vendor ID and serial number registers."""
    wb = await init_dut(dut)
    vendor = await wb.read(0x02C)
    # VENDOR="ON"=0x4F4E (lower 16), PRODUCT=0x0001 (upper 16)
    assert (vendor & 0xFFFF) == 0x4F4E, f"VENDOR_ID: {vendor:#010x}"
    assert ((vendor >> 16) & 0xFFFF) == 0x0001, f"PRODUCT_ID: {vendor:#010x}"

    serial_lo = await wb.read(0x024)
    assert serial_lo == 0x0000_0001, f"SERIAL_LO: {serial_lo:#010x}"
    serial_hi = await wb.read(0x028)
    assert serial_hi == 0x0000_0000, f"SERIAL_HI: {serial_hi:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: STATUS register reflects hardware inputs
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_status_register(dut):
    """STATUS register should reflect hw_busy, hw_dma_busy, hw_curr_layer."""
    wb = await init_dut(dut)

    # Set busy
    dut.hw_busy.value = 1
    dut.hw_curr_layer.value = 42
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    val = await wb.read(0x004)
    assert val & 1, f"BUSY not set: {val:#010x}"
    curr_layer = (val >> 8) & 0xFF
    assert curr_layer == 42, f"CURR_LAYER: got {curr_layer}, expected 42"

    # Clear busy
    dut.hw_busy.value = 0
    await RisingEdge(dut.clk)
    val = await wb.read(0x004)
    assert (val & 1) == 0, f"BUSY still set after clear: {val:#010x}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Performance counter readback
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_perf_counters(dut):
    """PERF_CNT and MAC_CNT should reflect hardware inputs."""
    wb = await init_dut(dut)
    dut.hw_perf_cnt.value = 123456
    dut.hw_mac_cnt.value = 789012
    await RisingEdge(dut.clk)

    perf = await wb.read(0x01C)
    assert perf == 123456, f"PERF_CNT: got {perf}, expected 123456"
    mac = await wb.read(0x020)
    assert mac == 789012, f"MAC_CNT: got {mac}, expected 789012"
