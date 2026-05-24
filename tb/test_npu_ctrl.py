# Open-NPU RTL — cocotb Tests for npu_ctrl (NPU Controller)
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer


async def init_dut(dut):
    """Initialize DUT: clock, reset, zero all inputs."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    dut.ctrl_start.value = 0
    dut.ctrl_abort.value = 0
    dut.ctrl_soft_rst.value = 0
    dut.dma_busy.value = 0
    dut.dma_done.value = 0
    dut.compute_done.value = 0
    dut.ppu_done.value = 0
    dut.cfg_dma_in_addr.value = 0x1000_0000
    dut.cfg_dma_out_addr.value = 0x2000_0000
    dut.cfg_dma_wgt_addr.value = 0x3000_0000
    dut.cfg_dma_param_addr.value = 0x4000_0000
    dut.cfg_dma_in_size.value = 256      # 256 bytes = 64 words
    dut.cfg_dma_wgt_size.value = 512     # 512 bytes = 128 words
    dut.cfg_param_count.value = 16       # 16 channels × 14 = 224 bytes = 56 words
    dut.cfg_layer_mode.value = 0

    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def pulse(sig):
    """Assert signal for 1 cycle."""
    sig.value = 1
    await RisingEdge(sig._path.split('.')[0] if hasattr(sig, '_path') else sig)


async def wait_for_dma_start(dut, timeout=50):
    """Wait until dma_start is asserted."""
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.dma_start.value):
            await Timer(1, unit="step")
            return True
        await Timer(1, unit="step")
    return False


async def complete_dma(dut):
    """Simulate DMA completion: pulse dma_done."""
    dut.dma_done.value = 1
    await RisingEdge(dut.clk)
    dut.dma_done.value = 0


async def complete_compute(dut):
    """Simulate compute completion."""
    dut.compute_done.value = 1
    await RisingEdge(dut.clk)
    dut.compute_done.value = 0


async def complete_ppu(dut):
    """Simulate PPU completion."""
    dut.ppu_done.value = 1
    await RisingEdge(dut.clk)
    dut.ppu_done.value = 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Idle state after reset
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_idle_after_reset(dut):
    """After reset, controller should be idle with no busy/done."""
    await init_dut(dut)
    await ReadOnly()
    assert int(dut.hw_busy.value) == 0, "Busy after reset"
    assert int(dut.hw_done.value) == 0, "Done after reset"
    assert int(dut.hw_error.value) == 0, "Error after reset"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: START → busy asserted
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_start_busy(dut):
    """After START pulse, hw_busy should be asserted."""
    await init_dut(dut)

    dut.ctrl_start.value = 1
    await RisingEdge(dut.clk)
    dut.ctrl_start.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.hw_busy.value) == 1, "Not busy after START"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Full layer sequence (START → DONE)
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_full_layer_sequence(dut):
    """Complete layer: DMA loads → compute → PPU → DMA store → DONE."""
    await init_dut(dut)

    # START
    dut.ctrl_start.value = 1
    await RisingEdge(dut.clk)
    dut.ctrl_start.value = 0

    # Phase 1: Wait for weight DMA start
    found = await wait_for_dma_start(dut)
    assert found, "Weight DMA start not issued"
    await ReadOnly()
    assert int(dut.dma_dir.value) == 0, "Weight DMA should be load (dir=0)"
    await Timer(1, unit="step")
    await complete_dma(dut)

    # Phase 2: Wait for activation DMA start
    found = await wait_for_dma_start(dut)
    assert found, "Activation DMA start not issued"
    await complete_dma(dut)

    # Phase 3: Wait for param DMA start
    found = await wait_for_dma_start(dut)
    assert found, "Param DMA start not issued"
    await complete_dma(dut)

    # Phase 4: Wait for compute start
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.compute_start.value):
            break
        await Timer(1, unit="step")
    await Timer(1, unit="step")
    await complete_compute(dut)

    # Phase 5: Wait for PPU start
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.ppu_start.value):
            break
        await Timer(1, unit="step")
    await Timer(1, unit="step")
    await complete_ppu(dut)

    # Phase 6: Wait for output DMA start
    found = await wait_for_dma_start(dut)
    assert found, "Output DMA start not issued"
    await ReadOnly()
    assert int(dut.dma_dir.value) == 1, "Output DMA should be store (dir=1)"
    await Timer(1, unit="step")
    await complete_dma(dut)

    # Should get hw_done pulse
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.hw_done.value):
            break
        await Timer(1, unit="step")

    assert int(dut.hw_done.value) == 1, "hw_done not pulsed after full sequence"
    await Timer(1, unit="step")

    # Next cycle: should be idle
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.hw_busy.value) == 0, "Still busy after done"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Abort during DMA load
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_abort_during_dma(dut):
    """Abort during weight DMA should go to DONE."""
    await init_dut(dut)

    dut.ctrl_start.value = 1
    await RisingEdge(dut.clk)
    dut.ctrl_start.value = 0

    # Wait for weight DMA
    await wait_for_dma_start(dut)

    # Abort
    dut.ctrl_abort.value = 1
    await RisingEdge(dut.clk)
    dut.ctrl_abort.value = 0

    # Should get done (not error)
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.hw_done.value):
            break
        await Timer(1, unit="step")

    assert int(dut.hw_done.value) == 1, "hw_done not pulsed after abort"
    await Timer(1, unit="step")
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.hw_busy.value) == 0, "Still busy after abort"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Soft reset
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_soft_reset(dut):
    """Soft reset should immediately return to idle."""
    await init_dut(dut)

    # Start a layer
    dut.ctrl_start.value = 1
    await RisingEdge(dut.clk)
    dut.ctrl_start.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.hw_busy.value) == 1
    await Timer(1, unit="step")

    # Soft reset
    dut.ctrl_soft_rst.value = 1
    await RisingEdge(dut.clk)
    dut.ctrl_soft_rst.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.hw_busy.value) == 0, "Busy after soft reset"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Layer counter increments
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_layer_counter(dut):
    """hw_curr_layer should increment after each completed layer."""
    await init_dut(dut)

    for layer_num in range(3):
        dut.ctrl_start.value = 1
        await RisingEdge(dut.clk)
        dut.ctrl_start.value = 0

        # Complete full sequence quickly
        await wait_for_dma_start(dut)  # weight
        await complete_dma(dut)
        await wait_for_dma_start(dut)  # act
        await complete_dma(dut)
        await wait_for_dma_start(dut)  # param
        await complete_dma(dut)

        # compute
        for _ in range(5):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.compute_start.value):
                break
            await Timer(1, unit="step")
        await Timer(1, unit="step")
        await complete_compute(dut)

        # ppu
        for _ in range(5):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.ppu_start.value):
                break
            await Timer(1, unit="step")
        await Timer(1, unit="step")
        await complete_ppu(dut)

        # store
        await wait_for_dma_start(dut)
        await complete_dma(dut)

        # Wait for done
        for _ in range(5):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.hw_done.value):
                break
            await Timer(1, unit="step")
        await Timer(1, unit="step")
        await RisingEdge(dut.clk)

    # Check layer counter
    await ReadOnly()
    layer = int(dut.hw_curr_layer.value)
    assert layer == 3, f"hw_curr_layer: got {layer}, expected 3"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: DMA address correctness
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_dma_addresses(dut):
    """Verify DMA external addresses match CSR config for each phase."""
    await init_dut(dut)

    dut.ctrl_start.value = 1
    await RisingEdge(dut.clk)
    dut.ctrl_start.value = 0

    # Weight DMA
    await wait_for_dma_start(dut)
    await ReadOnly()
    assert int(dut.dma_ext_addr.value) == 0x3000_0000, "Weight addr mismatch"
    await Timer(1, unit="step")
    await complete_dma(dut)

    # Activation DMA
    await wait_for_dma_start(dut)
    await ReadOnly()
    assert int(dut.dma_ext_addr.value) == 0x1000_0000, "Input addr mismatch"
    await Timer(1, unit="step")
    await complete_dma(dut)

    # Param DMA
    await wait_for_dma_start(dut)
    await ReadOnly()
    assert int(dut.dma_ext_addr.value) == 0x4000_0000, "Param addr mismatch"
    await Timer(1, unit="step")
    await complete_dma(dut)

    # Skip compute+ppu
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.compute_start.value):
            break
        await Timer(1, unit="step")
    await Timer(1, unit="step")
    await complete_compute(dut)
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.ppu_start.value):
            break
        await Timer(1, unit="step")
    await Timer(1, unit="step")
    await complete_ppu(dut)

    # Output DMA
    await wait_for_dma_start(dut)
    await ReadOnly()
    assert int(dut.dma_ext_addr.value) == 0x2000_0000, "Output addr mismatch"
    assert int(dut.dma_dir.value) == 1, "Output should be store"
    await Timer(1, unit="step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Zero-size weight skip
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_zero_weight_skip(dut):
    """If weight size is 0, skip weight DMA and go straight to activation."""
    await init_dut(dut)
    dut.cfg_dma_wgt_size.value = 0  # No weights

    dut.ctrl_start.value = 1
    await RisingEdge(dut.clk)
    dut.ctrl_start.value = 0

    # Should skip weight DMA and go to activation DMA
    await wait_for_dma_start(dut)
    await ReadOnly()
    # The address should be the input address, not weight
    addr = int(dut.dma_ext_addr.value)
    assert addr == 0x1000_0000, f"Expected input addr, got {addr:#010x}"
    await Timer(1, unit="step")
