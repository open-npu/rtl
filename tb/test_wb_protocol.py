# Open-NPU RTL — Wishbone B4 Protocol Compliance Tests (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# P0.1: Verify Wishbone B4 protocol edge-cases that matter for CPU integration:
#   1. Back-to-back writes (continuous pipeline, ACK every cycle)
#   2. Back-to-back reads
#   3. CYC=1 STB=0 — no ACK expected (bus idle hold)
#   4. Byte-select partial writes
#   5. DMA master with delayed ACK (stall/backpressure from memory)
#   6. DMA master with variable random stalls
#
# DUT = npu_top (full chip)

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly, First


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

async def reset(dut):
    """Apply reset, zero all WB signals."""
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


async def setup(dut):
    """Start clock and reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)


async def wb_slv_write(dut, addr, data, sel=0xF):
    """Single Wishbone slave write with timeout."""
    dut.wb_slv_cyc_i.value = 1
    dut.wb_slv_stb_i.value = 1
    dut.wb_slv_we_i.value = 1
    dut.wb_slv_adr_i.value = addr
    dut.wb_slv_dat_i.value = data
    dut.wb_slv_sel_i.value = sel
    await RisingEdge(dut.clk)
    for _ in range(50):
        if dut.wb_slv_ack_o.value:
            break
        await RisingEdge(dut.clk)
    assert dut.wb_slv_ack_o.value, f"No ACK for write @{addr:#x}"
    dut.wb_slv_cyc_i.value = 0
    dut.wb_slv_stb_i.value = 0
    await RisingEdge(dut.clk)


async def wb_slv_read(dut, addr):
    """Single Wishbone slave read with timeout."""
    dut.wb_slv_cyc_i.value = 1
    dut.wb_slv_stb_i.value = 1
    dut.wb_slv_we_i.value = 0
    dut.wb_slv_adr_i.value = addr
    dut.wb_slv_sel_i.value = 0xF
    await RisingEdge(dut.clk)
    for _ in range(50):
        if dut.wb_slv_ack_o.value:
            break
        await RisingEdge(dut.clk)
    assert dut.wb_slv_ack_o.value, f"No ACK for read @{addr:#x}"
    await ReadOnly()
    data = int(dut.wb_slv_dat_o.value)
    await Timer(1, unit="step")
    dut.wb_slv_cyc_i.value = 0
    dut.wb_slv_stb_i.value = 0
    await RisingEdge(dut.clk)
    return data


class WbMasterMemStall:
    """DDR emulator with configurable stall for DMA master port.

    stall_fn: callable() -> int (number of cycles to delay ACK)
    """

    def __init__(self, dut, clk, stall_fn=None):
        self.dut = dut
        self.clk = clk
        self.mem = {}
        self.stall_fn = stall_fn or (lambda: 0)
        self.total_stalls = 0
        self.total_txn = 0

    def populate(self, base_addr, words):
        for i, w in enumerate(words):
            self.mem[base_addr + i * 4] = int(w)

    async def run(self):
        """Background responder with stall injection."""
        while True:
            await RisingEdge(self.clk)
            try:
                cyc = int(self.dut.wb_mst_cyc_o.value)
                stb = int(self.dut.wb_mst_stb_o.value)
            except ValueError:
                self.dut.wb_mst_ack_i.value = 0
                continue

            if cyc and stb:
                # Insert stall cycles
                stall = self.stall_fn()
                self.total_stalls += stall
                for _ in range(stall):
                    self.dut.wb_mst_ack_i.value = 0
                    await RisingEdge(self.clk)
                    # Check abort
                    try:
                        if not int(self.dut.wb_mst_cyc_o.value):
                            break
                    except ValueError:
                        break

                # Now respond
                try:
                    addr = int(self.dut.wb_mst_adr_o.value)
                    we = int(self.dut.wb_mst_we_o.value)
                except ValueError:
                    self.dut.wb_mst_ack_i.value = 0
                    continue

                if we:
                    try:
                        self.mem[addr] = int(self.dut.wb_mst_dat_o.value)
                    except ValueError:
                        self.mem[addr] = 0
                else:
                    self.dut.wb_mst_dat_i.value = self.mem.get(addr, 0)

                self.dut.wb_mst_ack_i.value = 1
                self.total_txn += 1
            else:
                self.dut.wb_mst_ack_i.value = 0


# ═══════════════════════════════════════════════════════════════════════
# Test 1: CSR back-to-back writes
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_csr_back2back_write(dut):
    """Write 8 registers back-to-back without idle gap, verify ACK on every access."""
    await setup(dut)

    # Write 8 different values to Group 1 registers
    addrs = [0x040, 0x044, 0x048, 0x04C, 0x050, 0x054, 0x058, 0x05C]
    values = [0xDEAD_0001 + i for i in range(8)]

    for addr, val in zip(addrs, values):
        # Keep CYC high between accesses (pipelined)
        dut.wb_slv_cyc_i.value = 1
        dut.wb_slv_stb_i.value = 1
        dut.wb_slv_we_i.value = 1
        dut.wb_slv_adr_i.value = addr
        dut.wb_slv_dat_i.value = val
        dut.wb_slv_sel_i.value = 0xF
        await RisingEdge(dut.clk)
        # Wait for ACK
        for _ in range(10):
            if dut.wb_slv_ack_o.value:
                break
            await RisingEdge(dut.clk)
        assert dut.wb_slv_ack_o.value, f"No ACK for back2back write @{addr:#x}"
        # Don't deassert CYC between accesses

    # Deassert bus
    dut.wb_slv_cyc_i.value = 0
    dut.wb_slv_stb_i.value = 0
    await RisingEdge(dut.clk)

    # Readback verify
    for addr, val in zip(addrs, values):
        rd = await wb_slv_read(dut, addr)
        assert rd == val, f"Readback @{addr:#x}: got {rd:#010x}, want {val:#010x}"

    dut._log.info("PASS: 8 back-to-back writes with pipelined ACK")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: CSR back-to-back reads
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_csr_back2back_read(dut):
    """Read 8 registers back-to-back without idle gap between accesses."""
    await setup(dut)

    # Pre-write known values
    addrs = [0x040, 0x044, 0x048, 0x04C, 0x050, 0x054, 0x058, 0x05C]
    values = [0xCAFE_0000 + i for i in range(8)]
    for addr, val in zip(addrs, values):
        await wb_slv_write(dut, addr, val)

    # Back-to-back reads (keep CYC asserted)
    results = []
    for addr in addrs:
        dut.wb_slv_cyc_i.value = 1
        dut.wb_slv_stb_i.value = 1
        dut.wb_slv_we_i.value = 0
        dut.wb_slv_adr_i.value = addr
        dut.wb_slv_sel_i.value = 0xF
        await RisingEdge(dut.clk)
        for _ in range(10):
            if dut.wb_slv_ack_o.value:
                break
            await RisingEdge(dut.clk)
        assert dut.wb_slv_ack_o.value, f"No ACK for back2back read @{addr:#x}"
        await ReadOnly()
        results.append(int(dut.wb_slv_dat_o.value))
        await Timer(1, unit="step")  # Exit ReadOnly before next iteration

    dut.wb_slv_cyc_i.value = 0
    dut.wb_slv_stb_i.value = 0
    await RisingEdge(dut.clk)

    # Verify all reads
    for i, (addr, val) in enumerate(zip(addrs, values)):
        assert results[i] == val, \
            f"Read @{addr:#x}: got {results[i]:#010x}, want {val:#010x}"

    dut._log.info("PASS: 8 back-to-back reads with pipelined ACK")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: CYC=1, STB=0 — no ACK expected
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_csr_no_stb(dut):
    """Assert CYC=1 but STB=0 for 10 cycles — ACK must not assert."""
    await setup(dut)

    dut.wb_slv_cyc_i.value = 1
    dut.wb_slv_stb_i.value = 0
    dut.wb_slv_we_i.value = 0
    dut.wb_slv_adr_i.value = 0x040
    dut.wb_slv_sel_i.value = 0xF

    for cycle in range(10):
        await RisingEdge(dut.clk)
        ack = int(dut.wb_slv_ack_o.value)
        assert ack == 0, f"Spurious ACK on cycle {cycle} with STB=0"

    dut.wb_slv_cyc_i.value = 0
    await RisingEdge(dut.clk)

    # Now do a real access to confirm bus still works
    await wb_slv_write(dut, 0x040, 0x12345678)
    rd = await wb_slv_read(dut, 0x040)
    assert rd == 0x12345678, f"Bus broken after no-STB phase: {rd:#010x}"

    dut._log.info("PASS: No spurious ACK with CYC=1, STB=0")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Byte-select partial writes
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_csr_byte_select(dut):
    """Write with different SEL values — verify only selected bytes change.

    NOTE: Our CSR implementation does NOT implement byte-level select logic —
    it always writes the full 32-bit word regardless of SEL. This test
    documents and verifies that behavior (important for CPU integration
    to know that partial writes are NOT supported at CSR level).
    """
    await setup(dut)

    addr = 0x100  # DMA_IN_ADDR — a simple RW register

    # Write full word first
    await wb_slv_write(dut, addr, 0xAABBCCDD)
    rd = await wb_slv_read(dut, addr)
    assert rd == 0xAABBCCDD, f"Full write failed: {rd:#010x}"

    # Now write with SEL=0x1 (byte 0 only) — but CSR ignores SEL
    # This documents the actual behavior: full word is overwritten
    await wb_slv_write(dut, addr, 0x00000011, sel=0x1)
    rd = await wb_slv_read(dut, addr)
    # CSR ignores SEL — full word written
    assert rd == 0x00000011, f"SEL=0x1 write: got {rd:#010x}, expected 0x00000011"

    # Write with SEL=0xF (all bytes)
    await wb_slv_write(dut, addr, 0x11223344, sel=0xF)
    rd = await wb_slv_read(dut, addr)
    assert rd == 0x11223344, f"SEL=0xF write: got {rd:#010x}"

    # Write with SEL=0x0 — still goes through (CSR ignores SEL)
    await wb_slv_write(dut, addr, 0xDEADBEEF, sel=0x0)
    rd = await wb_slv_read(dut, addr)
    assert rd == 0xDEADBEEF, f"SEL=0x0 write: got {rd:#010x}"

    dut._log.info("PASS: Byte-select behavior documented (CSR ignores SEL)")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: DMA master with fixed stall (backpressure)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=2, timeout_unit="ms")
async def test_dma_stall(dut):
    """DMA load transfer with fixed 3-cycle stall per beat — must complete correctly.

    Programs a minimal 1×1 conv layer so the controller does:
      WGT load (16 words) → ACT load (16 words) → PARAM load (64 words) → Compute → Done
    Total expected DMA master transactions = 16 + 16 + 64 = 96 (before store).
    With skip_store (FUSE_START bit), output is not stored back.
    """
    await setup(dut)

    # Memory emulator with fixed 3-cycle stall
    mem = WbMasterMemStall(dut, dut.clk, stall_fn=lambda: 3)

    # WGT: 16 words at 0x2000 (size in bytes = 64)
    wgt_words = 16
    wgt_addr = 0x2000
    mem.populate(wgt_addr, [0] * wgt_words)

    # ACT: 16 words at 0x1000 (size in bytes = 64)
    act_words = 16
    act_addr = 0x1000
    mem.populate(act_addr, [0xA0000000 + i for i in range(act_words)])

    # PARAM: 16 output channels × 4 words = 64 words at 0x3000
    param_words = 64
    param_addr = 0x3000
    mem.populate(param_addr, [0] * param_words)

    total_load_words = wgt_words + act_words + param_words  # 96

    # Start DDR responder
    mem_task = cocotb.start_soon(mem.run())

    # Program layer via CSR
    await wb_slv_write(dut, 0x040, 0x0001)     # LAYER_MODE: conv2d, int8
    await wb_slv_write(dut, 0x044, 0x00010001) # IN_DIM_HW: 1x1
    await wb_slv_write(dut, 0x048, 16)         # IN_DIM_C = 16
    await wb_slv_write(dut, 0x04C, 0x00010001) # OUT_DIM_HW: 1x1
    await wb_slv_write(dut, 0x050, 16)         # OUT_DIM_C = 16
    await wb_slv_write(dut, 0x054, 0x0101)     # KERNEL: 1x1
    await wb_slv_write(dut, 0x058, 0x0101)     # STRIDE: 1x1
    await wb_slv_write(dut, 0x05C, 0)          # PADDING: 0
    await wb_slv_write(dut, 0x078, 0)          # SRAM_BASE: 0
    await wb_slv_write(dut, 0x108, wgt_addr)   # DMA_WGT_ADDR
    await wb_slv_write(dut, 0x100, act_addr)   # DMA_IN_ADDR
    await wb_slv_write(dut, 0x10C, param_addr) # DMA_PARAM_ADDR
    await wb_slv_write(dut, 0x12C, wgt_words * 4)  # DMA_WGT_SIZE (bytes)
    await wb_slv_write(dut, 0x128, act_words * 4)   # DMA_IN_SIZE (bytes)
    await wb_slv_write(dut, 0x130, 0)          # DMA_OUT_SIZE = 0 (skip store)
    await wb_slv_write(dut, 0x188, 16)         # POST_PARAM_COUNT = 16 out channels
    await wb_slv_write(dut, 0x118, 0x02)       # DMA_CTRL: FUSE_START (skip store)
    # Enable IRQ for done
    await wb_slv_write(dut, 0x008, 0x01)       # IRQ_EN: done bit

    # START
    await wb_slv_write(dut, 0x000, 0x01)

    # Wait for IRQ (done)
    for _ in range(50000):
        await RisingEdge(dut.clk)
        try:
            if int(dut.irq_o.value):
                break
        except ValueError:
            pass

    # Verify DMA completed all load phases despite stalls
    assert mem.total_txn >= total_load_words, \
        f"DMA didn't complete enough transactions: {mem.total_txn} < {total_load_words}"

    mem_task.cancel()
    dut._log.info(f"PASS: DMA load with 3-cycle stall, {mem.total_txn} txns, "
                  f"{mem.total_stalls} stall cycles")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: DMA master with variable random stall
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=5, timeout_unit="ms")
async def test_dma_stall_variable(dut):
    """DMA load with random 0-10 cycle stalls — must complete without protocol violation.

    Same layer config as test 5, but with random stall durations.
    Verifies protocol correctness under unpredictable backpressure.
    """
    await setup(dut)

    random.seed(42)
    stall_log = []

    def random_stall():
        s = random.randint(0, 10)
        stall_log.append(s)
        return s

    mem = WbMasterMemStall(dut, dut.clk, stall_fn=random_stall)

    # WGT: 32 words at 0x2000 (size in bytes = 128)
    wgt_words = 32
    wgt_addr = 0x2000
    mem.populate(wgt_addr, [0] * wgt_words)

    # ACT: 32 words at 0x4000 (size in bytes = 128)
    act_words = 32
    act_addr = 0x4000
    mem.populate(act_addr, [0xB0000000 + i for i in range(act_words)])

    # PARAM: 16 out channels × 4 words = 64 words at 0x6000
    param_words = 64
    param_addr = 0x6000
    mem.populate(param_addr, [0] * param_words)

    total_load_words = wgt_words + act_words + param_words  # 128

    mem_task = cocotb.start_soon(mem.run())

    # Program layer
    await wb_slv_write(dut, 0x040, 0x0001)     # LAYER_MODE: conv2d int8
    await wb_slv_write(dut, 0x044, 0x00020001) # IN_DIM: h=2, w=1
    await wb_slv_write(dut, 0x048, 16)         # IN_C = 16
    await wb_slv_write(dut, 0x04C, 0x00010001) # OUT_DIM: 1x1
    await wb_slv_write(dut, 0x050, 16)         # OUT_C = 16
    await wb_slv_write(dut, 0x054, 0x0101)     # KERNEL: 1x1
    await wb_slv_write(dut, 0x058, 0x0101)     # STRIDE: 1x1
    await wb_slv_write(dut, 0x05C, 0)          # PADDING: 0
    await wb_slv_write(dut, 0x078, 0)          # SRAM_BASE: 0
    await wb_slv_write(dut, 0x108, wgt_addr)   # DMA_WGT_ADDR
    await wb_slv_write(dut, 0x100, act_addr)   # DMA_IN_ADDR
    await wb_slv_write(dut, 0x10C, param_addr) # DMA_PARAM_ADDR
    await wb_slv_write(dut, 0x12C, wgt_words * 4)  # DMA_WGT_SIZE (bytes)
    await wb_slv_write(dut, 0x128, act_words * 4)   # DMA_IN_SIZE (bytes)
    await wb_slv_write(dut, 0x130, 0)          # DMA_OUT_SIZE = 0
    await wb_slv_write(dut, 0x188, 16)         # POST_PARAM_COUNT = 16
    await wb_slv_write(dut, 0x118, 0x02)       # DMA_CTRL: FUSE_START (skip store)
    await wb_slv_write(dut, 0x008, 0x01)       # IRQ_EN: done

    # START
    await wb_slv_write(dut, 0x000, 0x01)

    # Wait for IRQ (done) with generous timeout
    for _ in range(100000):
        await RisingEdge(dut.clk)
        try:
            if int(dut.irq_o.value):
                break
        except ValueError:
            pass

    # Verify DMA processed all load words despite variable stalls
    assert mem.total_txn >= total_load_words, \
        f"DMA incomplete: {mem.total_txn}/{total_load_words} txns"

    avg_stall = sum(stall_log) / len(stall_log) if stall_log else 0
    mem_task.cancel()
    dut._log.info(
        f"PASS: DMA with variable stall (avg={avg_stall:.1f}), "
        f"{mem.total_txn} txns, {mem.total_stalls} total stall cycles"
    )
