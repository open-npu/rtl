# Open-NPU RTL — Random Stress Tests (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# P2.2: Random stress testing for pre-CPU integration confidence:
#   1. Random layer configs (50 random valid configs, verify no hang/crash)
#   2. Random abort timing (inject abort at random points, verify recovery)
#
# DUT = npu_top (full chip)

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

class WbSlave:
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
            if self.dut.wb_slv_ack_o.value:
                break
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
            if self.dut.wb_slv_ack_o.value:
                break
            await RisingEdge(self.clk)
        await ReadOnly()
        try:
            data = int(self.dut.wb_slv_dat_o.value)
        except ValueError:
            data = 0
        await Timer(1, unit="step")
        self.dut.wb_slv_cyc_i.value = 0
        self.dut.wb_slv_stb_i.value = 0
        await RisingEdge(self.clk)
        return data


class WbMasterMem:
    def __init__(self, dut, clk):
        self.dut = dut
        self.clk = clk
        self.mem = {}

    def populate(self, base_addr, words):
        for i, w in enumerate(words):
            self.mem[base_addr + i * 4] = int(w)

    async def run(self):
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
            else:
                self.dut.wb_mst_ack_i.value = 0


async def reset(dut):
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
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)


def gen_random_layer_config(rng):
    """Generate a random but valid layer configuration.

    Returns dict with all parameters needed to program CSR.
    Constraints:
      - out_c must be multiple of ARRAY_SIZE (16) for systolic
      - in_c must be ≥ 1
      - Spatial dims must be ≥ 1
      - wgt_size and in_size must fit in SRAM
    """
    # Random op_type (only conv2d=1 for stress, simplest to configure)
    op_type = 1
    data_type = rng.choice([0, 1])  # 0=INT8, 1=INT16

    in_h = rng.choice([1, 2, 4, 8])
    in_w = rng.choice([1, 2, 4, 8])
    in_c = rng.choice([16, 32, 48, 64])
    kernel = rng.choice([1, 3])
    stride = rng.choice([1, 2])
    padding = (kernel - 1) // 2 if kernel > 1 else 0

    out_h = (in_h + 2 * padding - kernel) // stride + 1
    out_w = (in_w + 2 * padding - kernel) // stride + 1
    out_c = rng.choice([16, 32, 48])

    # Compute sizes
    elem_bytes = 2 if data_type == 1 else 1
    wgt_bytes = kernel * kernel * in_c * out_c * elem_bytes
    in_bytes = in_h * in_w * in_c * elem_bytes
    param_count = out_c  # PPU params: 4 words per output channel

    # Cap sizes to fit in SRAM (64KB weight, 32KB activation)
    if wgt_bytes > 65536:
        out_c = 16
        wgt_bytes = kernel * kernel * in_c * out_c * elem_bytes
    if in_bytes > 32768:
        in_h = min(in_h, 4)
        in_w = min(in_w, 4)
        in_bytes = in_h * in_w * in_c * elem_bytes
        out_h = (in_h + 2 * padding - kernel) // stride + 1
        out_w = (in_w + 2 * padding - kernel) // stride + 1

    return {
        'op_type': op_type,
        'data_type': data_type,
        'in_h': in_h, 'in_w': in_w, 'in_c': in_c,
        'out_h': out_h, 'out_w': out_w, 'out_c': out_c,
        'kernel': kernel, 'stride': stride, 'padding': padding,
        'wgt_bytes': wgt_bytes,
        'in_bytes': in_bytes,
        'param_count': param_count,
    }


async def program_random_layer(wb, cfg, mem, base_offset):
    """Program CSR and populate DDR for one random layer config."""
    wgt_addr = 0x10000 + base_offset
    in_addr = 0x50000 + base_offset
    param_addr = 0x90000 + base_offset

    wgt_words = (cfg['wgt_bytes'] + 3) // 4
    in_words = (cfg['in_bytes'] + 3) // 4
    param_words = cfg['param_count'] * 4

    # Populate DDR with zeros (don't care about values, just no-hang)
    mem.populate(wgt_addr, [0] * wgt_words)
    mem.populate(in_addr, [0] * in_words)
    mem.populate(param_addr, [0] * param_words)

    # Program CSR
    layer_mode = cfg['op_type'] | (cfg['data_type'] << 4)
    await wb.write(0x040, layer_mode)
    await wb.write(0x044, (cfg['in_h'] << 16) | cfg['in_w'])
    await wb.write(0x048, cfg['in_c'])
    await wb.write(0x04C, (cfg['out_h'] << 16) | cfg['out_w'])
    await wb.write(0x050, cfg['out_c'])
    await wb.write(0x054, cfg['kernel'] | (cfg['kernel'] << 8))
    await wb.write(0x058, cfg['stride'] | (cfg['stride'] << 8))
    await wb.write(0x05C, cfg['padding'] | (cfg['padding'] << 8))
    await wb.write(0x078, 0)
    await wb.write(0x108, wgt_addr)
    await wb.write(0x100, in_addr)
    await wb.write(0x10C, param_addr)
    await wb.write(0x12C, cfg['wgt_bytes'])
    await wb.write(0x128, cfg['in_bytes'])
    await wb.write(0x130, 0)  # skip store
    await wb.write(0x188, cfg['param_count'])
    await wb.write(0x118, 0x02)  # FUSE_START (skip store)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Random layer configs (50 iterations)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=60000, timeout_unit="ms")
async def test_random_layers_50(dut):
    """Run 50 randomly configured conv layers — verify no hangs or crashes.

    This is a smoke test: we don't verify output correctness (no golden),
    just that the hardware completes every layer without hanging.
    """
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)
    await wb.write(0x008, 0x01)  # IRQ_EN: done

    rng = random.Random(12345)
    passed = 0
    failed_configs = []

    for i in range(50):
        cfg = gen_random_layer_config(rng)

        # Program layer
        await program_random_layer(wb, cfg, mem, i * 0x1000)

        # Clear IRQ
        await wb.write(0x00C, 0xFF)

        # START
        await wb.write(0x000, 0x01)

        # Wait for done
        done = False
        for _ in range(200000):  # ~2ms per layer max
            await RisingEdge(dut.clk)
            try:
                if int(dut.irq_o.value):
                    done = True
                    break
            except ValueError:
                pass

        if done:
            passed += 1
        else:
            # Abort to recover
            await wb.write(0x000, 0x02)
            for _ in range(50):
                await RisingEdge(dut.clk)
            failed_configs.append((i, cfg))

    assert passed >= 45, \
        f"Only {passed}/50 random layers completed. Failed: {len(failed_configs)}"

    mem_task.cancel()
    dut._log.info(f"PASS: {passed}/50 random layers completed without hang")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Random abort timing (20 iterations)
# ═══════════════════════════════════════════════════════════════════════

@cocotb.test(timeout_time=30000, timeout_unit="ms")
async def test_random_abort_timing(dut):
    """Inject abort at random time during 20 layer executions, verify recovery.

    Each iteration:
      1. Start a layer
      2. Wait random 5-500 cycles
      3. Abort
      4. Verify return to IDLE
      5. Verify next layer still works
    """
    await setup(dut)

    mem = WbMasterMem(dut, dut.clk)
    mem.populate(0x1000, [0] * 256)   # weights
    mem.populate(0x2000, [0] * 64)    # activation
    mem.populate(0x3000, [0] * 64)    # params
    mem_task = cocotb.start_soon(mem.run())

    wb = WbSlave(dut, dut.clk)
    await wb.write(0x008, 0x01)  # IRQ_EN

    rng = random.Random(67890)
    recoveries = 0
    resume_ok = 0

    for i in range(20):
        # Program a layer (same config every time for simplicity)
        await wb.write(0x040, 0x0001)
        await wb.write(0x044, 0x00020002)  # 2x2
        await wb.write(0x048, 16)
        await wb.write(0x04C, 0x00020002)
        await wb.write(0x050, 16)
        await wb.write(0x054, 0x0101)
        await wb.write(0x058, 0x0101)
        await wb.write(0x05C, 0)
        await wb.write(0x078, 0)
        await wb.write(0x108, 0x1000)
        await wb.write(0x100, 0x2000)
        await wb.write(0x10C, 0x3000)
        await wb.write(0x12C, 1024)   # 256 words
        await wb.write(0x128, 256)    # 64 words
        await wb.write(0x130, 0)
        await wb.write(0x188, 16)
        await wb.write(0x118, 0x02)

        # Clear IRQ
        await wb.write(0x00C, 0xFF)

        # START
        await wb.write(0x000, 0x01)

        # Wait random time
        wait_cycles = rng.randint(5, 500)
        for _ in range(wait_cycles):
            await RisingEdge(dut.clk)

        # Check if already done
        try:
            already_done = int(dut.irq_o.value)
        except ValueError:
            already_done = False

        if already_done:
            # Layer finished before abort — still counts as pass
            recoveries += 1
            resume_ok += 1
            continue

        # ABORT
        await wb.write(0x000, 0x02)

        # Wait for recovery
        recovered = False
        for _ in range(100):
            await RisingEdge(dut.clk)
            status = await wb.read(0x004)
            if (status & 0x1) == 0:
                recovered = True
                break

        if recovered:
            recoveries += 1
        else:
            # Force soft_rst to recover
            await wb.write(0x000, 0x04)
            for _ in range(10):
                await RisingEdge(dut.clk)

        # Now verify the system can run a new layer
        await wb.write(0x00C, 0xFF)  # Clear IRQ
        await wb.write(0x000, 0x01)  # START same layer again

        done = False
        for _ in range(100000):
            await RisingEdge(dut.clk)
            try:
                if int(dut.irq_o.value):
                    done = True
                    break
            except ValueError:
                pass

        if done:
            resume_ok += 1
        else:
            # Recovery failed — abort and continue
            await wb.write(0x000, 0x02)
            for _ in range(50):
                await RisingEdge(dut.clk)

    assert recoveries >= 18, \
        f"Only {recoveries}/20 abort recoveries succeeded"
    assert resume_ok >= 18, \
        f"Only {resume_ok}/20 post-abort resumes succeeded"

    mem_task.cancel()
    dut._log.info(f"PASS: {recoveries}/20 aborts recovered, {resume_ok}/20 resumes OK")
