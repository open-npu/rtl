"""Unit tests for the psum-chained weight-stationary systolic array.

PE[r][c] holds W[r][c]. One ROWS-wide activation vector is presented per cycle;
the array emits, ROWS cycles later, the COLS dot products

    out[c] = sum_r act[r] * W[r][c]

one vector per cycle.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

# The build picks the array dimension via -DARRAY_SIZE; the Makefile default
# is 8. Pinning 16 here silently overflows the flattened operand buses.
ROWS = int(os.environ.get("ARRAY_SIZE", "8"))
COLS = ROWS
DATA_W = 16
ACC_W = 44

MODE_IDLE = 0
MODE_WGT_LOAD = 1
MODE_COMPUTE = 2


def pack(values, width):
    v = 0
    for i, x in enumerate(values):
        v |= (int(x) & ((1 << width) - 1)) << (width * i)
    return v


def unpack_signed(flat, width, count):
    out = []
    mask = (1 << width) - 1
    for i in range(count):
        raw = (int(flat) >> (width * i)) & mask
        if raw >> (width - 1):
            raw -= 1 << width
        out.append(raw)
    return out


async def reset(dut):
    dut.rst_n.value = 0
    dut.cmd.value = MODE_IDLE
    dut.cmd_valid.value = 0
    dut.wgt_valid.value = 0
    dut.act_valid.value = 0
    dut.wgt_data_flat.value = 0
    dut.act_data_flat.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, W):
    """W[r][c]. Column c is presented on cycle c over the row-broadcast bus."""
    dut.cmd.value = MODE_WGT_LOAD
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0
    for c in range(COLS):
        dut.wgt_data_flat.value = pack([W[r][c] for r in range(ROWS)], DATA_W)
        dut.wgt_valid.value = 1
        await RisingEdge(dut.clk)
    dut.wgt_valid.value = 0
    # wgt_load_done is registered, then the FSM moves to S_READY.
    for _ in range(3):
        await RisingEdge(dut.clk)


async def stream(dut, vectors):
    """Feed one activation vector per cycle; collect emitted result vectors."""
    dut.cmd.value = MODE_COMPUTE
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0

    results = []
    # ROWS cycles of pipeline fill plus a couple of cycles of margin.
    for i in range(len(vectors) + ROWS + 4):
        if i < len(vectors):
            dut.act_data_flat.value = pack(vectors[i], DATA_W)
            dut.act_valid.value = 1
        else:
            dut.act_valid.value = 0
        await RisingEdge(dut.clk)
        if dut.psum_out_valid.value == 1:
            results.append(unpack_signed(dut.psum_out_flat.value, ACC_W, COLS))
    return results


def reference(W, act):
    return [sum(act[r] * W[r][c] for r in range(ROWS)) for c in range(COLS)]


@cocotb.test()
async def test_single_vector(dut):
    """One activation vector produces one correct result vector."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    W = [[(r + 1) if c == r else 0 for c in range(COLS)] for r in range(ROWS)]
    act = [r + 1 for r in range(ROWS)]

    await load_weights(dut, W)
    got = await stream(dut, [act])

    assert len(got) == 1, f"expected 1 result vector, got {len(got)}"
    assert got[0] == reference(W, act), f"{got[0]} != {reference(W, act)}"


@cocotb.test()
async def test_pipelined_stream(dut):
    """Back-to-back vectors come out one per cycle, in order."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(1)
    W = [[random.randint(-128, 127) for _ in range(COLS)] for _ in range(ROWS)]
    vectors = [[random.randint(-128, 127) for _ in range(ROWS)]
               for _ in range(8)]

    await load_weights(dut, W)
    got = await stream(dut, vectors)

    assert len(got) == len(vectors), \
        f"expected {len(vectors)} result vectors, got {len(got)}"
    for i, (g, v) in enumerate(zip(got, vectors)):
        assert g == reference(W, v), f"vector {i}: {g} != {reference(W, v)}"


@cocotb.test()
async def test_zero_padded_rows(dut):
    """Rows beyond the ragged last k_pass are zeroed via their weights."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(2)
    k_valid = 5
    W = [[random.randint(-128, 127) if r < k_valid else 0
          for _ in range(COLS)] for r in range(ROWS)]
    act = [random.randint(-128, 127) for _ in range(ROWS)]

    await load_weights(dut, W)
    got = await stream(dut, [act])

    expect = [sum(act[r] * W[r][c] for r in range(k_valid))
              for c in range(COLS)]
    assert got[0] == expect, f"{got[0]} != {expect}"


@cocotb.test()
async def test_weight_reload_between_passes(dut):
    """A second weight load must not disturb results already in flight."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(3)
    W1 = [[random.randint(-50, 50) for _ in range(COLS)] for _ in range(ROWS)]
    W2 = [[random.randint(-50, 50) for _ in range(COLS)] for _ in range(ROWS)]
    act = [random.randint(-50, 50) for _ in range(ROWS)]

    await load_weights(dut, W1)
    got1 = await stream(dut, [act])
    assert got1[0] == reference(W1, act)

    await load_weights(dut, W2)
    got2 = await stream(dut, [act])
    assert got2[0] == reference(W2, act)


@cocotb.test()
async def test_negative_and_wide(dut):
    """Full INT16 operands exercise sign handling and accumulator width."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    W = [[-32768 if (r + c) % 2 else 32767 for c in range(COLS)]
         for r in range(ROWS)]
    act = [-32768 if r % 2 else 32767 for r in range(ROWS)]

    await load_weights(dut, W)
    got = await stream(dut, [act])

    expect = reference(W, act)
    # Results must fit the 44-bit accumulator for this comparison to be valid.
    for v in expect:
        assert -(1 << (ACC_W - 1)) <= v < (1 << (ACC_W - 1)), \
            f"test vector overflows ACC_W: {v}"
    assert got[0] == expect, f"{got[0]} != {expect}"
