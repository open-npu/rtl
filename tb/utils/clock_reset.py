# Open-NPU RTL — Clock and Reset Helper
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def clock_reset(dut, clk_period_ns=10, reset_cycles=5):
    """Start clock and perform active-low reset sequence.

    Args:
        dut: cocotb DUT handle
        clk_period_ns: Clock period in nanoseconds (default 100MHz)
        reset_cycles: Number of clock cycles to hold reset

    Returns:
        None (clock continues running as a background task)
    """
    # Start clock
    cocotb.start_soon(Clock(dut.clk, clk_period_ns, unit="ns").start())

    # Active-low reset
    dut.rst_n.value = 0
    for _ in range(reset_cycles):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
