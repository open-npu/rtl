# Open-NPU RTL — Wishbone Bus Driver (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight Wishbone B4 single-cycle master driver for testbenches.
# Drives signals named from the SLAVE perspective (wb_cyc_i, wb_ack_o, etc.)

import cocotb
from cocotb.triggers import RisingEdge


class WishboneMaster:
    """Simple Wishbone B4 single-cycle master driver.

    Drives slave-side signal names:
        wb_cyc_i, wb_stb_i, wb_we_i, wb_adr_i, wb_dat_i, wb_sel_i  (master drives)
        wb_ack_o, wb_dat_o  (master reads)
    """

    def __init__(self, dut, clk, prefix="wb"):
        self.dut = dut
        self.clk = clk
        self.prefix = prefix

    def _sig(self, name):
        return getattr(self.dut, f"{self.prefix}_{name}")

    async def write(self, addr, data, sel=0xF):
        """Single-cycle Wishbone write."""
        self._sig("cyc_i").value = 1
        self._sig("stb_i").value = 1
        self._sig("we_i").value = 1
        self._sig("adr_i").value = addr
        self._sig("dat_i").value = data
        self._sig("sel_i").value = sel
        await RisingEdge(self.clk)
        # Wait for ack
        while not self._sig("ack_o").value:
            await RisingEdge(self.clk)
        self._sig("cyc_i").value = 0
        self._sig("stb_i").value = 0
        await RisingEdge(self.clk)

    async def read(self, addr):
        """Single-cycle Wishbone read. Returns data."""
        self._sig("cyc_i").value = 1
        self._sig("stb_i").value = 1
        self._sig("we_i").value = 0
        self._sig("adr_i").value = addr
        self._sig("sel_i").value = 0xF
        await RisingEdge(self.clk)
        while not self._sig("ack_o").value:
            await RisingEdge(self.clk)
        data = int(self._sig("dat_o").value)
        self._sig("cyc_i").value = 0
        self._sig("stb_i").value = 0
        await RisingEdge(self.clk)
        return data
