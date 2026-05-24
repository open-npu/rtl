# Open-NPU RTL — Wishbone Bus Driver (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight Wishbone B4 pipelined master for testbenches.
# Will be expanded when npu_csr and npu_dma modules are implemented.

import cocotb
from cocotb.triggers import RisingEdge


class WishboneMaster:
    """Simple Wishbone B4 single-cycle master driver.

    Signals assumed on DUT:
        wb_cyc_o, wb_stb_o, wb_we_o, wb_adr_o, wb_dat_o, wb_sel_o
        wb_ack_i, wb_dat_i
    """

    def __init__(self, dut, clk, prefix="wb"):
        self.dut = dut
        self.clk = clk
        self.prefix = prefix

    def _sig(self, name):
        return getattr(self.dut, f"{self.prefix}_{name}")

    async def write(self, addr, data, sel=0xF):
        """Single-cycle Wishbone write."""
        self._sig("cyc_o").value = 1
        self._sig("stb_o").value = 1
        self._sig("we_o").value = 1
        self._sig("adr_o").value = addr
        self._sig("dat_o").value = data
        self._sig("sel_o").value = sel
        await RisingEdge(self.clk)
        # Wait for ack
        while not self._sig("ack_i").value:
            await RisingEdge(self.clk)
        self._sig("cyc_o").value = 0
        self._sig("stb_o").value = 0
        await RisingEdge(self.clk)

    async def read(self, addr):
        """Single-cycle Wishbone read. Returns data."""
        self._sig("cyc_o").value = 1
        self._sig("stb_o").value = 1
        self._sig("we_o").value = 0
        self._sig("adr_o").value = addr
        self._sig("sel_o").value = 0xF
        await RisingEdge(self.clk)
        while not self._sig("ack_i").value:
            await RisingEdge(self.clk)
        data = int(self._sig("dat_i").value)
        self._sig("cyc_o").value = 0
        self._sig("stb_o").value = 0
        await RisingEdge(self.clk)
        return data
