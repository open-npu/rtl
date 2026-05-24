# Open-NPU RTL — SRAM Behavioral Model (cocotb)
# SPDX-License-Identifier: Apache-2.0
#
# Simple single-port SRAM model for testbench use.
# Will be expanded when npu_sram_act/wgt modules are implemented.


class SramModel:
    """Software SRAM model for verification.

    Provides read/write methods that mirror what the RTL SRAM modules do,
    allowing testbenches to pre-load data and verify DMA transfers.
    """

    def __init__(self, size_bytes=32768):
        self.size = size_bytes
        self.mem = bytearray(size_bytes)

    def write_byte(self, addr, data):
        assert 0 <= addr < self.size, f"SRAM write OOB: addr={addr:#x} size={self.size:#x}"
        self.mem[addr] = data & 0xFF

    def read_byte(self, addr):
        assert 0 <= addr < self.size, f"SRAM read OOB: addr={addr:#x} size={self.size:#x}"
        return self.mem[addr]

    def write_block(self, base_addr, data):
        """Write a block of bytes starting at base_addr."""
        for i, b in enumerate(data):
            self.write_byte(base_addr + i, b)

    def read_block(self, base_addr, length):
        """Read a block of bytes."""
        return bytes(self.read_byte(base_addr + i) for i in range(length))

    def clear(self):
        self.mem = bytearray(self.size)
