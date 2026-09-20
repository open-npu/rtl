// Open-NPU RTL — Asymmetric dual-port SRAM (A32 / B256)
// SPDX-License-Identifier: Apache-2.0
//
// Same DEPTH as npu_sram: number of 32-bit words. Capacity is unchanged.
//   Port A: 32-bit R/W  — DMA / Wishbone
//   Port B: 32-bit write (compute writeback) + 256-bit read of 8 consecutive
//           words starting at b_addr. Word 0 of the beat is mem[b_addr], so
//           existing 32-bit consumers (DW / pool / add / resize / RMW) still
//           see the addressed word in b_rdata[31:0].
//
// Storage stays a linear 32-bit array so cocotb can keep poking `.mem[addr]`.
// Eight parallel reads typically infer replicated BRAM in a generic FPGA
// flow; remap to 8 banks when targeting area.
//
// Read: synchronous, 1-cycle latency (same as npu_sram).
// Same-port write+read returns OLD data (NBA). Cross-port: reader sees OLD.

`include "npu_defines.vh"

module npu_sram_wide #(
    parameter DEPTH  = 1024,
    parameter ADDR_W = $clog2(DEPTH),
    parameter BEAT_W = `SRAM_B_WIDTH
)(
    input  wire                 clk,

    // ─── Port A (32-bit read/write) ───
    input  wire                 a_en,
    input  wire                 a_we,
    input  wire [ADDR_W-1:0]   a_addr,
    input  wire [31:0]          a_wdata,
    output reg  [31:0]          a_rdata,

    // ─── Port B (32-bit write, BEAT_W-bit read) ───
    input  wire                 b_en,
    input  wire                 b_we,
    input  wire [ADDR_W-1:0]   b_addr,
    input  wire [31:0]          b_wdata,
    output reg  [BEAT_W-1:0]   b_rdata
);

    reg [31:0] mem [0:DEPTH-1];

    integer ii;
    initial begin
        for (ii = 0; ii < DEPTH; ii = ii + 1)
            mem[ii] = 32'd0;
    end

    always @(posedge clk) begin
        if (a_en) begin
            if (a_we)
                mem[a_addr] <= a_wdata;
            a_rdata <= mem[a_addr];
        end
    end

    // Write is a separate process from the 8-word read so simulators do not
    // treat the whole array as a single variable being read and written
    // in one block (Icarus then X's every location).
    always @(posedge clk) begin
        if (b_en && b_we)
            mem[b_addr] <= b_wdata;
    end

    always @(posedge clk) begin
        if (b_en) begin
            b_rdata[31:0]    <= mem[b_addr];
            b_rdata[63:32]   <= mem[b_addr + 1];
            b_rdata[95:64]   <= mem[b_addr + 2];
            b_rdata[127:96]  <= mem[b_addr + 3];
            b_rdata[159:128] <= mem[b_addr + 4];
            b_rdata[191:160] <= mem[b_addr + 5];
            b_rdata[223:192] <= mem[b_addr + 6];
            b_rdata[255:224] <= mem[b_addr + 7];
        end
    end

endmodule
