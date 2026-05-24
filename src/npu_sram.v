// Open-NPU RTL — Generic Dual-Port Synchronous SRAM
// SPDX-License-Identifier: Apache-2.0
//
// True dual-port SRAM: Port A (read/write), Port B (read/write).
// Parameterized data width and depth.
// Synthesizable inference style for FPGA block RAM / ASIC SRAM.
//
// Read behavior: synchronous read (1-cycle latency).
// Write-first: if read and write to same address on same port, read returns new data.
// Cross-port: if Port A writes while Port B reads same address, Port B returns OLD data.

`include "npu_defines.vh"

module npu_sram #(
    parameter DATA_W = 32,           // Data width in bits
    parameter DEPTH  = 1024,         // Number of entries
    parameter ADDR_W = $clog2(DEPTH) // Address width (auto-computed)
)(
    input  wire                 clk,

    // ─── Port A (read/write) ───
    input  wire                 a_en,
    input  wire                 a_we,
    input  wire [ADDR_W-1:0]   a_addr,
    input  wire [DATA_W-1:0]   a_wdata,
    output reg  [DATA_W-1:0]   a_rdata,

    // ─── Port B (read/write) ───
    input  wire                 b_en,
    input  wire                 b_we,
    input  wire [ADDR_W-1:0]   b_addr,
    input  wire [DATA_W-1:0]   b_wdata,
    output reg  [DATA_W-1:0]   b_rdata
);

    // ─── Memory array ───
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    // ─── Port A logic ───
    always @(posedge clk) begin
        if (a_en) begin
            if (a_we)
                mem[a_addr] <= a_wdata;
            a_rdata <= mem[a_addr];
        end
    end

    // ─── Port B logic ───
    always @(posedge clk) begin
        if (b_en) begin
            if (b_we)
                mem[b_addr] <= b_wdata;
            b_rdata <= mem[b_addr];
        end
    end

endmodule
