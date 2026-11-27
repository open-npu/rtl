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
    parameter DATA_W   = 32,           // Data width in bits
    parameter DEPTH    = 1024,         // Number of entries
    parameter ADDR_W   = $clog2(DEPTH),// Address width (auto-computed)
    parameter PORT_A_W = DATA_W        // Port A write width (DATA_W, 2*DATA_W, or 4*DATA_W)
)(
    input  wire                         clk,

    // ─── Port A (DMA write / read) ───
    input  wire                         a_en,
    input  wire [(PORT_A_W/DATA_W)-1:0] a_we,
    input  wire [ADDR_W-1:0]            a_addr,
    input  wire [PORT_A_W-1:0]          a_wdata,
    output reg  [DATA_W-1:0]            a_rdata,

    // ─── Port B (read/write) ───
    input  wire                 b_en,
    input  wire                 b_we,
    input  wire [ADDR_W-1:0]   b_addr,
    input  wire [DATA_W-1:0]   b_wdata,
    output reg  [DATA_W-1:0]   b_rdata
);

    localparam PORT_A_WORDS = PORT_A_W / DATA_W;

    // ─── Memory array ───
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    // ─── Port A logic ───
    generate
        if (PORT_A_WORDS == 1) begin : gen_porta_narrow
            always @(posedge clk) begin
                if (a_en) begin
                    if (a_we[0])
                        mem[a_addr] <= a_wdata;
                    a_rdata <= mem[a_addr];
                end
            end
        end else begin : gen_porta_wide
            integer aw;
            always @(posedge clk) begin
                if (a_en) begin
                    for (aw = 0; aw < PORT_A_WORDS; aw = aw + 1)
                        if (a_we[aw])
                            mem[a_addr + aw] <= a_wdata[DATA_W*aw +: DATA_W];
                end
            end
            always @(posedge clk) begin
                if (a_en)
                    a_rdata <= mem[a_addr];
            end
        end
    endgenerate

    // ─── Port B logic ───
    always @(posedge clk) begin
        if (b_en) begin
            if (b_we)
                mem[b_addr] <= b_wdata;
            b_rdata <= mem[b_addr];
        end
    end

endmodule
