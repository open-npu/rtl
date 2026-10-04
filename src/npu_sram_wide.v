// Open-NPU RTL — Asymmetric dual-port SRAM (A32 / B-wide)
// SPDX-License-Identifier: Apache-2.0
//
// Same DEPTH as npu_sram: number of 32-bit words. Capacity is unchanged.
//   Port A: 32-bit R/W  — DMA / Wishbone
//   Port B: 32-bit write (DW / pool / add / resize / RMW) or, when b_wmask
//           is nonzero, a byte-masked wide write (conv PPU).
//           BEAT_W-bit read of consecutive words starting at b_addr. Word 0
//           of the beat is mem[b_addr], so 32-bit consumers still see the
//           addressed word in b_rdata[31:0].
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

    // ─── Port B (32-bit or byte-masked wide write, BEAT_W-bit read) ───
    input  wire                 b_en,
    input  wire                 b_we,
    input  wire [ADDR_W-1:0]   b_addr,
    input  wire [31:0]          b_wdata,
    input  wire [BEAT_W/8-1:0]  b_wmask,
    input  wire [BEAT_W-1:0]    b_wdata_wide,
    output reg  [BEAT_W-1:0]   b_rdata
);

    reg [31:0] mem [0:DEPTH-1];

    integer ii;
    integer wb;
    integer bb;
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

    // Write is a separate process from the wide read so simulators do not
    // treat the whole array as a single variable being read and written
    // in one block (Icarus then X's every location).
    always @(posedge clk) begin
        if (b_en && (b_wmask != {BEAT_W/8{1'b0}})) begin
            for (wb = 0; wb < BEAT_W/32; wb = wb + 1)
                for (bb = 0; bb < 4; bb = bb + 1)
                    if (b_wmask[4*wb + bb])
                        mem[b_addr + wb][8*bb +: 8]
                            <= b_wdata_wide[32*wb + 8*bb +: 8];
        end else if (b_en && b_we)
            mem[b_addr] <= b_wdata;
    end

    always @(posedge clk) begin
        if (b_en) begin
            for (ii = 0; ii < BEAT_W/32; ii = ii + 1)
                b_rdata[32*ii +: 32] <= mem[b_addr + ii];
        end
    end

endmodule
