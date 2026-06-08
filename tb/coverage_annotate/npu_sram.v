//      // verilator_coverage annotation
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
 004959     input  wire                 clk,
        
            // ─── Port A (read/write) ───
 000256     input  wire                 a_en,
 000448     input  wire                 a_we,
~000223     input  wire [ADDR_W-1:0]   a_addr,
~000024     input  wire [DATA_W-1:0]   a_wdata,
%000006     output reg  [DATA_W-1:0]   a_rdata,
        
            // ─── Port B (read/write) ───
 000256     input  wire                 b_en,
~000064     input  wire                 b_we,
~000127     input  wire [ADDR_W-1:0]   b_addr,
~000016     input  wire [DATA_W-1:0]   b_wdata,
%000008     output reg  [DATA_W-1:0]   b_rdata
        );
        
            // ─── Memory array ───
            reg [DATA_W-1:0] mem [0:DEPTH-1];
        
            // ─── Port A logic ───
 002480     always @(posedge clk) begin
 002432         if (a_en) begin
~000128             if (a_we)
 000128                 mem[a_addr] <= a_wdata;
 000128             a_rdata <= mem[a_addr];
                end
            end
        
            // ─── Port B logic ───
 002480     always @(posedge clk) begin
 002448         if (b_en) begin
~000128             if (b_we)
~000032                 mem[b_addr] <= b_wdata;
 000128             b_rdata <= mem[b_addr];
                end
            end
        
        endmodule
        
