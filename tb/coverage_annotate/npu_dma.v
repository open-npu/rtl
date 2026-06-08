//      // verilator_coverage annotation
        // Open-NPU RTL — DMA Engine
        // SPDX-License-Identifier: Apache-2.0
        //
        // Simple DMA controller for transferring data between external memory
        // (Wishbone master interface) and internal SRAMs (direct SRAM interface).
        //
        // Features:
        //   - Configurable source address, destination address, transfer length
        //   - Auto-increment addressing
        //   - Configurable burst length (4/8/16/32 words)
        //   - Direction: MEM→SRAM (load) or SRAM→MEM (store)
        //   - Busy/done status signals
        //   - Abort capability
        //
        // The DMA operates on 32-bit words. For byte-level transfers,
        // the controller module handles packing/unpacking.
        
        `include "npu_defines.vh"
        
        module npu_dma #(
            parameter ADDR_W = 32,           // External memory address width
            parameter DATA_W = 32,           // Data bus width
            parameter SRAM_ADDR_W = 16       // Internal SRAM address width
        )(
 004959     input  wire                 clk,
 000011     input  wire                 rst_n,
        
            // ─── Control Interface (from CSR/Controller) ───
 000012     input  wire                 start,          // Pulse to begin transfer
%000000     input  wire                 abort,          // Pulse to abort
%000000     input  wire                 dir,            // 0=MEM→SRAM (load), 1=SRAM→MEM (store)
%000007     input  wire [ADDR_W-1:0]   ext_addr,       // External memory start address
%000000     input  wire [SRAM_ADDR_W-1:0] sram_addr,   // Internal SRAM start address
%000003     input  wire [15:0]         xfer_len,       // Transfer length (words)
%000000     input  wire [1:0]          burst_cfg,      // Burst: 0=4, 1=8, 2=16, 3=32
        
            // ─── Status Outputs ───
 000012     output reg                  busy,
 000012     output reg                  done_pulse,     // 1-cycle pulse on completion
~000224     output reg  [15:0]         xfer_count,     // Words transferred so far
        
            // ─── Wishbone Master Interface (to external memory) ───
 000228     output reg                  wb_cyc_o,
 000228     output reg                  wb_stb_o,
%000000     output reg                  wb_we_o,
~000223     output reg  [ADDR_W-1:0]   wb_adr_o,
%000000     output reg  [DATA_W-1:0]   wb_dat_o,
%000001     output wire [3:0]          wb_sel_o,
 000450     input  wire                 wb_ack_i,
~000024     input  wire [DATA_W-1:0]   wb_dat_i,
        
            // ─── SRAM Interface (to internal SRAM) ───
 000448     output reg                  sram_en,
 000448     output reg                  sram_we,
~000223     output reg  [SRAM_ADDR_W-1:0] sram_addr_o,
~000024     output reg  [DATA_W-1:0]   sram_wdata,
%000006     input  wire [DATA_W-1:0]   sram_rdata
        );
        
            // Always select all bytes
            assign wb_sel_o = 4'hF;
        
            // ─── FSM States ───
            localparam S_IDLE    = 3'd0;
            localparam S_LOAD    = 3'd1;  // MEM→SRAM: issue WB read
            localparam S_LOAD_WR = 3'd2;  // MEM→SRAM: write to SRAM
            localparam S_STORE_RD= 3'd3;  // SRAM→MEM: read from SRAM
            localparam S_STORE   = 3'd4;  // SRAM→MEM: issue WB write
            localparam S_DONE    = 3'd5;
        
 000460     reg [2:0] state;
~000224     reg [ADDR_W-1:0]      r_ext_addr;
~000224     reg [SRAM_ADDR_W-1:0] r_sram_addr;
%000003     reg [15:0]            r_xfer_len;
%000000     reg                   r_dir;
~000024     reg [DATA_W-1:0]      r_data_buf;  // temporary data buffer
        
            // Burst length decode (not used for now — single-beat transfers)
            // wire [5:0] burst_len = (burst_cfg == 2'd0) ? 6'd4 :
            //                        (burst_cfg == 2'd1) ? 6'd8 :
            //                        (burst_cfg == 2'd2) ? 6'd16 : 6'd32;
        
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n) begin
 000035             state      <= S_IDLE;
 000035             busy       <= 1'b0;
 000035             done_pulse <= 1'b0;
 000035             xfer_count <= 16'd0;
 000035             wb_cyc_o   <= 1'b0;
 000035             wb_stb_o   <= 1'b0;
 000035             wb_we_o    <= 1'b0;
 000035             wb_adr_o   <= {ADDR_W{1'b0}};
 000035             wb_dat_o   <= {DATA_W{1'b0}};
 000035             sram_en    <= 1'b0;
 000035             sram_we    <= 1'b0;
 000035             sram_addr_o<= {SRAM_ADDR_W{1'b0}};
 000035             sram_wdata <= {DATA_W{1'b0}};
 000035             r_ext_addr <= {ADDR_W{1'b0}};
 000035             r_sram_addr<= {SRAM_ADDR_W{1'b0}};
 000035             r_xfer_len <= 16'd0;
 000035             r_dir      <= 1'b0;
 000035             r_data_buf <= {DATA_W{1'b0}};
 002450         end else begin
 002450             done_pulse <= 1'b0;  // default: clear pulse
 002450             sram_en    <= 1'b0;  // default: SRAM idle
 002450             sram_we    <= 1'b0;
        
 002450             case (state)
 001301                 S_IDLE: begin
~002444                     if (start && xfer_len != 0) begin
%000006                         busy       <= 1'b1;
%000006                         r_ext_addr <= ext_addr;
%000006                         r_sram_addr<= sram_addr;
%000006                         r_xfer_len <= xfer_len;
%000006                         r_dir      <= dir;
%000006                         xfer_count <= 16'd0;
%000006                         if (!dir)
%000006                             state <= S_LOAD;
                                else
%000000                             state <= S_STORE_RD;
                            end
                        end
        
                        // ─── LOAD path: MEM → SRAM ───
 000919                 S_LOAD: begin
~000919                     if (abort) begin
%000000                         wb_cyc_o <= 1'b0;
%000000                         wb_stb_o <= 1'b0;
%000000                         state    <= S_DONE;
 000919                     end else begin
                                // Issue Wishbone read
 000919                         wb_cyc_o <= 1'b1;
 000919                         wb_stb_o <= 1'b1;
 000919                         wb_we_o  <= 1'b0;
 000919                         wb_adr_o <= r_ext_addr;
 000695                         if (wb_ack_i) begin
 000224                             r_data_buf <= wb_dat_i;
 000224                             wb_cyc_o   <= 1'b0;
 000224                             wb_stb_o   <= 1'b0;
 000224                             state      <= S_LOAD_WR;
                                end
                            end
                        end
        
 000224                 S_LOAD_WR: begin
                            // Write fetched data to SRAM
 000224                     sram_en     <= 1'b1;
 000224                     sram_we     <= 1'b1;
 000224                     sram_addr_o <= r_sram_addr;
 000224                     sram_wdata  <= r_data_buf;
                            // Advance pointers
 000224                     r_ext_addr  <= r_ext_addr + 4;
 000224                     r_sram_addr <= r_sram_addr + 1;
 000224                     xfer_count  <= xfer_count + 1;
~000218                     if (xfer_count + 1 >= r_xfer_len)
%000006                         state <= S_DONE;
                            else
 000218                         state <= S_LOAD;
                        end
        
                        // ─── STORE path: SRAM → MEM ───
%000000                 S_STORE_RD: begin
%000000                     if (abort) begin
%000000                         state <= S_DONE;
%000000                     end else begin
                                // Read from SRAM (address presented this cycle, data next)
%000000                         sram_en     <= 1'b1;
%000000                         sram_we     <= 1'b0;
%000000                         sram_addr_o <= r_sram_addr;
%000000                         state       <= S_STORE;
                            end
                        end
        
%000000                 S_STORE: begin
%000000                     if (abort) begin
%000000                         wb_cyc_o <= 1'b0;
%000000                         wb_stb_o <= 1'b0;
%000000                         state    <= S_DONE;
%000000                     end else begin
                                // Issue Wishbone write with SRAM read data
%000000                         wb_cyc_o <= 1'b1;
%000000                         wb_stb_o <= 1'b1;
%000000                         wb_we_o  <= 1'b1;
%000000                         wb_adr_o <= r_ext_addr;
%000000                         wb_dat_o <= sram_rdata;
%000000                         if (wb_ack_i) begin
%000000                             wb_cyc_o    <= 1'b0;
%000000                             wb_stb_o    <= 1'b0;
%000000                             r_ext_addr  <= r_ext_addr + 4;
%000000                             r_sram_addr <= r_sram_addr + 1;
%000000                             xfer_count  <= xfer_count + 1;
%000000                             if (xfer_count + 1 >= r_xfer_len)
%000000                                 state <= S_DONE;
                                    else
%000000                                 state <= S_STORE_RD;
                                end
                            end
                        end
        
%000006                 S_DONE: begin
%000006                     busy       <= 1'b0;
%000006                     done_pulse <= 1'b1;
%000006                     wb_cyc_o   <= 1'b0;
%000006                     wb_stb_o   <= 1'b0;
%000006                     state      <= S_IDLE;
                        end
        
%000000                 default: state <= S_IDLE;
                    endcase
                end
            end
        
        endmodule
        
