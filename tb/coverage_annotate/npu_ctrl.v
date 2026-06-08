//      // verilator_coverage annotation
        // Open-NPU RTL — NPU Controller / Sequencer
        // SPDX-License-Identifier: Apache-2.0
        //
        // Main FSM that orchestrates a single layer computation:
        //   1. Receive START from CSR
        //   2. Load weights via DMA (ext → weight SRAM)
        //   3. Load input activations via DMA (ext → act SRAM) [skip if fused mid/end]
        //   4. Load per-channel params via DMA (ext → param SRAM)
        //   5. Trigger compute engine (handles systolic/DW + PPU + writeback internally)
        //   6. Store output via DMA (act SRAM → ext) [skip if fused start/mid]
        //   7. Signal DONE
        //
        // V2: Fusion-aware — respects sched_ctrl bits to skip DMA phases.
        // The compute engine (npu_compute) handles PPU internally, so no
        // separate PPU phase is needed in this controller.
        
        `include "npu_defines.vh"
        
        module npu_ctrl (
 004959     input  wire         clk,
 000011     input  wire         rst_n,
        
            // ─── CSR Interface ───
%000004     input  wire         ctrl_start,      // START pulse from CSR
%000000     input  wire         ctrl_abort,      // ABORT pulse from CSR
%000000     input  wire         ctrl_soft_rst,   // Soft reset from CSR
%000004     output reg          hw_busy,         // Busy status to CSR
%000004     output reg          hw_done,         // Done pulse to CSR
%000000     output reg          hw_error,        // Error pulse to CSR
%000000     output reg  [3:0]   hw_error_code,   // Error code to CSR
%000003     output reg  [7:0]   hw_curr_layer,   // Current layer number
        
            // ─── DMA Control Interface ───
 000012     output reg          dma_start,       // Start DMA transfer
%000000     output reg          dma_dir,         // 0=load, 1=store
%000007     output reg  [31:0]  dma_ext_addr,    // External address
%000000     output reg  [15:0]  dma_sram_addr,   // SRAM address
%000003     output reg  [15:0]  dma_xfer_len,    // Transfer length (words)
 000012     input  wire         dma_busy,        // DMA busy
 000012     input  wire         dma_done,        // DMA done pulse
        
            // ─── Compute Control ───
%000004     output reg          compute_start,   // Start compute engine
%000004     input  wire         compute_done,    // Compute done
        
            // ─── Layer Configuration (from CSR register file) ───
%000004     input  wire [31:0]  cfg_dma_in_addr,
%000000     input  wire [31:0]  cfg_dma_out_addr,
%000003     input  wire [31:0]  cfg_dma_wgt_addr,
%000003     input  wire [31:0]  cfg_dma_param_addr,
%000002     input  wire [31:0]  cfg_dma_in_size,     // in bytes
%000002     input  wire [31:0]  cfg_dma_wgt_size,    // in bytes
%000000     input  wire [31:0]  cfg_dma_out_size,    // in bytes
%000003     input  wire [15:0]  cfg_param_count,     // number of output channels
%000003     input  wire [31:0]  cfg_dma_ctrl,        // sched_ctrl: [0]=DB_EN, [1]=FUSE_START, [2]=FUSE_MID, [3]=FUSE_END
%000006     input  wire [31:0]  cfg_layer_mode,      // OP type + data type
%000000     input  wire [15:0]  cfg_out_base,        // Output base address in SRAM (word addr)
%000000     input  wire [31:0]  cfg_dma_add_b_addr,  // DDR address of Add Branch B data
        
            // ─── Auto-Restart Interface ───
%000000     input  wire         ctrl_auto_next,      // AUTO_NEXT latch from CSR
%000000     input  wire [7:0]   cfg_layer_count      // Total layer count for auto-next
        );
        
            // ─── FSM States ───
            localparam S_IDLE       = 4'd0;
            localparam S_LOAD_WGT   = 4'd1;   // DMA: load weights
            localparam S_WAIT_WGT   = 4'd2;
            localparam S_LOAD_ACT   = 4'd3;   // DMA: load activations
            localparam S_WAIT_ACT   = 4'd4;
            localparam S_LOAD_PARAM = 4'd5;   // DMA: load PPU parameters
            localparam S_WAIT_PARAM = 4'd6;
            localparam S_COMPUTE    = 4'd7;   // Run compute engine
            localparam S_WAIT_COMP  = 4'd8;
            localparam S_STORE_OUT  = 4'd9;   // DMA: store output
            localparam S_WAIT_STORE = 4'd10;
            localparam S_DONE       = 4'd11;
            localparam S_ERROR      = 4'd12;
            localparam S_LOAD_ADD_B = 4'd13;  // DMA: load Add Branch B
            localparam S_WAIT_ADD_B = 4'd14;
        
~000020     reg [3:0] state;
%000000     reg       aborted;  // Latched abort flag — prevents auto-restart in S_DONE
        
            // Per-channel param size: 4 words (16 bytes) per channel
%000003     wire [15:0] param_words = cfg_param_count * 4;
        
            // Weight words = wgt_size / 4
%000002     wire [15:0] wgt_words = cfg_dma_wgt_size[17:2];
        
            // Input words = in_size / 4
%000002     wire [15:0] in_words = cfg_dma_in_size[17:2];
        
            // Output words = out_size / 4
%000000     wire [15:0] out_words = cfg_dma_out_size[17:2];
        
            // ─── Fusion control bits ───
%000003     wire fuse_start = cfg_dma_ctrl[1];
%000000     wire fuse_mid   = cfg_dma_ctrl[2];
%000000     wire fuse_end   = cfg_dma_ctrl[3];
%000000     wire skip_act_load = fuse_mid | fuse_end;   // Input already in SRAM
%000003     wire skip_store    = fuse_start | fuse_mid;  // Output stays in SRAM
        
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n) begin
 000035             state         <= S_IDLE;
 000035             hw_busy       <= 1'b0;
 000035             hw_done       <= 1'b0;
 000035             hw_error      <= 1'b0;
 000035             hw_error_code <= 4'd0;
 000035             hw_curr_layer <= 8'd0;
 000035             aborted       <= 1'b0;
 000035             dma_start     <= 1'b0;
 000035             dma_dir       <= 1'b0;
 000035             dma_ext_addr  <= 32'd0;
 000035             dma_sram_addr <= 16'd0;
 000035             dma_xfer_len  <= 16'd0;
 000035             compute_start <= 1'b0;
~002450         end else if (ctrl_soft_rst) begin
%000000             state         <= S_IDLE;
%000000             hw_busy       <= 1'b0;
%000000             hw_done       <= 1'b0;
%000000             hw_error      <= 1'b0;
%000000             hw_error_code <= 4'd0;
%000000             aborted       <= 1'b0;
%000000             dma_start     <= 1'b0;
%000000             compute_start <= 1'b0;
 002450         end else begin
                    // Default: clear single-cycle pulses
 002450             hw_done       <= 1'b0;
 002450             hw_error      <= 1'b0;
 002450             dma_start     <= 1'b0;
 002450             compute_start <= 1'b0;
        
                    // Latch abort while busy (prevents auto-restart in S_DONE)
~002450             if (ctrl_abort && hw_busy)
%000000                 aborted <= 1'b1;
        
 002450             case (state)
 000245                 S_IDLE: begin
~000243                     if (ctrl_start) begin
%000002                         hw_busy  <= 1'b1;
%000002                         state    <= S_LOAD_WGT;
                            end
                        end
        
                        // ─── Phase 1: Load Weights ───
%000002                 S_LOAD_WGT: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
%000002                     end else if (wgt_words != 0) begin
%000002                         dma_start     <= 1'b1;
%000002                         dma_dir       <= 1'b0;  // load
%000002                         dma_ext_addr  <= cfg_dma_wgt_addr;
%000002                         dma_sram_addr <= 16'd0;
%000002                         dma_xfer_len  <= wgt_words;
%000002                         state         <= S_WAIT_WGT;
%000000                     end else begin
%000000                         state <= S_LOAD_ACT;
                            end
                        end
        
 000248                 S_WAIT_WGT: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
~000246                     end else if (dma_done) begin
%000002                         state <= S_LOAD_ACT;
                            end
                        end
        
                        // ─── Phase 2: Load Activations ───
%000002                 S_LOAD_ACT: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
%000002                     end else if (skip_act_load || in_words == 0) begin
                                // Fused mid/end: input already in SRAM, skip load
%000000                         if (cfg_layer_mode[3:0] == 4'd4)
%000000                             state <= S_LOAD_ADD_B;
                                else
%000000                             state <= S_LOAD_PARAM;
%000002                     end else begin
%000002                         dma_start     <= 1'b1;
%000002                         dma_dir       <= 1'b0;
%000002                         dma_ext_addr  <= cfg_dma_in_addr;
%000002                         dma_sram_addr <= 16'd0;
%000002                         dma_xfer_len  <= in_words;
%000002                         state         <= S_WAIT_ACT;
                            end
                        end
        
 000246                 S_WAIT_ACT: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
~000244                     end else if (dma_done) begin
%000002                         if (cfg_layer_mode[3:0] == 4'd4)
%000000                             state <= S_LOAD_ADD_B;
                                else
%000002                             state <= S_LOAD_PARAM;
                            end
                        end
        
                        // ─── Phase 3: Load PPU Parameters ───
%000002                 S_LOAD_PARAM: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
%000002                     end else if (param_words != 0) begin
%000002                         dma_start     <= 1'b1;
%000002                         dma_dir       <= 1'b0;
%000002                         dma_ext_addr  <= cfg_dma_param_addr;
%000002                         dma_sram_addr <= 16'd0;
%000002                         dma_xfer_len  <= param_words;
%000002                         state         <= S_WAIT_PARAM;
%000000                     end else begin
%000000                         state <= S_COMPUTE;
                            end
                        end
        
 000667                 S_WAIT_PARAM: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
~000665                     end else if (dma_done) begin
%000002                         state <= S_COMPUTE;
                            end
                        end
        
                        // ─── Phase 4: Compute (includes PPU + writeback) ───
%000002                 S_COMPUTE: begin
%000002                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
%000002                     end else begin
%000002                         compute_start <= 1'b1;
%000002                         state         <= S_WAIT_COMP;
                            end
                        end
        
 001034                 S_WAIT_COMP: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
~001032                     end else if (compute_done) begin
%000002                         if (skip_store)
%000002                             state <= S_DONE;  // Fused: output stays in SRAM
                                else
%000000                             state <= S_STORE_OUT;
                            end
                        end
        
                        // ─── Phase 5: Store Output ───
%000000                 S_STORE_OUT: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
%000000                     end else if (out_words != 0) begin
%000000                         dma_start     <= 1'b1;
%000000                         dma_dir       <= 1'b1;  // store
%000000                         dma_ext_addr  <= cfg_dma_out_addr;
%000000                         dma_sram_addr <= cfg_out_base;
%000000                         dma_xfer_len  <= out_words;
%000000                         state         <= S_WAIT_STORE;
%000000                     end else begin
%000000                         state <= S_DONE;
                            end
                        end
        
%000000                 S_WAIT_STORE: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
%000000                     end else if (dma_done) begin
%000000                         state <= S_DONE;
                            end
                        end
        
                        // ─── Phase 2b: Load Add Branch B (op_type==4 only) ───
%000000                 S_LOAD_ADD_B: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
%000000                     end else if (cfg_dma_add_b_addr != 0 && in_words != 0) begin
%000000                         dma_start     <= 1'b1;
%000000                         dma_dir       <= 1'b0;  // load
%000000                         dma_ext_addr  <= cfg_dma_add_b_addr;
%000000                         dma_sram_addr <= cfg_out_base;
%000000                         dma_xfer_len  <= in_words;
%000000                         state         <= S_WAIT_ADD_B;
%000000                     end else begin
%000000                         state <= S_LOAD_PARAM;
                            end
                        end
        
%000000                 S_WAIT_ADD_B: begin
%000000                     if (ctrl_abort) begin
%000000                         state <= S_DONE;
%000000                     end else if (dma_done) begin
%000000                         state <= S_LOAD_PARAM;
                            end
                        end
        
                        // ─── Done ───
%000002                 S_DONE: begin
%000002                     hw_done       <= 1'b1;
%000002                     hw_curr_layer <= hw_curr_layer + 1;
~002450                     if (!aborted && ctrl_auto_next && (hw_curr_layer + 1 < cfg_layer_count)) begin
                                // Auto-restart: stay busy, start next layer
%000000                         state <= S_LOAD_WGT;
%000002                     end else begin
                                // Normal: deassert busy, return to idle
%000002                         hw_busy <= 1'b0;
%000002                         aborted <= 1'b0;
%000002                         state   <= S_IDLE;
                            end
                        end
        
%000000                 S_ERROR: begin
%000000                     hw_busy  <= 1'b0;
%000000                     hw_error <= 1'b1;
%000000                     state    <= S_IDLE;
                        end
        
%000000                 default: state <= S_IDLE;
                    endcase
                end
            end
        
        endmodule
        
