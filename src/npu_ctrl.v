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
    input  wire         clk,
    input  wire         rst_n,

    // ─── CSR Interface ───
    input  wire         ctrl_start,      // START pulse from CSR
    input  wire         ctrl_abort,      // ABORT pulse from CSR
    input  wire         ctrl_soft_rst,   // Soft reset from CSR
    output reg          hw_busy,         // Busy status to CSR
    output reg          hw_done,         // Done pulse to CSR
    output reg          hw_error,        // Error pulse to CSR
    output reg  [3:0]   hw_error_code,   // Error code to CSR
    output reg  [7:0]   hw_curr_layer,   // Current layer number

    // ─── DMA Control Interface ───
    output reg          dma_start,       // Start DMA transfer
    output reg          dma_dir,         // 0=load, 1=store
    output reg  [31:0]  dma_ext_addr,    // External address
    output reg  [15:0]  dma_sram_addr,   // SRAM address
    output reg  [15:0]  dma_xfer_len,    // Transfer length (words)
    input  wire         dma_busy,        // DMA busy
    input  wire         dma_done,        // DMA done pulse

    // ─── Compute Control ───
    output reg          compute_start,   // Start compute engine
    input  wire         compute_done,    // Compute done

    // ─── Layer Configuration (from CSR register file) ───
    input  wire [31:0]  cfg_dma_in_addr,
    input  wire [31:0]  cfg_dma_out_addr,
    input  wire [31:0]  cfg_dma_wgt_addr,
    input  wire [31:0]  cfg_dma_param_addr,
    input  wire [31:0]  cfg_dma_in_size,     // in bytes
    input  wire [31:0]  cfg_dma_wgt_size,    // in bytes
    input  wire [31:0]  cfg_dma_out_size,    // in bytes
    input  wire [15:0]  cfg_param_count,     // number of output channels
    input  wire [31:0]  cfg_dma_ctrl,        // sched_ctrl: [0]=DB_EN, [1]=FUSE_START, [2]=FUSE_MID, [3]=FUSE_END
    input  wire [31:0]  cfg_layer_mode,      // OP type + data type
    input  wire [15:0]  cfg_out_base         // Output base address in SRAM (word addr)
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

    reg [3:0] state;

    // Per-channel param size: 4 words (16 bytes) per channel
    wire [15:0] param_words = cfg_param_count * 4;

    // Weight words = wgt_size / 4
    wire [15:0] wgt_words = cfg_dma_wgt_size[17:2];

    // Input words = in_size / 4
    wire [15:0] in_words = cfg_dma_in_size[17:2];

    // Output words = out_size / 4
    wire [15:0] out_words = cfg_dma_out_size[17:2];

    // ─── Fusion control bits ───
    wire fuse_start = cfg_dma_ctrl[1];
    wire fuse_mid   = cfg_dma_ctrl[2];
    wire fuse_end   = cfg_dma_ctrl[3];
    wire skip_act_load = fuse_mid | fuse_end;   // Input already in SRAM
    wire skip_store    = fuse_start | fuse_mid;  // Output stays in SRAM

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            hw_busy       <= 1'b0;
            hw_done       <= 1'b0;
            hw_error      <= 1'b0;
            hw_error_code <= 4'd0;
            hw_curr_layer <= 8'd0;
            dma_start     <= 1'b0;
            dma_dir       <= 1'b0;
            dma_ext_addr  <= 32'd0;
            dma_sram_addr <= 16'd0;
            dma_xfer_len  <= 16'd0;
            compute_start <= 1'b0;
        end else if (ctrl_soft_rst) begin
            state         <= S_IDLE;
            hw_busy       <= 1'b0;
            hw_done       <= 1'b0;
            hw_error      <= 1'b0;
            hw_error_code <= 4'd0;
            dma_start     <= 1'b0;
            compute_start <= 1'b0;
        end else begin
            // Default: clear single-cycle pulses
            hw_done       <= 1'b0;
            hw_error      <= 1'b0;
            dma_start     <= 1'b0;
            compute_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (ctrl_start) begin
                        hw_busy  <= 1'b1;
                        state    <= S_LOAD_WGT;
                    end
                end

                // ─── Phase 1: Load Weights ───
                S_LOAD_WGT: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (wgt_words != 0) begin
                        dma_start     <= 1'b1;
                        dma_dir       <= 1'b0;  // load
                        dma_ext_addr  <= cfg_dma_wgt_addr;
                        dma_sram_addr <= 16'd0;
                        dma_xfer_len  <= wgt_words;
                        state         <= S_WAIT_WGT;
                    end else begin
                        state <= S_LOAD_ACT;
                    end
                end

                S_WAIT_WGT: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done) begin
                        state <= S_LOAD_ACT;
                    end
                end

                // ─── Phase 2: Load Activations ───
                S_LOAD_ACT: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (skip_act_load || in_words == 0) begin
                        // Fused mid/end: input already in SRAM, skip load
                        state <= S_LOAD_PARAM;
                    end else begin
                        dma_start     <= 1'b1;
                        dma_dir       <= 1'b0;
                        dma_ext_addr  <= cfg_dma_in_addr;
                        dma_sram_addr <= 16'd0;
                        dma_xfer_len  <= in_words;
                        state         <= S_WAIT_ACT;
                    end
                end

                S_WAIT_ACT: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done) begin
                        state <= S_LOAD_PARAM;
                    end
                end

                // ─── Phase 3: Load PPU Parameters ───
                S_LOAD_PARAM: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (param_words != 0) begin
                        dma_start     <= 1'b1;
                        dma_dir       <= 1'b0;
                        dma_ext_addr  <= cfg_dma_param_addr;
                        dma_sram_addr <= 16'd0;
                        dma_xfer_len  <= param_words;
                        state         <= S_WAIT_PARAM;
                    end else begin
                        state <= S_COMPUTE;
                    end
                end

                S_WAIT_PARAM: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done) begin
                        state <= S_COMPUTE;
                    end
                end

                // ─── Phase 4: Compute (includes PPU + writeback) ───
                S_COMPUTE: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else begin
                        compute_start <= 1'b1;
                        state         <= S_WAIT_COMP;
                    end
                end

                S_WAIT_COMP: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (compute_done) begin
                        if (skip_store)
                            state <= S_DONE;  // Fused: output stays in SRAM
                        else
                            state <= S_STORE_OUT;
                    end
                end

                // ─── Phase 5: Store Output ───
                S_STORE_OUT: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (out_words != 0) begin
                        dma_start     <= 1'b1;
                        dma_dir       <= 1'b1;  // store
                        dma_ext_addr  <= cfg_dma_out_addr;
                        dma_sram_addr <= cfg_out_base;
                        dma_xfer_len  <= out_words;
                        state         <= S_WAIT_STORE;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_WAIT_STORE: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done) begin
                        state <= S_DONE;
                    end
                end

                // ─── Done ───
                S_DONE: begin
                    hw_busy       <= 1'b0;
                    hw_done       <= 1'b1;
                    hw_curr_layer <= hw_curr_layer + 1;
                    state         <= S_IDLE;
                end

                S_ERROR: begin
                    hw_busy  <= 1'b0;
                    hw_error <= 1'b1;
                    state    <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
