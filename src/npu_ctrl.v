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
//     DB_EN-aware — overlaps DMA prefetch with compute via ping-pong.
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
    input  wire [31:0]  cfg_dma_in_stride,   // external address stride for load (bytes)
    input  wire [31:0]  cfg_dma_out_stride,  // external address stride for store (bytes)
    input  wire [31:0]  cfg_layer_mode,      // OP type + data type
    input  wire [15:0]  cfg_out_base,        // Output base address in SRAM (word addr)
    input  wire [31:0]  cfg_dma_add_b_addr,  // DDR address of Add Branch B data

    // ─── Double-Buffer Interface ───
    output reg          ping_pong_flag,      // 0=compute reads lower half, 1=upper half
    input  wire         tile_done,           // Compute finished a non-final tile
    input  wire [15:0]  cfg_act_bank_offset, // ACT_DEPTH/2 word offset, from npu_top

    // ─── Auto-Restart Interface ───
    input  wire         ctrl_auto_next,      // AUTO_NEXT latch from CSR
    input  wire [7:0]   cfg_layer_count      // Total layer count for auto-next
);

    // ─── FSM States ───
    localparam S_IDLE         = 5'd0;
    localparam S_LOAD_WGT     = 5'd1;   // DMA: load weights
    localparam S_WAIT_WGT     = 5'd2;
    localparam S_LOAD_ACT     = 5'd3;   // DMA: load activations
    localparam S_WAIT_ACT     = 5'd4;
    localparam S_LOAD_PARAM   = 5'd5;   // DMA: load PPU parameters
    localparam S_WAIT_PARAM   = 5'd6;
    localparam S_COMPUTE      = 5'd7;   // Run compute engine
    localparam S_WAIT_COMP    = 5'd8;
    localparam S_STORE_OUT    = 5'd9;   // DMA: store output
    localparam S_WAIT_STORE   = 5'd10;
    localparam S_DONE         = 5'd11;
    localparam S_ERROR        = 5'd12;
    localparam S_LOAD_ADD_B   = 5'd13;  // DMA: load Add Branch B
    localparam S_WAIT_ADD_B   = 5'd14;
    localparam S_WAIT_PREFETCH= 5'd15;  // DB_EN: wait for prefetch DMA to complete

    reg [4:0] state;
    reg       aborted;  // Latched abort flag — prevents auto-restart in S_DONE

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

    // ─── Double-buffer control ───
    wire db_en = cfg_dma_ctrl[0];

    reg        prefetch_active;     // Prefetch DMA is in flight
    reg        prefetch_pending;    // Flag flip pending after prefetch completes
    reg [31:0] next_tile_ddr_addr;  // Running DDR address for next tile's input
    reg [15:0] next_sram_offset;    // SRAM offset for prefetch target bank

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            hw_busy          <= 1'b0;
            hw_done          <= 1'b0;
            hw_error         <= 1'b0;
            hw_error_code    <= 4'd0;
            hw_curr_layer    <= 8'd0;
            aborted          <= 1'b0;
            dma_start        <= 1'b0;
            dma_dir          <= 1'b0;
            dma_ext_addr     <= 32'd0;
            dma_sram_addr    <= 16'd0;
            dma_xfer_len     <= 16'd0;
            compute_start    <= 1'b0;
            ping_pong_flag   <= 1'b0;
            prefetch_active  <= 1'b0;
            prefetch_pending <= 1'b0;
            next_tile_ddr_addr <= 32'd0;
            next_sram_offset <= 16'd0;
        end else if (ctrl_soft_rst) begin
            state            <= S_IDLE;
            hw_busy          <= 1'b0;
            hw_done          <= 1'b0;
            hw_error         <= 1'b0;
            hw_error_code    <= 4'd0;
            aborted          <= 1'b0;
            dma_start        <= 1'b0;
            compute_start    <= 1'b0;
            ping_pong_flag   <= 1'b0;
            prefetch_active  <= 1'b0;
            prefetch_pending <= 1'b0;
        end else begin
            // Default: clear single-cycle pulses
            hw_done       <= 1'b0;
            hw_error      <= 1'b0;
            dma_start     <= 1'b0;
            compute_start <= 1'b0;

            // Latch abort while busy (prevents auto-restart in S_DONE)
            if (ctrl_abort && hw_busy)
                aborted <= 1'b1;

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
                        if (cfg_layer_mode[3:0] == 4'd4)
                            state <= S_LOAD_ADD_B;
                        else
                            state <= S_LOAD_PARAM;
                    end else begin
                        dma_start     <= 1'b1;
                        dma_dir       <= 1'b0;
                        dma_ext_addr  <= cfg_dma_in_addr;
                        dma_sram_addr <= 16'd0;  // First tile always loads to Bank[0]
                        dma_xfer_len  <= in_words;
                        state         <= S_WAIT_ACT;
                    end
                end

                S_WAIT_ACT: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done) begin
                        if (cfg_layer_mode[3:0] == 4'd4)
                            state <= S_LOAD_ADD_B;
                        else
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
                        // DB_EN: initialize for first tile (no prefetch for tile 0)
                        if (db_en) begin
                            ping_pong_flag     <= 1'b0;  // Tile 0: compute reads Bank[0]
                            prefetch_active    <= 1'b0;
                            prefetch_pending   <= 1'b0;
                            next_tile_ddr_addr <= cfg_dma_in_addr + cfg_dma_in_size;
                            next_sram_offset   <= cfg_act_bank_offset;  // Prefetch to Bank[1]
                        end
                    end
                end

                S_WAIT_COMP: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else begin
                        // ─── Prefetch trigger: fire DMA on tile_done ───
                        if (db_en && tile_done && !prefetch_active) begin
                            dma_start      <= 1'b1;
                            dma_dir        <= 1'b0;  // load
                            dma_ext_addr   <= next_tile_ddr_addr;
                            dma_sram_addr  <= next_sram_offset;
                            dma_xfer_len   <= in_words;
                            prefetch_active  <= 1'b1;
                            prefetch_pending <= 1'b1;
                            // Advance DDR address for the tile after next
                            next_tile_ddr_addr <= next_tile_ddr_addr + cfg_dma_in_size;
                            // Toggle target bank for the NEXT prefetch
                            next_sram_offset <= (next_sram_offset == 16'd0) ?
                                                cfg_act_bank_offset : 16'd0;
                        end

                        // ─── Prefetch DMA completion: flip ping_pong_flag ───
                        if (dma_done && prefetch_active) begin
                            prefetch_active <= 1'b0;
                            if (prefetch_pending) begin
                                ping_pong_flag  <= ~ping_pong_flag;
                                prefetch_pending <= 1'b0;
                            end
                        end

                        // ─── Compute done ───
                        if (compute_done) begin
                            if (db_en && prefetch_active) begin
                                // Prefetch still in flight — wait for it
                                state <= S_WAIT_PREFETCH;
                            end else begin
                                if (skip_store)
                                    state <= S_DONE;
                                else
                                    state <= S_STORE_OUT;
                            end
                        end
                    end
                end

                // ─── DB_EN: Wait for prefetch DMA to complete ───
                S_WAIT_PREFETCH: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done && prefetch_active) begin
                        prefetch_active <= 1'b0;
                        if (prefetch_pending) begin
                            ping_pong_flag  <= ~ping_pong_flag;
                            prefetch_pending <= 1'b0;
                        end
                        if (skip_store)
                            state <= S_DONE;
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

                // ─── Phase 2b: Load Add Branch B (op_type==4 only) ───
                S_LOAD_ADD_B: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (cfg_dma_add_b_addr != 0 && in_words != 0) begin
                        dma_start     <= 1'b1;
                        dma_dir       <= 1'b0;  // load
                        dma_ext_addr  <= cfg_dma_add_b_addr;
                        dma_sram_addr <= cfg_out_base;
                        dma_xfer_len  <= in_words;
                        state         <= S_WAIT_ADD_B;
                    end else begin
                        state <= S_LOAD_PARAM;
                    end
                end

                S_WAIT_ADD_B: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done) begin
                        state <= S_LOAD_PARAM;
                    end
                end

                // ─── Done ───
                S_DONE: begin
                    hw_done       <= 1'b1;
                    hw_curr_layer <= hw_curr_layer + 1;
                    if (!aborted && ctrl_auto_next && (hw_curr_layer + 1 < cfg_layer_count)) begin
                        // Auto-restart: stay busy, start next layer
                        state <= S_LOAD_WGT;
                    end else begin
                        // Normal: deassert busy, return to idle
                        hw_busy <= 1'b0;
                        aborted <= 1'b0;
                        state   <= S_IDLE;
                    end
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
