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
    input  wire [31:0]  cfg_tile_in_size,    // per-tile input size (bytes), 0=use full in_size
    input  wire [15:0]  cfg_param_count,     // number of output channels
    input  wire [31:0]  cfg_dma_ctrl,        // sched_ctrl: [0]=DB_EN, [1]=FUSE_START, [2]=FUSE_MID, [3]=FUSE_END
    input  wire [31:0]  cfg_dma_in_stride,   // external address stride for load (bytes)
    input  wire [31:0]  cfg_dma_out_stride,  // external address stride for store (bytes)
    input  wire [31:0]  cfg_layer_mode,      // OP type + data type
    input  wire [15:0]  cfg_out_base,        // Output base address in SRAM (word addr)
    input  wire [31:0]  cfg_dma_add_b_addr,  // DDR address of Add Branch B data

    // ─── Per-tile Store Configuration (2D DMA for NHWC layout) ───
    input  wire [31:0]  cfg_dma_store_mode,    // 0x140: bit[0]=PER_TILE_STORE_EN
    input  wire [31:0]  cfg_dma_tile_out_size, // 0x138: per-tile output size (bytes)
    input  wire [15:0]  cfg_out_w,             // output width (for NHWC row stride)
    input  wire [15:0]  cfg_out_c,             // output channels (for NHWC row stride)
    input  wire [15:0]  cfg_tile_h,            // tile height (for tile sequencing)
    input  wire [15:0]  cfg_tile_w,            // tile width (for tile sequencing)
    input  wire [15:0]  cfg_tile_num_h,        // tile count H (for last-tile detection)
    input  wire [15:0]  cfg_tile_num_w,        // tile count W (for tile sequencing)
    input  wire         cfg_int16,             // 1=INT16 (elem_bytes=2), 0=INT8 (elem_bytes=1)
    input  wire [15:0]  tile_out_h_actual,     // Actual (border-clipped) tile height from compute
    input  wire [15:0]  tile_out_w_actual,     // Actual (border-clipped) tile width from compute
    output reg  [15:0]  dma_row_len,           // 2D DMA: words per row (0=1D mode)
    output reg  [15:0]  dma_row_count,         // 2D DMA: row count (0=1D mode)
    output reg  [31:0]  dma_out_stride,        // 2D DMA: row stride (bytes) for store

    // ─── Double-Buffer Interface ───
    output reg          ping_pong_flag,      // 0=compute reads lower half, 1=upper half
    output reg          db_prefetch_done,    // Prefetch complete; compute may start next tile
    input  wire         tile_done,           // Compute finished a non-final tile
    input  wire [15:0]  cfg_act_bank_offset, // ACT_DEPTH/2 word offset, from npu_top

    // ─── Auto-Restart Interface ───
    input  wire         ctrl_auto_next,      // AUTO_NEXT latch from CSR
    input  wire [7:0]   cfg_layer_count      // Total layer count for auto-next
);

    // ─── FSM States ───
    localparam S_IDLE         = 6'd0;
    localparam S_LOAD_WGT     = 6'd1;   // DMA: load weights
    localparam S_WAIT_WGT     = 6'd2;
    localparam S_LOAD_ACT     = 6'd3;   // DMA: load activations
    localparam S_WAIT_ACT     = 6'd4;
    localparam S_LOAD_PARAM   = 6'd5;   // DMA: load PPU parameters
    localparam S_WAIT_PARAM   = 6'd6;
    localparam S_COMPUTE      = 6'd7;   // Run compute engine
    localparam S_WAIT_COMP    = 6'd8;
    localparam S_STORE_OUT    = 6'd9;   // DMA: store output
    localparam S_WAIT_STORE   = 6'd10;
    localparam S_DONE         = 6'd11;
    localparam S_ERROR        = 6'd12;
    localparam S_LOAD_ADD_B   = 6'd13;  // DMA: load Add Branch B
    localparam S_WAIT_ADD_B   = 6'd14;
    localparam S_WAIT_PREFETCH= 6'd15;  // DB_EN: wait for prefetch DMA to complete
    localparam S_TILE_STORE   = 6'd16;  // Per-tile store: fire 2D DMA store for current tile
    localparam S_TILE_STORE_WAIT = 6'd17; // Per-tile store: wait for DMA done

    reg [5:0] state;
    reg       aborted;  // Latched abort flag — prevents auto-restart in S_DONE

    // Per-channel param size: 4 words (16 bytes) per channel
    wire [15:0] param_words = (cfg_layer_mode[3:0] == 4'd4 || cfg_layer_mode[3:0] == 4'd7)
                              ? cfg_param_count        // Add/Concat: direct word count
                              : cfg_param_count * 4;  // Conv2D: 4 words per channel

    // Weight words = wgt_size / 4
    wire [15:0] wgt_words = cfg_dma_wgt_size[17:2];

    // Input words = in_size / 4
    wire [15:0] in_words = cfg_dma_in_size[17:2];

    // Per-tile input words (for DB_EN): use tile_in_size when non-zero, else full in_words
    wire [15:0] tile_in_words = (db_en && cfg_tile_in_size[17:2] != 0)
                                ? cfg_tile_in_size[17:2]
                                : in_words;

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
    wire per_tile_store_en = cfg_dma_store_mode[0];  // Per-tile store to NHWC DDR layout

    reg        prefetch_active;     // Prefetch DMA is in flight
    reg        prefetch_pending;    // Flag flip pending after prefetch completes
    reg [31:0] next_tile_ddr_addr;  // Running DDR address for next tile's input
    reg [31:0] cur_tile_ddr_offset; // Current tile's DDR offset from base
    reg [15:0] next_sram_offset;    // SRAM offset for prefetch target bank
    reg        add_b_reload;        // 1=Add B reload after compute_done (→ S_STORE_OUT)
    reg        store_bank;          // Bank to store from (final tile's ping_pong)
    reg        last_tile_store;     // 1=S_TILE_STORE is for final tile (→ S_DONE, no prefetch)

    // ─── Per-tile store tile sequencing ───
    // Track tile (y,x) position for NHWC DDR offset computation
    reg [15:0] tile_y_seq, tile_x_seq;  // Current tile indices
    // Computed NHWC output row stride (bytes) = out_w * out_c * elem_bytes
    wire [31:0] nhwc_row_stride = {16'd0, cfg_out_w} * {16'd0, cfg_out_c} * (cfg_int16 ? 32'd2 : 32'd1);
    // Per-tile output words (total) — use clipped row_len * row_count for
    // correct border tile DMA transfer length (avoid over-writing adjacent tiles)
    wire [15:0] tile_out_words_padded = cfg_dma_tile_out_size[17:2];
    wire [15:0] tile_out_words = tile_row_len * tile_row_count;
    // Per-tile row length (words) = actual_tile_w * out_c * eb / 4
    // Uses actual (border-clipped) tile width from compute for correct border tile store
    wire [15:0] tile_row_len = (tile_out_w_actual != 0) ?
        (({16'd0, tile_out_w_actual} * {16'd0, cfg_out_c} * (cfg_int16 ? 32'd2 : 32'd1)) >> 2) : tile_out_words;
    // Per-tile row count = actual tile height (clipped at border)
    wire [15:0] tile_row_count = (tile_out_h_actual != 0) ? tile_out_h_actual : cfg_tile_h;
    // DDR offset for current tile = (tile_y*tile_h*out_w + tile_x*tile_w) * out_c * eb
    wire [31:0] tile_ddr_offset = ({16'd0, tile_y_seq} * {16'd0, cfg_tile_h} * {16'd0, cfg_out_w}
                                   + {16'd0, tile_x_seq} * {16'd0, cfg_tile_w})
                                  * {16'd0, cfg_out_c} * (cfg_int16 ? 32'd2 : 32'd1);

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
            db_prefetch_done <= 1'b1;
            next_tile_ddr_addr <= 32'd0;
            cur_tile_ddr_offset <= 32'd0;
            add_b_reload <= 1'b0;
            next_sram_offset <= 16'd0;
            tile_y_seq <= 16'd0;
            tile_x_seq <= 16'd0;
            last_tile_store <= 1'b0;
            dma_row_len <= 16'd0;
            dma_row_count <= 16'd0;
            dma_out_stride <= 32'd0;
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
            db_prefetch_done <= 1'b1;
        end else begin
            // Default: clear single-cycle pulses
            hw_done       <= 1'b0;
            hw_error      <= 1'b0;
            dma_start     <= 1'b0;
            compute_start <= 1'b0;
            // Default: 1D DMA mode (2D only active in S_TILE_STORE)
            dma_row_len   <= 16'd0;
            dma_row_count <= 16'd0;
            // Default out_stride from CSR (used in 1D S_STORE_OUT; overridden in S_TILE_STORE)
            dma_out_stride <= cfg_dma_out_stride;

            // Latch abort while busy (prevents auto-restart in S_DONE)
            if (ctrl_abort && hw_busy)
                aborted <= 1'b1;

            case (state)
                S_IDLE: begin
                    if (ctrl_start) begin
                        hw_busy  <= 1'b1;
                        state    <= S_LOAD_WGT;
                        // Reset per-tile store sequencing
                        tile_y_seq <= 16'd0;
                        tile_x_seq <= 16'd0;
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
                        dma_xfer_len  <= tile_in_words;  // per-tile when DB_EN, full when not
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
                        // For fused MID/END layers, skip_act_load is asserted:
                        // input is already in SRAM, no DDR load or prefetch needed.
                        if (db_en) begin
                            if (!skip_act_load) begin
                                ping_pong_flag     <= 1'b0;  // Tile 0: reads Bank[0]
                                prefetch_active    <= 1'b0;
                                prefetch_pending   <= 1'b0;
                                cur_tile_ddr_offset <= 32'd0;  // Tile 0: offset 0
                                // Advance by tile_in_size when DB_EN active (per-tile), else full in_size
                                next_tile_ddr_addr <= cfg_dma_in_addr
                                    + (db_en && cfg_tile_in_size != 0 ? cfg_tile_in_size : cfg_dma_in_size);
                                // Prefetch tile 1 to Bank[1] at offset 0
                                // No intra-bank word offset: compute always reads from offset 0
                                next_sram_offset   <= cfg_act_bank_offset;
                            end else begin
                                // Fused: input is contiguous in SRAM, no prefetch
                                ping_pong_flag     <= 1'b0;
                                prefetch_active    <= 1'b0;
                                prefetch_pending   <= 1'b0;
                            end
                        end
                    end
                end

                S_WAIT_COMP: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else begin
                        // ─── tile_done: non-final tile boundary ───
                        // Per-tile store path: store current tile output to DDR (NHWC) first,
                        // then prefetch next tile input.
                        // Non-per-tile path (last-tile-only store): prefetch directly.
                        if (db_en && tile_done && !prefetch_active && !skip_act_load) begin
                            if (per_tile_store_en) begin
                                // Latch current tile's bank for store
                                store_bank <= ping_pong_flag;
                                last_tile_store <= 1'b0;  // non-final tile
                                db_prefetch_done <= 1'b0;  // Block compute during store+prefetch
                                `ifndef SYNTHESIS
                                $display("[PTS_TILEDONE] t=%0t tile_done fired, ping_pong=%0d → S_TILE_STORE", $time, ping_pong_flag);
                                `endif
                                state <= S_TILE_STORE;
                            end else begin
                                // Original path: prefetch next tile directly
                                dma_start      <= 1'b1;
                                dma_dir        <= 1'b0;  // load
                                dma_ext_addr   <= next_tile_ddr_addr;
                                dma_sram_addr  <= next_sram_offset;
                                dma_xfer_len   <= tile_in_words;
                                prefetch_active  <= 1'b1;
                                prefetch_pending <= 1'b1;
                                db_prefetch_done <= 1'b0;
                                cur_tile_ddr_offset <= next_tile_ddr_addr - cfg_dma_in_addr;
                                next_tile_ddr_addr <= next_tile_ddr_addr
                                    + (db_en && cfg_tile_in_size != 0 ? cfg_tile_in_size : cfg_dma_in_size);
                                next_sram_offset <= (next_sram_offset >= cfg_act_bank_offset) ?
                                                    16'd0 : cfg_act_bank_offset;
                            end
                        end

                        // ─── Prefetch DMA completion: flip ping_pong_flag ───
                        if (dma_done && prefetch_active) begin
                            prefetch_active  <= 1'b0;
                            if (prefetch_pending) begin
                                ping_pong_flag  <= ~ping_pong_flag;
                                prefetch_pending <= 1'b0;
                                `ifndef SYNTHESIS
                                $display("[PP_FLIP] t=%0t ping_pong %0d→%0d db_prefetch_done=1",
                                         $time, ping_pong_flag, ~ping_pong_flag);
                                `endif
                            end
                            if (cfg_dma_add_b_addr != 0) begin
                                add_b_reload <= 1'b1;
                                state <= S_LOAD_ADD_B;
                            end else begin
                                db_prefetch_done <= 1'b1;
                            end
                        end

                        // ─── Compute done (final tile) ───
                        if (compute_done) begin
                            store_bank <= ping_pong_flag;
                            if (db_en && prefetch_active) begin
                                state <= S_WAIT_PREFETCH;
                            end else begin
                                if (skip_store)
                                    state <= S_DONE;
                                else if (per_tile_store_en) begin
                                    // Last tile NOT yet stored (tile_done doesn't fire
                                    // for final tile). Go to S_TILE_STORE to store it,
                                    // then S_TILE_STORE_WAIT → S_DONE (skip prefetch).
                                    last_tile_store <= 1'b1;
                                    state <= S_TILE_STORE;
                                end
                                else
                                    state <= S_STORE_OUT;
                            end
                        end
                    end
                end

                // ─── Per-tile store: fire 2D DMA to store current tile to NHWC DDR ───
                S_TILE_STORE: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else begin
                        // 2D DMA store: current tile output → DDR at NHWC offset
                        dma_start     <= 1'b1;
                        dma_dir       <= 1'b1;  // store
                        dma_ext_addr  <= cfg_dma_out_addr + tile_ddr_offset;
                        // Source SRAM: output region in current bank
                        if (cfg_dma_add_b_addr != 0)
                            dma_sram_addr <= (db_en && store_bank ? cfg_act_bank_offset : 16'd0);
                        else
                            dma_sram_addr <= cfg_out_base + (db_en && store_bank ? cfg_act_bank_offset : 16'd0);
                        dma_xfer_len  <= tile_out_words;
                        // 2D parameters: row_len words/row, row_count rows, row_stride bytes
                        // All layers use tile_out_h/w_actual (border-clipped from compute)
                        dma_row_len    <= tile_row_len;
                        dma_row_count  <= tile_row_count;
                        dma_out_stride <= nhwc_row_stride;
                        `ifndef SYNTHESIS
                        $display("[PTS_STORE] t=%0t ty=%0d tx=%0d ddr=0x%08x sram=%0d len=%0d row_len=%0d row_cnt=%0d stride=%0d bank=%0d",
                                 $time, tile_y_seq, tile_x_seq,
                                 cfg_dma_out_addr + tile_ddr_offset,
                                 cfg_out_base + (db_en && store_bank ? cfg_act_bank_offset : 16'd0),
                                 tile_out_words, tile_row_len, tile_row_count, nhwc_row_stride, store_bank);
                        `endif
                        state          <= S_TILE_STORE_WAIT;
                    end
                end

                S_TILE_STORE_WAIT: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done) begin
                        // Clear 2D mode (revert to 1D for prefetch)
                        dma_row_len   <= 16'd0;
                        dma_row_count <= 16'd0;
                        if (last_tile_store) begin
                            // Final tile stored — done (no prefetch needed)
                            state <= S_DONE;
                        end else begin
                            // Non-final tile: advance tile indices and prefetch next
                            if (tile_x_seq + 1 >= cfg_tile_num_w) begin
                                tile_x_seq <= 16'd0;
                                tile_y_seq <= tile_y_seq + 1;
                            end else begin
                                tile_x_seq <= tile_x_seq + 1;
                            end
                            // Fire prefetch for next tile
                            if (!skip_act_load) begin
                                dma_start      <= 1'b1;
                                dma_dir        <= 1'b0;
                                dma_ext_addr   <= next_tile_ddr_addr;
                                dma_sram_addr  <= next_sram_offset;
                                dma_xfer_len   <= tile_in_words;
                                prefetch_active  <= 1'b1;
                                prefetch_pending <= 1'b1;
                                db_prefetch_done <= 1'b0;
                                cur_tile_ddr_offset <= next_tile_ddr_addr - cfg_dma_in_addr;
                                `ifndef SYNTHESIS
                                $display("[PTS_PREFETCH] t=%0t ddr=0x%08x sram=%0d len=%0d offset=%0d",
                                         $time, next_tile_ddr_addr, next_sram_offset, tile_in_words,
                                         next_tile_ddr_addr - cfg_dma_in_addr);
                                `endif
                                next_tile_ddr_addr <= next_tile_ddr_addr
                                    + (db_en && cfg_tile_in_size != 0 ? cfg_tile_in_size : cfg_dma_in_size);
                                // Toggle target bank for next prefetch
                                next_sram_offset <= (next_sram_offset >= cfg_act_bank_offset) ?
                                                    16'd0 : cfg_act_bank_offset;
                            end else begin
                                db_prefetch_done <= 1'b1;
                            end
                            state <= S_WAIT_COMP;
                        end
                    end
                end

                // ─── DB_EN: Wait for prefetch DMA to complete ───
                S_WAIT_PREFETCH: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done && prefetch_active) begin
                        prefetch_active  <= 1'b0;
                        db_prefetch_done <= 1'b1;
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
                        // Conv2D: store from cfg_out_base; Add: store from cfg_act_base (in-place)
                        if (cfg_dma_add_b_addr != 0)
                            dma_sram_addr <= (db_en && store_bank ? cfg_act_bank_offset : 16'd0);
                        else
                            dma_sram_addr <= cfg_out_base + (db_en && store_bank ? cfg_act_bank_offset : 16'd0);
                        // For DB_EN Add: use tile_in_words (same as output size per tile)
                        dma_xfer_len  <= (db_en && cfg_dma_add_b_addr != 0) ? tile_in_words : out_words;
                        // 1D mode for final store (clear any 2D params)
                        dma_row_len   <= 16'd0;
                        dma_row_count <= 16'd0;
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
                    end else if (cfg_dma_add_b_addr != 0 && tile_in_words != 0) begin
                        dma_start     <= 1'b1;
                        dma_dir       <= 1'b0;  // load
                        dma_ext_addr  <= cfg_dma_add_b_addr + cur_tile_ddr_offset;
                        dma_sram_addr <= cfg_out_base + (db_en && ping_pong_flag ? cfg_act_bank_offset : 16'd0);
                        dma_xfer_len  <= tile_in_words;
                        state         <= S_WAIT_ADD_B;
                    end else begin
                        state <= S_LOAD_PARAM;
                    end
                end

                S_WAIT_ADD_B: begin
                    if (ctrl_abort) begin
                        state <= S_DONE;
                    end else if (dma_done) begin
                        if (add_b_reload) begin
                            // Prefetch reload (after tile_done): unblock compute
                            add_b_reload <= 1'b0;
                            db_prefetch_done <= 1'b1;
                            state <= S_WAIT_COMP;
                        end else begin
                            // First load (tile 0): proceed to param load
                            state <= S_LOAD_PARAM;
                        end
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
