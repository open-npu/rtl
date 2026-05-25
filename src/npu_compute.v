// Open-NPU RTL — Compute Micro-Sequencer
// SPDX-License-Identifier: Apache-2.0
//
// Orchestrates the systolic array (Conv2D/FC) or DW conv engine:
//   1. Reads weights from Weight SRAM, unpacks INT8, feeds to systolic
//   2. Reads activations from Act SRAM, unpacks INT8, streams to systolic
//   3. Drains accumulator results column-by-column
//   4. Reads per-channel params from Param SRAM, feeds PPU
//   5. Packs PPU INT8 outputs, writes back to Act SRAM
//   6. Loops over OC groups and spatial tiles
//
// Tiling loops (outer → inner):
//   tile_y → tile_x → oc_group
//
// Key simplification for V2 first-pass:
//   - k_depth <= ARRAY_SIZE (single weight-load + stream pass)
//   - Spatial output count is handled by repeated compute passes
//   - No partial-sum external accumulation yet
//
// Op types: 0=Conv2D, 1=DWConv, 2=FC

`include "npu_defines.vh"

module npu_compute #(
    parameter ARRAY_SIZE   = `ARRAY_SIZE,
    parameter ACT_ADDR_W   = 13,
    parameter WGT_ADDR_W   = 14,
    parameter PARAM_ADDR_W = 11,
    parameter DATA_W       = `DATA_WIDTH,
    parameter ACC_W        = `ACC_WIDTH
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // ─── Controller handshake ───
    input  wire                         start,
    output reg                          done,

    // ─── Layer configuration (from CSR) ───
    input  wire [7:0]                   cfg_op_type,
    input  wire [15:0]                  cfg_in_c,
    input  wire [15:0]                  cfg_out_h,
    input  wire [15:0]                  cfg_out_w,
    input  wire [15:0]                  cfg_out_c,
    input  wire [7:0]                   cfg_kernel_h,
    input  wire [7:0]                   cfg_kernel_w,
    input  wire [7:0]                   cfg_stride_h,
    input  wire [7:0]                   cfg_stride_w,
    input  wire [7:0]                   cfg_pad_top,
    input  wire [7:0]                   cfg_pad_left,
    input  wire [15:0]                  cfg_tile_h,
    input  wire [15:0]                  cfg_tile_w,
    input  wire [15:0]                  cfg_tile_num_h,
    input  wire [15:0]                  cfg_tile_num_w,
    input  wire [15:0]                  cfg_in_w,
    input  wire [15:0]                  cfg_in_h,

    // ─── Weight SRAM Port B (read-only) ───
    output reg                          wgt_rd_en,
    output reg  [WGT_ADDR_W-1:0]       wgt_rd_addr,
    input  wire [31:0]                  wgt_rd_data,

    // ─── Activation SRAM Port B (read + write) ───
    output reg                          act_rd_en,
    output reg  [ACT_ADDR_W-1:0]       act_rd_addr,
    input  wire [31:0]                  act_rd_data,
    output reg                          act_wr_en,
    output reg  [ACT_ADDR_W-1:0]       act_wr_addr,
    output reg  [31:0]                  act_wr_data,

    // ─── Parameter SRAM Port B (read-only) ───
    output reg                          param_rd_en,
    output reg  [PARAM_ADDR_W-1:0]     param_rd_addr,
    input  wire [31:0]                  param_rd_data,

    // ─── Systolic Array ───
    output reg  [1:0]                   sa_cmd,
    output reg                          sa_cmd_valid,
    output reg  signed [DATA_W-1:0]    sa_wgt_data  [0:ARRAY_SIZE-1],
    output reg                          sa_wgt_valid,
    output reg  signed [DATA_W-1:0]    sa_act_data  [0:ARRAY_SIZE-1],
    output reg                          sa_act_valid,
    output reg  [$clog2(ARRAY_SIZE)-1:0] sa_drain_col_sel,
    input  wire signed [ACC_W-1:0]     sa_acc_out   [0:ARRAY_SIZE-1],
    input  wire                         sa_acc_out_valid,
    input  wire                         sa_busy,
    input  wire                         sa_ready,

    // ─── DW Conv ───
    output reg                          dw_wgt_load,
    output reg                          dw_wgt_valid,
    output reg  signed [DATA_W-1:0]    dw_wgt_data,
    output reg                          dw_in_valid,
    output reg  signed [DATA_W-1:0]    dw_in_data,
    output reg                          dw_acc_clear,
    input  wire signed [ACC_W-1:0]     dw_acc_out,
    input  wire                         dw_out_valid,

    // ─── PPU ───
    output reg  signed [ACC_W-1:0]     ppu_acc_in,
    output reg                          ppu_in_valid,
    output reg  signed [ACC_W-1:0]     ppu_bias,
    output reg  [14:0]                  ppu_mult_m,
    output reg  [5:0]                   ppu_shift_s,
    output reg  signed [15:0]          ppu_zero_point,
    input  wire signed [DATA_W-1:0]    ppu_out_data,
    input  wire                         ppu_out_valid
);

    // ─── Systolic command encoding ───
    localparam MODE_IDLE     = 2'b00;
    localparam MODE_WGT_LOAD = 2'b01;
    localparam MODE_COMPUTE  = 2'b10;
    localparam MODE_DRAIN    = 2'b11;

    // ─── Derived constants ───
    localparam COL_W = $clog2(ARRAY_SIZE);
    localparam [$clog2(ARRAY_SIZE)-1:0] COL_MAX = ARRAY_SIZE - 1;  // last column index
    localparam [15:0] ARRAY_SIZE_16 = ARRAY_SIZE;  // 16-bit for comparisons

    // ─── FSM States ───
    localparam [4:0]
        S_IDLE        = 5'd0,
        S_TILE_SETUP  = 5'd1,
        S_OC_SETUP    = 5'd2,
        S_WGT_CMD     = 5'd3,
        S_WGT_LOAD    = 5'd4,   // Read+fill wgt_data for one column
        S_WGT_EMIT    = 5'd5,   // Pulse wgt_valid
        S_ACT_CMD     = 5'd6,
        S_ACT_LOAD    = 5'd7,   // Read activation word from SRAM
        S_ACT_EMIT    = 5'd8,   // Pulse act_valid with one byte
        S_ACT_FLUSH   = 5'd9,
        S_DRAIN_CMD   = 5'd10,
        S_DRAIN_WAIT  = 5'd11,
        S_PARAM_LOAD  = 5'd12,
        S_PPU_FEED    = 5'd13,
        S_PPU_WAIT    = 5'd14,
        S_WRITEBACK   = 5'd15,
        S_OC_NEXT     = 5'd16,
        S_TILE_NEXT   = 5'd17,
        S_DONE        = 5'd18,
        S_DW_WGT_LOAD = 5'd19,
        S_DW_COMPUTE  = 5'd20,
        S_DW_DRAIN    = 5'd21,
        S_DW_PARAM    = 5'd22,
        S_DW_PPU      = 5'd23;

    reg [4:0] state;

    // ─── Tile iteration ───
    reg [15:0] tile_y, tile_x;
    reg [15:0] oc_group;

    // ─── Latched config ───
    reg [15:0] oc_groups_total;
    reg [15:0] k_depth;         // kh * kw * in_c
    reg [15:0] out_tile_h, out_tile_w;

    // ─── Weight load state ───
    // Load one column at a time: read ceil(ARRAY_SIZE/4) words, fill sa_wgt_data
    reg [$clog2(ARRAY_SIZE)-1:0] wgt_col_idx;     // current column (0..ARRAY_SIZE-1)
    reg [$clog2(ARRAY_SIZE):0]   wgt_byte_idx;    // byte index within column (0..ARRAY_SIZE-1)
    reg [15:0]                   wgt_word_addr;    // current SRAM address
    reg                          wgt_read_issued;  // 1-cycle read latency tracker

    // ─── Activation stream state ───
    reg [15:0] act_cnt;         // activation byte counter (0..k_depth-1)
    reg [15:0] act_word_addr;   // current SRAM address
    reg        act_read_issued;
    reg [31:0] act_buf;         // buffered SRAM word
    reg [1:0]  act_byte_sel;    // byte position within word

    // ─── Drain state ───
    reg [$clog2(ARRAY_SIZE)-1:0] drain_col;

    // ─── PPU state ───
    reg [$clog2(ARRAY_SIZE):0] ppu_feed_cnt;  // how many acc values fed to PPU
    reg [15:0]                  ppu_wait_cnt;   // PPU flush wait counter
    reg signed [ACC_W-1:0]     acc_buf [0:ARRAY_SIZE-1];

    // ─── Param read state ───
    reg [2:0]  param_word_idx;
    reg [31:0] param_buf [0:3];
    reg        param_read_issued;

    // ─── Writeback state ───
    reg [$clog2(ARRAY_SIZE):0] wb_cnt;   // output bytes collected
    reg [31:0] wb_pack;                   // pack buffer
    reg [ACT_ADDR_W-1:0] wb_addr;

    // ─── Address bases ───
    reg [WGT_ADDR_W-1:0]   wgt_base;
    reg [ACT_ADDR_W-1:0]   act_base;
    reg [PARAM_ADDR_W-1:0] param_base;
    reg [ACT_ADDR_W-1:0]   out_base;

    // ─── DW state ───
    reg [15:0] dw_ch_idx;
    reg [5:0]  dw_cnt;
    reg        dw_read_issued;

    // ─── Flush counter (reused) ───
    reg [15:0] flush_cnt;

    integer i;

    // ════════════════════════════════════════════════════════════════════
    // Main FSM
    // ════════════════════════════════════════════════════════════════════

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            // All outputs idle
            sa_cmd       <= MODE_IDLE;
            sa_cmd_valid <= 1'b0;
            sa_wgt_valid <= 1'b0;
            sa_act_valid <= 1'b0;
            sa_drain_col_sel <= 0;
            wgt_rd_en   <= 1'b0;
            wgt_rd_addr <= 0;
            act_rd_en   <= 1'b0;
            act_rd_addr <= 0;
            act_wr_en   <= 1'b0;
            act_wr_addr <= 0;
            act_wr_data <= 32'd0;
            param_rd_en <= 1'b0;
            param_rd_addr <= 0;
            ppu_acc_in   <= 0;
            ppu_in_valid <= 1'b0;
            ppu_bias     <= 0;
            ppu_mult_m   <= 0;
            ppu_shift_s  <= 0;
            ppu_zero_point <= 0;
            dw_wgt_load  <= 1'b0;
            dw_wgt_valid <= 1'b0;
            dw_wgt_data  <= 0;
            dw_in_valid  <= 1'b0;
            dw_in_data   <= 0;
            dw_acc_clear <= 1'b0;
            // Internal state
            tile_y <= 0; tile_x <= 0; oc_group <= 0;
            oc_groups_total <= 1; k_depth <= 1;
            out_tile_h <= 1; out_tile_w <= 1;
            wgt_col_idx <= 0; wgt_byte_idx <= 0;
            wgt_word_addr <= 0; wgt_read_issued <= 0;
            act_cnt <= 0; act_word_addr <= 0;
            act_read_issued <= 0; act_buf <= 0; act_byte_sel <= 0;
            drain_col <= 0;
            ppu_feed_cnt <= 0; ppu_wait_cnt <= 0;
            param_word_idx <= 0; param_read_issued <= 0;
            wb_cnt <= 0; wb_pack <= 0; wb_addr <= 0;
            wgt_base <= 0; act_base <= 0; param_base <= 0; out_base <= 0;
            dw_ch_idx <= 0; dw_cnt <= 0; dw_read_issued <= 0;
            flush_cnt <= 0;
            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                sa_wgt_data[i] <= 0;
                sa_act_data[i] <= 0;
                acc_buf[i] <= 0;
            end
            for (i = 0; i < 4; i = i + 1)
                param_buf[i] <= 0;
        end else begin
            // ─── Default pulse deassertion ───
            done         <= 1'b0;
            sa_cmd_valid <= 1'b0;
            sa_wgt_valid <= 1'b0;
            sa_act_valid <= 1'b0;
            wgt_rd_en   <= 1'b0;
            act_rd_en   <= 1'b0;
            act_wr_en   <= 1'b0;
            param_rd_en <= 1'b0;
            ppu_in_valid <= 1'b0;
            dw_wgt_valid <= 1'b0;
            dw_in_valid  <= 1'b0;
            dw_acc_clear <= 1'b0;

            case (state)

            // ══════════════════════════════════════════════════════════════
            S_IDLE: begin
                if (start) begin
                    if (cfg_op_type == 8'd1)
                        oc_groups_total <= cfg_out_c;
                    else
                        oc_groups_total <= (cfg_out_c + ARRAY_SIZE_16 - 1) / ARRAY_SIZE_16;

                    k_depth <= {8'd0, cfg_kernel_h} * {8'd0, cfg_kernel_w} * cfg_in_c;

                    if (cfg_tile_h == 16'd0) begin
                        out_tile_h <= cfg_out_h;
                        out_tile_w <= cfg_out_w;
                    end else begin
                        out_tile_h <= cfg_tile_h;
                        out_tile_w <= cfg_tile_w;
                    end

                    tile_y <= 0;
                    tile_x <= 0;
                    state  <= S_TILE_SETUP;
                end
            end

            // ══════════════════════════════════════════════════════════════
            S_TILE_SETUP: begin
                // Compute base addresses
                wgt_base <= 0;  // Weight base fixed (all OC weights from start)
                param_base <= 0;
                // Input activation base = tile_y * tile_h * stride_h * in_w * in_c / 4
                //                        + tile_x * tile_w * stride_w * in_c / 4
                act_base <= (tile_y * out_tile_h * cfg_stride_h[ACT_ADDR_W-1:0]
                            * cfg_in_w[ACT_ADDR_W-1:0] * cfg_in_c[ACT_ADDR_W-1:0]
                           + tile_x * out_tile_w * cfg_stride_w[ACT_ADDR_W-1:0]
                            * cfg_in_c[ACT_ADDR_W-1:0]) >> 2;
                // Output base
                out_base <= (tile_y * out_tile_h * cfg_out_w[ACT_ADDR_W-1:0]
                            * cfg_out_c[ACT_ADDR_W-1:0]
                           + tile_x * out_tile_w * cfg_out_c[ACT_ADDR_W-1:0]) >> 2;

                oc_group <= 0;

                if (cfg_op_type == 8'd1)
                    state <= S_DW_WGT_LOAD;
                else
                    state <= S_OC_SETUP;
            end

            // ══════════════════════════════════════════════════════════════
            S_OC_SETUP: begin
                // Weight base for this OC group:
                //   oc_group * ARRAY_SIZE channels * k_depth bytes / 4 bytes_per_word
                wgt_base <= (oc_group * ARRAY_SIZE_16 * k_depth) >> 2;
                // Param base: 4 words per channel, ARRAY_SIZE channels per group
                param_base <= oc_group * ARRAY_SIZE_16 * 4;

                wgt_col_idx <= 0;
                state <= S_WGT_CMD;
            end

            // ══════════════════════════════════════════════════════════════
            // WEIGHT LOAD: load ARRAY_SIZE columns, one per wgt_valid pulse
            // ══════════════════════════════════════════════════════════════
            S_WGT_CMD: begin
                sa_cmd       <= MODE_WGT_LOAD;
                sa_cmd_valid <= 1'b1;
                // Begin loading column 0
                wgt_byte_idx    <= 0;
                wgt_read_issued <= 1'b0;
                // Address for column wgt_col_idx, byte 0:
                //   wgt_base + wgt_col_idx * ceil(k_depth/4)
                // But actually weights are in OHWI order contiguously:
                //   column c -> channel (oc_group*ARRAY_SIZE + c) weights at
                //   offset c * k_depth bytes from wgt_base
                wgt_word_addr <= wgt_base + (wgt_col_idx * k_depth[WGT_ADDR_W-1:0]) / 4;
                state <= S_WGT_LOAD;
            end

            S_WGT_LOAD: begin
                // Fill sa_wgt_data[0..ARRAY_SIZE-1] for current column
                // Read SRAM words (4 bytes each) and unpack
                if (!wgt_read_issued) begin
                    // Issue SRAM read
                    wgt_rd_en   <= 1'b1;
                    wgt_rd_addr <= wgt_word_addr[WGT_ADDR_W-1:0];
                    wgt_read_issued <= 1'b1;
                end else begin
                    // Data available from wgt_rd_data (1-cycle latency)
                    // Unpack up to 4 bytes
                    if (wgt_byte_idx < ARRAY_SIZE) begin
                        sa_wgt_data[wgt_byte_idx[COL_W-1:0]] <=
                            $signed(wgt_rd_data[7:0]);
                    end
                    if (wgt_byte_idx + 1 < ARRAY_SIZE) begin
                        sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 1] <=
                            $signed(wgt_rd_data[15:8]);
                    end
                    if (wgt_byte_idx + 2 < ARRAY_SIZE) begin
                        sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 2] <=
                            $signed(wgt_rd_data[23:16]);
                    end
                    if (wgt_byte_idx + 3 < ARRAY_SIZE) begin
                        sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 3] <=
                            $signed(wgt_rd_data[31:24]);
                    end

                    wgt_byte_idx <= wgt_byte_idx + 4;

                    if (wgt_byte_idx + 4 >= ARRAY_SIZE) begin
                        // All bytes for this column loaded → emit
                        state <= S_WGT_EMIT;
                    end else begin
                        // Need more words
                        wgt_word_addr <= wgt_word_addr + 1;
                        wgt_read_issued <= 1'b0;
                    end
                end
            end

            S_WGT_EMIT: begin
                // Pulse wgt_valid for this column
                sa_wgt_valid <= 1'b1;

                if (wgt_col_idx == COL_MAX) begin
                    // All columns loaded → activate streaming
                    state <= S_ACT_CMD;
                end else begin
                    // Next column
                    wgt_col_idx <= wgt_col_idx + 1;
                    wgt_byte_idx <= 0;
                    wgt_read_issued <= 1'b0;
                    wgt_word_addr <= wgt_base +
                        ((wgt_col_idx + 1) * k_depth[WGT_ADDR_W-1:0]) / 4;
                    state <= S_WGT_LOAD;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // ACTIVATION STREAM: broadcast k_depth values to all rows
            // ══════════════════════════════════════════════════════════════
            S_ACT_CMD: begin
                // Wait for systolic to be ready before issuing COMPUTE
                if (sa_ready) begin
                    sa_cmd       <= MODE_COMPUTE;
                    sa_cmd_valid <= 1'b1;
                    act_cnt      <= 0;
                    act_byte_sel <= 2'd0;
                    act_read_issued <= 1'b0;
                    act_word_addr <= act_base[15:0];
                    state <= S_ACT_LOAD;
                end
            end

            S_ACT_LOAD: begin
                // Read one SRAM word (4 activation bytes)
                if (!act_read_issued) begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= act_word_addr[ACT_ADDR_W-1:0];
                    act_read_issued <= 1'b1;
                end else begin
                    // Data available
                    act_buf <= act_rd_data;
                    act_byte_sel <= 2'd0;
                    state <= S_ACT_EMIT;
                end
            end

            S_ACT_EMIT: begin
                // Broadcast one byte to all rows
                begin : act_emit_blk
                    reg signed [DATA_W-1:0] abyte;
                    case (act_byte_sel)
                        2'd0: abyte = $signed(act_buf[7:0]);
                        2'd1: abyte = $signed(act_buf[15:8]);
                        2'd2: abyte = $signed(act_buf[23:16]);
                        2'd3: abyte = $signed(act_buf[31:24]);
                        default: abyte = 0;
                    endcase
                    for (i = 0; i < ARRAY_SIZE; i = i + 1)
                        sa_act_data[i] <= abyte;
                end
                sa_act_valid <= 1'b1;
                act_cnt <= act_cnt + 1;

                if (act_cnt + 1 >= k_depth) begin
                    // All K elements streamed → flush
                    flush_cnt <= 0;
                    state <= S_ACT_FLUSH;
                end else if (act_byte_sel == 2'd3) begin
                    // Need next SRAM word
                    act_byte_sel <= 2'd0;
                    act_word_addr <= act_word_addr + 1;
                    act_read_issued <= 1'b0;
                    state <= S_ACT_LOAD;
                end else begin
                    act_byte_sel <= act_byte_sel + 1;
                end
            end

            S_ACT_FLUSH: begin
                // Wait ARRAY_SIZE-1 cycles for pipeline propagation
                flush_cnt <= flush_cnt + 1;
                if (flush_cnt >= ARRAY_SIZE_16 - 1) begin
                    drain_col <= 0;
                    state <= S_DRAIN_CMD;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // DRAIN: iterate columns, each takes 2 cycles (DRAIN + DRAIN_OUT)
            // ══════════════════════════════════════════════════════════════
            S_DRAIN_CMD: begin
                // Systolic accepts DRAIN from S_COMPUTE or S_READY
                sa_cmd           <= MODE_DRAIN;
                sa_cmd_valid     <= 1'b1;
                sa_drain_col_sel <= drain_col;
                state            <= S_DRAIN_WAIT;
            end

            S_DRAIN_WAIT: begin
                if (sa_acc_out_valid) begin
                    // Capture results
                    for (i = 0; i < ARRAY_SIZE; i = i + 1)
                        acc_buf[i] <= sa_acc_out[i];
                    // Read params for this channel
                    param_word_idx <= 0;
                    param_read_issued <= 1'b0;
                    state <= S_PARAM_LOAD;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // PARAM READ: 4 words per channel (14 bytes padded to 16)
            // ══════════════════════════════════════════════════════════════
            S_PARAM_LOAD: begin
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= param_base + drain_col * 4 + param_word_idx;
                    param_read_issued <= 1'b1;
                end else begin
                    param_buf[param_word_idx] <= param_rd_data;
                    if (param_word_idx == 3'd3) begin
                        // Extract params and setup PPU
                        // Word 0: [M_lo(8), M_hi(7+1pad), S(6+2pad), pad(8)]
                        //   M = param_buf[0][14:0], S = param_buf[0][21:16]
                        // Word 1: [zp_lo(8), zp_hi(8), bias_0(8), bias_1(8)]
                        //   zp = param_buf[1][15:0]
                        // Word 2: [bias_2..bias_5]
                        // Word 3: [bias_6, bias_7, pad, pad]
                        //   bias = {word3[15:0], word2, word1[31:16]} = 48 bits (lower of 64-bit)
                        ppu_mult_m     <= param_buf[0][14:0];
                        ppu_shift_s    <= param_buf[0][21:16];
                        ppu_zero_point <= $signed(param_buf[1][15:0]);
                        ppu_bias       <= $signed({param_buf[3][15:0], param_buf[2],
                                                   param_buf[1][31:16]});
                        ppu_feed_cnt <= 0;
                        state <= S_PPU_FEED;
                    end else begin
                        param_word_idx <= param_word_idx + 1;
                        param_read_issued <= 1'b0;
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            // PPU FEED: send ARRAY_SIZE acc values one per cycle
            // ══════════════════════════════════════════════════════════════
            S_PPU_FEED: begin
                ppu_acc_in   <= acc_buf[ppu_feed_cnt[$clog2(ARRAY_SIZE)-1:0]];
                ppu_in_valid <= 1'b1;
                ppu_feed_cnt <= ppu_feed_cnt + 1;

                if (ppu_feed_cnt + 1 >= ARRAY_SIZE) begin
                    wb_cnt  <= 0;
                    wb_pack <= 0;
                    // Compute writeback address
                    wb_addr <= out_base + (oc_group * ARRAY_SIZE_16 +
                               drain_col) / 4;
                    // Go directly to writeback — outputs arrive after 4-cycle PPU latency
                    state <= S_WRITEBACK;
                end
            end

            S_PPU_WAIT: begin
                // Unused — kept for state encoding compatibility
                state <= S_WRITEBACK;
            end

            // ══════════════════════════════════════════════════════════════
            // WRITEBACK: collect PPU INT8 outputs as they arrive
            // ══════════════════════════════════════════════════════════════
            S_WRITEBACK: begin
                if (ppu_out_valid) begin
                    case (wb_cnt[1:0])
                        2'd0: wb_pack[7:0]   <= ppu_out_data;
                        2'd1: wb_pack[15:8]  <= ppu_out_data;
                        2'd2: wb_pack[23:16] <= ppu_out_data;
                        2'd3: begin
                            // Write full word
                            act_wr_en   <= 1'b1;
                            act_wr_addr <= wb_addr;
                            act_wr_data <= {ppu_out_data, wb_pack[23:0]};
                            wb_addr     <= wb_addr + 1;
                        end
                    endcase
                    wb_cnt <= wb_cnt + 1;
                end

                // All outputs collected?
                if (wb_cnt >= ARRAY_SIZE) begin
                    // Flush remaining partial word if needed
                    if (wb_cnt[1:0] != 2'd0) begin
                        act_wr_en   <= 1'b1;
                        act_wr_addr <= wb_addr;
                        act_wr_data <= wb_pack;
                    end

                    // Next drain column or next OC group
                    if (drain_col == COL_MAX) begin
                        state <= S_OC_NEXT;
                    end else begin
                        drain_col <= drain_col + 1;
                        state <= S_DRAIN_CMD;
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            S_OC_NEXT: begin
                if (oc_group + 1 >= oc_groups_total) begin
                    state <= S_TILE_NEXT;
                end else begin
                    oc_group <= oc_group + 1;
                    state <= S_OC_SETUP;
                end
            end

            S_TILE_NEXT: begin
                if (cfg_tile_h == 16'd0) begin
                    state <= S_DONE;
                end else if (tile_x + 1 >= cfg_tile_num_w) begin
                    if (tile_y + 1 >= cfg_tile_num_h) begin
                        state <= S_DONE;
                    end else begin
                        tile_x <= 0;
                        tile_y <= tile_y + 1;
                        state <= S_TILE_SETUP;
                    end
                end else begin
                    tile_x <= tile_x + 1;
                    state <= S_TILE_SETUP;
                end
            end

            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            // ══════════════════════════════════════════════════════════════
            // DW Conv Path (placeholder — basic structure)
            // ══════════════════════════════════════════════════════════════
            S_DW_WGT_LOAD: begin
                dw_wgt_load  <= 1'b1;
                dw_acc_clear <= 1'b1;
                dw_ch_idx    <= oc_group[15:0];
                dw_cnt       <= 0;
                dw_read_issued <= 1'b0;
                state <= S_DW_COMPUTE;
            end

            S_DW_COMPUTE: begin
                dw_wgt_load <= 1'b0;
                // Simplified: just signal done for now (DW path needs more work)
                // TODO: implement proper DW weight load + compute streaming
                state <= S_OC_NEXT;
            end

            S_DW_DRAIN: state <= S_DW_PARAM;
            S_DW_PARAM: state <= S_DW_PPU;
            S_DW_PPU:   state <= S_OC_NEXT;

            default: state <= S_IDLE;

            endcase
        end
    end

endmodule
