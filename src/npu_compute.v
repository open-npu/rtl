// Open-NPU RTL — Compute Micro-Sequencer
// SPDX-License-Identifier: Apache-2.0
//
// Orchestrates the systolic array (Conv2D/FC) or DW conv engine:
//   1. Reads weights from Weight SRAM, unpacks INT8, feeds to systolic
//   2. Reads activations from Act SRAM, unpacks INT8, streams to systolic
//   3. Collects the array's reduced psum vector (no drain phase)
//   4. Reads per-channel params from Param SRAM, feeds PPU
//   5. Packs PPU INT8 outputs, writes back to Act SRAM
//   6. Loops over OC groups and spatial tiles
//
// Tiling loops (outer → inner):
//   tile_y → tile_x → oc_group
//
// Key simplification for V2 first-pass:
//   - k_depth > ARRAY_SIZE supported via multi-pass partial sum accumulation
//   - Spatial output count is handled by repeated compute passes
//   - Weights reused across a 16-pixel block; activation vectors are packed
//     from the 256-bit compute SRAM beat (up to 32 INT8 / 16 INT16 lanes)
//     and up to ARRAY_SIZE vectors may be in flight (array latency = ROWS).
//
// Op types: 0=Conv2D, 1=DWConv, 2=FC, 3=Pooling, 4=Add, 5=Resize, 6=Deconv, 7=Concat

`include "npu_defines.vh"

module npu_compute #(
    parameter ARRAY_SIZE   = `ARRAY_SIZE,
    parameter ACT_ADDR_W   = 13,  // $clog2(SPAD_KB*64) for SPAD_KB=128 → $clog2(8192)=13
    parameter WGT_ADDR_W   = $clog2(`SPAD_KB * 128),  // $clog2(WGT_DEPTH)
    parameter PARAM_ADDR_W = 11,  // $clog2(SPAD_KB*16) for SPAD_KB=128 → $clog2(2048)=11
    parameter DATA_W       = `DATA_WIDTH,
    parameter ACC_W        = `ACC_WIDTH
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // ─── Controller handshake ───
    input  wire                         start,
    output reg                          done,
    output wire                         tile_done,  // 1-cycle pulse at non-final tile boundary
    output reg                          oc_group_done, // 1-cycle pulse: oc_group finished, request weight reload
    output wire [15:0]                  oc_group_out,   // Current oc_group index (for controller weight reload)
    input  wire                         wgt_reload_done, // Controller loaded next oc_group's weights
    input  wire [31:0]                  cfg_wgt_per_oc,  // Per-oc weight words (0=all weights fit, skip reload)
    input  wire                         db_prefetch_done,  // DB_EN: prefetch complete, safe to start next tile
    // Per-tile store support: actual (clipped) tile output dimensions
    output wire [15:0]                  tile_out_h_actual, // Current tile's actual output height (clipped at border)
    output wire [15:0]                  tile_out_w_actual, // Current tile's actual output width (clipped at border)

    // ─── Layer configuration (from CSR) ───
    input  wire [7:0]                   cfg_op_type,
    input  wire                         cfg_int16,      // 1=INT16 mode, 0=INT8 mode
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
    input  wire signed [15:0]           cfg_in_zp,      // Input zero-point for padding
    input  wire [ACT_ADDR_W-1:0]        cfg_act_base,   // Input activation base (word addr)
    input  wire [ACT_ADDR_W-1:0]        cfg_out_base,   // Output base (word addr in act SRAM)
    input  wire [31:0]                  cfg_pool_cfg,   // Pooling config register
    input  wire [31:0]                  cfg_resize_cfg, // Resize config register
    input  wire [31:0]                  cfg_deconv_cfg, // Deconv config: [7:0]=INSERT_H, [15:8]=INSERT_W
    input  wire [31:0]                  cfg_concat_cfg, // Concat config: [15:0]=OFFSET, [31:16]=TOTAL_C
    input  wire                         cfg_2d_load,    // 2D DMA load mode (chain: SRAM row 0 = input row 0, not padding)

    // ─── Weight SRAM Port B (read-only, 256-bit beat) ───
    output reg                          wgt_rd_en,
    output reg  [WGT_ADDR_W-1:0]       wgt_rd_addr,
    input  wire [`SRAM_B_WIDTH-1:0]     wgt_rd_data,

    // ─── Activation SRAM Port B (256-bit IFM read + 32-bit OFM write) ───
    output reg                          act_rd_en,
    output reg  [ACT_ADDR_W-1:0]       act_rd_addr,
    input  wire [`SRAM_B_WIDTH-1:0]     act_rd_data,
    output reg                          act_rd_ofm,    // 1 = RMW read from OFM, 0 = IFM
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
    output wire [DATA_W*ARRAY_SIZE-1:0] sa_wgt_data_flat,
    output reg                          sa_wgt_valid,
    output wire [DATA_W*ARRAY_SIZE-1:0] sa_act_data_flat,
    output reg                          sa_act_valid,
    input  wire [ACC_W*ARRAY_SIZE-1:0]  sa_psum_out_flat,
    input  wire                         sa_psum_out_valid,
    input  wire                         sa_busy,
    input  wire                         sa_ready,

    // ─── DW Conv (scalar + ARRAY_SIZE-wide) ───
    output reg                          dw_wgt_load,
    output reg                          dw_wgt_valid,
    output reg  signed [DATA_W-1:0]    dw_wgt_data,
    output reg                          dw_in_valid,
    output reg  signed [DATA_W-1:0]    dw_in_data,
    output reg                          dw_acc_clear,
    input  wire signed [ACC_W-1:0]     dw_acc_out,
    input  wire                         dw_out_valid,
    output reg  [ARRAY_SIZE-1:0]       dw_wgt_valid_w,
    output reg  [DATA_W*ARRAY_SIZE-1:0] dw_wgt_data_w,
    output reg  [ARRAY_SIZE-1:0]       dw_in_valid_w,
    output reg  [DATA_W*ARRAY_SIZE-1:0] dw_in_data_w,
    input  wire [ACC_W*ARRAY_SIZE-1:0] dw_acc_w,
    input  wire [ARRAY_SIZE-1:0]       dw_out_valid_w,

    // ─── PPU (scalar lane 0 + ARRAY_SIZE-wide) ───
    output reg  signed [ACC_W-1:0]     ppu_acc_in /* verilator public */,
    output reg                          ppu_in_valid,
    output reg  signed [ACC_W-1:0]     ppu_bias,
    output reg  [14:0]                  ppu_mult_m,
    output reg  [5:0]                   ppu_shift_s,
    output reg  signed [15:0]          ppu_zero_point,
    input  wire signed [DATA_W-1:0]    ppu_out_data,
    input  wire                         ppu_out_valid,
    output reg  [ACC_W*ARRAY_SIZE-1:0] ppu_acc_w,
    output reg  [ARRAY_SIZE-1:0]       ppu_valid_w,
    output reg  [ACC_W*ARRAY_SIZE-1:0] ppu_bias_w,
    output reg  [15*ARRAY_SIZE-1:0]    ppu_mult_w,
    output reg  [6*ARRAY_SIZE-1:0]     ppu_shift_w,
    output reg  [16*ARRAY_SIZE-1:0]    ppu_zp_w,
    input  wire [DATA_W*ARRAY_SIZE-1:0] ppu_out_w,
    input  wire [ARRAY_SIZE-1:0]       ppu_vout_w
);

    // ─── Internal unpacked arrays for systolic interface ───
    reg  signed [DATA_W-1:0] sa_wgt_data [0:ARRAY_SIZE-1];
    reg  signed [DATA_W-1:0] sa_act_data [0:ARRAY_SIZE-1];
    wire signed [ACC_W-1:0]  sa_psum_out [0:ARRAY_SIZE-1];
    genvar gi;
    generate
        for (gi = 0; gi < ARRAY_SIZE; gi = gi + 1) begin : unpack_sa
            assign sa_wgt_data_flat[DATA_W*gi +: DATA_W] = sa_wgt_data[gi];
            assign sa_act_data_flat[DATA_W*gi +: DATA_W] = sa_act_data[gi];
            assign sa_psum_out[gi] = sa_psum_out_flat[ACC_W*gi +: ACC_W];
        end
    endgenerate

    // ─── Systolic command encoding ───
    localparam MODE_IDLE     = 2'b00;
    localparam MODE_WGT_LOAD = 2'b01;
    localparam MODE_COMPUTE  = 2'b10;
    localparam MODE_DRAIN    = 2'b11;

    // ─── Derived constants ───
    localparam COL_W = $clog2(ARRAY_SIZE);
    localparam [$clog2(ARRAY_SIZE)-1:0] COL_MAX = ARRAY_SIZE - 1;  // last column index
    localparam [15:0] ARRAY_SIZE_16 = ARRAY_SIZE;  // 16-bit for comparisons

    // Cycles spent in S_ACT_FLUSH after the last activation is pushed.
    // npu_systolic broadcasts act to all columns, so this is a fixed pipeline
    // drain, not a function of ARRAY_SIZE. Two cycles: one for the trailing
    // registered sa_act_valid, one for the PE accumulator update.
    localparam [15:0] ACT_FLUSH_CYCLES = 16'd2;

    // ─── Reciprocal LUT for AvgPool division (supported counts through 64) ───
    // Q32 fixed-point: result = (dividend * recip) >>> 32
    // pool_count=1 is bypassed; count=2 uses 0x40000000 with a 31-bit shift.
    function [31:0] recip_pool;
        input [6:0] idx;
        case (idx)
            7'd1:  recip_pool = 32'h4000_0000;  // 1/1 identity (unused, count=1 special-cased)
            7'd2:  recip_pool = 32'h4000_0000;  // 1/2 with >>31 (avoid 0x80000000 signed)
            7'd3:  recip_pool = 32'h5555_5556;
            7'd4:  recip_pool = 32'h4000_0000;
            7'd5:  recip_pool = 32'h3333_3333;
            7'd6:  recip_pool = 32'h2AAA_AAAB;
            7'd7:  recip_pool = 32'h2492_4925;
            7'd8:  recip_pool = 32'h2000_0000;
            7'd9:  recip_pool = 32'h1C71_C71C;
            7'd10: recip_pool = 32'h1999_999A;
            7'd11: recip_pool = 32'h1745_D174;
            7'd12: recip_pool = 32'h1555_5555;
            7'd13: recip_pool = 32'h13B1_3B14;
            7'd14: recip_pool = 32'h1249_2492;
            7'd15: recip_pool = 32'h1111_1111;
            7'd16: recip_pool = 32'h1000_0000;
            7'd25: recip_pool = 32'h0A3D_70A4;  // 1/25
            7'd36: recip_pool = 32'h071C_71C7;  // 1/36
            7'd49: recip_pool = 32'h0540_5405;  // 1/49 (7x7 global pool)
            7'd64: recip_pool = 32'h0400_0000;  // 1/64 (8x8 global pool)
            default: recip_pool = 32'h0100_0000;  // fallback: 1/256 (avoid div-by-zero)
        endcase
    endfunction

    // ─── FSM States ───
    localparam [6:0]
        S_IDLE        = 7'd0,
        S_TILE_SETUP  = 7'd1,
        S_OC_SETUP    = 7'd2,
        S_WGT_CMD     = 7'd3,
        S_WGT_LOAD    = 7'd4,   // Read+fill wgt_data for one column
        S_WGT_EMIT    = 7'd5,   // Pulse wgt_valid
        S_ACT_CMD     = 7'd6,
        S_ACT_LOAD    = 7'd7,   // Read activation word from SRAM
        S_ACT_EMIT    = 7'd8,   // Pulse act_valid with one byte
        S_ACT_FLUSH   = 7'd9,
        S_PSUM_COLLECT= 7'd11,  // was S_DRAIN_WAIT; 7'd10 (S_DRAIN_CMD) retired
        S_PARAM_LOAD  = 7'd12,
        S_PPU_FEED    = 7'd13,
        S_PPU_WAIT    = 7'd14,
        S_WRITEBACK   = 7'd15,
        S_OC_NEXT     = 7'd16,
        S_TILE_NEXT   = 7'd17,
        S_DONE        = 7'd18,
        S_DW_WGT_LOAD = 7'd19,
        S_DW_COMPUTE  = 7'd20,
        S_DW_DRAIN    = 7'd21,
        S_DW_PARAM    = 7'd22,
        S_DW_PPU      = 7'd23,
        S_DW_ACT_STREAM = 7'd24,
        S_DW_PPU_WAIT   = 7'd25,
        S_DW_WB         = 7'd26,
        S_SPATIAL_SETUP = 7'd27,
        S_PIXEL_NEXT    = 7'd29,
        // Pooling states
        S_POOL_SETUP    = 7'd30,
        S_POOL_CH_SETUP = 7'd31,
        S_POOL_READ     = 7'd32,
        S_POOL_ACC      = 7'd33,
        S_POOL_DIV      = 7'd34,
        S_POOL_PPU      = 7'd35,
        S_POOL_PPU_WAIT = 7'd36,
        S_POOL_WB       = 7'd37,
        S_POOL_PIX_NEXT = 7'd38,
        S_POOL_CH_NEXT  = 7'd39,
        // Eltwise Add states
        S_ADD_SETUP     = 7'd40,
        S_ADD_PARAM     = 7'd41,
        S_ADD_READ_A    = 7'd42,
        S_ADD_READ_B    = 7'd43,
        S_ADD_COMPUTE   = 7'd44,
        S_ADD_PPU       = 7'd45,
        S_ADD_PPU_WAIT  = 7'd46,
        S_ADD_WB        = 7'd47,
        S_ADD_NEXT      = 7'd48,
        // Resize states
        S_RESIZE_SETUP    = 7'd49,
        S_RESIZE_CH_SETUP = 7'd50,
        S_RESIZE_COORD    = 7'd51,
        S_RESIZE_READ0    = 7'd52,
        S_RESIZE_READ1    = 7'd53,
        S_RESIZE_READ2    = 7'd54,
        S_RESIZE_READ3    = 7'd55,
        S_RESIZE_INTERP   = 7'd56,
        S_RESIZE_PPU      = 7'd57,
        S_RESIZE_PPU_WAIT = 7'd58,
        S_RESIZE_WB       = 7'd59,
        S_RESIZE_PIX_NEXT = 7'd60,
        S_RESIZE_CH_NEXT  = 7'd61,
        S_TILE_WAIT_DB    = 7'd62, // Wait for DB_EN prefetch before next tile
        S_WAIT_WGT_RELOAD = 7'd63, // Wait for controller to reload next oc_group weights
        S_RESIZE_INTERP1  = 7'd64, // Bilinear interp cycle 1 (4 mults: top/bot)
        S_RESIZE_INTERP2  = 7'd65, // Bilinear interp cycle 2 (2 mults: val64)
        S_PARAM_CACHE     = 7'd66, // Burst-load per-group PPU params into cache
        S_PPU_STREAM      = 7'd67; // Streamed PPU+WB for one 16-px block
    (* fsm_encoding = "one_hot" *)
    reg [6:0] state;

    // ─── Tile iteration ───
    reg [15:0] tile_y, tile_x;
    reg        tile_done_r;
    reg        tile_wait_delay;  // DB_EN wait: skip 1 cycle, then wait for prefetch
    assign tile_done = tile_done_r;
    // Pool stream iterates pool_ch (not oc_group) — report the pool group
    // base so the controller reloads the right slice.
    assign oc_group_out = pool_stream ? pool_grp_base : oc_group;
    // Expose actual (border-clipped) tile output dims for per-tile store DMA
    assign tile_out_h_actual = out_tile_h;
    assign tile_out_w_actual = out_tile_w;
    reg [15:0] oc_group;

    // ─── Latched config ───
    reg [15:0] oc_groups_total;
    reg [15:0] k_depth;         // kh * kw * in_c
    reg [15:0] out_tile_h, out_tile_w;

    // ─── Reciprocal registers for Conv k_pass decomposition ───
    reg [31:0] recip_kw_x_inc;  // Q32 reciprocal of kw*in_c
    reg [31:0] recip_in_c;      // Q32 reciprocal of in_c
    reg [15:0] kw_x_inc_r;     // latched kw*in_c
    reg [15:0] in_c_r;         // latched in_c

    // ─── Weight load state ───
    // Load one column at a time: one 256-bit beat fills a 16-lane INT8 column
    // (INT16 unaligned may need a second beat).
    localparam SRAM_B_W     = `SRAM_B_WIDTH;
    localparam SRAM_B_WORDS = `SRAM_B_WORDS;
    reg [$clog2(ARRAY_SIZE)-1:0] wgt_col_idx;     // current column (0..ARRAY_SIZE-1)
    reg [$clog2(ARRAY_SIZE):0]   wgt_byte_idx;    // byte index within column (0..ARRAY_SIZE-1)
    reg [15:0]                   wgt_word_addr;    // current SRAM address
    reg                          wgt_read_issued;  // 1-cycle read latency tracker
    reg                          wgt_data_ready;   // 2nd cycle: SRAM data available
    reg [1:0]                    wgt_bsel;         // byte offset within first word of the beat

    // ─── Activation stream state ───
    reg [15:0] act_cnt;         // activation byte counter (0..k_depth-1)
    reg [15:0] act_word_addr;   // current SRAM address
    reg        act_read_issued;
    reg        act_data_ready;  // 2nd cycle: SRAM data available
    reg        act_use_rd;      // 1×1: EMIT consumes held act_rd_data
    reg        act_ahead;       // next 1×1 pixel already issued (2-cycle SRAM)
    reg [4:0]  ahead_bsel;
    reg        ahead_pad;
    reg [SRAM_B_W-1:0] act_buf; // buffered 256-bit SRAM beat
    reg [4:0]  act_byte_sel;    // byte position within the 256-bit beat

    // ─── Drain state ───
    // 4-bit so {px[3:0], col[3:0]} indexing is defined for ARRAY_SIZE < 16.
    reg [3:0] drain_col;
    reg [3:0] tree_col;   // column captured, being tree-reduced
    // Act word prefetch age: 0=invalid, 1=addr issued this/last cycle,
    // 2=data valid in act_rd_data. Saturating up-counter.
    reg [1:0]    pf_age;

    // In-flight psum FIFO. The array emits ARRAY_SIZE cycles after sa_act_valid,
    // so depth ARRAY_SIZE lets S_ACT_EMIT issue one vector/cycle. Each slot
    // remembers which pixel / k_pass the returning psum belongs to.
    localparam IF_DEPTH = ARRAY_SIZE;
    localparam IF_CNT_W = $clog2(ARRAY_SIZE + 1);
    localparam [IF_CNT_W-1:0] IF_DEPTH_CNT = ARRAY_SIZE;
    reg [4:0]    if_px [0:IF_DEPTH-1];
    reg [15:0]   if_kp [0:IF_DEPTH-1];
    reg [COL_W-1:0] if_wr, if_rd;
    reg [IF_CNT_W-1:0] if_count;
    wire if_collect = sa_psum_out_valid && (if_count != {IF_CNT_W{1'b0}});
    wire if_full    = (if_count == IF_DEPTH_CNT);
    wire if_empty   = (if_count == {IF_CNT_W{1'b0}});

    // ─── A: param cache + streamed PPU/WB (per oc_group params are
    // pixel-invariant — load once per group, then feed PPU back-to-back)
    reg [31:0]   param_cache [0:63];   // 16 ch × 4 words
    reg [6:0]    cache_issue, cache_cap;
    reg          feed_left;            // PPU stream feed side active
    reg [1:0]    ppu_st;               // 0=feed 1=wait 2=wb
    reg signed [DATA_W-1:0] ppu_lat [0:15];
    reg signed [DATA_W-1:0] ppu_lat2 [0:15]; // 2nd queued PPU result
    reg [4:0]    ppu_wb_left;
    reg [4:0]    ppu_wb_idx;
    reg          act_armed;            // 1 after first ACT_CMD of this k_pass
    reg          ppu_ovl;              // PPU/WB running overlapped with next-block WGT
    reg          ppu_bg;               // PPU stream runs while state is WGT/ACT
    reg [4:0]    ppu_blk_save;         // blk_px_cnt latched at PPU start (PIXEL_NEXT overwrites)
    reg [15:0]   ppu_oc_save;          // oc_group latched at PPU start (OC may advance)
    reg [$clog2(ARRAY_SIZE)-1:0] ppu_col_last;
    reg          ppu_ahead;            // q0 (ppu_lat) holds a completed pixel
    reg          ppu_pend;             // q1 (ppu_lat2) holds a second result
    reg          param_pend;           // next-OC param fill waits until PPU feed done
    // Next-k_pass weight shadow (col-major: col*ARRAY_SIZE + row)
    reg signed [DATA_W-1:0] wgt_sh [0:255];
    reg          wgt_sh_ok;
    reg [15:0]   wgt_sh_pass;
    reg [4:0]    wgt_pf_col;
    reg [1:0]    wgt_pf_ph;
    reg          wgt_from_sh;
    reg          wgt_held;             // PE array still holds wgt_held_pass
    reg [15:0]   wgt_held_pass;
    reg [15:0]   k_first;              // first k_pass of this spatial block
    reg [15:0]   k_fin_cnt;            // completed k_passes in this block
    reg [4:0]    wb_px;                // WB-side pixel within block
    reg [3:0]    wb_ch;                // WB-side channel
    reg [15:0]   spw_oh, spw_ow;       // WB-side pixel coords
    reg [$clog2(ARRAY_SIZE)-1:0] col_last;  // last valid drain column for current oc_group

    // ─── PPU state ───
    reg [$clog2(ARRAY_SIZE):0] ppu_feed_cnt;  // how many acc values fed to PPU
    reg [15:0]                  ppu_wait_cnt;   // PPU flush wait counter
    reg signed [ACC_W-1:0]      last_ppu_acc /* verilator public */;  // Last PPU acc (debug)
    reg [15:0]                  last_drain_col /* verilator public */; // Channel index in OC group
    reg [15:0]                  last_tile_x /* verilator public */;
    reg [15:0]                  last_tile_y /* verilator public */;
    reg signed [ACC_W-1:0]     acc_buf [0:ARRAY_SIZE-1];

    // ─── Param read state ───
    reg [2:0]  param_word_idx;
    reg [31:0] param_buf [0:3];
    reg        param_read_issued;
    reg        param_data_ready;   // 2nd cycle: SRAM data available

    // ─── Writeback state ───
    reg [$clog2(ARRAY_SIZE):0] wb_cnt;   // output bytes collected
    reg [31:0] wb_pack;                   // pack buffer
    reg [1:0]  wb_pos;                    // byte position within current word (0-3 for INT8)
    reg [ACT_ADDR_W-1:0] wb_addr;

    // ─── Address bases ───
    reg [WGT_ADDR_W-1:0]   wgt_base;
    reg [ACT_ADDR_W-1:0]   act_base;
    reg [PARAM_ADDR_W-1:0] param_base;
    reg [ACT_ADDR_W-1:0]   out_base;

    // ─── DW state ───
    reg [15:0] dw_ch_idx;
    reg [7:0]  dw_cnt;
    reg        dw_read_issued;
    reg [1:0]  dw_init_phase;          // 0=setup, 1=acc_clear, 2=feeding
    reg [15:0] dw_oh, dw_ow;          // Output pixel coordinates
    reg [3:0]  dw_fh, dw_fw;          // Filter position (0..6)
    reg signed [ACC_W-1:0] dw_acc_buf; // Captured DW output accumulator
    reg [7:0]  dw_kernel_size;         // kh * kw (cached, up to 16x16=256)
    reg [1:0]  dw_wb_phase;            // 0=issue read, 1=wait, 2=merge+write
    reg [15:0] dw_wb_byte;             // PPU output element to write (up to 16-bit)
    reg [1:0]  dw_wb_bytesel;          // byte position within word
    reg [ACT_ADDR_W-1:0] dw_wb_addr;  // target word address
    reg [1:0]  dw_wgt_bsel_base;       // starting byte offset for weight reads
    reg [15:0] dw_grp_base;            // stream mode: first channel of resident 16-ch group
    reg [15:0] pool_grp_base;          // pool stream mode: ditto

    // ─── Spatial pixel loop (Conv2D Plan A) ───
    reg [15:0] sp_oh, sp_ow;                  // Current output pixel coordinates (tile-local)
    reg [15:0] tile_oh_origin, tile_ow_origin; // Global origin of current tile
    reg [15:0] tile_in_h, tile_in_w;           // Input tile dimensions (including halo)
    reg [15:0] kw_eff;                         // Effective kernel width: (kw-1)*dw+1
    reg signed [ACC_W-1:0] dot_acc;           // Reduction accumulator
    reg [$clog2(ARRAY_SIZE):0] reduce_cnt;    // Reduction counter
    reg [ACT_ADDR_W-1:0] pixel_act_base;     // Per-pixel activation base address
    reg signed [ACC_W-1:0] dot_buf [0:ARRAY_SIZE-1]; // Reduced dot products per column

    // ─── 1b pixel-block weight reuse ───
    // Partial sums for a 16-pixel block × 16 columns. Weights are loaded once
    // per (oc_group, k_pass) and reused across the block's pixels instead of
    // being re-streamed per pixel (baseline: ~92% of conv cycles were weight
    // re-streaming). Accumulation order per output element is unchanged
    // (pass-sequential, wraparound add) → bit-exact with the old schedule.
    // Two banks: PPU feeds bank rd while the next spatial block accumulates
    // into bank wr. Index is {bank, px_in_blk[3:0], col[3:0]}.
    reg signed [ACC_W-1:0] px_acc_buf [0:511];
    reg          acc_wr_bank;
    reg          acc_rd_bank;
    reg [15:0] blk_oh_start, blk_ow_start;       // sp of first pixel in block
    reg [15:0] px_remaining;                     // pixels left in this oc_group/tile
    reg [4:0]  px_in_blk;                        // pixel index within block (0..15)
    reg [4:0]  blk_px_cnt;                       // pixels in current block (1..16)
    reg [4:0]  ppu_px;                           // PPU-loop pixel index within block

    // ─── Multi-pass (k_depth > ARRAY_SIZE) ───
    reg [15:0] k_pass;           // Current pass index (0-based)
    reg [15:0] k_pass_max;       // Total passes - 1
    reg [15:0] k_pass_remain;    // Elements in current pass
`ifdef DBG_DOTBUF
    integer dbg_fh;
    integer dbg_trace_fh;
    reg signed [ACC_W-1:0] dbg_prev_db9;
    initial begin
        dbg_fh = $fopen("/tmp/rtl_dotbuf.log", "w");
        dbg_trace_fh = $fopen("/tmp/rtl_db9_trace.log", "w");
        dbg_prev_db9 = 0;
    end
`endif

    // ─── Conv2D kernel window iteration ───
    reg [7:0]  conv_fh, conv_fw;             // Current filter position
    reg [15:0] conv_ch_cnt;                  // Channel counter within (fh, fw)
    reg [7:0]  pass_fh, pass_fw;             // k_pass start (fh, fw) — stable for the pass
    reg [15:0] pass_ch;                      // k_pass start channel
    reg signed [15:0] conv_ih_base;          // Input row origin: sp_oh*stride_h - pad_top
    reg signed [15:0] conv_iw_base;          // Input col origin: sp_ow*stride_w - pad_left
    reg        conv_is_pad;                  // Current (fh, fw) is padding
    reg [15:0] conv_elem_cnt;                // Total elements emitted in this pass

    // ─── Flush counter (reused) ───
    reg [15:0] flush_cnt;

    // ─── Pooling state ───
    wire        pool_mode     = cfg_pool_cfg[0];      // 0=Max, 1=Avg
    wire [3:0]  pool_cfg_h    = cfg_pool_cfg[7:4];
    wire [3:0]  pool_cfg_w    = cfg_pool_cfg[11:8];
    wire [3:0]  pool_cfg_sh   = cfg_pool_cfg[15:12];
    wire [3:0]  pool_cfg_sw   = cfg_pool_cfg[19:16];
    wire        global_pool   = cfg_pool_cfg[20];
    // ─── DW global-pool streaming detect ───
    // Non-tiled DW Conv with kernel == whole input and 1x1 output whose
    // input/weight tensor (H*W*C words each) overflows act SRAM
    // (SPAD_KB*64 words). Streamed in 16-channel groups (see npu_ctrl).
    wire [31:0] dws_total_words = ({16'd0, cfg_in_h} * {16'd0, cfg_in_w}
                                   * {16'd0, cfg_in_c}) >> (cfg_int16 ? 2'd1 : 2'd2);
    wire        dw_stream = (cfg_op_type == 8'd1) && (cfg_tile_h == 16'd0)
                         && ({8'd0, cfg_kernel_h} == cfg_in_h)
                         && ({8'd0, cfg_kernel_w} == cfg_in_w)
                         && (cfg_out_h == 16'd1) && (cfg_out_w == 16'd1)
                         && (dws_total_words > (`SPAD_KB * 64));

    // ─── Per-OC 8-channel sub-group detect ───
    // When a 16-channel weight block would overflow the weight SRAM
    // (SPAD_KB*128 words), the packer emits per-oc blocks of 8 channels
    // (e.g. model_e int16 L23: 16ch×3x3x512×2B = 36864 words > 24576).
    // Detect from config: per-oc reload active AND 16-ch block doesn't fit.
    // grp_oc drives group count, param/WB channel indexing. Lanes ≥ grp_oc
    // read garbage weights but are never drained (col_last bounds the WB
    // loop), so no feed-path change is needed.
    wire [31:0] kd_bytes = ({24'd0, cfg_kernel_h} * {24'd0, cfg_kernel_w}
                            * {16'd0, cfg_in_c}) << (cfg_int16 ? 1 : 0);
    wire        grp8_mode = (cfg_wgt_per_oc != 32'd0)
                         && ((kd_bytes << 2) > (`SPAD_KB * 128));  // 16ch words = kd_bytes*16/4
    wire [4:0]  grp_oc    = grp8_mode ? 5'd8 : 5'd16;
    wire [4:0]  param_nch = {1'b0, col_last} + 5'd1;
    wire [6:0]  param_tgt = {param_nch, 2'b00}; // nch * 4 words
    wire        param_feed_busy = ppu_bg && (ppu_px < ppu_blk_save);
    wire        param_ready = !param_pend
                            && (cache_issue >= param_tgt)
                            && (cache_cap + 1 >= param_tgt);
    wire        param_bg_ok = (cfg_op_type == 8'd0 || cfg_op_type == 8'd2)
                            && !param_feed_busy && !param_pend
                            && (state == S_WGT_CMD || state == S_WGT_LOAD
                                || state == S_WGT_EMIT || state == S_ACT_CMD
                                || state == S_ACT_LOAD || state == S_ACT_EMIT
                                || state == S_ACT_FLUSH || state == S_SPATIAL_SETUP
                                || state == S_PSUM_COLLECT || state == S_PIXEL_NEXT);
    wire [15:0] dw_ch_base = oc_group * ARRAY_SIZE_16;
    wire [4:0]  dw_nch     = ((dw_ch_base + ARRAY_SIZE_16) > cfg_out_c)
                           ? (cfg_out_c - dw_ch_base)
                           : ARRAY_SIZE[4:0];

    // ─── Pool global-pool streaming detect ───
    // Same SRAM-capacity problem as dw_stream but on the Pool path
    // (model_e int16 L29: AvgPool 7x7x512→1x1x512, input 12544 words >
    // 12288-word act SRAM; int8 6272 fits). Same 16-channel slice
    // streaming; Pool has no weights so the reload only refetches the
    // act slice.
    wire        pool_stream = (cfg_op_type == 8'd3) && global_pool
                           && (cfg_tile_h == 16'd0)
                           && (dws_total_words > (`SPAD_KB * 64));
    wire        slice_stream = dw_stream | pool_stream;
    wire        resize_mode   = cfg_resize_cfg[0];   // 0=nearest, 1=bilinear
    // ─── Deconv state ───
    wire [7:0]  cfg_insert_h  = cfg_deconv_cfg[7:0];
    wire [7:0]  cfg_insert_w  = cfg_deconv_cfg[15:8];
    wire        is_deconv     = (cfg_op_type == 8'd6);
    // 1×1 (not deconv): next pixel is just +Cin in NHWC — skip SPATIAL/CMD.
    wire        is_1x1_fast   = (cfg_kernel_h == 8'd1) && (cfg_kernel_w == 8'd1)
                                && !is_deconv;
    // One k_pass vector lives in a single (fh,fw) tap. 1×1 always; 3×3
    // when IC covers the whole pass (e.g. IC=16, ARRAY=8). Then the 1×1
    // stream (skip SETUP/CMD, 2-ahead, skip LOAD sample) is safe.
    wire [16:0] tap_ch_end    = {1'b0, pass_ch} + {1'b0, k_pass_remain};
    wire        is_tap_fast   = !is_deconv
                                && (k_pass_remain <= ARRAY_SIZE_16)
                                && (tap_ch_end <= {1'b0, cfg_in_c});
    // Byte offset + pad for output pixel (oh,ow) at the current pass tap.
    // 1×1 is fh=fw=0. Packed as {pad, byte_off}.
    function [32:0] conv_tap_byte;
        input [15:0] oh, ow;
        input [7:0]  fh, fw;
        input [15:0] ch;
        reg signed [15:0] ihb, iwb, ih, iw;
        reg [31:0] elem;
        begin
            ihb = $signed({1'b0, tile_oh_origin + oh})
                * $signed({1'b0, cfg_stride_h[7:0]})
                - $signed({1'b0, cfg_pad_top[7:0]});
            iwb = $signed({1'b0, tile_ow_origin + ow})
                * $signed({1'b0, cfg_stride_w[7:0]})
                - $signed({1'b0, cfg_pad_left[7:0]});
            ih = ihb + $signed({8'd0, fh});
            iw = iwb + $signed({8'd0, fw});
            if ((ih < 0) || (ih >= $signed({1'b0, cfg_in_h}))
                    || (iw < 0) || (iw >= $signed({1'b0, cfg_in_w})))
                conv_tap_byte = {1'b1, 32'd0};
            else begin
                if (cfg_tile_h == 17'd0)
                    elem = (ih[15:0] * cfg_in_w + iw[15:0]) * cfg_in_c + ch;
                else
                    elem = (({8'd0, oh} * cfg_stride_h + {8'd0, fh}) * tile_in_w
                         + {8'd0, ow} * cfg_stride_w + {8'd0, fw}) * cfg_in_c
                         + {16'd0, ch};
                conv_tap_byte = {1'b0, cfg_int16 ? (elem << 1) : elem};
            end
        end
    endfunction
    // Single-beat vector: whole k-vector in the current 256-bit beat.
    // 2-ahead (LOAD-wait issues N+1, EMIT issues N+2) stays in EMIT at
    // 1 vec/cyc. Falling back to LOAD is the 2 cyc/vec path.
    wire [15:0] emit_rw = cfg_int16
        ? ((16'd32 - {11'd0, act_byte_sel}) >> 1)
        : (16'd32 - {11'd0, act_byte_sel});
    wire        emit_1x1_can  = is_tap_fast && (state == S_ACT_EMIT)
                                && act_use_rd && (act_cnt == 16'd0)
                                && (k_pass_remain <= ARRAY_SIZE_16)
                                && (emit_rw >= k_pass_remain)
                                && (cfg_in_c >= conv_ch_cnt + k_pass_remain)
                                && (px_in_blk + 1 < blk_px_cnt)
                                && !conv_is_pad && !deconv_skip
                                && !(if_full && !if_collect);
    wire        emit_1x1_stream = emit_1x1_can && act_ahead && !ahead_pad;
    wire        emit_1x1_next   = emit_1x1_can && !emit_1x1_stream;
    wire        emit_1x1_last   = is_tap_fast && (state == S_ACT_EMIT)
                                && act_use_rd && (act_cnt == 16'd0)
                                && (k_pass_remain <= ARRAY_SIZE_16)
                                && (emit_rw >= k_pass_remain)
                                && (cfg_in_c >= conv_ch_cnt + k_pass_remain)
                                && (px_in_blk + 1 >= blk_px_cnt)
                                && !conv_is_pad && !deconv_skip
                                && !(if_full && !if_collect);
    wire        if_push = (state == S_ACT_FLUSH) || emit_1x1_next
                        || emit_1x1_stream || emit_1x1_last;
    // Next k_pass we will actually WGT-load (skips the resident first pass
    // of a block). Same rule as S_PSUM_COLLECT / PIXEL_NEXT.
    function [15:0] wgt_pass_after;
        input [15:0] cur, first, fin, maxp;
        begin
            if (fin >= maxp) begin
                if (cur == 16'd0)
                    wgt_pass_after = 16'd1;
                else
                    wgt_pass_after = 16'd0;
            end else if (cur == first) begin
                if (first == 16'd0)
                    wgt_pass_after = 16'd1;
                else
                    wgt_pass_after = 16'd0;
            end else if ((cur + 16'd1) == first) begin
                wgt_pass_after = cur + 16'd2;
            end else begin
                wgt_pass_after = cur + 16'd1;
            end
        end
    endfunction
    wire [15:0] wgt_pf_tgt  = wgt_pass_after(k_pass, k_first, k_fin_cnt,
                                             k_pass_max);
    wire        wgt_pf_more = px_remaining > {11'd0, blk_px_cnt};
    wire        wgt_pf_want = !wgt_sh_ok
                            && (cfg_op_type == 8'd0 || cfg_op_type == 8'd2)
                            && ((k_fin_cnt < k_pass_max)
                                || (wgt_pf_more && (k_pass_max != 16'd0)));
    wire [15:0] deconv_exp_h  = cfg_in_h + (cfg_in_h - 17'd1) * {8'd0, cfg_insert_h};
    wire [15:0] deconv_exp_w  = cfg_in_w + (cfg_in_w - 17'd1) * {8'd0, cfg_insert_w};
    wire [8:0]  deconv_step_h = {1'b0, cfg_insert_h} + 9'd1; // ins_h + 1
    wire [8:0]  deconv_step_w = {1'b0, cfg_insert_w} + 9'd1; // ins_w + 1
    reg         deconv_skip;  // 1 = current (fh, fw) maps to zero-inserted position
    // Deconv precomputed address (avoids deep combinational path in S_ACT_CMD)
    reg [31:0]  deconv_elem_off;  // precomputed elem_off for deconv
    reg         deconv_addr_valid; // 1 = deconv_elem_off is valid (not skip)

    // ─── Deconv reciprocal LUT (step=1,2,3,4) ───
    // Q32: ih = (eh * recip) >> 32
    wire [31:0] recip_deconv_h = (cfg_insert_h == 8'd0) ? 32'h8000_0000 :  // step=1, shift 31
                                 (cfg_insert_h == 8'd1) ? 32'h8000_0000 :  // step=2, shift 32
                                 (cfg_insert_h == 8'd2) ? 32'h5555_5556 :  // step=3
                                 (cfg_insert_h == 8'd3) ? 32'h4000_0000 :  // step=4
                                 32'h0;
    wire [31:0] recip_deconv_w = (cfg_insert_w == 8'd0) ? 32'h8000_0000 :
                                 (cfg_insert_w == 8'd1) ? 32'h8000_0000 :
                                 (cfg_insert_w == 8'd2) ? 32'h5555_5556 :
                                 (cfg_insert_w == 8'd3) ? 32'h4000_0000 :
                                 32'h0;
    wire deconv_h_shift1 = (cfg_insert_h == 8'd0);  // step=1: shift 31
    wire deconv_w_shift1 = (cfg_insert_w == 8'd0);
    // ─── Concat state ───
    wire [15:0] concat_offset  = cfg_concat_cfg[15:0];
    wire [15:0] concat_total_c = cfg_concat_cfg[31:16];
    wire        is_concat      = (cfg_op_type == 8'd7);
    reg signed [ACC_W-1:0] pool_acc;     // Running sum or max
    reg signed [ACC_W-1:0] pool_val;     // Current pool element value
    reg [15:0]             pool_count;   // Valid element count (AvgPool)
    reg [3:0]              pool_fh, pool_fw; // Window position
    reg [15:0]             pool_oh, pool_ow; // Output pixel coords
    reg [15:0]             pool_ch;          // Current channel
    reg [7:0]              pool_kh, pool_kw; // Effective kernel size
    reg [7:0]              pool_sh, pool_sw; // Effective stride
    reg [1:0]              pool_rd_phase;    // SRAM read phasing
    reg [1:0]              pool_wb_phase;    // Writeback phasing
    reg [ACT_ADDR_W-1:0]  pool_wb_addr;     // Writeback word address
    reg [1:0]              pool_wb_bytesel;  // Writeback byte select
    reg [15:0]             pool_wb_byte;     // PPU output to write

    // ─── Eltwise Add state ───
    reg [14:0] add_M_A, add_M_B;
    reg [5:0]  add_S_A, add_S_B;
    reg signed [ACC_W-1:0] add_val_a, add_val_b;
    reg [15:0] add_elem_cnt;      // Current element index (flat)
    reg [15:0] add_tile_elem_cnt; // Element index within current tile
    // Concat pixel/ch counters (avoids division in writeback)
    reg [15:0] concat_pixel_cnt;  // pixel = elem_cnt / in_c
    reg [15:0] concat_ch_cnt;     // ch = elem_cnt % in_c
    reg [15:0] add_total_elems;   // H * W * C
    reg [1:0]  add_rd_phase;      // SRAM read phasing
    reg [1:0]  add_param_phase;   // Param read phasing
    reg [1:0]  add_param_idx;     // Which param word (0 or 1)
    reg [1:0]  add_wb_phase;      // Writeback phasing
    reg [ACT_ADDR_W-1:0]  add_wb_addr;     // Writeback word address
    reg [1:0]             add_wb_bytesel;  // Writeback byte select
    reg [15:0]            add_wb_byte;     // PPU output to write

    // ─── Resize state ───
    reg [15:0]             rsz_oh, rsz_ow;
    reg [15:0]             rsz_ch;
    reg signed [ACC_W-1:0] rsz_v00, rsz_v01, rsz_v10, rsz_v11;
    reg [7:0]              rsz_frac_h, rsz_frac_w;
    reg [15:0]             rsz_ih0, rsz_iw0, rsz_ih1, rsz_iw1;
    reg [1:0]              rsz_rd_phase;
    reg [1:0]              rsz_wb_phase;
    reg [ACT_ADDR_W-1:0]   rsz_wb_addr;
    reg [1:0]              rsz_wb_bytesel;
    reg [15:0]             rsz_wb_byte;
    // Resize reciprocal registers (precomputed per layer)
    reg [39:0]             recip_out_h;     // Q40 reciprocal of cfg_out_h
    reg [39:0]             recip_out_w;     // Q40 reciprocal of cfg_out_w
    reg [39:0]             recip_out_h_m1;  // Q40 reciprocal of (cfg_out_h-1)
    reg [39:0]             recip_out_w_m1;  // Q40 reciprocal of (cfg_out_w-1)
    // Combined reciprocals (in_h * recip_out_h) — single multiply per pixel
    reg signed [55:0]      recip_scale_h;      // cfg_in_h * recip_out_h
    reg signed [55:0]      recip_scale_w;      // cfg_in_w * recip_out_w
    reg signed [55:0]      recip_scale_h_m1;   // ((cfg_in_h-1)<<8) * recip_out_h_m1
    reg signed [55:0]      recip_scale_w_m1;   // ((cfg_in_w-1)<<8) * recip_out_w_m1
    // Tile-origin hoist (avoid 4x redundant division per pixel)
    reg [15:0]             rsz_tile_ih_origin;  // (tile_oh_origin * in_h) / out_h
    reg [15:0]             rsz_tile_iw_origin;  // (tile_ow_origin * in_w) / out_w
    // Bilinear interp pipeline register (for 2-cycle interp split)
    reg signed [63:0]      rsz_top_r, rsz_bot_r;

    integer i;

    // ════════════════════════════════════════════════════════════════════
    // Main FSM
    // ════════════════════════════════════════════════════════════════════

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            tile_done_r <= 1'b0;
            oc_group_done <= 1'b0;
            tile_wait_delay <= 1'b0;
            // All outputs idle
            sa_cmd       <= MODE_IDLE;
            sa_cmd_valid <= 1'b0;
            sa_wgt_valid <= 1'b0;
            sa_act_valid <= 1'b0;
            wgt_rd_en   <= 1'b0;
            wgt_rd_addr <= 0;
            act_rd_en   <= 1'b0;
            act_rd_addr <= 0;
            act_rd_ofm  <= 1'b0;
            pf_age      <= 2'd0;
            if_wr        <= {COL_W{1'b0}};
            if_rd        <= {COL_W{1'b0}};
            if_count     <= {IF_CNT_W{1'b0}};
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
            ppu_acc_w    <= 0;
            ppu_valid_w  <= 0;
            ppu_bias_w   <= 0;
            ppu_mult_w   <= 0;
            ppu_shift_w  <= 0;
            ppu_zp_w     <= 0;
            dw_wgt_load  <= 1'b0;
            dw_wgt_valid <= 1'b0;
            dw_wgt_data  <= 0;
            dw_in_valid  <= 1'b0;
            dw_in_data   <= 0;
            dw_acc_clear <= 1'b0;
            dw_wgt_valid_w <= 0;
            dw_wgt_data_w  <= 0;
            dw_in_valid_w  <= 0;
            dw_in_data_w   <= 0;
            // Internal state
            tile_y <= 0; tile_x <= 0; oc_group <= 0;
            oc_groups_total <= 1; k_depth <= 1;
            out_tile_h <= 1; out_tile_w <= 1;
            wgt_col_idx <= 0; wgt_byte_idx <= 0;
            wgt_word_addr <= 0; wgt_read_issued <= 0; wgt_data_ready <= 0;
            act_cnt <= 0; act_word_addr <= 0;
            act_read_issued <= 0; act_data_ready <= 0; act_use_rd <= 0;
            act_ahead <= 1'b0; ahead_bsel <= 5'd0; ahead_pad <= 1'b0;
            act_buf <= 0; act_byte_sel <= 0;
            drain_col <= 0; col_last <= COL_MAX;
            ppu_feed_cnt <= 0; ppu_wait_cnt <= 0;
            param_word_idx <= 0; param_read_issued <= 0; param_data_ready <= 0;
            wb_cnt <= 0; wb_pack <= 0; wb_addr <= 0;
            wgt_base <= 0; act_base <= 0; param_base <= 0; out_base <= 0;
            dw_ch_idx <= 0; dw_cnt <= 0; dw_read_issued <= 0; dw_init_phase <= 0;
            dw_oh <= 0; dw_ow <= 0; dw_fh <= 0; dw_fw <= 0;
            dw_acc_buf <= 0; dw_kernel_size <= 0;
            dw_wb_phase <= 0; dw_wb_byte <= 0; dw_wb_bytesel <= 0; dw_wb_addr <= 0;
            dw_wgt_bsel_base <= 0;
            dw_grp_base <= 0;
            pool_grp_base <= 0;
            flush_cnt <= 0;
            ppu_st <= 0;
            ppu_wb_left <= 0;
            ppu_wb_idx <= 0;
            act_armed <= 1'b0;
            ppu_ovl <= 1'b0;
            ppu_bg <= 1'b0;
            ppu_blk_save <= 5'd0;
            ppu_oc_save <= 16'd0;
            ppu_col_last <= COL_MAX;
            param_pend <= 1'b0;
            ppu_ahead <= 1'b0;
            ppu_pend <= 1'b0;
            wgt_sh_ok <= 1'b0;
            wgt_sh_pass <= 16'd0;
            wgt_pf_col <= 0;
            wgt_pf_ph <= 0;
            wgt_from_sh <= 1'b0;
            wgt_held <= 1'b0;
            wgt_held_pass <= 16'd0;
            k_first <= 16'd0;
            k_fin_cnt <= 16'd0;
            acc_wr_bank <= 1'b0;
            acc_rd_bank <= 1'b0;
            sp_oh <= 0; sp_ow <= 0; tile_oh_origin <= 0; tile_ow_origin <= 0;
            dot_acc <= 0; reduce_cnt <= 0; pixel_act_base <= 0;
            k_pass <= 0; k_pass_max <= 0; k_pass_remain <= 0;
            conv_fh <= 0; conv_fw <= 0; conv_ch_cnt <= 0;
            pass_fh <= 0; pass_fw <= 0; pass_ch <= 0;
            conv_ih_base <= 0; conv_iw_base <= 0; conv_is_pad <= 0;
            conv_elem_cnt <= 0;
            // Pooling resets
            pool_acc <= 0; pool_count <= 0;
            pool_fh <= 0; pool_fw <= 0; pool_oh <= 0; pool_ow <= 0;
            pool_ch <= 0; pool_kh <= 0; pool_kw <= 0; pool_sh <= 0; pool_sw <= 0;
            pool_rd_phase <= 0; pool_wb_phase <= 0;
            pool_wb_addr <= 0; pool_wb_bytesel <= 0; pool_wb_byte <= 0;
            // Add resets
            add_M_A <= 0; add_M_B <= 0; add_S_A <= 0; add_S_B <= 0;
            add_val_a <= 0; add_val_b <= 0;
            add_elem_cnt <= 0; add_total_elems <= 0; add_tile_elem_cnt <= 0;
            add_rd_phase <= 0; add_param_phase <= 0; add_param_idx <= 0;
            add_wb_phase <= 0; add_wb_addr <= 0; add_wb_bytesel <= 0; add_wb_byte <= 0;
            // Resize resets
            rsz_oh <= 0; rsz_ow <= 0; rsz_ch <= 0;
            rsz_v00 <= 0; rsz_v01 <= 0; rsz_v10 <= 0; rsz_v11 <= 0;
            rsz_frac_h <= 0; rsz_frac_w <= 0;
            rsz_ih0 <= 0; rsz_iw0 <= 0; rsz_ih1 <= 0; rsz_iw1 <= 0;
            rsz_rd_phase <= 0; rsz_wb_phase <= 0;
            rsz_wb_addr <= 0; rsz_wb_bytesel <= 0; rsz_wb_byte <= 0;
            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                sa_wgt_data[i] <= 0;
                sa_act_data[i] <= 0;
                acc_buf[i] <= 0;
                dot_buf[i] <= 0;
            end
            for (i = 0; i < 4; i = i + 1)
                param_buf[i] <= 0;
        end else begin
            // ─── Default pulse deassertion ───
            if (pf_age == 2'd1)
                pf_age <= 2'd2;   // data valid in act_rd_data now
            done         <= 1'b0;
            tile_done_r  <= 1'b0;
            sa_cmd_valid <= 1'b0;
            sa_wgt_valid <= 1'b0;
            sa_act_valid <= 1'b0;
            wgt_rd_en   <= 1'b0;
            act_rd_en   <= 1'b0;
            act_rd_ofm  <= 1'b0;
            act_wr_en   <= 1'b0;
            param_rd_en <= 1'b0;
            ppu_in_valid <= 1'b0;
            ppu_valid_w  <= {ARRAY_SIZE{1'b0}};
            dw_wgt_valid <= 1'b0;
            dw_in_valid  <= 1'b0;
            dw_acc_clear <= 1'b0;
            dw_wgt_valid_w <= {ARRAY_SIZE{1'b0}};
            dw_in_valid_w  <= {ARRAY_SIZE{1'b0}};

            // Collect whenever the array produces a result, regardless of
            // which gather state we are in. Depth-ARRAY_SIZE FIFO holds the
            // (pixel, k_pass) of each in-flight vector.
            if (if_collect) begin
                for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                    if (if_kp[if_rd] == k_first)
                        px_acc_buf[{acc_wr_bank, if_px[if_rd][3:0], i[3:0]}]
                            <= sa_psum_out[i];
                    else
                        px_acc_buf[{acc_wr_bank, if_px[if_rd][3:0], i[3:0]}]
                            <= px_acc_buf[{acc_wr_bank, if_px[if_rd][3:0], i[3:0]}]
                               + sa_psum_out[i];
                end
                if_rd <= if_rd + 1'b1;
            end
            if (if_push) begin
                if_px[if_wr] <= px_in_blk;
                if_kp[if_wr] <= k_pass;
                if_wr <= if_wr + 1'b1;
            end
            if (if_collect && if_push)
                if_count <= if_count;
            else if (if_collect)
                if_count <= if_count - 1'b1;
            else if (if_push)
                if_count <= if_count + 1'b1;

            // PPU stream: standalone so it can run while the next spatial
            // block reloads weights / streams ACT into the other acc bank.
            if (ppu_bg || (state == S_PPU_STREAM)) begin
                begin : ppu_stream_blk
                    reg [31:0] elem_off;
                    reg [4:0]  nch;
                    reg        last_wb;
                    reg        vout;
                    reg [4:0]  inflight;
                    nch = {1'b0, ppu_col_last} + 5'd1;
                    last_wb = ppu_ahead && ((cfg_int16 && (ppu_wb_idx + 5'd2 >= nch))
                                         || (!cfg_int16 && (ppu_wb_idx + 5'd4 >= nch)));
                    vout = ppu_vout_w[0] || ppu_out_valid;
                    inflight = ppu_px - wb_px;

                    if (vout) begin
                        if (!ppu_ahead || last_wb) begin
                            for (i = 0; i < ARRAY_SIZE; i = i + 1)
                                ppu_lat[i] <= ppu_out_w[DATA_W*i +: DATA_W];
                            if (!ppu_vout_w[0])
                                ppu_lat[0] <= ppu_out_data;
                        end else begin
                            for (i = 0; i < ARRAY_SIZE; i = i + 1)
                                ppu_lat2[i] <= ppu_out_w[DATA_W*i +: DATA_W];
                            if (!ppu_vout_w[0])
                                ppu_lat2[0] <= ppu_out_data;
                            ppu_pend <= 1'b1;
                        end
                    end

                    if (ppu_ahead) begin
                        elem_off = (spw_oh * out_tile_w + spw_ow) * cfg_out_c
                                 + ppu_oc_save * {11'd0, grp_oc}
                                 + {11'd0, ppu_wb_idx};
                        act_wr_en <= 1'b1;
                        if (cfg_int16) begin
                            act_wr_addr <= out_base + elem_off[17:1];
                            act_wr_data <= {ppu_lat[ppu_wb_idx[3:0] + 4'd1],
                                            ppu_lat[ppu_wb_idx[3:0]]};
                            ppu_wb_idx <= ppu_wb_idx + 5'd2;
                        end else begin
                            act_wr_addr <= out_base + elem_off[17:2];
                            act_wr_data <= {ppu_lat[ppu_wb_idx[3:0] + 4'd3][7:0],
                                            ppu_lat[ppu_wb_idx[3:0] + 4'd2][7:0],
                                            ppu_lat[ppu_wb_idx[3:0] + 4'd1][7:0],
                                            ppu_lat[ppu_wb_idx[3:0]][7:0]};
                            ppu_wb_idx <= ppu_wb_idx + 5'd4;
                        end
                        if (last_wb) begin
                            if (wb_px + 1 >= ppu_blk_save) begin
                                if (!ppu_bg) begin
                                    sp_oh <= spw_oh;
                                    sp_ow <= spw_ow;
                                    state <= S_PIXEL_NEXT;
                                end
                                ppu_ovl    <= 1'b0;
                                ppu_bg     <= 1'b0;
                                ppu_ahead  <= 1'b0;
                                ppu_pend   <= 1'b0;
                            end else begin
                                wb_px <= wb_px + 1;
                                if (spw_ow + 1 >= out_tile_w) begin
                                    spw_ow <= 0;
                                    spw_oh <= spw_oh + 1;
                                end else begin
                                    spw_ow <= spw_ow + 1;
                                end
                                if (ppu_pend) begin
                                    for (i = 0; i < ARRAY_SIZE; i = i + 1)
                                        ppu_lat[i] <= ppu_lat2[i];
                                    ppu_pend <= 1'b0;
                                    ppu_wb_idx <= 0;
                                    ppu_ahead <= 1'b1;
                                end else if (vout) begin
                                    ppu_wb_idx <= 0;
                                    ppu_ahead <= 1'b1;
                                end else begin
                                    ppu_ahead <= 1'b0;
                                end
                            end
                        end
                    end else if (vout) begin
                        ppu_ahead <= 1'b1;
                        ppu_wb_idx <= 0;
                    end

                    if ((ppu_px < ppu_blk_save) && (inflight < 5'd2)) begin
                        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                            if (i[3:0] <= ppu_col_last) begin
                                ppu_acc_w[ACC_W*i +: ACC_W]
                                    <= px_acc_buf[{acc_rd_bank, ppu_px[3:0], i[3:0]}];
                                ppu_mult_w[15*i +: 15]
                                    <= param_cache[{i[3:0], 2'b00}][14:0];
                                ppu_shift_w[6*i +: 6]
                                    <= param_cache[{i[3:0], 2'b00}][21:16];
                                ppu_zp_w[16*i +: 16]
                                    <= param_cache[{i[3:0], 2'b00} + 1][15:0];
                                ppu_bias_w[ACC_W*i +: ACC_W]
                                    <= $signed({param_cache[{i[3:0], 2'b00} + 3][15:0],
                                                param_cache[{i[3:0], 2'b00} + 2],
                                                param_cache[{i[3:0], 2'b00} + 1][31:16]});
                                ppu_valid_w[i] <= 1'b1;
                            end
                        end
                        ppu_in_valid <= 1'b1;
                        ppu_acc_in   <= px_acc_buf[{acc_rd_bank, ppu_px[3:0], 4'd0}];
                        ppu_px <= ppu_px + 1;
                    end
                end
            end

            // Param cache fill: Param SRAM is free during WGT/ACT. Do not
            // overwrite the cache while a background PPU is still feeding.
            if (param_pend && !param_feed_busy) begin
                cache_issue <= 7'd0;
                cache_cap   <= 7'd0;
                param_pend  <= 1'b0;
            end else if (param_bg_ok && (cache_issue < param_tgt
                    || (cache_issue >= 7'd2 && cache_cap < param_tgt))) begin
                if (cache_issue < param_tgt) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= param_base + {10'd0, cache_issue};
                    cache_issue   <= cache_issue + 1;
                end
                if (cache_issue >= 2 && cache_cap < param_tgt) begin
                    param_cache[cache_cap[5:0]] <= param_rd_data;
                    cache_cap <= cache_cap + 1;
                end
            end

            (* parallel_case, full_case *)
            case (state)

            // ══════════════════════════════════════════════════════════════
            S_IDLE: begin
                if (start) begin
                    if_wr    <= {COL_W{1'b0}};
                    if_rd    <= {COL_W{1'b0}};
                    if_count <= {IF_CNT_W{1'b0}};
                    if (cfg_op_type == 8'd1)
                        oc_groups_total <= (cfg_out_c + ARRAY_SIZE_16 - 1) / ARRAY_SIZE_16;
                    else if (cfg_op_type == 8'd3 || cfg_op_type == 8'd5)
                        oc_groups_total <= cfg_out_c;
                    else if (grp8_mode)
                        oc_groups_total <= (cfg_out_c + 16'd7) >> 3;
                    else
                        oc_groups_total <= (cfg_out_c + ARRAY_SIZE_16 - 1) / ARRAY_SIZE_16;

                    k_depth <= {8'd0, cfg_kernel_h} * {8'd0, cfg_kernel_w} * cfg_in_c;
                    kw_eff  <= cfg_kernel_w;

                    // Precompute reciprocals for Conv k_pass decomposition
                    // (divisors are layer-constant, computed once here)
                    begin : recip_setup_blk
                        reg [15:0] kw_inc;
                        reg [31:0] rem;
                        integer i;
                        kw_inc = {8'd0, cfg_kernel_w} * cfg_in_c;
                        kw_x_inc_r <= kw_inc;
                        in_c_r <= cfg_in_c;
                        `ifndef SYNTHESIS
                        $display("[S_IDLE_RECIP] kw=%0d in_c=%0d kw_inc=%0d recip=%0d",
                                cfg_kernel_w, cfg_in_c, kw_inc,
                                (kw_inc > 1) ? (32'hFFFF_FFFF / kw_inc) + 1 :
                                (kw_inc == 1) ? 32'hFFFF_FFFF : 0);
                        `endif
                        // Compute recip_kw_x_inc = ceil((1<<32) / kw_inc) iteratively
                        // Simple: use division here (once per layer, not critical path)
                        recip_kw_x_inc <= (kw_inc > 1) ? (32'hFFFF_FFFF / kw_inc) + 1 :
                                          (kw_inc == 1) ? 32'hFFFF_FFFF : 0;
                        // For in_c=1: (0xFFFFFFFF/1)+1 overflows to 0. Use 0xFFFFFFFF.
                        recip_in_c <= (cfg_in_c > 1) ? (32'hFFFF_FFFF / cfg_in_c) + 1 :
                                      (cfg_in_c == 1) ? 32'hFFFF_FFFF : 0;
                    end

                    if (cfg_tile_h == 17'd0) begin
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
                // Input: always use cfg_act_base; tile offset is folded into
                // conv_ih_base/conv_iw_base via tile_oh_origin/tile_ow_origin
                act_base <= cfg_act_base;
                // Reset tile dims to full before clipping.
                // Without this, border-tile clipping persists into next tile
                // (e.g., tile(0,3) clips out_tile_w to 4, tile(1,0) inherits 4).
                // For non-tiled (cfg_tile_h=0): use full output dims.
                if (cfg_tile_h == 17'd0) begin
                    out_tile_h <= cfg_out_h;
                    out_tile_w <= cfg_out_w;
                    tile_oh_origin <= 17'd0;
                    tile_ow_origin <= 17'd0;
                    rsz_tile_ih_origin <= 17'd0;
                    rsz_tile_iw_origin <= 17'd0;
                end else begin
                    out_tile_h <= cfg_tile_h;
                    out_tile_w <= cfg_tile_w;
                    // Compute tile origin using FULL tile dims (not clipped)
                    tile_oh_origin <= tile_y * cfg_tile_h;
                    tile_ow_origin <= tile_x * cfg_tile_w;
                    // Hoist tile-input origin for Resize (avoid 4x redundant division)
                    if (cfg_op_type == 8'd5) begin
                        begin : rsz_tile_origin_blk
                            reg [71:0] prod_th, prod_tw;
                            prod_th = (tile_y * cfg_tile_h) * cfg_in_h * recip_out_h;
                            prod_tw = (tile_x * cfg_tile_w) * cfg_in_w * recip_out_w;
                            rsz_tile_ih_origin <= prod_th[71:40];
                            rsz_tile_iw_origin <= prod_tw[71:40];
                        end
                    end
                end
                `ifndef SYNTHESIS
                if (cfg_tile_h != 0)
                    $display("[CMP_TILE] t=%0t tile(%0d,%0d) act_base=%0d out_base=%0d ow_origin_reg=%0d out_tw=%0d tile_in_w=%0d",
                             $time, tile_y, tile_x, act_base, act_base + cfg_out_base,
                             tile_ow_origin, out_tile_w, tile_in_w);
                `endif
                // Clip tile dimensions to image boundary (after reset + origin)
                if (cfg_tile_h != 17'd0) begin
                    if (cfg_tile_w > cfg_out_w - tile_x * cfg_tile_w)
                        out_tile_w <= cfg_out_w - tile_x * cfg_tile_w;
                    if (cfg_tile_h > cfg_out_h - tile_y * cfg_tile_h)
                        out_tile_h <= cfg_out_h - tile_y * cfg_tile_h;
                end
                // Compute input tile dimensions (including halo)
                // tile_in_h/w for activation addressing:
                //   tiled mode: max DMA stride
                //   non-tiled: full image width (entire image in SRAM)
                if (cfg_tile_h == 17'd0) begin
                    tile_in_h <= 1'b0;  // unused in non-tiled mode
                    tile_in_w <= cfg_in_w;
                end else begin
                    // Pooling uses pool_cfg params, Conv uses cfg_stride/kw_eff
                    if (cfg_op_type == 3) begin  // POOLING
                        tile_in_h <= cfg_tile_h * {12'd0, pool_cfg_sh} + {12'd0, pool_cfg_h} - {12'd0, pool_cfg_sh};
                        tile_in_w <= cfg_tile_w * {12'd0, pool_cfg_sw} + {12'd0, pool_cfg_w} - {12'd0, pool_cfg_sw};
                    end else begin
                        tile_in_h <= cfg_tile_h * cfg_stride_h + kw_eff - cfg_stride_h;
                        tile_in_w <= cfg_tile_w * cfg_stride_w + kw_eff - cfg_stride_w;
                    end
                end
                // Output base: add bank offset so output doesn't overlap input
                // For DB_EN: out_base = effective_act_base + cfg_out_base
                // This ensures output is written after input within the same bank
                // DW stream mode: act SRAM[0] holds the resident 16-ch slice;
                // the 1x1xC output accumulates in the fixed high region
                // (npu_ctrl stores from DW_STREAM_OUT_BASE for these layers).
                if (slice_stream) begin
                    out_base    <= `DW_STREAM_OUT_BASE;
                    dw_grp_base <= 16'd0;
                end else begin
                    out_base <= cfg_act_base + cfg_out_base;
                end

                oc_group <= 0;

                if (cfg_op_type == 8'd1) begin
                    dw_cnt <= 0;
                    dw_ch_idx <= 16'd0;
                    dw_read_issued <= 1'b0;
                    dw_init_phase <= 2'd0;
                    state <= S_DW_WGT_LOAD;
                end else if (cfg_op_type == 8'd3) begin
                    state <= S_POOL_SETUP;
                end else if (cfg_op_type == 8'd4 || cfg_op_type == 8'd7) begin
                    state <= S_ADD_SETUP;
                end else if (cfg_op_type == 8'd5) begin
                    state <= S_RESIZE_SETUP;
                end else begin
                    // Conv2D/FC: if per-oc reload, need to reload oc_group 0 weights
                    // (SRAM still has last oc_group's weights from previous tile)
                    if (cfg_wgt_per_oc != 0) begin
                        oc_group_done <= 1'b1;  // Request reload of oc_group 0
                        state <= S_WAIT_WGT_RELOAD;
                    end else begin
                        state <= S_OC_SETUP;
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            S_OC_SETUP: begin
                // Weight base: 0 if per-oc reload, else offset into full weight SRAM
                if (cfg_wgt_per_oc != 0)
                    wgt_base <= 0;  // per-oc: weights reloaded to SRAM[0]
                else if (cfg_int16)
                    wgt_base <= (oc_group * ARRAY_SIZE_16 * k_depth) >> 1;
                else
                    wgt_base <= (oc_group * ARRAY_SIZE_16 * k_depth) >> 2;
`ifdef DBG_DOTBUF
                $fwrite(dbg_fh, "[OC_SETUP] oc_group=%0d k_depth=%0d wgt_base=%0d param_base=%0d\n",
                        oc_group, k_depth, (oc_group * ARRAY_SIZE_16 * k_depth) >> 1, oc_group * ARRAY_SIZE_16 * 4);
`endif
                `ifndef SYNTHESIS
                if (cfg_tile_h != 0)
                    $display("[CMP_OC] t=%0t tile(%0d,%0d) oc_group=%0d k_depth=%0d k_pass_max=%0d col_last=%0d out_base=%0d",
                             $time, tile_y, tile_x, oc_group, k_depth,
                             (k_depth - 1) / ARRAY_SIZE_16, col_last, act_base + cfg_out_base);
                `endif
                // Param base: 4 words per channel, ARRAY_SIZE channels per group.
                // When per-oc PARAM reload is active (weights reload per oc_group
                // AND out_c exceeds the param SRAM capacity of SPAD_KB*16/4
                // channels), the controller reloads this oc_group's params to
                // param SRAM[0], so param_base must be 0 (not oc_group*64).
                if (cfg_wgt_per_oc != 0 && ({17'd0, cfg_out_c} * 4 > (`SPAD_KB * 16)))
                    param_base <= 16'd0;  // per-oc param reload → params at SRAM[0]
                else
                    // 4 words per channel; grp_oc channels per group (8 in grp8 mode)
                    param_base <= oc_group * {11'd0, grp_oc} * 16'd4;

                // Multi-pass setup
                k_pass <= 0;
                k_pass_max <= (k_depth - 1) / ARRAY_SIZE_16;
                k_first <= 16'd0;
                k_fin_cnt <= 16'd0;
                wgt_held <= 1'b0;

                // Last valid drain column for this oc_group
                begin : col_last_blk
                    reg [15:0] remaining_oc;
                    remaining_oc = cfg_out_c - oc_group * {11'd0, grp_oc};
                    if (remaining_oc >= {11'd0, grp_oc})
                        col_last <= grp_oc[3:0] - 4'd1;
                    else
                        col_last <= remaining_oc[$clog2(ARRAY_SIZE)-1:0] - 1;
                end

                // Reset spatial coords for first pixel
                sp_oh <= 0;
                sp_ow <= 0;

                // 1b pixel-block init (block 0 starts at sp(0,0))
                px_in_blk   <= 0;
                if (!ppu_bg)
                    ppu_px      <= 0;
                blk_oh_start <= 0;
                blk_ow_start <= 0;
                begin : blk_init_blk
                    reg [31:0] tpx;
                    tpx = {16'd0, out_tile_h} * {16'd0, out_tile_w};
                    px_remaining <= tpx[15:0];
                    blk_px_cnt   <= (tpx >= 32'd16) ? 5'd16 : tpx[4:0];
                end

                // Reset writeback packing state for this OC group
                wb_pack <= 0;
                wb_pos  <= 2'd0;

                wgt_col_idx <= 0;
                if (param_feed_busy) begin
                    param_pend <= 1'b1;
                end else begin
                    cache_issue <= 0;
                    cache_cap   <= 0;
                    param_pend  <= 1'b0;
                end
                if (!ppu_bg) begin
                    acc_wr_bank <= 1'b0;
                    acc_rd_bank <= 1'b0;
                end
                state <= S_WGT_CMD;
            end

            // ══════════════════════════════════════════════════════════════
            // PARAM CACHE: burst-read this group's PPU params (grp_oc × 4
            // words) once per oc_group — they are identical for every pixel.
            // ══════════════════════════════════════════════════════════════
            S_PARAM_CACHE: begin
                if (param_ready)
                    state <= S_WGT_CMD;
            end

            // Body hoisted above the case so it can run under ppu_bg.
            S_PPU_STREAM: begin
            end

            // ══════════════════════════════════════════════════════════════
            // WEIGHT LOAD: load ARRAY_SIZE columns, one per wgt_valid pulse
            // ══════════════════════════════════════════════════════════════
            S_WGT_CMD: begin
                sa_cmd       <= MODE_WGT_LOAD;
                sa_cmd_valid <= 1'b1;
                act_armed    <= 1'b0;
                act_ahead    <= 1'b0;
                // Begin loading column 0
                wgt_byte_idx    <= 0;
                wgt_read_issued <= 1'b0;
                wgt_data_ready  <= 1'b0;
                // Compute k_pass_remain for this pass
                k_pass_remain <= (k_pass == k_pass_max)
                    ? (k_depth - k_pass * ARRAY_SIZE_16)
                    : ARRAY_SIZE_16;
                // Address for column wgt_col_idx, starting at k_pass offset:
                //   INT8: byte_offset = col * k_depth + k_pass * ARRAY_SIZE, word=byte_off/4
                //   INT16: byte_offset = (col * k_depth + k_pass * ARRAY_SIZE) * 2, word=byte_off/4
                begin : wgt_cmd_blk
                    reg [31:0] elem_off;
                    reg [31:0] byte_off;
                    elem_off = wgt_col_idx * k_depth[15:0] + k_pass * ARRAY_SIZE_16;
                    byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                    wgt_word_addr <= wgt_base + byte_off[17:2];
                    wgt_bsel <= byte_off[1:0];
                end
                wgt_pf_col <= 0;
                wgt_pf_ph  <= 2'd0;
                if (wgt_sh_ok && wgt_sh_pass == k_pass) begin
                    wgt_from_sh <= 1'b1;
                    wgt_sh_ok   <= 1'b0;
                    for (i = 0; i < ARRAY_SIZE; i = i + 1)
                        sa_wgt_data[i] <= wgt_sh[{wgt_col_idx, i[COL_W-1:0]}];
                    state <= S_WGT_EMIT;
                end else begin
                    wgt_from_sh <= 1'b0;
                    state <= S_WGT_LOAD;
                end
            end

            S_WGT_LOAD: begin
                // Fill sa_wgt_data[0..k_pass_remain-1] for current column, zero-pad rest
                // One 256-bit beat holds 32 INT8 / 16 INT16 starting at wgt_bsel.
                if (!wgt_read_issued) begin
                    // Phase 0: Issue SRAM read
                    wgt_rd_en   <= 1'b1;
                    wgt_rd_addr <= wgt_word_addr[WGT_ADDR_W-1:0];
                    wgt_read_issued <= 1'b1;
                    wgt_data_ready  <= 1'b0;
                end else if (!wgt_data_ready) begin
                    // Phase 1: Wait for SRAM read latency
                    wgt_data_ready <= 1'b1;
                end else begin
                    // Phase 2: Data available from wgt_rd_data
`ifdef DBG_DOTBUF
                    if (wgt_byte_idx == 0 && wgt_col_idx == 0 && sp_oh == 0 && sp_ow == 0 && k_pass == 0)
                        $fwrite(dbg_fh, "[WGT_RD] oc=%0d col=%0d pass=%0d addr=%0d data=0x%08x bsel=%0d\n",
                                oc_group, wgt_col_idx, k_pass, wgt_word_addr, wgt_rd_data[31:0], wgt_bsel);
`endif
                    begin : wgt_unpack_blk
                        integer ei;
                        reg [SRAM_B_W-1:0] shifted;
                        reg [5:0] avail_b;
                        reg [5:0] elems_this_word;
                        shifted = wgt_rd_data >> (wgt_bsel * 8);
                        avail_b = 6'd32 - {4'd0, wgt_bsel};
                        if (cfg_int16)
                            elems_this_word = {1'b0, avail_b[5:1]};
                        else
                            elems_this_word = avail_b;
                        for (ei = 0; ei < ARRAY_SIZE; ei = ei + 1) begin
                            if ((wgt_byte_idx + ei[15:0] < k_pass_remain)
                                    && (ei < elems_this_word)) begin
                                if (cfg_int16)
                                    sa_wgt_data[wgt_byte_idx[COL_W-1:0] + ei[COL_W-1:0]]
                                        <= $signed(shifted[16*ei +: 16]);
                                else
                                    sa_wgt_data[wgt_byte_idx[COL_W-1:0] + ei[COL_W-1:0]]
                                        <= {{8{shifted[8*ei+7]}}, shifted[8*ei +: 8]};
                            end
                        end
                        wgt_byte_idx <= wgt_byte_idx + elems_this_word[$clog2(ARRAY_SIZE):0];

                        if (wgt_byte_idx + elems_this_word >= k_pass_remain) begin
                            if (k_pass_remain < ARRAY_SIZE_16) begin
                                for (i = 0; i < ARRAY_SIZE; i = i + 1)
                                    if (i[COL_W-1:0] >= k_pass_remain[COL_W-1:0])
                                        sa_wgt_data[i[COL_W-1:0]] <= 0;
                            end
                            state <= S_WGT_EMIT;
                        end else begin
                            wgt_word_addr <= wgt_word_addr + SRAM_B_WORDS[15:0];
                            wgt_bsel <= 2'd0;
                            wgt_read_issued <= 1'b0;
                        end
                    end
                end
            end

            S_WGT_EMIT: begin
                // Pulse wgt_valid for this column
                sa_wgt_valid <= 1'b1;

                if (wgt_col_idx == COL_MAX) begin
                    // All columns loaded → go to spatial setup (compute act addr)
                    wgt_from_sh <= 1'b0;
                    wgt_held <= 1'b1;
                    wgt_held_pass <= k_pass;
                    state <= S_SPATIAL_SETUP;
                end else if (wgt_from_sh) begin
                    wgt_col_idx <= wgt_col_idx + 1;
                    for (i = 0; i < ARRAY_SIZE; i = i + 1)
                        sa_wgt_data[i] <= wgt_sh[{(wgt_col_idx + 1'b1), i[COL_W-1:0]}];
                    state <= S_WGT_EMIT;
                end else begin
                    // Next column
                    wgt_col_idx <= wgt_col_idx + 1;
                    wgt_byte_idx <= 0;
                    wgt_read_issued <= 1'b0;
                    wgt_data_ready  <= 1'b0;
                    begin : wgt_emit_next_blk
                        reg [31:0] elem_off;
                        reg [31:0] byte_off;
                        elem_off = (wgt_col_idx + 1) * k_depth[15:0]
                                 + k_pass * ARRAY_SIZE_16;
                        byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                        wgt_word_addr <= wgt_base + byte_off[17:2];
                        wgt_bsel <= byte_off[1:0];
                    end
                    state <= S_WGT_LOAD;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // ACTIVATION STREAM: send k_depth values, one per target row
            // ══════════════════════════════════════════════════════════════
            S_ACT_CMD: begin
                // First pixel of a k_pass waits for sa_ready and issues COMPUTE.
                // Later pixels (act_armed) skip that — the array stays in COMPUTE.
                if (act_armed || sa_ready) begin
                    if (!act_armed) begin
                        sa_cmd       <= MODE_COMPUTE;
                        sa_cmd_valid <= 1'b1;
                        act_armed    <= 1'b1;
                    end
                    act_cnt      <= 0;
                    act_byte_sel <= 2'd0;
                    act_read_issued <= 1'b0;
                    act_data_ready  <= 1'b0;
                    pf_age          <= 2'd0;

                    // Compute activation address for current (conv_fh, conv_fw)
                    begin : act_addr_blk
                        reg signed [15:0] ih, iw;
                        reg [31:0] elem_off;
                        reg [31:0] byte_off;
                        if (is_deconv) begin
                            // Deconv: address precomputed in S_SPATIAL_SETUP
                            if (deconv_skip) begin
                                // Already set by S_SPATIAL_SETUP
                            end else if (deconv_addr_valid) begin
                                elem_off = deconv_elem_off;
                                byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                                act_word_addr <= {2'd0, act_base} + byte_off[17:2];
                                act_byte_sel <= byte_off[1:0];
                            end
                        end else begin
                            ih = conv_ih_base + $signed({8'd0, conv_fh});
                            iw = conv_iw_base + $signed({8'd0, conv_fw});
                            conv_is_pad <= (ih < 0) || (ih >= $signed({1'b0, cfg_in_h}))
                                        || (iw < 0) || (iw >= $signed({1'b0, cfg_in_w}));
                            deconv_skip <= 1'b0;
                            // Tile-local or absolute image address based on mode
                            if (cfg_tile_h == 17'd0) begin
                                // Non-tiled: full image in SRAM, absolute coords
                                elem_off = (ih[15:0] * cfg_in_w + iw[15:0]) * cfg_in_c + conv_ch_cnt;
                            end else begin
                                // Tiled: use row/col within input tile, with channel offset
                                `ifdef DBG_DOTBUF
                                if (sp_oh == 0 && sp_ow == 0 && k_pass == 0 && conv_fh == 0 && conv_fw == 0)
                                    $display("[TILED_DBG] cfg_2d=%0d tile_h=%0d pad=%0d/%0d", cfg_2d_load, cfg_tile_h, cfg_pad_top, cfg_pad_left);
                                `endif
                                if (cfg_2d_load) begin
                                    // SRAM is packed-tile layout (leading pad rows/cols
                                    // left zero by 2D DMA). Same elem_off as 1D packed.
                                    elem_off = (({8'd0, sp_oh} * cfg_stride_h + {8'd0, conv_fh}) * tile_in_w
                                             + {8'd0, sp_ow} * cfg_stride_w + {8'd0, conv_fw}) * cfg_in_c
                                             + {8'd0, conv_ch_cnt};
                                    `ifdef DBG_DOTBUF
                                    if (sp_oh == 0 && sp_ow == 0 && k_pass == 0)
                                        $display("[2D_EOFF] cfg_2d=%0d fh=%0d fw=%0d pad=%0d/%0d elem_off=%0d",
                                                cfg_2d_load, conv_fh, conv_fw, cfg_pad_top, cfg_pad_left, elem_off);
                                    `endif
                                end else begin
                                    elem_off = (({8'd0, sp_oh} * cfg_stride_h + {8'd0, conv_fh}) * tile_in_w
                                             + {8'd0, sp_ow} * cfg_stride_w + {8'd0, conv_fw}) * cfg_in_c
                                             + {8'd0, conv_ch_cnt};
                                end
                            end
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            act_word_addr <= {2'd0, act_base} + byte_off[17:2];
`ifdef NPU_SIM_DEBUG
                            if (cfg_2d_load && sp_oh == 0 && sp_ow == 0 && k_pass < 20)
                                $display("[L2CMD] ty=%0d tx=%0d pass=%0d fh=%0d fw=%0d ch=%0d in_c=%0d elem=%0d addr=%0d remain=%0d",
                                        tile_y, tile_x, k_pass, conv_fh, conv_fw, conv_ch_cnt, cfg_in_c, elem_off,
                                        {2'd0, act_base} + byte_off[17:2], k_pass_remain);
`endif
                            act_byte_sel <= byte_off[1:0];
                        end
                    end
                    state <= S_ACT_LOAD;
                end
            end

            S_ACT_LOAD: begin
                // Read one 256-bit beat (8 consecutive 32-bit words)
                // If padding or deconv_skip, skip read and go directly to emit zeros
                `ifdef DBG_DOTBUF
                if (cfg_2d_load && sp_oh == 0 && sp_ow == 0 && (k_pass == 0 || k_pass == 16 || k_pass == 17))
                    $display("[ACT_LD2D] kp=%0d fh=%0d fw=%0d ih=%0d iw=%0d pad=%0d addr=%0d",
                            k_pass, conv_fh, conv_fw, conv_ih_base + $signed({8'd0, conv_fh}),
                            conv_iw_base + $signed({8'd0, conv_fw}), conv_is_pad, act_word_addr);
                `endif
                if (conv_is_pad || deconv_skip) begin
                    // Use cfg_in_zp for padding, matching CSIM dma_extract_tile behavior
                    pf_age <= 2'd0;
                    if (cfg_int16)
                        act_buf <= {16{cfg_in_zp}};
                    else
                        act_buf <= {32{cfg_in_zp[7:0]}};
                    state <= S_ACT_EMIT;
                end else if (!act_read_issued) begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= act_word_addr[ACT_ADDR_W-1:0];
                    act_read_issued <= 1'b1;
                    act_data_ready  <= 1'b0;
                end else if (!act_data_ready) begin
                    // Issue was last cycle; SRAM updates rdata on this
                    // posedge. Sample next cycle (NBA) — skipping this
                    // wait reads the previous beat.
                    act_data_ready <= 1'b1;
                    // Single-tap pass: EMIT can consume the held rdata next
                    // cycle (2 cycles after issue). Skip the extra sample beat.
                    if (is_tap_fast && !conv_is_pad && !deconv_skip) begin
                        act_use_rd <= 1'b1;
                        state <= S_ACT_EMIT;
                        // Issue pixel N+1 this cycle so EMIT of N (next
                        // cycle) can stay in EMIT: SRAM needs 2 cycles.
                        if (px_in_blk + 1 < blk_px_cnt) begin
                            begin : act_ahead_fill
                                reg [15:0] n_oh, n_ow;
                                reg [32:0] pack_n;
                                reg [31:0] byte_off_n;
                                reg [15:0] n_rw;
                                if (sp_ow + 1 >= out_tile_w) begin
                                    n_ow = 16'd0;
                                    n_oh = sp_oh + 16'd1;
                                end else begin
                                    n_ow = sp_ow + 16'd1;
                                    n_oh = sp_oh;
                                end
                                pack_n = conv_tap_byte(n_oh, n_ow, pass_fh,
                                                       pass_fw, pass_ch);
                                if (pack_n[32]) begin
                                    act_ahead  <= 1'b1;
                                    ahead_pad  <= 1'b1;
                                end else begin
                                    byte_off_n = pack_n[31:0];
                                    n_rw = cfg_int16
                                        ? ((16'd32 - {14'd0, byte_off_n[1:0]}) >> 1)
                                        : (16'd32 - {14'd0, byte_off_n[1:0]});
                                    if (n_rw >= k_pass_remain) begin
                                        act_rd_en    <= 1'b1;
                                        act_rd_addr  <= ({2'd0, act_base} + byte_off_n[17:2]);
                                        act_ahead    <= 1'b1;
                                        ahead_bsel   <= byte_off_n[1:0];
                                        ahead_pad    <= 1'b0;
                                    end
                                end
                            end
                        end
                    end
                end else begin
                    act_buf <= act_rd_data;
                    act_data_ready <= 1'b1;
                    // Prefetch the next 256-bit beat (used by INT16 unaligned
                    // leftover). Data arrives 2 cycles later.
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= act_word_addr[ACT_ADDR_W-1:0] + SRAM_B_WORDS[ACT_ADDR_W-1:0];
                    pf_age <= 2'd1;
`ifndef SYNTHESIS
                    if (cfg_2d_load && sp_oh == 0 && sp_ow == 0 && k_pass == 16 && tile_x == 0 && tile_y == 0 && act_cnt < 4)
                        $display("[L2RD] addr=%0d data=0x%08x cnt=%0d", act_word_addr[ACT_ADDR_W-1:0], act_rd_data[31:0], act_cnt);
`endif
`ifdef DBG_DOTBUF
                    if (sp_oh == 0 && sp_ow == 0 && k_pass < 2 && ((tile_x == 0 && tile_y == 0) || (tile_x == 1 && tile_y == 0)))
                        $fwrite(dbg_fh, "[RTL_RD] t=%0d tile(%0d,%0d) sp(%0d,%0d) pass=%0d act_addr=%0d act_data=0x%08x\n",
                                $time, tile_y, tile_x, sp_oh, sp_ow, k_pass, act_word_addr[ACT_ADDR_W-1:0], act_rd_data[31:0]);
`endif
                    state <= S_ACT_EMIT;
                end
            end

            S_ACT_EMIT: begin
                // Pack as many consecutive k-slots as the current 256-bit SRAM
                // beat still holds (up to 32 INT8 / 16 INT16), clipped by the
                // remaining channels at this (fh,fw) and the remaining rows of
                // this pass. sa_act_valid fires once the ROWS-wide vector is
                // complete, but only after the previous in-flight psum has
                // been collected.
                begin : act_emit_blk
                    integer npack, ei, remain_word, remain_k, remain_ch;
                    reg signed [DATA_W-1:0] lanes [0:ARRAY_SIZE-1];
                    reg [SRAM_B_W-1:0] shifted;
                    reg vec_done;

                    remain_k  = k_pass_remain - act_cnt;
                    remain_ch = cfg_in_c - conv_ch_cnt;
                    if (remain_k < 1) remain_k = 1;
                    if (remain_ch < 1) remain_ch = 1;

                    // 1×1 skipped the ACT_LOAD sample beat: rdata was
                    // registered last cycle and is stable this cycle.
                    if (act_use_rd) begin
                        shifted = act_rd_data >> (act_byte_sel * 8);
                        act_buf <= act_rd_data;
                        act_use_rd <= 1'b0;
                        // Next-pixel issue owns the SRAM port this cycle —
                        // do not prefetch the following beat.
                        if (!emit_1x1_next && !emit_1x1_stream) begin
                            act_rd_en   <= 1'b1;
                            act_rd_addr <= act_word_addr[ACT_ADDR_W-1:0]
                                         + SRAM_B_WORDS[ACT_ADDR_W-1:0];
                            pf_age <= 2'd1;
                        end
                    end else begin
                        shifted = act_buf >> (act_byte_sel * 8);
                    end
                    if (cfg_int16)
                        remain_word = (32 - act_byte_sel) >> 1;
                    else
                        remain_word = 32 - act_byte_sel;

                    for (ei = 0; ei < ARRAY_SIZE; ei = ei + 1) begin
                        if (cfg_int16)
                            lanes[ei] = $signed(shifted[16*ei +: 16]);
                        else
                            lanes[ei] = {{8{shifted[8*ei+7]}}, shifted[8*ei +: 8]};
                    end

                    npack = remain_word;
                    if (remain_k  < npack) npack = remain_k;
                    if (remain_ch < npack) npack = remain_ch;
                    if (npack > ARRAY_SIZE) npack = ARRAY_SIZE;

                    for (ei = 0; ei < ARRAY_SIZE; ei = ei + 1) begin
                        if (ei < npack)
                            sa_act_data[act_cnt + ei[15:0]] <= lanes[ei];
                    end
                    if (act_cnt == 16'd0) begin
                        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                            if (i >= npack)
                                sa_act_data[i] <= {DATA_W{1'b0}};
                        end
                    end

                    vec_done = (act_cnt + npack >= k_pass_remain);

                    // Vector ready, but the in-flight FIFO is full — hold
                    // the assembled lanes until a psum is collected.
                    if (vec_done && if_full && !if_collect) begin
                        if (act_use_rd) begin
                            act_buf    <= act_rd_data;
                            act_use_rd <= 1'b0;
                            act_ahead  <= 1'b0;
                        end
                    end else begin
                        sa_act_valid <= vec_done;
                        act_cnt      <= act_cnt + npack[15:0];
                        conv_ch_cnt  <= conv_ch_cnt + npack[15:0];

                        if (emit_1x1_stream) begin
                            flush_cnt <= 0;
                            pf_age    <= 2'd0;
                            act_use_rd <= 1'b1;
                            px_in_blk <= px_in_blk + 1;
                            begin : emit_1x1_stream_blk
                                reg [15:0] n_oh, n_ow, n2_oh, n2_ow;
                                reg [32:0] pack2;
                                reg [31:0] byte_off_n;
                                reg [15:0] n_rw;
                                if (sp_ow + 1 >= out_tile_w) begin
                                    n_ow = 16'd0;
                                    n_oh = sp_oh + 16'd1;
                                end else begin
                                    n_ow = sp_ow + 16'd1;
                                    n_oh = sp_oh;
                                end
                                if (n_ow + 1 >= out_tile_w) begin
                                    n2_ow = 16'd0;
                                    n2_oh = n_oh + 16'd1;
                                end else begin
                                    n2_ow = n_ow + 16'd1;
                                    n2_oh = n_oh;
                                end
                                sp_ow <= n_ow;
                                sp_oh <= n_oh;
                                act_byte_sel <= ahead_bsel;
                                conv_is_pad  <= ahead_pad;
                                deconv_skip  <= 1'b0;
                                conv_fh <= pass_fh;
                                conv_fw <= pass_fw;
                                conv_ch_cnt <= pass_ch;
                                conv_ih_base <= $signed({1'b0, tile_oh_origin + n_oh})
                                              * $signed({1'b0, cfg_stride_h[7:0]})
                                              - $signed({1'b0, cfg_pad_top[7:0]});
                                conv_iw_base <= $signed({1'b0, tile_ow_origin + n_ow})
                                              * $signed({1'b0, cfg_stride_w[7:0]})
                                              - $signed({1'b0, cfg_pad_left[7:0]});
                                act_cnt <= 16'd0;
                                if (px_in_blk + 2 < blk_px_cnt) begin
                                    pack2 = conv_tap_byte(n2_oh, n2_ow, pass_fh,
                                                          pass_fw, pass_ch);
                                    if (pack2[32]) begin
                                        act_ahead <= 1'b1;
                                        ahead_pad <= 1'b1;
                                    end else begin
                                        byte_off_n = pack2[31:0];
                                        n_rw = cfg_int16
                                            ? ((16'd32 - {14'd0, byte_off_n[1:0]}) >> 1)
                                            : (16'd32 - {14'd0, byte_off_n[1:0]});
                                        if (n_rw >= k_pass_remain) begin
                                            act_rd_en    <= 1'b1;
                                            act_rd_addr  <= ({2'd0, act_base} + byte_off_n[17:2]);
                                            act_word_addr <= {2'd0, act_base} + byte_off_n[17:2];
                                            act_ahead    <= 1'b1;
                                            ahead_bsel   <= byte_off_n[1:0];
                                            ahead_pad    <= 1'b0;
                                        end else begin
                                            act_ahead <= 1'b0;
                                        end
                                    end
                                end else begin
                                    act_ahead <= 1'b1;
                                end
                                state <= S_ACT_EMIT;
                            end
                        end else if (emit_1x1_next) begin
                            flush_cnt <= 0;
                            pf_age    <= 2'd0;
                            act_use_rd <= 1'b0;
                            act_ahead  <= 1'b0;
                            px_in_blk <= px_in_blk + 1;
                            begin : emit_fold_flush
                                reg [15:0] next_oh, next_ow;
                                reg [32:0] pack_n;
                                reg [31:0] byte_off_n;
                                if (sp_ow + 1 >= out_tile_w) begin
                                    next_ow = 16'd0;
                                    next_oh = sp_oh + 16'd1;
                                    sp_ow <= 16'd0;
                                    sp_oh <= sp_oh + 16'd1;
                                end else begin
                                    next_ow = sp_ow + 16'd1;
                                    next_oh = sp_oh;
                                    sp_ow <= sp_ow + 16'd1;
                                end
                                pack_n = conv_tap_byte(next_oh, next_ow, pass_fh,
                                                       pass_fw, pass_ch);
                                conv_is_pad <= pack_n[32];
                                deconv_skip <= 1'b0;
                                conv_fh <= pass_fh;
                                conv_fw <= pass_fw;
                                conv_ch_cnt <= pass_ch;
                                conv_ih_base <= $signed({1'b0, tile_oh_origin + next_oh})
                                              * $signed({1'b0, cfg_stride_h[7:0]})
                                              - $signed({1'b0, cfg_pad_top[7:0]});
                                conv_iw_base <= $signed({1'b0, tile_ow_origin + next_ow})
                                              * $signed({1'b0, cfg_stride_w[7:0]})
                                              - $signed({1'b0, cfg_pad_left[7:0]});
                                act_cnt <= 16'd0;
                                byte_off_n = pack_n[31:0];
                                act_word_addr <= {2'd0, act_base} + byte_off_n[17:2];
                                act_byte_sel <= byte_off_n[1:0];
                                if (pack_n[32]) begin
                                    act_read_issued <= 1'b0;
                                    act_data_ready  <= 1'b0;
                                end else begin
                                    act_rd_en       <= 1'b1;
                                    act_rd_addr     <= ({2'd0, act_base} + byte_off_n[17:2]);
                                    act_read_issued <= 1'b1;
                                    act_data_ready  <= 1'b0;
                                end
                                state <= S_ACT_LOAD;
                            end
                        end else if (emit_1x1_last) begin
                            flush_cnt <= 0;
                            pf_age    <= 2'd0;
                            act_use_rd <= 1'b0;
                            act_ahead  <= 1'b0;
                            state     <= S_PSUM_COLLECT;
                        end else if (vec_done) begin
                            flush_cnt <= 0;
                            pf_age    <= 2'd0;
                            act_use_rd <= 1'b0;
                            state     <= S_ACT_FLUSH;
                        end else if (conv_ch_cnt + npack[15:0] >= cfg_in_c) begin
                            conv_ch_cnt <= 16'd0;
                            if (conv_fw + 1 >= {8'd0, cfg_kernel_w}) begin
                                conv_fw <= 0;
                                conv_fh <= conv_fh + 1;
                            end else begin
                                conv_fw <= conv_fw + 1;
                            end
                            act_read_issued <= 1'b0;
                            act_data_ready  <= 1'b0;
                            pf_age          <= 2'd0;
                            state <= S_ACT_LOAD;
                            begin : next_pos_blk
                                reg signed [15:0] nih, niw;
                                reg signed [15:0] neh, new_;
                                reg [31:0] elem_off_n;
                                reg [31:0] byte_off_n;
                                reg [7:0] next_fw, next_fh;
                                if (conv_fw + 1 >= {8'd0, cfg_kernel_w}) begin
                                    next_fw = 0;
                                    next_fh = conv_fh + 1;
                                end else begin
                                    next_fw = conv_fw + 1;
                                    next_fh = conv_fh;
                                end
                                if (is_deconv) begin
                                    neh = conv_ih_base - $signed({8'd0, next_fh});
                                    new_ = conv_iw_base - $signed({8'd0, next_fw});
                                    begin : deconv_next_addr_blk
                                        reg [47:0] qh_n, qw_n;
                                        reg [15:0] nih_u, niw_u;
                                        qh_n = (neh[15:0] * recip_deconv_h);
                                        if (deconv_h_shift1) nih_u = qh_n[46:31];
                                        else                  nih_u = qh_n[47:32];
                                        qw_n = (new_[15:0] * recip_deconv_w);
                                        if (deconv_w_shift1) niw_u = qw_n[46:31];
                                        else                  niw_u = qw_n[47:32];
                                        if ((neh < 0) || (neh >= $signed({1'b0, deconv_exp_h}))
                                            || (new_ < 0) || (new_ >= $signed({1'b0, deconv_exp_w}))
                                            || (neh[15:0] != nih_u * deconv_step_h[8:0])
                                            || (new_[15:0] != niw_u * deconv_step_w[8:0])) begin
                                            conv_is_pad  <= 1'b0;
                                            deconv_skip  <= 1'b1;
                                        end else begin
                                            nih = $signed({17'd0, nih_u});
                                            niw = $signed({17'd0, niw_u});
                                            conv_is_pad <= (nih < 0) || (nih >= $signed({1'b0, cfg_in_h}))
                                                        || (niw < 0) || (niw >= $signed({1'b0, cfg_in_w}));
                                            deconv_skip <= 1'b0;
                                            elem_off_n = (nih[15:0] * cfg_in_w + niw[15:0]) * cfg_in_c;
                                            byte_off_n = cfg_int16 ? (elem_off_n << 1) : elem_off_n;
                                            act_word_addr <= {2'd0, act_base} + byte_off_n[17:2];
                                            act_byte_sel <= byte_off_n[1:0];
                                        end
                                    end
                                end else begin
                                    nih = conv_ih_base + $signed({8'd0, next_fh});
                                    niw = conv_iw_base + $signed({8'd0, next_fw});
                                    conv_is_pad <= (nih < 0) || (nih >= $signed({1'b0, cfg_in_h}))
                                                || (niw < 0) || (niw >= $signed({1'b0, cfg_in_w}));
                                    deconv_skip <= 1'b0;
                                    if (cfg_tile_h == 17'd0) begin
                                        elem_off_n = (nih[15:0] * cfg_in_w + niw[15:0]) * cfg_in_c;
                                    end else begin
                                        elem_off_n = (({8'd0, sp_oh} * cfg_stride_h + {8'd0, next_fh}) * tile_in_w
                                                   + {8'd0, sp_ow} * cfg_stride_w + {8'd0, next_fw}) * cfg_in_c;
                                    end
                                    byte_off_n = cfg_int16 ? (elem_off_n << 1) : elem_off_n;
                                    act_word_addr <= {2'd0, act_base} + byte_off_n[17:2];
                                    act_byte_sel <= byte_off_n[1:0];
                                end
                            end
                        end else if (npack == remain_word) begin
                            if (pf_age == 2'd2 && !cfg_int16) begin
                                act_buf <= act_rd_data;
                                act_byte_sel <= 2'd0;
                                act_word_addr <= act_word_addr + SRAM_B_WORDS[15:0];
                                act_rd_en   <= 1'b1;
                                act_rd_addr <= act_word_addr[ACT_ADDR_W-1:0]
                                             + (SRAM_B_WORDS[ACT_ADDR_W-1:0] << 1);
                                pf_age <= 2'd1;
                            end else if (pf_age == 2'd2) begin
                                act_read_issued <= 1'b1;
                                act_data_ready  <= 1'b1;
                                act_byte_sel <= 2'd0;
                                act_word_addr <= act_word_addr + SRAM_B_WORDS[15:0];
                                state <= S_ACT_LOAD;
                            end else begin
                                act_byte_sel <= 2'd0;
                                act_word_addr <= act_word_addr + SRAM_B_WORDS[15:0];
                                act_read_issued <= 1'b0;
                                state <= S_ACT_LOAD;
                            end
                        end else begin
                            act_byte_sel <= cfg_int16
                                ? (act_byte_sel + {npack[3:0], 1'b0})
                                : (act_byte_sel + npack[4:0]);
                        end
                    end
                end
            end

            S_ACT_FLUSH: begin
                // Vector has been presented. The prefix pusher records
                // (px_in_blk, k_pass) this cycle. If more pixels remain,
                // start gathering the next one immediately.
                if (px_in_blk + 1 < blk_px_cnt) begin
                    px_in_blk <= px_in_blk + 1;
                    begin : flush_next_px
                        reg [15:0] next_oh, next_ow;
                        reg [32:0] pack_n;
                        reg [31:0] byte_off_n;
                        if (sp_ow + 1 >= out_tile_w) begin
                            next_ow = 16'd0;
                            next_oh = sp_oh + 16'd1;
                            sp_ow <= 16'd0;
                            sp_oh <= sp_oh + 16'd1;
                        end else begin
                            next_ow = sp_ow + 16'd1;
                            next_oh = sp_oh;
                            sp_ow <= sp_ow + 16'd1;
                        end
                        if (is_tap_fast) begin
                            // Same tap as this k_pass — skip SPATIAL_SETUP
                            // and ACT_CMD (2 cycles) and kick the SRAM read.
                            pack_n = conv_tap_byte(next_oh, next_ow, pass_fh,
                                                   pass_fw, pass_ch);
                            conv_is_pad <= pack_n[32];
                            deconv_skip <= 1'b0;
                            conv_fh <= pass_fh;
                            conv_fw <= pass_fw;
                            conv_ch_cnt <= pass_ch;
                            conv_ih_base <= $signed({1'b0, tile_oh_origin + next_oh})
                                          * $signed({1'b0, cfg_stride_h[7:0]})
                                          - $signed({1'b0, cfg_pad_top[7:0]});
                            conv_iw_base <= $signed({1'b0, tile_ow_origin + next_ow})
                                          * $signed({1'b0, cfg_stride_w[7:0]})
                                          - $signed({1'b0, cfg_pad_left[7:0]});
                            act_cnt <= 16'd0;
                            byte_off_n = pack_n[31:0];
                            act_word_addr <= {2'd0, act_base} + byte_off_n[17:2];
                            act_byte_sel <= byte_off_n[1:0];
                            if (pack_n[32]) begin
                                act_read_issued <= 1'b0;
                                act_data_ready  <= 1'b0;
                            end else begin
                                act_rd_en       <= 1'b1;
                                act_rd_addr     <= ({2'd0, act_base} + byte_off_n[17:2]);
                                act_read_issued <= 1'b1;
                                act_data_ready  <= 1'b0;
                            end
                            state <= S_ACT_LOAD;
                        end else begin
                            state <= S_SPATIAL_SETUP;
                        end
                    end
                end else begin
                    state <= S_PSUM_COLLECT;
                end
            end

            // Wait until the last in-flight psum of this block has been
            // written into px_acc_buf (the write itself happens in the
            // prefix collector).
            S_PSUM_COLLECT: begin
                if (if_empty || (if_count == {{(IF_CNT_W-1){1'b0}}, 1'b1} && if_collect)) begin
                    if (k_fin_cnt >= k_pass_max) begin
                        // Wait if a previous block's PPU WB is still draining
                        // or this OC's param cache is not ready.
                        if (ppu_bg || !param_ready) begin
                        end else begin
                        ppu_px <= 0;
                        drain_col <= 0;
                        wb_px <= 0;
                        wb_ch <= 0;
                        spw_oh <= blk_oh_start;
                        spw_ow <= blk_ow_start;
                        feed_left <= 1'b1;
                        ppu_st <= 2'd0;
                        ppu_ahead <= 1'b0;
                        ppu_pend <= 1'b0;
                        ppu_blk_save <= blk_px_cnt;
                        ppu_oc_save <= oc_group;
                        ppu_col_last <= col_last;
                        acc_rd_bank <= acc_wr_bank;
                        acc_wr_bank <= ~acc_wr_bank;
                        if (px_remaining > {11'd0, blk_px_cnt}) begin
                            ppu_bg  <= 1'b1;
                            ppu_ovl <= 1'b1;
                            state   <= S_PIXEL_NEXT;
                        end else if (oc_group + 1 < oc_groups_total) begin
                            ppu_bg  <= 1'b1;
                            ppu_ovl <= 1'b1;
                            state   <= S_PIXEL_NEXT;
                        end else begin
                            ppu_bg  <= 1'b0;
                            ppu_ovl <= 1'b0;
                            state   <= S_PPU_STREAM;
                        end
                        end
                    end else begin
                        begin : k_pass_next_blk
                            reg [15:0] nxt;
                            nxt = wgt_pass_after(k_pass, k_first, k_fin_cnt,
                                                 k_pass_max);
                            k_pass <= nxt;
                            k_fin_cnt <= k_fin_cnt + 16'd1;
                            wgt_col_idx <= 0;
                            px_in_blk <= 0;
                            act_armed <= 1'b0;
                            sp_oh <= blk_oh_start;
                            sp_ow <= blk_ow_start;
                            if (wgt_held && (nxt == wgt_held_pass)) begin
                                k_pass_remain <= (nxt == k_pass_max)
                                    ? (k_depth - nxt * ARRAY_SIZE_16)
                                    : ARRAY_SIZE_16;
                                state <= S_SPATIAL_SETUP;
                            end else
                                state <= S_WGT_CMD;
                        end
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            // PARAM READ: 4 words per channel — now iterates drain_col
            // as the output channel index (0..ARRAY_SIZE-1)
            // ══════════════════════════════════════════════════════════════
            S_PARAM_LOAD: begin
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= param_base + drain_col * 4 + param_word_idx;
                    param_read_issued <= 1'b1;
                    param_data_ready  <= 1'b0;
                end else if (!param_data_ready) begin
                    // Wait for SRAM read latency (1 cycle)
                    param_data_ready <= 1'b1;
                end else begin
                    param_buf[param_word_idx] <= param_rd_data;
`ifndef SYNTHESIS
                    if (cfg_2d_load && drain_col == 0 && tile_x == 0 && tile_y == 0 && sp_oh == 0 && sp_ow == 0)
                        $display("[L2PRM] idx=%0d addr=%0d data=0x%08x", param_word_idx, param_rd_addr, param_rd_data);
`endif
                    if (param_word_idx == 3'd3) begin
                        ppu_mult_m     <= param_buf[0][14:0];
                        ppu_shift_s    <= param_buf[0][21:16];
                        ppu_zero_point <= $signed(param_buf[1][15:0]);
                        // Use param_rd_data for word 3 (param_buf[3] not yet updated this cycle)
                        ppu_bias       <= $signed({param_rd_data[15:0], param_buf[2],
                                                   param_buf[1][31:16]});
                        state <= S_PPU_FEED;
                    end else begin
                        param_word_idx <= param_word_idx + 1;
                        param_read_issued <= 1'b0;
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            // PPU FEED: send ONE dot product (dot_buf[drain_col]) to PPU
            // ══════════════════════════════════════════════════════════════
            S_PPU_FEED: begin
                ppu_acc_in   <= px_acc_buf[{acc_rd_bank, ppu_px[3:0], drain_col[3:0]}];
                ppu_in_valid <= 1'b1;
`ifndef SYNTHESIS
                if (cfg_2d_load && drain_col == 0 && tile_x == 0 && tile_y == 0 && sp_oh == 0 && sp_ow == 0)
                    $display("[L2PPU] drain=%0d acc=%0d bias=%0d M=%0d S=%0d zp=%0d",
                            drain_col, px_acc_buf[{acc_rd_bank, ppu_px[3:0], drain_col[3:0]}],
                            $signed({param_buf[3][15:0], param_buf[2], param_buf[1][31:16]}),
                            param_buf[0][14:0], param_buf[0][21:16],
                            $signed(param_buf[1][15:0]));
`endif
                state        <= S_PPU_WAIT;
            end

            S_PPU_WAIT: begin
                // Wait for PPU output (4-cycle pipeline)
                if (ppu_out_valid) begin
                    state <= S_WRITEBACK;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // WRITEBACK: pack output bytes and write SRAM words
            // Output NHWC layout: contiguous bytes across pixels.
            // wb_pos tracks byte position within the current 32-bit word,
            // persisting across pixels to support OUT_C not a multiple of 4.
            // ══════════════════════════════════════════════════════════════
            S_WRITEBACK: begin
                // Address for the current word (same formula, correct for all OUT_C)
                // Pack output elements into wb_pack using wb_pos for byte position
                if (cfg_int16) begin
                    case (wb_pos[0])
                        1'b0: wb_pack[15:0] <= ppu_out_data;
                        1'b1: begin
                            // Write full word (2 elements accumulated)
                            act_wr_en   <= 1'b1;
                            act_wr_addr <= out_base +
                                (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                  + oc_group * {11'd0, grp_oc}
                                  + ({12'd0, drain_col} & ~17'd1)) >> 1);
                            act_wr_data <= {ppu_out_data, wb_pack[15:0]};
                            `ifndef SYNTHESIS
                            if (tile_x == 1 && tile_y == 0 && sp_oh == 0 && sp_ow == 0)
                                $display("[CMP_WB] t=%0t drain=%0d col_last=%0d addr=%0d data=0x%04x%04x",
                                         $time, drain_col, col_last,
                                         out_base + (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                           + oc_group * {11'd0, grp_oc}
                                           + ({12'd0, drain_col} & ~17'd1)) >> 1),
                                         ppu_out_data, wb_pack[15:0]);
                            if (tile_x == 3 && tile_y == 0 && sp_oh == 0 && sp_ow == 0)
                                $display("[CMP_WB_BORDER] t=%0t drain=%0d col_last=%0d out_tw=%0d addr=%0d data=0x%04x%04x",
                                         $time, drain_col, col_last, out_tile_w,
                                         out_base + (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                           + oc_group * {11'd0, grp_oc}
                                           + ({12'd0, drain_col} & ~17'd1)) >> 1),
                                         ppu_out_data, wb_pack[15:0]);
                            if (tile_x == 0 && tile_y == 1 && sp_oh == 0 && sp_ow == 0)
                                $display("[CMP_WB_ROW2] t=%0t drain=%0d col_last=%0d out_th=%0d out_tw=%0d addr=%0d data=0x%04x%04x",
                                         $time, drain_col, col_last, out_tile_h, out_tile_w,
                                         out_base + (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                           + oc_group * {11'd0, grp_oc}
                                           + ({12'd0, drain_col} & ~17'd1)) >> 1),
                                         ppu_out_data, wb_pack[15:0]);
                            `endif
                        end
                    endcase
                    wb_pos <= {1'b0, ~wb_pos[0]};  // toggle: 0->1->0->1...
                end else begin
                    case (wb_pos)
                        2'd0: wb_pack[7:0]   <= ppu_out_data[7:0];
                        2'd1: wb_pack[15:8]  <= ppu_out_data[7:0];
                        2'd2: wb_pack[23:16] <= ppu_out_data[7:0];
                        2'd3: begin
                            // Write full word (4 bytes accumulated)
                            act_wr_en   <= 1'b1;
                            act_wr_addr <= out_base +
                                (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                  + oc_group * {11'd0, grp_oc}
                                  + ({12'd0, drain_col} & ~17'd3)) >> 2);
                            act_wr_data <= {ppu_out_data[7:0], wb_pack[23:0]};
                        end
                    endcase
                    wb_pos <= wb_pos + 2'd1;
                end

                // Advance to next channel or finish pixel
                if (drain_col == col_last) begin
                    // Last channel of this PPU pixel — partial-word flush only
                    // when this is the LAST pixel of the whole tile (sp tracks
                    // the PPU pixel during the block PPU loop).
                    if (sp_ow + 1 >= out_tile_w && sp_oh + 1 >= out_tile_h) begin
                        // Last pixel — flush any partial word
                        if (cfg_int16) begin
                            if (wb_pos[0] != 1'b1) begin
                                act_wr_en   <= 1'b1;
                                act_wr_addr <= out_base +
                                    (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                      + oc_group * {11'd0, grp_oc}
                                      + ({12'd0, drain_col} & ~17'd1)) >> 1);
                                act_wr_data <= {17'd0, ppu_out_data};
                            end
                        end else begin
                            if (wb_pos != 2'd3) begin
                                act_wr_en   <= 1'b1;
                                // Use (drain_col - wb_pos) instead of (drain_col & ~3)
                                // to correctly compute the start-of-word byte offset
                                // when wb_pack carries over bytes from previous pixels
                                // for non-word-aligned OUT_C.
                                act_wr_addr <= out_base +
                                    (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                      + oc_group * {11'd0, grp_oc}
                                      + ({14'd0, drain_col} - {14'd0, wb_pos})) >> 2);
                                case (wb_pos)
                                    2'd0: act_wr_data <= {24'd0, ppu_out_data[7:0]};
                                    2'd1: act_wr_data <= {17'd0, ppu_out_data[7:0], wb_pack[7:0]};
                                    2'd2: act_wr_data <= {8'd0,  ppu_out_data[7:0], wb_pack[15:0]};
                                    default: act_wr_data <= 32'd0; // unreachable
                                endcase
                            end
                        end
                    end
                    // 1b: more PPU pixels left in this block?
                    if (ppu_px + 1 < blk_px_cnt) begin
                        ppu_px <= ppu_px + 1;
                        drain_col <= 0;
                        param_word_idx <= 0;
                        param_read_issued <= 1'b0;
                        param_data_ready  <= 1'b0;
                        if (sp_ow + 1 >= out_tile_w) begin
                            sp_ow <= 0;
                            sp_oh <= sp_oh + 1;
                        end else begin
                            sp_ow <= sp_ow + 1;
                        end
                        state <= S_PARAM_LOAD;
                    end else begin
                        state <= S_PIXEL_NEXT;  // block done → block advance
                    end
                end else begin
                    drain_col <= drain_col + 1;
                    param_word_idx <= 0;
                    param_read_issued <= 1'b0;
                    param_data_ready  <= 1'b0;
                    state <= S_PARAM_LOAD;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // SPATIAL SETUP: compute per-pixel activation address
            // ══════════════════════════════════════════════════════════════
            S_SPATIAL_SETUP: begin
                // Compute input window origin for output pixel (sp_oh, sp_ow)
                // Use global output coordinates for input address & padding check
                if (is_deconv) begin
                    // Deconv: ih_base = oh + pad_top (fh subtracted later)
                    conv_ih_base <= $signed({1'b0, tile_oh_origin + sp_oh})
                                  + $signed({1'b0, cfg_pad_top[7:0]});
                    conv_iw_base <= $signed({1'b0, tile_ow_origin + sp_ow})
                                  + $signed({1'b0, cfg_pad_left[7:0]});
                    // Precompute deconv address here (avoid deep combinational
                    // path in S_ACT_CMD). conv_ih_base is non-blocking so use
                    // local blocking vars for the deconv calc.
                    // FIX (2026-08-18 deconv bit-rot): fh/fw/ch must be the
                    // PASS-START values — conv_fh/conv_fw/conv_ch_cnt still
                    // hold the previous pixel's last tap here (they are only
                    // reassigned by spatial_setup_blk below, NBA). Decompose
                    // flat_start locally, mirroring spatial_setup_blk.
                    begin : deconv_precompute_blk
                        reg signed [15:0] eh_local, ew_local;
                        reg [47:0] qh, qw;
                        reg [15:0] ih_u, iw_u;
                        reg signed [15:0] ih_s, iw_s;
                        reg [15:0] flat_loc, rem_kw_loc;
                        reg [15:0] kw_x_inc_loc;
                        reg [47:0] q_full_loc, q_fw_loc, q_ch_loc;
                        reg [7:0]  fh_loc, fw_loc;
                        reg [15:0] ch_loc;
                        flat_loc = k_pass * ARRAY_SIZE_16;
                        kw_x_inc_loc = {8'd0, cfg_kernel_w} * cfg_in_c;
                        if (cfg_kernel_h == 8'd1 && cfg_kernel_w == 8'd1) begin
                            fh_loc = 8'd0;
                            fw_loc = 8'd0;
                            ch_loc = flat_loc;
                        end else begin
                            q_full_loc = (flat_loc * recip_kw_x_inc) >> 32;
                            fh_loc = q_full_loc[7:0];
                            rem_kw_loc = flat_loc - q_full_loc[15:0] * kw_x_inc_r;
                            if (in_c_r == 16'd1) begin
                                fw_loc = rem_kw_loc[7:0];
                                ch_loc = 16'd0;
                            end else begin
                                q_fw_loc = (rem_kw_loc * recip_in_c) >> 32;
                                fw_loc = q_fw_loc[7:0];
                                q_ch_loc = (flat_loc * recip_in_c) >> 32;
                                ch_loc = flat_loc - q_ch_loc[15:0] * in_c_r;
                            end
                        end
                        eh_local = $signed({1'b0, tile_oh_origin + sp_oh})
                                 + $signed({1'b0, cfg_pad_top[7:0]})
                                 - $signed({8'd0, fh_loc});
                        ew_local = $signed({1'b0, tile_ow_origin + sp_ow})
                                 + $signed({1'b0, cfg_pad_left[7:0]})
                                 - $signed({8'd0, fw_loc});
                        qh = (eh_local[15:0] * recip_deconv_h);
                        if (deconv_h_shift1) ih_u = qh[46:31];
                        else                  ih_u = qh[47:32];
                        qw = (ew_local[15:0] * recip_deconv_w);
                        if (deconv_w_shift1) iw_u = qw[46:31];
                        else                  iw_u = qw[47:32];
                        if ((eh_local < 0) || (eh_local >= $signed({1'b0, deconv_exp_h}))
                            || (ew_local < 0) || (ew_local >= $signed({1'b0, deconv_exp_w}))
                            || (eh_local[15:0] != ih_u * deconv_step_h[8:0])
                            || (ew_local[15:0] != iw_u * deconv_step_w[8:0])) begin
                            deconv_skip  <= 1'b1;
                            deconv_addr_valid <= 1'b0;
                        end else begin
                            ih_s = $signed({17'd0, ih_u});
                            iw_s = $signed({17'd0, iw_u});
                            conv_is_pad <= (ih_s < 0) || (ih_s >= $signed({1'b0, cfg_in_h}))
                                        || (iw_s < 0) || (iw_s >= $signed({1'b0, cfg_in_w}));
                            deconv_skip <= 1'b0;
                            deconv_addr_valid <= 1'b1;
                            deconv_elem_off <= (ih_u * cfg_in_w + iw_u) * cfg_in_c + ch_loc;
                        end
                    end
                end else begin
                    conv_ih_base <= $signed({1'b0, tile_oh_origin + sp_oh}) * $signed({1'b0, cfg_stride_h[7:0]})
                                  - $signed({1'b0, cfg_pad_top[7:0]});
                    conv_iw_base <= $signed({1'b0, tile_ow_origin + sp_ow}) * $signed({1'b0, cfg_stride_w[7:0]})
                                  - $signed({1'b0, cfg_pad_left[7:0]});
                end

                // Compute starting (fh, fw, ch) from k_pass offset
                // flat_start = k_pass * ARRAY_SIZE
                // fh = flat_start / (kw * in_c)
                // fw = (flat_start / in_c) % kw
                // ch = flat_start % in_c
                begin : spatial_setup_blk
                    reg [15:0] flat_start;
                    reg [15:0] kw_x_inc;
                    flat_start = k_pass * ARRAY_SIZE_16;
                    kw_x_inc = {8'd0, cfg_kernel_w} * cfg_in_c;
                    if (cfg_kernel_h == 8'd1 && cfg_kernel_w == 8'd1) begin
                        // 1×1 conv: simple contiguous addressing
                        conv_fh <= 0;
                        conv_fw <= 0;
                        conv_ch_cnt <= flat_start;
                        pass_fh <= 8'd0;
                        pass_fw <= 8'd0;
                        pass_ch <= flat_start;
                    end else begin
                        // General conv: decompose flat_start into (fh, fw, ch)
                        // using multiply-by-reciprocal (no division)
                        begin : recip_decomp_blk
                            reg [47:0] q_full;    // flat_start / kw_x_inc (16b*32b=48b)
                            reg [15:0] rem_kw;     // flat_start % kw_x_inc
                            reg [47:0] q_fw;       // rem_kw / in_c
                            reg [47:0] q_ch;       // flat_start / in_c
                            // fh = flat_start / kw_x_inc
                            q_full = (flat_start * recip_kw_x_inc) >> 32;
                            conv_fh <= q_full[7:0];
                            pass_fh <= q_full[7:0];
                            rem_kw = flat_start - q_full[15:0] * kw_x_inc_r;
                            `ifdef DBG_DOTBUF
                            if (sp_oh == 0 && sp_ow == 0 && (k_pass == 0 || k_pass == 8 || k_pass == 16))
                                $display("[KP_DECOMP] kp=%0d flat=%0d recip=%0d kw_inc=%0d q=%0d fh=%0d rem=%0d",
                                        k_pass, flat_start, recip_kw_x_inc, kw_x_inc_r, q_full, q_full[7:0], rem_kw);
                            `endif
                            // fw = rem_kw / in_c (special case in_c=1)
                            if (in_c_r == 16'd1) begin
                                conv_fw <= rem_kw[7:0];
                                pass_fw <= rem_kw[7:0];
                                conv_ch_cnt <= 16'd0;
                                pass_ch <= 16'd0;
                            end else begin
                                q_fw = (rem_kw * recip_in_c) >> 32;
                                conv_fw <= q_fw[7:0];
                                pass_fw <= q_fw[7:0];
                                // ch = flat_start % in_c
                                q_ch = (flat_start * recip_in_c) >> 32;
                                conv_ch_cnt <= flat_start - q_ch[15:0] * in_c_r;
                                pass_ch <= flat_start - q_ch[15:0] * in_c_r;
                            end
                        end
                    end
                end

                conv_elem_cnt <= 0;
                drain_col <= 0;
                // Note: wb_pack and wb_pos persist across pixels for non-aligned OUT_C
                // Clear partial_sum on first pass of new pixel
                if (k_pass == 0) begin
                    for (i = 0; i < ARRAY_SIZE; i = i + 1)
                        dot_buf[i] <= 0;
                end
`ifdef DBG_DOTBUF
                if (tile_x == 0 && tile_y == 0)
                    $fwrite(dbg_fh, "[RTL_SP] t=%0d sp(%0d,%0d) k_pass=%0d conv_fh=%0d conv_fw=%0d\n",
                            $time, sp_oh, sp_ow, k_pass, conv_fh, conv_fw);
`endif
                state <= S_ACT_CMD;
            end

            // ══════════════════════════════════════════════════════════════
            // PIXEL NEXT (1b): advance to the next 16-pixel block, or OC_NEXT
            // when all blocks of this oc_group/tile are done.
            // sp currently sits at the last pixel of the finished block.
            // ══════════════════════════════════════════════════════════════
            S_PIXEL_NEXT: begin
                px_in_blk <= 0;
                k_fin_cnt <= 16'd0;
                act_armed <= 1'b0;
                if (!ppu_bg)
                    ppu_px <= 0;
                begin : blk_adv_blk
                    reg [15:0] rem_next;
                    rem_next = px_remaining - {11'd0, blk_px_cnt};
                    if (rem_next == 16'd0) begin
                        // PPU WB uses latched oc/col; safe to enter next OC now.
                        state <= S_OC_NEXT;
                    end else begin
                        px_remaining <= rem_next;
                        blk_px_cnt   <= (rem_next >= 16'd16) ? 5'd16 : rem_next[4:0];
                        // Advance sp to the next block's first pixel (row-major)
                        if (sp_ow + 1 >= out_tile_w) begin
                            sp_ow <= 0;
                            sp_oh <= sp_oh + 1;
                            blk_ow_start <= 0;
                            blk_oh_start <= sp_oh + 1;
                        end else begin
                            sp_ow <= sp_ow + 1;
                            blk_ow_start <= sp_ow + 1;
                            blk_oh_start <= sp_oh;
                        end
                        wgt_col_idx <= 0;
                        if (wgt_held) begin
                            k_pass  <= wgt_held_pass;
                            k_first <= wgt_held_pass;
                            k_pass_remain <= (wgt_held_pass == k_pass_max)
                                ? (k_depth - wgt_held_pass * ARRAY_SIZE_16)
                                : ARRAY_SIZE_16;
                            state <= S_SPATIAL_SETUP;
                        end else begin
                            k_pass  <= 16'd0;
                            k_first <= 16'd0;
                            state <= S_WGT_CMD;
                        end
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            S_OC_NEXT: begin
                if (oc_group + 1 >= oc_groups_total) begin
                    state <= S_TILE_NEXT;
                end else begin
                    oc_group <= oc_group + 1;
                    if (cfg_op_type == 8'd1) begin
                        dw_ch_idx <= 16'd0;
                        if (dw_stream) begin
                            // Stream mode group boundary: oc_group+1 is the next
                            // 16-channel group's base. Request act slice + weight
                            // block reload from the controller.
                            dw_grp_base <= (oc_group + 1) * ARRAY_SIZE_16;
                            oc_group_done <= 1'b1;
                            state <= S_WAIT_WGT_RELOAD;
                        end else begin
                            // DW Conv: next channel
                            dw_cnt <= 0;
                            dw_read_issued <= 1'b0;
                            dw_init_phase <= 2'd0;
                            state <= S_DW_WGT_LOAD;
                        end
                    end else if (cfg_op_type == 8'd5) begin
                        rsz_ch <= rsz_ch + 1;
                        param_word_idx <= 0;
                        param_read_issued <= 1'b0;
                        state <= S_RESIZE_CH_SETUP;
                    end else begin
                        // Conv2D/FC: request weight reload if per-oc enabled
                        if (cfg_wgt_per_oc != 0) begin
                            oc_group_done <= 1'b1;
                            state <= S_WAIT_WGT_RELOAD;
                        end else begin
                            // All weights fit in SRAM, go directly
                            state <= S_OC_SETUP;
                        end
                    end
                end
            end

            S_WAIT_WGT_RELOAD: begin
                // Wait for controller to DMA next oc_group's weights
                // (DW stream mode: act slice + weight block for next 16-ch group)
                // (Pool stream mode: act slice only)
                if (wgt_reload_done) begin
                    oc_group_done <= 1'b0;
                    if (dw_stream) begin
                        dw_cnt <= 0;
                        dw_read_issued <= 1'b0;
                        dw_init_phase <= 2'd0;
                        state <= S_DW_WGT_LOAD;
                    end else if (pool_stream) begin
                        param_word_idx <= 0;
                        param_read_issued <= 1'b0;
                        state <= S_POOL_CH_SETUP;
                    end else begin
                        state <= S_OC_SETUP;
                    end
                end
            end

            S_TILE_NEXT: begin
                if (cfg_tile_h == 17'd0) begin
                    state <= S_DONE;
                end else if (tile_x + 1 >= cfg_tile_num_w) begin
                    if (tile_y + 1 >= cfg_tile_num_h) begin
                        state <= S_DONE;  // Final tile — no pulse, done handles it
                    end else begin
                        tile_x <= 0;
                        tile_y <= tile_y + 1;
                        tile_done_r <= 1'b1;  // PULSE: more tiles coming
                        state <= S_TILE_WAIT_DB;
                    end
                end else begin
                    tile_x <= tile_x + 1;
                    tile_done_r <= 1'b1;  // PULSE: more tiles coming
                    state <= S_TILE_WAIT_DB;
                end
            end

            // Wait for DB_EN prefetch to complete before starting next tile.
            // On entry, db_prefetch_done might still be deasserting (1 cycle lag
            // from controller). Skip one cycle, then check:
            //   - If db_prefetch_done=1 (no DB_EN or prefetch already done): go
            //   - If db_prefetch_done=0 (prefetch in flight): wait for reassert
            S_TILE_WAIT_DB: begin
                if (tile_wait_delay) begin
                    if (db_prefetch_done) begin
                        tile_wait_delay <= 1'b0;
                        if (cfg_op_type == 8'd4 || cfg_op_type == 8'd7) begin
                            // Add/Concat: clip tile dims for new tile, then read
                            // Re-latch act_base here since this path skips S_TILE_SETUP
                            act_base <= cfg_act_base;
                            if (cfg_tile_h != 17'd0) begin
                                out_tile_h <= cfg_tile_h;
                                out_tile_w <= cfg_tile_w;
                                if (cfg_tile_w > cfg_out_w - tile_x * cfg_tile_w)
                                    out_tile_w <= cfg_out_w - tile_x * cfg_tile_w;
                                if (cfg_tile_h > cfg_out_h - tile_y * cfg_tile_h)
                                    out_tile_h <= cfg_out_h - tile_y * cfg_tile_h;
                            end
                            add_elem_cnt <= add_elem_cnt + 1;
                            add_rd_phase <= 0;
                            state <= S_ADD_READ_A;
                        end else begin
                            state <= S_TILE_SETUP;
                        end
                    end
                end else begin
                    tile_wait_delay <= 1'b1;
                end
            end

            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            // ══════════════════════════════════════════════════════════════
            // DW Conv Path — Full Implementation
            // Flow: WGT_LOAD → PARAM → COMPUTE → (ACT_STREAM → PPU_WAIT)* → PPU → OC_NEXT
            // ══════════════════════════════════════════════════════════════
            S_DW_WGT_LOAD: begin
                // Load kh*kw weights for channel oc_group from Weight SRAM
                dw_wgt_load <= 1'b1;

                case (dw_init_phase)
                2'd0: begin
                    // Phase 0: setup
                    dw_kernel_size <= cfg_kernel_h[3:0] * cfg_kernel_w[3:0];
                    begin : dw_wgt_addr_setup
                        reg [15:0] wgt_elem_start;
                        reg [15:0] wgt_byte_start;
                        // Stream mode: weights for the resident 16-channel group
                        // were DMA-loaded as a block at wgt SRAM[0] — index by
                        // group-local channel. Normal mode: whole tensor resident,
                        // index by global channel.
                        if (dw_stream)
                            wgt_elem_start = (dw_ch_base + {11'd0, dw_ch_idx[4:0]} - dw_grp_base)
                                           * {2'd0, cfg_kernel_h[3:0]}
                                           * {2'd0, cfg_kernel_w[3:0]};
                        else
                            wgt_elem_start = (dw_ch_base + {11'd0, dw_ch_idx[4:0]})
                                           * {2'd0, cfg_kernel_h[3:0]}
                                           * {2'd0, cfg_kernel_w[3:0]};
                        wgt_byte_start = cfg_int16 ? (wgt_elem_start << 1) : wgt_elem_start;
                        wgt_word_addr <= {7'd0, wgt_base} + wgt_byte_start[15:2];
                        dw_wgt_bsel_base <= wgt_byte_start[1:0];
                    end
                    dw_init_phase <= 2'd1;
                end
                2'd1: begin
                    // Phase 1: send acc_clear, issue first SRAM read
                    dw_acc_clear <= 1'b1;
                    wgt_rd_en   <= 1'b1;
                    wgt_rd_addr <= wgt_word_addr[WGT_ADDR_W-1:0];
                    dw_read_issued <= 1'b0;
                    dw_init_phase <= 2'd2;
                end
                2'd2: begin
                    // Phase 2+: weight feeding loop
                    if (!dw_read_issued) begin
                        dw_read_issued <= 1'b1;
                    end else begin
                        // SRAM data available — extract element and feed
                        if (cfg_int16) begin : dw_wgt_extract_int16
                            reg [1:0] bsel;
                            bsel = {dw_cnt[0], 1'b0} + dw_wgt_bsel_base;
                            case (bsel[1])
                                1'b0: dw_wgt_data <= $signed(wgt_rd_data[15:0]);
                                1'b1: dw_wgt_data <= $signed(wgt_rd_data[31:16]);
                            endcase
                            dw_wgt_data_w[DATA_W*dw_ch_idx[COL_W-1:0] +: DATA_W]
                                <= (bsel[1] == 1'b0)
                                    ? $signed(wgt_rd_data[15:0])
                                    : $signed(wgt_rd_data[31:16]);
                        end else begin : dw_wgt_extract_int8
                            reg [1:0] bsel;
                            bsel = dw_cnt[1:0] + dw_wgt_bsel_base;
                            case (bsel)
                                2'd0: dw_wgt_data <= {{8{wgt_rd_data[7]}},  wgt_rd_data[7:0]};
                                2'd1: dw_wgt_data <= {{8{wgt_rd_data[15]}}, wgt_rd_data[15:8]};
                                2'd2: dw_wgt_data <= {{8{wgt_rd_data[23]}}, wgt_rd_data[23:16]};
                                2'd3: dw_wgt_data <= {{8{wgt_rd_data[31]}}, wgt_rd_data[31:24]};
                            endcase
                            case (bsel)
                                2'd0: dw_wgt_data_w[DATA_W*dw_ch_idx[COL_W-1:0] +: DATA_W]
                                    <= {{8{wgt_rd_data[7]}},  wgt_rd_data[7:0]};
                                2'd1: dw_wgt_data_w[DATA_W*dw_ch_idx[COL_W-1:0] +: DATA_W]
                                    <= {{8{wgt_rd_data[15]}}, wgt_rd_data[15:8]};
                                2'd2: dw_wgt_data_w[DATA_W*dw_ch_idx[COL_W-1:0] +: DATA_W]
                                    <= {{8{wgt_rd_data[23]}}, wgt_rd_data[23:16]};
                                default: dw_wgt_data_w[DATA_W*dw_ch_idx[COL_W-1:0] +: DATA_W]
                                    <= {{8{wgt_rd_data[31]}}, wgt_rd_data[31:24]};
                            endcase
                        end
                        dw_wgt_valid <= (dw_ch_idx == 16'd0);
                        dw_wgt_valid_w[dw_ch_idx[COL_W-1:0]] <= 1'b1;
                        dw_cnt <= dw_cnt + 1;

                        if (dw_cnt + 1 >= dw_kernel_size) begin
                            if (dw_ch_idx + 16'd1 < {11'd0, dw_nch}) begin
                                dw_ch_idx <= dw_ch_idx + 16'd1;
                                dw_cnt <= 16'd0;
                                dw_init_phase <= 2'd0;
                                dw_read_issued <= 1'b0;
                            end else begin
                                dw_ch_idx <= 16'd0;
                                state <= S_DW_DRAIN;
                            end
                        end else begin
                            // Check if need next SRAM word
                            if (cfg_int16) begin
                                // INT16: 2 elements per word; need next word when bsel[1]==1
                                if (({dw_cnt[0], 1'b0} + dw_wgt_bsel_base) >= 2'd2) begin
                                    wgt_word_addr <= wgt_word_addr + 1;
                                    wgt_rd_en    <= 1'b1;
                                    wgt_rd_addr  <= wgt_word_addr[WGT_ADDR_W-1:0] + 1;
                                    dw_read_issued <= 1'b0;
                                end
                            end else begin
                                // INT8: 4 elements per word
                                if ((dw_cnt[1:0] + dw_wgt_bsel_base) == 2'd3) begin
                                    wgt_word_addr <= wgt_word_addr + 1;
                                    wgt_rd_en    <= 1'b1;
                                    wgt_rd_addr  <= wgt_word_addr[WGT_ADDR_W-1:0] + 1;
                                    dw_read_issued <= 1'b0;
                                end
                            end
                        end
                    end
                end
                default: dw_init_phase <= 2'd0;
                endcase
            end

            S_DW_DRAIN: begin
                // Transition state: deassert wgt_load, go to param
                dw_wgt_load <= 1'b0;
                dw_cnt <= 0;
                param_word_idx <= 0;
                param_read_issued <= 1'b0;
                state <= S_DW_PARAM;
            end

            S_DW_PARAM: begin
                // Load 4 PPU param words for current channel
                // 3-phase per word: issue → wait → capture
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= (oc_group * 4) + param_word_idx;
                    param_read_issued <= 1'b1;
                    act_read_issued <= 1'b0;  // reuse as wait flag
                end else if (!act_read_issued) begin
                    // Wait for SRAM read latency
                    act_read_issued <= 1'b1;
                end else begin
                    param_buf[param_word_idx] <= param_rd_data;
                    if (param_word_idx == 3'd3) begin
                        // Extract params
                        ppu_mult_m     <= param_buf[0][14:0];
                        ppu_shift_s    <= param_buf[0][21:16];
                        ppu_zero_point <= $signed(param_buf[1][15:0]);
                        // Use param_rd_data for word 3 (param_buf[3] not yet updated this cycle)
                        ppu_bias       <= $signed({param_rd_data[15:0], param_buf[2],
                                                   param_buf[1][31:16]});
                        state <= S_DW_COMPUTE;
                    end else begin
                        param_word_idx <= param_word_idx + 1;
                        param_read_issued <= 1'b0;
                    end
                end
            end

            S_DW_COMPUTE: begin
                // Initialize pixel loop
                dw_oh <= 0;
                dw_ow <= 0;
                dw_fh <= 0;
                dw_fw <= 0;
                dw_read_issued <= 1'b0;
                state <= S_DW_ACT_STREAM;
            end

            S_DW_ACT_STREAM: begin
                // Stream window elements for pixel (dw_oh, dw_ow)
                begin : dw_act_stream_blk
                    reg signed [15:0] ih_s, iw_s;
                    reg               is_pad;
                    ih_s = $signed({1'b0, tile_oh_origin + dw_oh}) * $signed({1'b0, cfg_stride_h[7:0]})
                         - $signed({1'b0, cfg_pad_top[7:0]}) + $signed({1'b0, dw_fh});
                    iw_s = $signed({1'b0, tile_ow_origin + dw_ow}) * $signed({1'b0, cfg_stride_w[7:0]})
                         - $signed({1'b0, cfg_pad_left[7:0]}) + $signed({1'b0, dw_fw});
                    is_pad = (ih_s < 0) || (ih_s >= $signed({1'b0, cfg_in_h}))
                          || (iw_s < 0) || (iw_s >= $signed({1'b0, cfg_in_w}));

                    if (is_pad) begin
                        // Padding: feed zero to all live lanes
                        dw_in_valid <= 1'b1;
                        dw_in_data  <= {DATA_W{1'b0}};
                        dw_in_valid_w <= {ARRAY_SIZE{1'b0}};
                        dw_in_data_w  <= {DATA_W*ARRAY_SIZE{1'b0}};
                        begin : dw_pad_wide
                            integer ci;
                            for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1)
                                if (ci[4:0] < dw_nch)
                                    dw_in_valid_w[ci] <= 1'b1;
                        end
                        if (dw_fh == 0 && dw_fw == 0)
                            dw_acc_clear <= 1'b1;

                        // Advance filter position
                        if (dw_fw + 1 >= cfg_kernel_w[3:0]) begin
                            dw_fw <= 0;
                            if (dw_fh + 1 >= cfg_kernel_h[3:0]) begin
                                state <= S_DW_PPU_WAIT;
                                ppu_wait_cnt <= 0;
                            end else begin
                                dw_fh <= dw_fh + 1;
                            end
                        end else begin
                            dw_fw <= dw_fw + 1;
                        end
                        dw_read_issued <= 1'b0;
                    end else if (!dw_read_issued) begin
                        // In-bounds: issue SRAM read
                        begin : dw_addr_calc
                            reg [ACT_ADDR_W+15:0] elem_off;
                            reg [ACT_ADDR_W+15:0] byte_off;
                            if (dw_stream) begin
                                // Stream mode: resident slice is packed
                                // [pos][ch_local], ARRAY_SIZE channels per
                                // spatial position (ch_local = oc_group - base)
                                elem_off = (ih_s[15:0] * cfg_in_w + iw_s[15:0])
                                         * ARRAY_SIZE + (dw_ch_base - dw_grp_base);
                            end else if (cfg_tile_h == 17'd0) begin
                                // Non-tiled: full image address
                                elem_off = (ih_s[15:0] * cfg_in_w * cfg_in_c)
                                         + (iw_s[15:0] * cfg_in_c)
                                         + dw_ch_base;
                            end else begin
                                // Tiled: tile-local address (dw_oh/dw_ow are tile-local
                                // output coords; input row = dw_oh*stride + dw_fh,
                                // input col = dw_ow*stride + dw_fw, both within tile_in_h/w)
                                begin : dw_addr_2d_blk
                                    reg [17:0] row_rel, col_rel;
                                    row_rel = {2'd0, dw_oh} * {10'd0, cfg_stride_h} + {14'd0, dw_fh};
                                    col_rel = {2'd0, dw_ow} * {10'd0, cfg_stride_w} + {14'd0, dw_fw};
                                    elem_off = (row_rel[15:0] * tile_in_w + col_rel[15:0])
                                               * cfg_in_c + dw_ch_base;
                                end
                            end
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            act_rd_en   <= 1'b1;
                            act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                            act_byte_sel <= byte_off[1:0];
                        end
                        dw_read_issued <= 1'b1;
                        act_read_issued <= 1'b0;
                        if (dw_fh == 0 && dw_fw == 0)
                            dw_acc_clear <= 1'b1;
                    end else if (!act_read_issued) begin
                        act_read_issued <= 1'b1;
                    end else begin
                        // SRAM data available — extract element and feed
                        if (cfg_int16) begin : dw_act_extract_int16
                            case (act_byte_sel[1])
                                1'b0: dw_in_data <= $signed(act_rd_data[15:0]);
                                1'b1: dw_in_data <= $signed(act_rd_data[31:16]);
                            endcase
                        end else begin : dw_act_extract_int8
                            case (act_byte_sel)
                                2'd0: dw_in_data <= {{8{act_rd_data[7]}},  act_rd_data[7:0]};
                                2'd1: dw_in_data <= {{8{act_rd_data[15]}}, act_rd_data[15:8]};
                                2'd2: dw_in_data <= {{8{act_rd_data[23]}}, act_rd_data[23:16]};
                                2'd3: dw_in_data <= {{8{act_rd_data[31]}}, act_rd_data[31:24]};
                            endcase
                        end
                        dw_in_valid <= 1'b0;
                        begin : dw_act_wide
                            integer ci;
                            reg [SRAM_B_W-1:0] sh;
                            sh = act_rd_data >> (act_byte_sel * 8);
                            for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1) begin
                                if (ci[4:0] < dw_nch) begin
                                    if (cfg_int16)
                                        dw_in_data_w[DATA_W*ci +: DATA_W]
                                            <= $signed(sh[16*ci +: 16]);
                                    else
                                        dw_in_data_w[DATA_W*ci +: DATA_W]
                                            <= {{8{sh[8*ci+7]}}, sh[8*ci +: 8]};
                                    dw_in_valid_w[ci] <= 1'b1;
                                end
                            end
                            dw_in_data <= cfg_int16 ? $signed(sh[15:0])
                                                    : {{8{sh[7]}}, sh[7:0]};
                        end
                        dw_read_issued <= 1'b0;

                        // Advance filter position
                        if (dw_fw + 1 >= cfg_kernel_w[3:0]) begin
                            dw_fw <= 0;
                            if (dw_fh + 1 >= cfg_kernel_h[3:0]) begin
                                state <= S_DW_PPU_WAIT;
                                ppu_wait_cnt <= 0;
                            end else begin
                                dw_fh <= dw_fh + 1;
                            end
                        end else begin
                            dw_fw <= dw_fw + 1;
                        end
                    end
                end
            end

            S_DW_PPU_WAIT: begin
                // Wait for dw_out_valid, then feed PPU, wait for PPU output
                if (ppu_wait_cnt == 0) begin
                    if (dw_out_valid_w[0] || dw_out_valid) begin
                        ppu_acc_in   <= dw_acc_out;
                        ppu_in_valid <= 1'b1;
                        begin : dw_ppu_wide
                            integer ci;
                            for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1) begin
                                if (ci[4:0] < dw_nch) begin
                                    ppu_acc_w[ACC_W*ci +: ACC_W]
                                        <= dw_acc_w[ACC_W*ci +: ACC_W];
                                    ppu_valid_w[ci] <= 1'b1;
                                end
                            end
                        end
                        ppu_wait_cnt <= 1;
                    end
                end else begin
                    if (ppu_vout_w[0] || ppu_out_valid) begin
                        for (i = 0; i < ARRAY_SIZE; i = i + 1)
                            ppu_lat[i] <= ppu_out_w[DATA_W*i +: DATA_W];
                        if (!ppu_vout_w[0])
                            ppu_lat[0] <= ppu_out_data;
                        dw_ch_idx <= 16'd0;
                        // Compute NHWC address for this pixel/channel
                        begin : dw_wb_addr_calc
                            reg [31:0] elem_off;
                            reg [31:0] byte_off;
                            elem_off = (dw_oh * out_tile_w + dw_ow)
                                     * cfg_out_c + dw_ch_base;
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            dw_wb_addr    <= out_base + byte_off[ACT_ADDR_W+1:2];
                            dw_wb_bytesel <= byte_off[1:0];
                        end
                        dw_wb_byte  <= ppu_vout_w[0] ? ppu_out_w[DATA_W-1:0] : ppu_out_data;
                        dw_wb_phase <= 2'd0;
                        state <= S_DW_WB;
                    end else begin
                        ppu_wait_cnt <= ppu_wait_cnt + 1;
                    end
                end
            end

            S_DW_WB: begin
                // Read-modify-write: place output element at correct NHWC position
                act_rd_ofm <= 1'b1;
                case (dw_wb_phase)
                2'd0: begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= dw_wb_addr;
                    dw_wb_phase <= 2'd1;
                end
                2'd1: begin
                    dw_wb_phase <= 2'd2;
                end
                2'd2: begin
                    // Merge element into read word and write back
                    begin : dw_wb_merge
                        reg [31:0] merged;
                        merged = act_rd_data;
                        if (cfg_int16) begin
                            // INT16: merge half-word
                            case (dw_wb_bytesel[1])
                                1'b0: merged[15:0]  = dw_wb_byte;
                                1'b1: merged[31:16] = dw_wb_byte;
                            endcase
                        end else begin
                            // INT8: merge byte
                            case (dw_wb_bytesel)
                                2'd0: merged[7:0]   = dw_wb_byte[7:0];
                                2'd1: merged[15:8]  = dw_wb_byte[7:0];
                                2'd2: merged[23:16] = dw_wb_byte[7:0];
                                2'd3: merged[31:24] = dw_wb_byte[7:0];
                            endcase
                        end
                        act_wr_en   <= 1'b1;
                        act_wr_addr <= dw_wb_addr;
                        act_wr_data <= merged;
                    end

                    // Next channel of this pixel, or next pixel
                    if (dw_ch_idx + 16'd1 < {11'd0, dw_nch}) begin
                        dw_ch_idx <= dw_ch_idx + 16'd1;
                        dw_wb_byte <= ppu_lat[dw_ch_idx[3:0] + 4'd1];
                        begin : dw_wb_next_ch
                            reg [31:0] elem_off, byte_off;
                            elem_off = (dw_oh * out_tile_w + dw_ow) * cfg_out_c
                                     + dw_ch_base + dw_ch_idx + 16'd1;
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            dw_wb_addr    <= out_base + byte_off[ACT_ADDR_W+1:2];
                            dw_wb_bytesel <= byte_off[1:0];
                        end
                        dw_wb_phase <= 2'd0;
                    end else if (dw_ow + 1 >= out_tile_w) begin
                        dw_ow <= 0;
                        if (dw_oh + 1 >= out_tile_h) begin
                            state <= S_OC_NEXT;
                        end else begin
                            dw_oh <= dw_oh + 1;
                            dw_fh <= 0;
                            dw_fw <= 0;
                            dw_read_issued <= 1'b0;
                            ppu_wait_cnt <= 0;
                            state <= S_DW_ACT_STREAM;
                        end
                    end else begin
                        dw_ow <= dw_ow + 1;
                        dw_fh <= 0;
                        dw_fw <= 0;
                        dw_read_issued <= 1'b0;
                        ppu_wait_cnt <= 0;
                        state <= S_DW_ACT_STREAM;
                    end
                end
                default: dw_wb_phase <= 2'd0;
                endcase
            end

            S_DW_PPU: begin
                // No longer used (writeback handled in S_DW_WB)
                state <= S_OC_NEXT;
            end

            // ══════════════════════════════════════════════════════════════
            // POOLING Path (op_type=3)
            // Flow: SETUP → CH_SETUP → [READ → ACC]* → DIV → PPU → PPU_WAIT → WB → PIX_NEXT → CH_NEXT
            // ══════════════════════════════════════════════════════════════
            S_POOL_SETUP: begin
                // Latch effective pool kernel/stride (global override)
                if (global_pool) begin
                    pool_kh <= cfg_in_h[7:0];
                    pool_kw <= cfg_in_w[7:0];
                    pool_sh <= cfg_in_h[7:0];
                    pool_sw <= cfg_in_w[7:0];
                end else begin
                    pool_kh <= {4'd0, pool_cfg_h};
                    pool_kw <= {4'd0, pool_cfg_w};
                    pool_sh <= {4'd0, pool_cfg_sh};
                    pool_sw <= {4'd0, pool_cfg_sw};
                end
                // Set output base (same as S_TILE_SETUP — Pool also goes through S_TILE_SETUP
                // which sets tile_oh/ow_origin and out_tile_h/w, so don't override here)
                // Pool stream: fixed high region (slice at SRAM[0]).
                if (pool_stream) begin
                    out_base      <= `DW_STREAM_OUT_BASE;
                    pool_grp_base <= 16'd0;
                end else begin
                    out_base <= cfg_act_base + cfg_out_base;
                end
                pool_ch <= 0;
                param_word_idx <= 0;
                param_read_issued <= 1'b0;
                state <= S_POOL_CH_SETUP;
            end

            S_POOL_CH_SETUP: begin
                // Load 4 PPU param words for current channel (same as DW_PARAM)
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= (pool_ch * 4) + param_word_idx;
                    param_read_issued <= 1'b1;
                    act_read_issued <= 1'b0;
                end else if (!act_read_issued) begin
                    act_read_issued <= 1'b1;
                end else begin
                    param_buf[param_word_idx] <= param_rd_data;
                    if (param_word_idx == 3'd3) begin
                        ppu_mult_m     <= param_buf[0][14:0];
                        ppu_shift_s    <= param_buf[0][21:16];
                        ppu_zero_point <= $signed(param_buf[1][15:0]);
                        // Use param_rd_data for word 3 (param_buf[3] not yet updated this cycle)
                        ppu_bias       <= $signed({param_rd_data[15:0], param_buf[2],
                                                   param_buf[1][31:16]});
                        // Start spatial loop for this channel
                        pool_oh <= 0;
                        pool_ow <= 0;
                        pool_fh <= 0;
                        pool_fw <= 0;
                        pool_rd_phase <= 0;
                        // Initialize accumulator
                        if (pool_mode) begin
                            pool_acc <= 0;  // AvgPool: sum=0
                        end else begin
                            pool_acc <= -$signed({{(ACC_W-1){1'b0}}, 1'b1});  // MaxPool: -2^(ACC_W-1), signed
                        end
                        pool_count <= 0;
                        state <= S_POOL_READ;
                    end else begin
                        param_word_idx <= param_word_idx + 1;
                        param_read_issued <= 1'b0;
                    end
                end
            end

            S_POOL_READ: begin
                // Compute input coords, bounds-check, issue SRAM read
                begin : pool_read_blk
                    reg signed [15:0] ih_s, iw_s;
                    reg               is_oob;
                    ih_s = $signed({1'b0, tile_oh_origin + pool_oh}) * $signed({1'b0, pool_sh})
                         - $signed({1'b0, cfg_pad_top[7:0]}) + $signed({1'b0, pool_fh});
                    iw_s = $signed({1'b0, tile_ow_origin + pool_ow}) * $signed({1'b0, pool_sw})
                         - $signed({1'b0, cfg_pad_left[7:0]}) + $signed({1'b0, pool_fw});
                    is_oob = (ih_s < 0) || (ih_s >= $signed({1'b0, cfg_in_h}))
                          || (iw_s < 0) || (iw_s >= $signed({1'b0, cfg_in_w}));
                    `ifndef SYNTHESIS
                    if (pool_ch == 9 && pool_oh == 0 && pool_ow == 2 && tile_y == 0 && tile_x == 5)
                        $display("[POOL_DBG] t=%0t ch=%0d oh=%0d ow=%0d fh=%0d fw=%0d ih=%0d iw=%0d oob=%0d tile_in_w=%0d",
                                 $time, pool_ch, pool_oh, pool_ow, pool_fh, pool_fw,
                                 ih_s, iw_s, is_oob, tile_in_w);
                    `endif
`ifdef DBG_DOTBUF
                    if (pool_ch == 33 && pool_oh == 0 && pool_ow == 0 && tile_y == 0 && tile_x == 0)
                        $fwrite(dbg_fh, "[POOL_RD] t=%0d fh=%0d fw=%0d ih=%0d iw=%0d oob=%0d rd_ph=%0d\n",
                                $time, pool_fh, pool_fw, ih_s, iw_s, is_oob, pool_rd_phase);
`endif

                    if (is_oob) begin
                        // Out-of-bounds: skip, advance window position
                        if ({4'd0, pool_fw} + 1 >= pool_kw) begin
                            pool_fw <= 0;
                            if ({4'd0, pool_fh} + 1 >= pool_kh) begin
                                // Window complete
                                state <= S_POOL_DIV;
                            end else begin
                                pool_fh <= pool_fh + 1;
                            end
                        end else begin
                            pool_fw <= pool_fw + 1;
                        end
                    end else if (pool_rd_phase == 0) begin
                        // Issue SRAM read
                        begin : pool_addr_calc
                            reg [31:0] elem_off;
                            reg [31:0] byte_off;
                            if (pool_stream) begin
                                // Slice packed [pos][ch_local], 16 ch per position
                                elem_off = (ih_s[15:0] * cfg_in_w + iw_s[15:0])
                                         * ARRAY_SIZE + (pool_ch - pool_grp_base);
                            end else if (cfg_tile_h == 17'd0) begin
                                // Non-tiled: full image in SRAM, absolute coords
                                elem_off = (ih_s[15:0] * cfg_in_w * cfg_in_c)
                                         + (iw_s[15:0] * cfg_in_c)
                                         + pool_ch;
                            end else begin
                                // Tiled: use tile-local coords (matching Conv2D fix)
                                begin : pool_addr_2d_blk
                                    reg [17:0] row_rel, col_rel;
                                    row_rel = {2'd0, pool_oh} * {10'd0, pool_sh} + {14'd0, pool_fh};
                                    col_rel = {2'd0, pool_ow} * {10'd0, pool_sw} + {14'd0, pool_fw};
                                    elem_off = (row_rel[15:0] * tile_in_w + col_rel[15:0])
                                               * cfg_in_c + pool_ch;
                                end
                            end
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            act_rd_en   <= 1'b1;
                            act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                            act_byte_sel <= byte_off[1:0];
                        end
                        pool_rd_phase <= 1;
                    end else if (pool_rd_phase == 1) begin
                        // Wait for SRAM latency
                        pool_rd_phase <= 2;
                    end else begin
                        // Data available — go to ACC
                        pool_rd_phase <= 0;
                        state <= S_POOL_ACC;
                    end
                end
            end

            S_POOL_ACC: begin
                // Extract element and accumulate (module-level pool_val for Verilator compat)
                if (cfg_int16) begin
                    case (act_byte_sel[1])
                        1'b0: pool_val = $signed(act_rd_data[15:0]);
                        1'b1: pool_val = $signed(act_rd_data[31:16]);
                    endcase
                end else begin
                    case (act_byte_sel)
                        2'd0: pool_val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
                        2'd1: pool_val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
                        2'd2: pool_val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
                        2'd3: pool_val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                    endcase
                end

                if (pool_mode) begin
                    pool_acc <= pool_acc + pool_val;
                end else begin
                    if ($signed(pool_val) > $signed(pool_acc))
                        pool_acc <= pool_val;
                end
                pool_count <= pool_count + 1;
`ifdef DBG_DOTBUF
                if (pool_ch == 33)
                    $fwrite(dbg_fh, "[POOL_ACC] t=%0d fh=%0d fw=%0d pix_oh=%0d pix_ow=%0d val=%0d acc=%0d cnt=%0d kw=%0d kh=%0d\n",
                            $time, pool_fh, pool_fw, pool_oh, pool_ow, pool_val, pool_acc, pool_count, pool_kw, pool_kh);
`endif

                // Advance window position
                if ({4'd0, pool_fw} + 1 >= pool_kw) begin
                    pool_fw <= 0;
                    if ({4'd0, pool_fh} + 1 >= pool_kh) begin
                        // Window complete
                        state <= S_POOL_DIV;
                    end else begin
                        pool_fh <= pool_fh + 1;
                        state <= S_POOL_READ;
                    end
                end else begin
                    pool_fw <= pool_fw + 1;
                    state <= S_POOL_READ;
                end
            end

            S_POOL_DIV: begin
                // AvgPool: symmetric rounding division (reciprocal LUT)
                // MaxPool: pass through
                if (pool_mode && pool_count > 0) begin
                    // pool_count=1 is identity (e.g. 1x1 global avg pool): skip
                    // reciprocal multiply — LUT can't represent 1.0 in 32-bit
                    // signed Q0.32 (0x80000000 reads back as -2^31 under $signed).
                    if (pool_count == 17'd1) begin
                        pool_acc <= pool_acc;  // identity
                    end else begin
                        begin : pool_div_blk
                            reg signed [ACC_W-1:0] rounded;
                            reg signed [ACC_W-1:0] half_count;
                            reg signed [71:0] prod;  // 40-bit * 32-bit = 72-bit
                            half_count = pool_count >> 1;
                            if (pool_acc >= 0)
                                rounded = pool_acc + half_count;
                            else
                                rounded = pool_acc - half_count;
                            // Multiply by reciprocal: q = (rounded * recip) >>> N
                            // count=2 uses recip=0x40000000 (2^30) >>> 31 since
                            // 0x80000000 would be negative under $signed.
                            prod = rounded * $signed(recip_pool(pool_count[6:0]));
                            if (pool_count == 17'd2)
                                pool_acc <= prod >>> 31;
                            else
                                pool_acc <= prod >>> 32;
                        end
                    end
                end
                state <= S_POOL_PPU;
            end

            S_POOL_PPU: begin
                // Feed pool_acc to PPU (CONV_REQ mode)
                ppu_acc_in   <= pool_acc;
                ppu_in_valid <= 1'b1;
                state <= S_POOL_PPU_WAIT;
            end

            S_POOL_PPU_WAIT: begin
                if (ppu_out_valid) begin
                    // Compute writeback address
                    begin : pool_wb_addr_calc
                        reg [31:0] elem_off;
                        reg [31:0] byte_off;
                        elem_off = (pool_oh * out_tile_w + pool_ow)
                                 * cfg_out_c + pool_ch;
                        byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                        pool_wb_addr    <= out_base + byte_off[ACT_ADDR_W+1:2];
                        pool_wb_bytesel <= byte_off[1:0];
                    end
                    pool_wb_byte <= ppu_out_data;
                    pool_wb_phase <= 2'd0;
                    state <= S_POOL_WB;
                end
            end

            S_POOL_WB: begin
                // Read-modify-write (same pattern as S_DW_WB)
                act_rd_ofm <= 1'b1;
                case (pool_wb_phase)
                2'd0: begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= pool_wb_addr;
                    pool_wb_phase <= 2'd1;
                end
                2'd1: begin
                    pool_wb_phase <= 2'd2;
                end
                2'd2: begin
                    begin : pool_wb_merge
                        reg [31:0] merged;
                        merged = act_rd_data;
                        if (cfg_int16) begin
                            case (pool_wb_bytesel[1])
                                1'b0: merged[15:0]  = pool_wb_byte;
                                1'b1: merged[31:16] = pool_wb_byte;
                            endcase
                        end else begin
                            case (pool_wb_bytesel)
                                2'd0: merged[7:0]   = pool_wb_byte[7:0];
                                2'd1: merged[15:8]  = pool_wb_byte[7:0];
                                2'd2: merged[23:16] = pool_wb_byte[7:0];
                                2'd3: merged[31:24] = pool_wb_byte[7:0];
                            endcase
                        end
                        act_wr_en   <= 1'b1;
                        act_wr_addr <= pool_wb_addr;
                        act_wr_data <= merged;
                    end
                    state <= S_POOL_PIX_NEXT;
                end
                default: pool_wb_phase <= 2'd0;
                endcase
            end

            S_POOL_PIX_NEXT: begin
                // Advance output pixel, reset window for next pixel
                pool_fh <= 0;
                pool_fw <= 0;
                pool_rd_phase <= 0;
                if (pool_mode) begin
                    pool_acc <= 0;
                end else begin
                    pool_acc <= -$signed({{(ACC_W-1){1'b0}}, 1'b1});
                end
                pool_count <= 0;

                if (pool_ow + 1 >= out_tile_w) begin
                    pool_ow <= 0;
                    if (pool_oh + 1 >= out_tile_h) begin
                        state <= S_POOL_CH_NEXT;
                    end else begin
                        pool_oh <= pool_oh + 1;
                        state <= S_POOL_READ;
                    end
                end else begin
                    pool_ow <= pool_ow + 1;
                    state <= S_POOL_READ;
                end
            end

            S_POOL_CH_NEXT: begin
                if (pool_ch + 1 >= cfg_out_c) begin
                    state <= S_TILE_NEXT;
                end else if (pool_stream && ((pool_ch + 1) % ARRAY_SIZE == 0)) begin
                    // Stream group boundary: request next act slice reload
                    pool_ch <= pool_ch + 1;
                    pool_grp_base <= pool_grp_base + ARRAY_SIZE;
                    oc_group_done <= 1'b1;
                    state <= S_WAIT_WGT_RELOAD;
                end else begin
                    pool_ch <= pool_ch + 1;
                    param_word_idx <= 0;
                    param_read_issued <= 1'b0;
                    state <= S_POOL_CH_SETUP;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // Eltwise Add Path (op_type=4)
            // Flow: SETUP → PARAM → [READ_A → READ_B → COMPUTE → PPU → PPU_WAIT → WB → NEXT]*
            // ══════════════════════════════════════════════════════════════
            S_ADD_SETUP: begin
                // Concat scatters its INPUT elements (in_c per pixel) into a
                // wider output buffer — the loop must cover in_c, not out_c.
                // Add is element-wise 1:1 and legitimately uses out_c.
                add_total_elems <= cfg_out_h * cfg_out_w * (is_concat ? cfg_in_c : cfg_out_c);
                add_elem_cnt <= 0;
                add_tile_elem_cnt <= 0;
                concat_pixel_cnt <= 0;
                concat_ch_cnt <= 0;
                add_param_idx <= 0;
                add_param_phase <= 0;
                tile_x <= 0;
                tile_y <= 0;
                // Set tile dimensions for actual tile size calculation.
                // Clip border tiles to image boundary (same as S_TILE_SETUP).
                if (cfg_tile_h == 17'd0) begin
                    out_tile_h <= cfg_out_h;
                    out_tile_w <= cfg_out_w;
                end else begin
                    out_tile_h <= cfg_tile_h;
                    out_tile_w <= cfg_tile_w;
                    if (cfg_tile_w > cfg_out_w - tile_x * cfg_tile_w)
                        out_tile_w <= cfg_out_w - tile_x * cfg_tile_w;
                    if (cfg_tile_h > cfg_out_h - tile_y * cfg_tile_h)
                        out_tile_h <= cfg_out_h - tile_y * cfg_tile_h;
                end
                state <= S_ADD_PARAM;
            end

            S_ADD_PARAM: begin
                // Read 2 words from Param SRAM (global Add rescale params)
                if (add_param_phase == 0) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= add_param_idx;
                    add_param_phase <= 1;
                end else if (add_param_phase == 1) begin
                    // Wait SRAM latency
                    add_param_phase <= 2;
                end else begin
                    // Capture
                    if (add_param_idx == 0) begin
                        add_M_A <= param_rd_data[14:0];
                        add_S_A <= param_rd_data[21:16];
                        if (is_concat) begin
                            // Concat: only 1 param word needed
                            add_rd_phase <= 0;
                            state <= S_ADD_READ_A;
                        end else begin
                            add_param_idx <= 1;
                            add_param_phase <= 0;
                        end
                    end else begin
                        add_M_B <= param_rd_data[14:0];
                        add_S_B <= param_rd_data[21:16];
                        add_rd_phase <= 0;
                        state <= S_ADD_READ_A;
                    end
                end
            end

            S_ADD_READ_A: begin
                // Read element from Branch A (act_base region)
                if (add_rd_phase == 0) begin
                    begin : add_a_addr_calc
                        reg [31:0] byte_off;
                        byte_off = cfg_int16 ? ({17'd0, add_tile_elem_cnt} << 1) : {17'd0, add_tile_elem_cnt};
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                        act_byte_sel <= byte_off[1:0];
                    end
                    add_rd_phase <= 1;
                end else if (add_rd_phase == 1) begin
                    add_rd_phase <= 2;
                end else begin
                    // Extract element
                    if (cfg_int16) begin
                        case (act_byte_sel[1])
                            1'b0: add_val_a <= $signed(act_rd_data[15:0]);
                            1'b1: add_val_a <= $signed(act_rd_data[31:16]);
                        endcase
                    end else begin
                        case (act_byte_sel)
                            2'd0: add_val_a <= {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
                            2'd1: add_val_a <= {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
                            2'd2: add_val_a <= {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
                            2'd3: add_val_a <= {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                        endcase
                    end
                    add_rd_phase <= 0;
                    state <= is_concat ? S_ADD_COMPUTE : S_ADD_READ_B;
                end
            end

            S_ADD_READ_B: begin
                // Read element from Branch B (out_base region)
                if (add_rd_phase == 0) begin
                    begin : add_b_addr_calc
                        reg [31:0] byte_off;
                        byte_off = cfg_int16 ? ({17'd0, add_tile_elem_cnt} << 1) : {17'd0, add_tile_elem_cnt};
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= act_base + cfg_out_base + byte_off[ACT_ADDR_W+1:2];
                        act_byte_sel <= byte_off[1:0];
                    end
                    add_rd_phase <= 1;
                end else if (add_rd_phase == 1) begin
                    add_rd_phase <= 2;
                end else begin
                    // Extract element
                    if (cfg_int16) begin
                        case (act_byte_sel[1])
                            1'b0: add_val_b <= $signed(act_rd_data[15:0]);
                            1'b1: add_val_b <= $signed(act_rd_data[31:16]);
                        endcase
                    end else begin
                        case (act_byte_sel)
                            2'd0: add_val_b <= {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
                            2'd1: add_val_b <= {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
                            2'd2: add_val_b <= {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
                            2'd3: add_val_b <= {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                        endcase
                    end
                    add_rd_phase <= 0;
                    state <= S_ADD_COMPUTE;
                end
            end

            S_ADD_COMPUTE: begin
                // Dual rescale: rescaled_A = (val_A * M_A + round) >> S_A
                begin : add_compute_blk
                    reg signed [ACC_W-1:0] prod_a, prod_b;
                    reg signed [ACC_W-1:0] rescaled_a, rescaled_b;
                    prod_a = add_val_a * $signed({1'b0, add_M_A});
                    if (add_S_A > 0)
                        rescaled_a = (prod_a + (1 <<< (add_S_A - 1))) >>> add_S_A;
                    else
                        rescaled_a = prod_a;
                    if (is_concat) begin
                        ppu_acc_in <= rescaled_a;
                    end else begin
                        prod_b = add_val_b * $signed({1'b0, add_M_B});
                        if (add_S_B > 0)
                            rescaled_b = (prod_b + (1 <<< (add_S_B - 1))) >>> add_S_B;
                        else
                            rescaled_b = prod_b;
                        ppu_acc_in <= rescaled_a + rescaled_b;
                    end
                end
                state <= S_ADD_PPU;
            end

            S_ADD_PPU: begin
                ppu_in_valid <= 1'b1;
                state <= S_ADD_PPU_WAIT;
            end

            S_ADD_PPU_WAIT: begin
                if (ppu_out_valid) begin
                    // Compute writeback address
                    if (add_elem_cnt >= 16'd18816 && add_elem_cnt <= 16'd18820)
                        $display("[PPU_OUT_DBG] elem=%0d ppu_out_data=0x%04x ppu_acc_in=%0d",
                                 add_elem_cnt, ppu_out_data, ppu_acc_in);
                    begin : add_wb_addr_calc
                        reg [31:0] byte_off;
                        if (is_concat) begin
                            // Concat: output[tile_pixel * total_c + offset + ch]
                            // Use counters instead of division (pixel = elem/in_c, ch = elem%in_c)
                            byte_off = ({16'd0, concat_pixel_cnt} * {16'd0, concat_total_c}
                                     + {16'd0, concat_offset} + {16'd0, concat_ch_cnt})
                                       << (cfg_int16 ? 1 : 0);
                        end else begin
                            // Add: flat overwrite at input A region (tile-local)
                            byte_off = cfg_int16 ? ({17'd0, add_tile_elem_cnt} << 1) : {17'd0, add_tile_elem_cnt};
                        end
                        // Concat writes to output region (cfg_act_base + cfg_out_base),
                        // Add writes in-place to input A region (cfg_act_base)
                        if (is_concat)
                            add_wb_addr    <= act_base + cfg_out_base + byte_off[ACT_ADDR_W+1:2];
                        else
                            add_wb_addr    <= act_base + byte_off[ACT_ADDR_W+1:2];
                        add_wb_bytesel <= byte_off[1:0];
                    end
                    add_wb_byte <= ppu_out_data;
                    add_wb_phase <= 2'd0;
                    state <= S_ADD_WB;
                end
            end

            S_ADD_WB: begin
                // Read-modify-write (partial word lives in OFM)
                act_rd_ofm <= 1'b1;
                case (add_wb_phase)
                2'd0: begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= add_wb_addr;
                    add_wb_phase <= 2'd1;
                end
                2'd1: begin
                    add_wb_phase <= 2'd2;
                end
                2'd2: begin
                    begin : add_wb_merge
                        reg [31:0] merged;
                        merged = act_rd_data;
                        if (cfg_int16) begin
                            case (add_wb_bytesel[1])
                                1'b0: merged[15:0]  = add_wb_byte;
                                1'b1: merged[31:16] = add_wb_byte;
                            endcase
                        end else begin
                            case (add_wb_bytesel)
                                2'd0: merged[7:0]   = add_wb_byte[7:0];
                                2'd1: merged[15:8]  = add_wb_byte[7:0];
                                2'd2: merged[23:16] = add_wb_byte[7:0];
                                2'd3: merged[31:24] = add_wb_byte[7:0];
                            endcase
                        end
                        act_wr_en   <= 1'b1;
                        act_wr_addr <= add_wb_addr;
                        act_wr_data <= merged;
                        `ifndef SYNTHESIS
                        if (tile_x == 0 && tile_y == 0 && add_tile_elem_cnt >= 384 && add_tile_elem_cnt <= 390)
                            $display("[ADD_WB] t=%0t elem=%0d tile_elem=%0d wb_addr=%0d rd=0x%08x merged=0x%08x byte=0x%04x bytesel=%0d",
                                     $time, add_elem_cnt, add_tile_elem_cnt, add_wb_addr,
                                     act_rd_data, merged, add_wb_byte, add_wb_bytesel);
                        `endif
                        if (add_elem_cnt >= 16'd18816 && add_elem_cnt <= 16'd18820)
                            $display("[WB_DBG] elem=%0d tile(%0d,%0d) wb_addr=%0d cfg_act_base=%0d tile_elem=%0d merged=0x%08x wb_byte=0x%04x",
                                     add_elem_cnt, tile_y, tile_x, add_wb_addr, act_base, add_tile_elem_cnt, merged, add_wb_byte);
                    end
                    state <= S_ADD_NEXT;
                end
                default: add_wb_phase <= 2'd0;
                endcase
            end

            S_ADD_NEXT: begin
                reg [31:0] elems_per_tile;
                act_wr_en <= 1'b0;  // Clear write enable
                ppu_in_valid <= 1'b0;  // Clear PPU input valid
                // Use padded tile size for SRAM layout (DMA skips padding via src_row_len)
                // Concat iterates input elements (in_c per pixel), not out_c.
                elems_per_tile = {17'd0, cfg_tile_h} * {17'd0, cfg_tile_w}
                                 * {17'd0, (is_concat ? cfg_in_c : cfg_out_c)};
                if (cfg_tile_h == 17'd0 || elems_per_tile == 0) begin
                    // Non-tiled: original logic
                    if ({17'd0, add_elem_cnt} + 32'd1 >= {17'd0, add_total_elems}) begin
                        state <= S_DONE;
                    end else begin
                        add_elem_cnt <= add_elem_cnt + 17'd1;
                        add_tile_elem_cnt <= add_tile_elem_cnt + 17'd1;
                        // Concat pixel/ch counter increment
                        if (is_concat) begin
                            if (concat_ch_cnt + 1 >= cfg_in_c) begin
                                concat_ch_cnt <= 0;
                                concat_pixel_cnt <= concat_pixel_cnt + 1;
                            end else begin
                                concat_ch_cnt <= concat_ch_cnt + 1;
                            end
                        end
                        add_rd_phase <= 0;
                        state <= S_ADD_READ_A;
                    end
                end else if ({17'd0, add_tile_elem_cnt} + 32'd1 >= elems_per_tile) begin
                    // Current tile done — use tile-local counter for boundary check
                    if (tile_x + 1 >= cfg_tile_num_w) begin
                        if (tile_y + 1 >= cfg_tile_num_h) begin
                            state <= S_DONE;  // Final tile
                        end else begin
                            tile_x <= 0;
                            tile_y <= tile_y + 1;
                            tile_done_r <= 1'b1;
                            add_tile_elem_cnt <= 0;
                            concat_pixel_cnt <= 0;
                            concat_ch_cnt <= 0;
                            state <= S_TILE_WAIT_DB;
                        end
                    end else begin
                        tile_x <= tile_x + 1;
                        tile_done_r <= 1'b1;
                        add_tile_elem_cnt <= 0;
                        concat_pixel_cnt <= 0;
                        concat_ch_cnt <= 0;
                        state <= S_TILE_WAIT_DB;
                    end
                end else begin
                    // Same tile, next element
                    add_elem_cnt <= add_elem_cnt + 17'd1;
                    add_tile_elem_cnt <= add_tile_elem_cnt + 17'd1;
                    // Concat pixel/ch counter increment
                    if (is_concat) begin
                        if (concat_ch_cnt + 1 >= cfg_in_c) begin
                            concat_ch_cnt <= 0;
                            concat_pixel_cnt <= concat_pixel_cnt + 1;
                        end else begin
                            concat_ch_cnt <= concat_ch_cnt + 1;
                        end
                    end
                    add_rd_phase <= 0;
                    state <= S_ADD_READ_A;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // Resize Path (op_type=5)
            // Flow: SETUP → CH_SETUP → COORD → READ0[/1/2/3] → INTERP → PPU → WB → PIX_NEXT → CH_NEXT
            // ══════════════════════════════════════════════════════════════
            S_RESIZE_SETUP: begin
                rsz_ch <= 0;
                rsz_oh <= 0;
                rsz_ow <= 0;
                param_word_idx <= 0;
                param_read_issued <= 1'b0;
                rsz_rd_phase <= 0;
                // Precompute reciprocals (division here is once per layer, not critical path)
                recip_out_h <= (cfg_out_h > 0) ? (40'hFFFFFFFFFF / cfg_out_h) + 1 : 0;
                recip_out_w <= (cfg_out_w > 0) ? (40'hFFFFFFFFFF / cfg_out_w) + 1 : 0;
                recip_out_h_m1 <= (cfg_out_h > 1) ? (40'hFFFFFFFFFF / (cfg_out_h - 1)) + 1 : 0;
                recip_out_w_m1 <= (cfg_out_w > 1) ? (40'hFFFFFFFFFF / (cfg_out_w - 1)) + 1 : 0;
                // Precompute combined reciprocals: in_h * recip_out_h (single multiply per pixel)
                recip_scale_h <= {16'd0, cfg_in_h} * ((cfg_out_h > 0) ? (40'hFFFFFFFFFF / cfg_out_h) + 1 : 0);
                recip_scale_w <= {16'd0, cfg_in_w} * ((cfg_out_w > 0) ? (40'hFFFFFFFFFF / cfg_out_w) + 1 : 0);
                // Avoid unsized concat: use explicit sizing
                begin : resize_recip_m1_blk
                    reg [24:0] in_h_m1_shifted, in_w_m1_shifted;
                    in_h_m1_shifted = {9'd0, cfg_in_h[15:0]} - 17'd1;
                    in_h_m1_shifted = in_h_m1_shifted << 8;
                    in_w_m1_shifted = {9'd0, cfg_in_w[15:0]} - 17'd1;
                    in_w_m1_shifted = in_w_m1_shifted << 8;
                    recip_scale_h_m1 <= {16'd0, in_h_m1_shifted} * ((cfg_out_h > 17'd1) ? (40'hFFFFFFFFFF / (cfg_out_h - 17'd1)) + 1 : 0);
                    recip_scale_w_m1 <= {16'd0, in_w_m1_shifted} * ((cfg_out_w > 17'd1) ? (40'hFFFFFFFFFF / (cfg_out_w - 17'd1)) + 1 : 0);
                end
                state <= S_RESIZE_CH_SETUP;
            end

            S_RESIZE_CH_SETUP: begin
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= (rsz_ch * 4) + param_word_idx;
                    param_read_issued <= 1'b1;
                    act_read_issued <= 1'b0;
                end else if (!act_read_issued) begin
                    act_read_issued <= 1'b1;
                end else begin
                    param_buf[param_word_idx] <= param_rd_data;
                    if (param_word_idx == 3'd3) begin
                        ppu_mult_m     <= param_buf[0][14:0];
                        ppu_shift_s    <= param_buf[0][21:16];
                        ppu_zero_point <= $signed(param_buf[1][15:0]);
                        // Use param_rd_data for word 3 (param_buf[3] not yet updated this cycle)
                        ppu_bias       <= $signed({param_rd_data[15:0], param_buf[2],
                                                   param_buf[1][31:16]});
                        rsz_oh <= 0;
                        rsz_ow <= 0;
                        rsz_rd_phase <= 0;
                        state <= S_RESIZE_COORD;
                    end else begin
                        param_word_idx <= param_word_idx + 1;
                        param_read_issued <= 1'b0;
                    end
                end
            end

            S_RESIZE_COORD: begin
                // Single-cycle coord computation using combined reciprocals
                // prod = oh_global * recip_scale_h (one multiply, not two)
                begin : rsz_coord_blk
                    reg [31:0] oh_global, ow_global;
                    reg [71:0] prod_h, prod_w;
                    reg [15:0] ih_nearest, iw_nearest;
                    reg [31:0] src_h_q8, src_w_q8;
                    oh_global = tile_oh_origin + rsz_oh;
                    ow_global = tile_ow_origin + rsz_ow;

                    if (!resize_mode) begin
                        prod_h = oh_global * recip_scale_h;
                        prod_w = ow_global * recip_scale_w;
                        ih_nearest = prod_h[71:40];
                        iw_nearest = prod_w[71:40];
                        rsz_ih0 <= ih_nearest;
                        rsz_iw0 <= iw_nearest;
                        rsz_ih1 <= ih_nearest;
                        rsz_iw1 <= iw_nearest;
                        rsz_frac_h <= 0;
                        rsz_frac_w <= 0;
                    end else begin
                        if (cfg_out_h > 1) begin
                            prod_h = oh_global * recip_scale_h_m1;
                            src_h_q8 = prod_h[71:40];
                        end else
                            src_h_q8 = 0;
                        if (cfg_out_w > 1) begin
                            prod_w = ow_global * recip_scale_w_m1;
                            src_w_q8 = prod_w[71:40];
                        end else
                            src_w_q8 = 0;
                        rsz_ih0 <= src_h_q8[31:8];
                        rsz_iw0 <= src_w_q8[31:8];
                        rsz_frac_h <= src_h_q8[7:0];
                        rsz_frac_w <= src_w_q8[7:0];
                        if (src_h_q8[31:8] + 1 >= cfg_in_h)
                            rsz_ih1 <= cfg_in_h - 1;
                        else
                            rsz_ih1 <= src_h_q8[31:8] + 1;
                        if (src_w_q8[31:8] + 1 >= cfg_in_w)
                            rsz_iw1 <= cfg_in_w - 1;
                        else
                            rsz_iw1 <= src_w_q8[31:8] + 1;
                    end
                end
                rsz_rd_phase <= 0;
                state <= S_RESIZE_READ0;
            end

            S_RESIZE_READ0: begin
                case (rsz_rd_phase)
                2'd0: begin
                    begin : rsz_read0_addr
                        reg [31:0] elem_off;
                        reg [31:0] byte_off;
                        if (cfg_tile_h == 17'd0) begin
                            // Non-tiled: full image address
                            elem_off = (rsz_ih0 * cfg_in_w * cfg_in_c)
                                     + (rsz_iw0 * cfg_in_c)
                                     + rsz_ch;
                        end else begin
                            // Tiled: tile-local address
                            // Input tile origin = tile_oh_origin * in_h / out_h
                            // (Resize has no kernel/stride, scale = out/in)
                            elem_off = ((rsz_ih0 - rsz_tile_ih_origin) * tile_in_w
                                     + (rsz_iw0 - rsz_tile_iw_origin))
                                     * cfg_in_c + rsz_ch;
                        end
                        byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                        act_byte_sel <= byte_off[1:0];
                    end
                    rsz_rd_phase <= 2'd1;
                end
                2'd1: begin
                    rsz_rd_phase <= 2'd2;
                end
                2'd2: begin
                    begin : rsz_read0_cap
                        reg signed [ACC_W-1:0] val;
                        if (cfg_int16) begin
                            case (act_byte_sel[1])
                                1'b0: val = $signed(act_rd_data[15:0]);
                                1'b1: val = $signed(act_rd_data[31:16]);
                            endcase
                        end else begin
                            case (act_byte_sel)
                                2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
                                2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
                                2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
                                2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                            endcase
                        end
                        rsz_v00 <= val;
                    end
                    rsz_rd_phase <= 0;
                    if (resize_mode)
                        state <= S_RESIZE_READ1;
                    else
                        state <= S_RESIZE_PPU;
                end
                default: rsz_rd_phase <= 0;
                endcase
            end

            S_RESIZE_READ1: begin
                case (rsz_rd_phase)
                2'd0: begin
                    begin : rsz_read1_addr
                        reg [31:0] elem_off;
                        reg [31:0] byte_off;
                        if (cfg_tile_h == 17'd0) begin
                            elem_off = (rsz_ih0 * cfg_in_w * cfg_in_c)
                                     + (rsz_iw1 * cfg_in_c)
                                     + rsz_ch;
                        end else begin
                            elem_off = ((rsz_ih0 - rsz_tile_ih_origin) * tile_in_w
                                     + (rsz_iw1 - rsz_tile_iw_origin))
                                     * cfg_in_c + rsz_ch;
                        end
                        byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                        act_byte_sel <= byte_off[1:0];
                    end
                    rsz_rd_phase <= 2'd1;
                end
                2'd1: begin
                    rsz_rd_phase <= 2'd2;
                end
                2'd2: begin
                    begin : rsz_read1_cap
                        reg signed [ACC_W-1:0] val;
                        if (cfg_int16) begin
                            case (act_byte_sel[1])
                                1'b0: val = $signed(act_rd_data[15:0]);
                                1'b1: val = $signed(act_rd_data[31:16]);
                            endcase
                        end else begin
                            case (act_byte_sel)
                                2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
                                2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
                                2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
                                2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                            endcase
                        end
                        rsz_v01 <= val;
                    end
                    rsz_rd_phase <= 0;
                    state <= S_RESIZE_READ2;
                end
                default: rsz_rd_phase <= 0;
                endcase
            end

            S_RESIZE_READ2: begin
                case (rsz_rd_phase)
                2'd0: begin
                    begin : rsz_read2_addr
                        reg [31:0] elem_off;
                        reg [31:0] byte_off;
                        if (cfg_tile_h == 17'd0) begin
                            elem_off = (rsz_ih1 * cfg_in_w * cfg_in_c)
                                     + (rsz_iw0 * cfg_in_c)
                                     + rsz_ch;
                        end else begin
                            elem_off = ((rsz_ih1 - rsz_tile_ih_origin) * tile_in_w
                                     + (rsz_iw0 - rsz_tile_iw_origin))
                                     * cfg_in_c + rsz_ch;
                        end
                        byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                        act_byte_sel <= byte_off[1:0];
                    end
                    rsz_rd_phase <= 2'd1;
                end
                2'd1: begin
                    rsz_rd_phase <= 2'd2;
                end
                2'd2: begin
                    begin : rsz_read2_cap
                        reg signed [ACC_W-1:0] val;
                        if (cfg_int16) begin
                            case (act_byte_sel[1])
                                1'b0: val = $signed(act_rd_data[15:0]);
                                1'b1: val = $signed(act_rd_data[31:16]);
                            endcase
                        end else begin
                            case (act_byte_sel)
                                2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
                                2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
                                2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
                                2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                            endcase
                        end
                        rsz_v10 <= val;
                    end
                    rsz_rd_phase <= 0;
                    state <= S_RESIZE_READ3;
                end
                default: rsz_rd_phase <= 0;
                endcase
            end

            S_RESIZE_READ3: begin
                case (rsz_rd_phase)
                2'd0: begin
                    begin : rsz_read3_addr
                        reg [31:0] elem_off;
                        reg [31:0] byte_off;
                        if (cfg_tile_h == 17'd0) begin
                            elem_off = (rsz_ih1 * cfg_in_w * cfg_in_c)
                                     + (rsz_iw1 * cfg_in_c)
                                     + rsz_ch;
                        end else begin
                            elem_off = ((rsz_ih1 - rsz_tile_ih_origin) * tile_in_w
                                     + (rsz_iw1 - rsz_tile_iw_origin))
                                     * cfg_in_c + rsz_ch;
                        end
                        byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                        act_byte_sel <= byte_off[1:0];
                    end
                    rsz_rd_phase <= 2'd1;
                end
                2'd1: begin
                    rsz_rd_phase <= 2'd2;
                end
                2'd2: begin
                    begin : rsz_read3_cap
                        reg signed [ACC_W-1:0] val;
                        if (cfg_int16) begin
                            case (act_byte_sel[1])
                                1'b0: val = $signed(act_rd_data[15:0]);
                                1'b1: val = $signed(act_rd_data[31:16]);
                            endcase
                        end else begin
                            case (act_byte_sel)
                                2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
                                2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
                                2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
                                2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                            endcase
                        end
                        rsz_v11 <= val;
                    end
                    rsz_rd_phase <= 0;
                    state <= S_RESIZE_INTERP1;
                end
                default: rsz_rd_phase <= 0;
                endcase
            end

            S_RESIZE_INTERP1: begin
                // Bilinear interp cycle 1: compute top and bot (4 mults)
                begin : rsz_interp1_blk
                    reg [8:0] one_minus_w;
                    one_minus_w = 9'd256 - {1'b0, rsz_frac_w};
                    rsz_top_r <= rsz_v00 * $signed({1'b0, one_minus_w})
                               + rsz_v01 * $signed({1'b0, rsz_frac_w});
                    rsz_bot_r <= rsz_v10 * $signed({1'b0, one_minus_w})
                               + rsz_v11 * $signed({1'b0, rsz_frac_w});
                end
                state <= S_RESIZE_INTERP2;
            end

            S_RESIZE_INTERP2: begin
                // Bilinear interp cycle 2: compute val64 (2 mults + shift)
                begin : rsz_interp2_blk
                    reg [8:0] one_minus_h;
                    reg signed [63:0] val64;
                    one_minus_h = 9'd256 - {1'b0, rsz_frac_h};
                    val64 = rsz_top_r * $signed({1'b0, one_minus_h})
                          + rsz_bot_r * $signed({1'b0, rsz_frac_h});
                    ppu_acc_in <= $signed((val64 + 64'sd32768) >>> 16);
                end
                state <= S_RESIZE_PPU;
            end

            S_RESIZE_PPU: begin
                if (!resize_mode)
                    ppu_acc_in <= rsz_v00;
                ppu_in_valid <= 1'b1;
                state <= S_RESIZE_PPU_WAIT;
            end

            S_RESIZE_PPU_WAIT: begin
                if (ppu_out_valid) begin
                    begin : rsz_wb_addr_calc
                        reg [31:0] elem_off;
                        reg [31:0] byte_off;
                        elem_off = (rsz_oh * out_tile_w + rsz_ow)
                                 * cfg_out_c + rsz_ch;
                        byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                        rsz_wb_addr    <= out_base + byte_off[ACT_ADDR_W+1:2];
                        rsz_wb_bytesel <= byte_off[1:0];
                    end
                    rsz_wb_byte  <= ppu_out_data;
                    rsz_wb_phase <= 2'd0;
                    state <= S_RESIZE_WB;
                end
            end

            S_RESIZE_WB: begin
                act_rd_ofm <= 1'b1;
                case (rsz_wb_phase)
                2'd0: begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= rsz_wb_addr;
                    rsz_wb_phase <= 2'd1;
                end
                2'd1: begin
                    rsz_wb_phase <= 2'd2;
                end
                2'd2: begin
                    begin : rsz_wb_merge
                        reg [31:0] merged;
                        merged = act_rd_data;
                        if (cfg_int16) begin
                            case (rsz_wb_bytesel[1])
                                1'b0: merged[15:0]  = rsz_wb_byte;
                                1'b1: merged[31:16] = rsz_wb_byte;
                            endcase
                        end else begin
                            case (rsz_wb_bytesel)
                                2'd0: merged[7:0]   = rsz_wb_byte[7:0];
                                2'd1: merged[15:8]  = rsz_wb_byte[7:0];
                                2'd2: merged[23:16] = rsz_wb_byte[7:0];
                                2'd3: merged[31:24] = rsz_wb_byte[7:0];
                            endcase
                        end
                        act_wr_en   <= 1'b1;
                        act_wr_addr <= rsz_wb_addr;
                        act_wr_data <= merged;
                    end
                    state <= S_RESIZE_PIX_NEXT;
                end
                default: rsz_wb_phase <= 2'd0;
                endcase
            end

            S_RESIZE_PIX_NEXT: begin
                rsz_rd_phase <= 0;
                if (rsz_ow + 1 >= out_tile_w) begin
                    rsz_ow <= 0;
                    if (rsz_oh + 1 >= out_tile_h) begin
                        state <= S_RESIZE_CH_NEXT;
                    end else begin
                        rsz_oh <= rsz_oh + 1;
                        state <= S_RESIZE_COORD;
                    end
                end else begin
                    rsz_ow <= rsz_ow + 1;
                    state <= S_RESIZE_COORD;
                end
            end

            S_RESIZE_CH_NEXT: begin
                if (rsz_ch + 1 >= cfg_out_c) begin
                    state <= S_TILE_NEXT;
                end else begin
                    state <= S_OC_NEXT;
                end
            end

            default: state <= S_IDLE;

            endcase

            // Prefetch next k_pass (or wrap to pass 0 for the next spatial
            // block) into wgt_sh while the array is computing.
            if (wgt_pf_want
                    && (state == S_ACT_LOAD || state == S_ACT_EMIT
                        || state == S_ACT_FLUSH || state == S_ACT_CMD
                        || state == S_SPATIAL_SETUP
                        || state == S_PSUM_COLLECT)) begin
                if (wgt_pf_ph == 2'd0) begin
                    begin : wgt_pf_addr
                        reg [31:0] elem_off, byte_off;
                        elem_off = {11'd0, wgt_pf_col} * k_depth
                                 + wgt_pf_tgt * ARRAY_SIZE_16;
                        byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                        wgt_rd_en   <= 1'b1;
                        wgt_rd_addr <= wgt_base[WGT_ADDR_W-1:0] + byte_off[WGT_ADDR_W+1:2];
                    end
                    wgt_pf_ph <= 2'd1;
                end else if (wgt_pf_ph == 2'd1) begin
                    wgt_pf_ph <= 2'd2;
                end else begin
                    begin : wgt_pf_unpack
                        integer ei;
                        for (ei = 0; ei < ARRAY_SIZE; ei = ei + 1) begin
                            if (cfg_int16)
                                wgt_sh[{wgt_pf_col[COL_W-1:0], ei[COL_W-1:0]}]
                                    <= $signed(wgt_rd_data[16*ei +: 16]);
                            else
                                wgt_sh[{wgt_pf_col[COL_W-1:0], ei[COL_W-1:0]}]
                                    <= {{8{wgt_rd_data[8*ei+7]}}, wgt_rd_data[8*ei +: 8]};
                        end
                    end
                    if (wgt_pf_col >= {1'b0, COL_MAX}) begin
                        wgt_sh_ok   <= 1'b1;
                        wgt_sh_pass <= wgt_pf_tgt;
                        wgt_pf_col  <= 0;
                        wgt_pf_ph   <= 2'd0;
                    end else begin
                        wgt_pf_col <= wgt_pf_col + 1;
                        wgt_pf_ph  <= 2'd0;
                    end
                end
            end
        end
    end

`ifdef DBG_DOTBUF
// Independent tracker: log ALL dot_buf[9] changes
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dbg_prev_db9 <= 0;
    end else begin
        if (dbg_prev_db9 != dot_buf[9]) begin
            $fwrite(dbg_trace_fh, "[CHG9] t=%0d dot_buf[9]=%0d (0x%h) state=%0d tile(%0d,%0d) sp(%0d,%0d) drain=%0d pass=%0d\n",
                    $time, dot_buf[9], dot_buf[9], state, tile_y, tile_x, sp_oh, sp_ow, drain_col, k_pass);
            dbg_prev_db9 <= dot_buf[9];
        end
    end
end
`endif

endmodule
