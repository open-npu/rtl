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
//   - k_depth > ARRAY_SIZE supported via multi-pass partial sum accumulation
//   - Spatial output count is handled by repeated compute passes
//   - Weights reloaded per-pixel per-pass (correctness over efficiency)
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
    output wire [DATA_W*ARRAY_SIZE-1:0] sa_wgt_data_flat,
    output reg                          sa_wgt_valid,
    output wire [DATA_W*ARRAY_SIZE-1:0] sa_act_data_flat,
    output reg                          sa_act_valid,
    output reg  [$clog2(ARRAY_SIZE)-1:0] sa_drain_col_sel,
    input  wire [ACC_W*ARRAY_SIZE-1:0]  sa_acc_out_flat,
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
    output reg  signed [ACC_W-1:0]     ppu_acc_in /* verilator public */,
    output reg                          ppu_in_valid,
    output reg  signed [ACC_W-1:0]     ppu_bias,
    output reg  [14:0]                  ppu_mult_m,
    output reg  [5:0]                   ppu_shift_s,
    output reg  signed [15:0]          ppu_zero_point,
    input  wire signed [DATA_W-1:0]    ppu_out_data,
    input  wire                         ppu_out_valid
);

    // ─── Internal unpacked arrays for systolic interface ───
    reg  signed [DATA_W-1:0] sa_wgt_data [0:ARRAY_SIZE-1];
    reg  signed [DATA_W-1:0] sa_act_data [0:ARRAY_SIZE-1];
    wire signed [ACC_W-1:0]  sa_acc_out  [0:ARRAY_SIZE-1];
    genvar gi;
    generate
        for (gi = 0; gi < ARRAY_SIZE; gi = gi + 1) begin : unpack_sa
            assign sa_wgt_data_flat[DATA_W*gi +: DATA_W] = sa_wgt_data[gi];
            assign sa_act_data_flat[DATA_W*gi +: DATA_W] = sa_act_data[gi];
            assign sa_acc_out[gi] = sa_acc_out_flat[ACC_W*gi +: ACC_W];
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

    // ─── Reciprocal LUT for AvgPool division (pool_count 1..15) ───
    // Q32 fixed-point: result = (dividend * recip) >>> 32
    // For pool_count=1: recip = 0x80000000 (shift 31 instead of 32 to fit 32-bit)
    function [31:0] recip_pool;
        input [3:0] idx;
        case (idx)
            4'd1:  recip_pool = 32'h8000_0000;  // 1/1 with >>31
            4'd2:  recip_pool = 32'h8000_0000;  // 1/2 with >>32
            4'd3:  recip_pool = 32'h5555_5556;
            4'd4:  recip_pool = 32'h4000_0000;
            4'd5:  recip_pool = 32'h3333_3333;
            4'd6:  recip_pool = 32'h2AAA_AAAB;
            4'd7:  recip_pool = 32'h2492_4925;
            4'd8:  recip_pool = 32'h2000_0000;
            4'd9:  recip_pool = 32'h1C71_C71C;
            4'd10: recip_pool = 32'h1999_999A;
            4'd11: recip_pool = 32'h1745_D174;
            4'd12: recip_pool = 32'h1555_5555;
            4'd13: recip_pool = 32'h13B1_3B14;
            4'd14: recip_pool = 32'h1249_2492;
            4'd15: recip_pool = 32'h1111_1111;
            default: recip_pool = 32'h0;
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
        S_DRAIN_CMD   = 7'd10,
        S_DRAIN_WAIT  = 7'd11,
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
        S_REDUCE        = 7'd28,
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
        S_RESIZE_INTERP2  = 7'd65; // Bilinear interp cycle 2 (2 mults: val64)
    (* fsm_encoding = "one_hot" *)
    reg [6:0] state;

    // ─── Tile iteration ───
    reg [15:0] tile_y, tile_x;
    reg        tile_done_r;
    reg        tile_wait_delay;  // DB_EN wait: skip 1 cycle, then wait for prefetch
    assign tile_done = tile_done_r;
    assign oc_group_out = oc_group;
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
    // Load one column at a time: read ceil(k_pass_remain/4) words, fill sa_wgt_data
    reg [$clog2(ARRAY_SIZE)-1:0] wgt_col_idx;     // current column (0..ARRAY_SIZE-1)
    reg [$clog2(ARRAY_SIZE):0]   wgt_byte_idx;    // byte index within column (0..ARRAY_SIZE-1)
    reg [15:0]                   wgt_word_addr;    // current SRAM address
    reg                          wgt_read_issued;  // 1-cycle read latency tracker
    reg                          wgt_data_ready;   // 2nd cycle: SRAM data available
    reg [1:0]                    wgt_bsel;         // byte offset within first word

    // ─── Activation stream state ───
    reg [15:0] act_cnt;         // activation byte counter (0..k_depth-1)
    reg [15:0] act_word_addr;   // current SRAM address
    reg        act_read_issued;
    reg        act_data_ready;  // 2nd cycle: SRAM data available
    reg [31:0] act_buf;         // buffered SRAM word
    reg [1:0]  act_byte_sel;    // byte position within word

    // ─── Drain state ───
    reg [$clog2(ARRAY_SIZE)-1:0] drain_col;
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
    reg [5:0]  dw_cnt;
    reg        dw_read_issued;
    reg [1:0]  dw_init_phase;          // 0=setup, 1=acc_clear, 2=feeding
    reg [15:0] dw_oh, dw_ow;          // Output pixel coordinates
    reg [3:0]  dw_fh, dw_fw;          // Filter position (0..6)
    reg signed [ACC_W-1:0] dw_acc_buf; // Captured DW output accumulator
    reg [5:0]  dw_kernel_size;         // kh * kw (cached)
    reg [1:0]  dw_wb_phase;            // 0=issue read, 1=wait, 2=merge+write
    reg [15:0] dw_wb_byte;             // PPU output element to write (up to 16-bit)
    reg [1:0]  dw_wb_bytesel;          // byte position within word
    reg [ACT_ADDR_W-1:0] dw_wb_addr;  // target word address
    reg [1:0]  dw_wgt_bsel_base;       // starting byte offset for weight reads

    // ─── Spatial pixel loop (Conv2D Plan A) ───
    reg [15:0] sp_oh, sp_ow;                  // Current output pixel coordinates (tile-local)
    reg [15:0] tile_oh_origin, tile_ow_origin; // Global origin of current tile
    reg [15:0] tile_in_h, tile_in_w;           // Input tile dimensions (including halo)
    reg [15:0] kw_eff;                         // Effective kernel width: (kw-1)*dw+1
    reg signed [ACC_W-1:0] dot_acc;           // Reduction accumulator
    reg [$clog2(ARRAY_SIZE):0] reduce_cnt;    // Reduction counter
    reg [ACT_ADDR_W-1:0] pixel_act_base;     // Per-pixel activation base address
    reg signed [ACC_W-1:0] dot_buf [0:ARRAY_SIZE-1]; // Reduced dot products per column

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
    wire        resize_mode   = cfg_resize_cfg[0];   // 0=nearest, 1=bilinear
    // ─── Deconv state ───
    wire [7:0]  cfg_insert_h  = cfg_deconv_cfg[7:0];
    wire [7:0]  cfg_insert_w  = cfg_deconv_cfg[15:8];
    wire        is_deconv     = (cfg_op_type == 8'd6);
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
            wgt_word_addr <= 0; wgt_read_issued <= 0; wgt_data_ready <= 0;
            act_cnt <= 0; act_word_addr <= 0;
            act_read_issued <= 0; act_data_ready <= 0; act_buf <= 0; act_byte_sel <= 0;
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
            flush_cnt <= 0;
            sp_oh <= 0; sp_ow <= 0; tile_oh_origin <= 0; tile_ow_origin <= 0;
            dot_acc <= 0; reduce_cnt <= 0; pixel_act_base <= 0;
            k_pass <= 0; k_pass_max <= 0; k_pass_remain <= 0;
            conv_fh <= 0; conv_fw <= 0; conv_ch_cnt <= 0;
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
            done         <= 1'b0;
            tile_done_r  <= 1'b0;
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

            (* parallel_case, full_case *)
            case (state)

            // ══════════════════════════════════════════════════════════════
            S_IDLE: begin
                if (start) begin
                    if (cfg_op_type == 8'd1 || cfg_op_type == 8'd3 || cfg_op_type == 8'd5)
                        oc_groups_total <= cfg_out_c;
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
                out_base <= cfg_act_base + cfg_out_base;

                oc_group <= 0;

                if (cfg_op_type == 8'd1) begin
                    dw_cnt <= 0;
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
                // Param base: 4 words per channel, ARRAY_SIZE channels per group
                param_base <= oc_group * ARRAY_SIZE_16 * 4;

                // Multi-pass setup
                k_pass <= 0;
                k_pass_max <= (k_depth - 1) / ARRAY_SIZE_16;

                // Last valid drain column for this oc_group
                begin : col_last_blk
                    reg [15:0] remaining_oc;
                    remaining_oc = cfg_out_c - oc_group * ARRAY_SIZE_16;
                    if (remaining_oc >= ARRAY_SIZE_16)
                        col_last <= COL_MAX;
                    else
                        col_last <= remaining_oc[$clog2(ARRAY_SIZE)-1:0] - 1;
                end

                // Reset spatial coords for first pixel
                sp_oh <= 0;
                sp_ow <= 0;

                // Reset writeback packing state for this OC group
                wb_pack <= 0;
                wb_pos  <= 2'd0;

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
                state <= S_WGT_LOAD;
            end

            S_WGT_LOAD: begin
                // Fill sa_wgt_data[0..k_pass_remain-1] for current column, zero-pad rest
                // INT8: Read SRAM words (4 bytes each) and unpack with byte offset (wgt_bsel)
                // INT16: Read SRAM words (2 half-words each) and unpack
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
                                oc_group, wgt_col_idx, k_pass, wgt_word_addr, wgt_rd_data, wgt_bsel);
`endif
                    if (cfg_int16) begin : wgt_unpack_int16_blk
                        // INT16: extract 2 half-words per SRAM word
                        reg [31:0] shifted;
                        shifted = wgt_rd_data >> (wgt_bsel * 8);
                        if (wgt_byte_idx < k_pass_remain)
                            sa_wgt_data[wgt_byte_idx[COL_W-1:0]] <= $signed(shifted[15:0]);
                        if (wgt_byte_idx + 1 < k_pass_remain && wgt_bsel == 2'd0)
                            sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 1] <= $signed(shifted[31:16]);
                    end else begin : wgt_unpack_int8_blk
                        // INT8: extract 4 bytes, sign-extend each to 16-bit
                        reg [31:0] shifted;
                        shifted = wgt_rd_data >> (wgt_bsel * 8);
                        if (wgt_byte_idx < k_pass_remain)
                            sa_wgt_data[wgt_byte_idx[COL_W-1:0]] <= {{8{shifted[7]}}, shifted[7:0]};
                        if (wgt_byte_idx + 1 < k_pass_remain && wgt_bsel < 2'd3)
                            sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 1] <= {{8{shifted[15]}}, shifted[15:8]};
                        if (wgt_byte_idx + 2 < k_pass_remain && wgt_bsel < 2'd2)
                            sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 2] <= {{8{shifted[23]}}, shifted[23:16]};
                        if (wgt_byte_idx + 3 < k_pass_remain && wgt_bsel < 2'd1)
                            sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 3] <= {{8{shifted[31]}}, shifted[31:24]};
                    end

                    begin : wgt_advance_blk
                        reg [COL_W:0] elems_this_word;
                        if (cfg_int16)
                            elems_this_word = (wgt_bsel == 2'd0) ? 2 : 1;
                        else
                            elems_this_word = {2'd0, 3'd4 - {1'b0, wgt_bsel}};
                        wgt_byte_idx <= wgt_byte_idx + elems_this_word;

                        if (wgt_byte_idx + elems_this_word >= k_pass_remain) begin
                            // All k_pass_remain elements loaded; zero-pad remaining rows
                            if (k_pass_remain < ARRAY_SIZE_16) begin
                                for (i = 0; i < ARRAY_SIZE; i = i + 1)
                                    if (i[COL_W-1:0] >= k_pass_remain[COL_W-1:0])
                                        sa_wgt_data[i[COL_W-1:0]] <= 0;
                            end
                            // Emit
                            state <= S_WGT_EMIT;
                        end else begin
                            // Need more words (next word starts at offset 0)
                            wgt_word_addr <= wgt_word_addr + 1;
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
                    state <= S_SPATIAL_SETUP;
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
                // Wait for systolic to be ready before issuing COMPUTE
                if (sa_ready) begin
                    sa_cmd       <= MODE_COMPUTE;
                    sa_cmd_valid <= 1'b1;
                    act_cnt      <= 0;
                    act_byte_sel <= 2'd0;
                    act_read_issued <= 1'b0;
                    act_data_ready  <= 1'b0;

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
                                elem_off = (({8'd0, sp_oh} * cfg_stride_h + {8'd0, conv_fh}) * tile_in_w
                                         + {8'd0, sp_ow} * cfg_stride_w + {8'd0, conv_fw}) * cfg_in_c
                                         + {8'd0, conv_ch_cnt};
                            end
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            act_word_addr <= {2'd0, act_base} + byte_off[17:2];
`ifdef DBG_DOTBUF
                            if (sp_oh == 0 && sp_ow == 0 && k_pass < 2 && ((tile_x == 0 && tile_y == 0) || (tile_x == 1 && tile_y == 0)))
                                $fwrite(dbg_fh, "[RTL_CMD] t=%0d tile(%0d,%0d) pass=%0d fh=%0d fw=%0d elem_off=%0d act_addr=%0d\n",
                                        $time, tile_y, tile_x, k_pass, conv_fh, conv_fw, elem_off,
                                        {2'd0, act_base} + byte_off[17:2]);
`endif
                            act_byte_sel <= byte_off[1:0];
                        end
                    end
                    state <= S_ACT_LOAD;
                end
            end

            S_ACT_LOAD: begin
                // Read one SRAM word (4 activation bytes)
                // If padding or deconv_skip, skip read and go directly to emit zeros
                if (conv_is_pad || deconv_skip) begin
                    // Use cfg_in_zp for padding, matching CSIM dma_extract_tile behavior
                    if (cfg_int16)
                        act_buf <= {cfg_in_zp, cfg_in_zp};
                    else
                        act_buf <= {cfg_in_zp[7:0], cfg_in_zp[7:0],
                                    cfg_in_zp[7:0], cfg_in_zp[7:0]};
                    state <= S_ACT_EMIT;
                end else if (!act_read_issued) begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= act_word_addr[ACT_ADDR_W-1:0];
                    act_read_issued <= 1'b1;
                    act_data_ready  <= 1'b0;
                end else if (!act_data_ready) begin
                    // Wait for SRAM read latency
                    act_data_ready <= 1'b1;
                end else begin
                    // Data available
                    act_buf <= act_rd_data;
`ifdef DBG_DOTBUF
                    if (sp_oh == 0 && sp_ow == 0 && k_pass < 2 && ((tile_x == 0 && tile_y == 0) || (tile_x == 1 && tile_y == 0)))
                        $fwrite(dbg_fh, "[RTL_RD] t=%0d tile(%0d,%0d) sp(%0d,%0d) pass=%0d act_addr=%0d act_data=0x%08x\n",
                                $time, tile_y, tile_x, sp_oh, sp_ow, k_pass, act_word_addr[ACT_ADDR_W-1:0], act_rd_data);
`endif
                    state <= S_ACT_EMIT;
                end
            end

            S_ACT_EMIT: begin
                // Send activation to row act_cnt ONLY (per-row targeting)
                begin : act_emit_blk
                    reg signed [DATA_W-1:0] aval;
                    if (cfg_int16) begin
                        // INT16: extract half-word (2 elements per 32-bit word)
                        case (act_byte_sel[1])
                            1'b0: aval = $signed(act_buf[15:0]);
                            1'b1: aval = $signed(act_buf[31:16]);
                            default: aval = 0;
                        endcase
                    end else begin
                        // INT8: extract byte, sign-extend to 16-bit
                        case (act_byte_sel)
                            2'd0: aval = {{8{act_buf[7]}},  act_buf[7:0]};
                            2'd1: aval = {{8{act_buf[15]}}, act_buf[15:8]};
                            2'd2: aval = {{8{act_buf[23]}}, act_buf[23:16]};
                            2'd3: aval = {{8{act_buf[31]}}, act_buf[31:24]};
                            default: aval = 0;
                        endcase
                    end
                    for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                        if (i[COL_W-1:0] == act_cnt[COL_W-1:0])
                            sa_act_data[i] <= aval;
                        else
                            sa_act_data[i] <= 0;
                    end
                end
                sa_act_valid <= 1'b1;
                act_cnt <= act_cnt + 1;
                conv_ch_cnt <= conv_ch_cnt + 1;

                if (act_cnt + 1 >= k_pass_remain) begin
                    // All K elements for this pass streamed → flush
                    flush_cnt <= 0;
                    state <= S_ACT_FLUSH;
                end else if (conv_ch_cnt + 1 >= cfg_in_c) begin
                    // Reached end of channels for current (fh, fw)
                    // Advance to next filter position
                    conv_ch_cnt <= 0;
                    if (conv_fw + 1 >= {8'd0, cfg_kernel_w}) begin
                        conv_fw <= 0;
                        conv_fh <= conv_fh + 1;
                    end else begin
                        conv_fw <= conv_fw + 1;
                    end
                    // Need to recompute address for new (fh, fw)
                    act_read_issued <= 1'b0;
                    act_data_ready  <= 1'b0;
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
                            // Use tile-local or full-image coords, matching S_ACT_CMD
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
                end else if (cfg_int16 ? (act_byte_sel[1] == 1'b1) : (act_byte_sel == 2'd3)) begin
                    // Need next SRAM word
                    act_byte_sel <= 2'd0;
                    act_word_addr <= act_word_addr + 1;
                    act_read_issued <= 1'b0;
                    state <= S_ACT_LOAD;
                end else begin
                    // Next element in same word
                    act_byte_sel <= cfg_int16 ? (act_byte_sel + 2'd2) : (act_byte_sel + 2'd1);
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
                    // Capture per-row results for this column
                    for (i = 0; i < ARRAY_SIZE; i = i + 1)
                        acc_buf[i] <= sa_acc_out[i];
                    // Reduce: sum rows 0..k_depth-1 into dot product
                    reduce_cnt <= 0;
                    dot_acc <= 0;
                    state <= S_REDUCE;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // REDUCE: sum k_pass_remain partial products into one dot product
            // Accumulate across passes in dot_buf[drain_col]
            // ══════════════════════════════════════════════════════════════
            S_REDUCE: begin
                dot_acc <= dot_acc + acc_buf[reduce_cnt[COL_W-1:0]];
                reduce_cnt <= reduce_cnt + 1;
                if (reduce_cnt + 1 >= k_pass_remain) begin
                    // Accumulate reduced dot product into dot_buf[drain_col]
                    dot_buf[drain_col] <= dot_buf[drain_col]
                        + dot_acc + acc_buf[reduce_cnt[COL_W-1:0]];
`ifdef DBG_DOTBUF
                    if (drain_col == 0 && ((tile_x == 0 && tile_y == 0) || (tile_x == 1 && tile_y == 0)) && sp_oh == 0 && sp_ow == 0)
                        $fwrite(dbg_fh, "[RTL_DB] t=%0d tile(%0d,%0d) sp(%0d,%0d) col=0 pass=%0d kpr=%0d dot_acc=%0d acc_buf=%0d dot_buf_next=%0d\n",
                                $time, tile_y, tile_x, sp_oh, sp_ow, k_pass, k_pass_remain,
                                dot_acc, acc_buf[reduce_cnt[COL_W-1:0]],
                                dot_buf[drain_col] + dot_acc + acc_buf[reduce_cnt[COL_W-1:0]]);
`endif
                    // Next column or decide next step
                    if (drain_col == COL_MAX) begin
                        // All columns drained for this pass
                        if (k_pass >= k_pass_max) begin
                            // Final pass — proceed to PPU
                            param_word_idx <= 0;
                            param_read_issued <= 1'b0;
                            param_data_ready  <= 1'b0;
                            drain_col <= 0;
                            state <= S_PARAM_LOAD;
                        end else begin
                            // More passes needed — reload weights for next slice
                            k_pass <= k_pass + 1;
                            wgt_col_idx <= 0;
                            state <= S_WGT_CMD;
                        end
                    end else begin
                        drain_col <= drain_col + 1;
                        state <= S_DRAIN_CMD;
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
                ppu_acc_in   <= dot_buf[drain_col];
                ppu_in_valid <= 1'b1;
`ifdef DBG_DOTBUF
                if (drain_col == 9 && ((tile_x == 0 && tile_y == 0) || (tile_x == 3 && tile_y == 6)) && sp_oh == 0 && sp_ow == 0)
                    $fwrite(dbg_fh, "[RTL_PPU] t=%0d drain=%0d acc_in=%0d bias=%0d M=%0d S=%0d zp=%0d\n",
                            $time, drain_col, dot_buf[drain_col],
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
                                  + oc_group * ARRAY_SIZE_16
                                  + ({12'd0, drain_col} & ~17'd1)) >> 1);
                            act_wr_data <= {ppu_out_data, wb_pack[15:0]};
                            `ifndef SYNTHESIS
                            if (tile_x == 1 && tile_y == 0 && sp_oh == 0 && sp_ow == 0)
                                $display("[CMP_WB] t=%0t drain=%0d col_last=%0d addr=%0d data=0x%04x%04x",
                                         $time, drain_col, col_last,
                                         out_base + (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                           + oc_group * ARRAY_SIZE_16
                                           + ({12'd0, drain_col} & ~17'd1)) >> 1),
                                         ppu_out_data, wb_pack[15:0]);
                            if (tile_x == 3 && tile_y == 0 && sp_oh == 0 && sp_ow == 0)
                                $display("[CMP_WB_BORDER] t=%0t drain=%0d col_last=%0d out_tw=%0d addr=%0d data=0x%04x%04x",
                                         $time, drain_col, col_last, out_tile_w,
                                         out_base + (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                           + oc_group * ARRAY_SIZE_16
                                           + ({12'd0, drain_col} & ~17'd1)) >> 1),
                                         ppu_out_data, wb_pack[15:0]);
                            if (tile_x == 0 && tile_y == 1 && sp_oh == 0 && sp_ow == 0)
                                $display("[CMP_WB_ROW2] t=%0t drain=%0d col_last=%0d out_th=%0d out_tw=%0d addr=%0d data=0x%04x%04x",
                                         $time, drain_col, col_last, out_tile_h, out_tile_w,
                                         out_base + (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                           + oc_group * ARRAY_SIZE_16
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
                                  + oc_group * ARRAY_SIZE_16
                                  + ({12'd0, drain_col} & ~17'd3)) >> 2);
                            act_wr_data <= {ppu_out_data[7:0], wb_pack[23:0]};
                        end
                    endcase
                    wb_pos <= wb_pos + 2'd1;
                end

                // Advance to next channel or finish pixel
                if (drain_col == col_last) begin
                    // Last channel of this pixel — check if we need partial flush
                    // Flush only when this is the LAST pixel of the tile
                    // (next pixel will be in same word if OUT_C not word-aligned)
                    if (sp_ow + 1 >= out_tile_w && sp_oh + 1 >= out_tile_h) begin
                        // Last pixel — flush any partial word
                        if (cfg_int16) begin
                            if (wb_pos[0] != 1'b1) begin
                                act_wr_en   <= 1'b1;
                                act_wr_addr <= out_base +
                                    (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
                                      + oc_group * ARRAY_SIZE_16
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
                                      + oc_group * ARRAY_SIZE_16
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
                    state <= S_PIXEL_NEXT;
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
                    begin : deconv_precompute_blk
                        reg signed [15:0] eh_local, ew_local;
                        reg [47:0] qh, qw;
                        reg [15:0] ih_u, iw_u;
                        reg signed [15:0] ih_s, iw_s;
                        eh_local = $signed({1'b0, tile_oh_origin + sp_oh})
                                 + $signed({1'b0, cfg_pad_top[7:0]})
                                 - $signed({8'd0, conv_fh});
                        ew_local = $signed({1'b0, tile_ow_origin + sp_ow})
                                 + $signed({1'b0, cfg_pad_left[7:0]})
                                 - $signed({8'd0, conv_fw});
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
                            deconv_elem_off <= (ih_u * cfg_in_w + iw_u) * cfg_in_c + conv_ch_cnt;
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
                            rem_kw = flat_start - q_full[15:0] * kw_x_inc_r;
                            // fw = rem_kw / in_c (special case in_c=1)
                            if (in_c_r == 16'd1) begin
                                conv_fw <= rem_kw[7:0];
                                conv_ch_cnt <= 16'd0;
                            end else begin
                                q_fw = (rem_kw * recip_in_c) >> 32;
                                conv_fw <= q_fw[7:0];
                                // ch = flat_start % in_c
                                q_ch = (flat_start * recip_in_c) >> 32;
                                conv_ch_cnt <= flat_start - q_ch[15:0] * in_c_r;
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
            // PIXEL NEXT: advance spatial pixel or go to OC_NEXT
            // ══════════════════════════════════════════════════════════════
            S_PIXEL_NEXT: begin
                // Reset k_pass for next pixel
                k_pass <= 0;
                if (sp_ow + 1 >= out_tile_w) begin
                    if (sp_oh + 1 >= out_tile_h) begin
                        // All pixels done for this OC group
                        state <= S_OC_NEXT;
                    end else begin
                        sp_ow <= 0;
                        sp_oh <= sp_oh + 1;
                        if (k_pass_max > 0) begin
                            wgt_col_idx <= 0;
                            state <= S_WGT_CMD;
                        end else
                            state <= S_SPATIAL_SETUP;
                    end
                end else begin
                    sp_ow <= sp_ow + 1;
                    if (k_pass_max > 0) begin
                        wgt_col_idx <= 0;
                        state <= S_WGT_CMD;
                    end else
                        state <= S_SPATIAL_SETUP;
                end
            end

            // ══════════════════════════════════════════════════════════════
            S_OC_NEXT: begin
                if (oc_group + 1 >= oc_groups_total) begin
                    state <= S_TILE_NEXT;
                end else begin
                    oc_group <= oc_group + 1;
                    if (cfg_op_type == 8'd1) begin
                        // DW Conv: next channel
                        dw_cnt <= 0;
                        dw_read_issued <= 1'b0;
                        dw_init_phase <= 2'd0;
                        state <= S_DW_WGT_LOAD;
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
                if (wgt_reload_done) begin
                    oc_group_done <= 1'b0;
                    state <= S_OC_SETUP;
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
                        wgt_elem_start = oc_group * {2'd0, cfg_kernel_h[3:0]}
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
                        end else begin : dw_wgt_extract_int8
                            reg [1:0] bsel;
                            bsel = dw_cnt[1:0] + dw_wgt_bsel_base;
                            case (bsel)
                                2'd0: dw_wgt_data <= {{8{wgt_rd_data[7]}},  wgt_rd_data[7:0]};
                                2'd1: dw_wgt_data <= {{8{wgt_rd_data[15]}}, wgt_rd_data[15:8]};
                                2'd2: dw_wgt_data <= {{8{wgt_rd_data[23]}}, wgt_rd_data[23:16]};
                                2'd3: dw_wgt_data <= {{8{wgt_rd_data[31]}}, wgt_rd_data[31:24]};
                            endcase
                        end
                        dw_wgt_valid <= 1'b1;
                        dw_cnt <= dw_cnt + 1;

                        if (dw_cnt + 1 >= dw_kernel_size) begin
                            // All weights loaded
                            state <= S_DW_DRAIN;
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
                        ppu_bias       <= $signed({param_buf[3][15:0], param_buf[2],
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
                        // Padding: feed zero
                        dw_in_valid <= 1'b1;
                        dw_in_data  <= {DATA_W{1'b0}};
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
                            if (cfg_tile_h == 17'd0) begin
                                // Non-tiled: full image address
                                elem_off = (ih_s[15:0] * cfg_in_w * cfg_in_c)
                                         + (iw_s[15:0] * cfg_in_c)
                                         + oc_group;
                            end else begin
                                // Tiled: tile-local address (dw_oh/dw_ow are tile-local
                                // output coords; input row = dw_oh*stride + dw_fh,
                                // input col = dw_ow*stride + dw_fw, both within tile_in_h/w)
                                elem_off = ({17'd0, dw_oh} * cfg_stride_h + {17'd0, dw_fh}) * tile_in_w
                                         + ({17'd0, dw_ow} * cfg_stride_w + {17'd0, dw_fw})
                                         ;
                                elem_off = elem_off * cfg_in_c + oc_group;
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
                        dw_in_valid <= 1'b1;
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
                    if (dw_out_valid) begin
                        ppu_acc_in   <= dw_acc_out;
                        ppu_in_valid <= 1'b1;
                        ppu_wait_cnt <= 1;
                    end
                end else begin
                    if (ppu_out_valid) begin
                        // Compute NHWC address for this pixel/channel
                        begin : dw_wb_addr_calc
                            reg [31:0] elem_off;
                            reg [31:0] byte_off;
                            elem_off = (dw_oh * out_tile_w + dw_ow)
                                     * cfg_out_c + oc_group;
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            dw_wb_addr    <= out_base + byte_off[ACT_ADDR_W+1:2];
                            dw_wb_bytesel <= byte_off[1:0];
                        end
                        dw_wb_byte  <= ppu_out_data;
                        dw_wb_phase <= 2'd0;
                        state <= S_DW_WB;
                    end else begin
                        ppu_wait_cnt <= ppu_wait_cnt + 1;
                    end
                end
            end

            S_DW_WB: begin
                // Read-modify-write: place output element at correct NHWC position
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

                    // Advance to next pixel
                    if (dw_ow + 1 >= out_tile_w) begin
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
                out_base <= cfg_act_base + cfg_out_base;
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
                            if (cfg_tile_h == 17'd0) begin
                                // Non-tiled: full image in SRAM, absolute coords
                                elem_off = (ih_s[15:0] * cfg_in_w * cfg_in_c)
                                         + (iw_s[15:0] * cfg_in_c)
                                         + pool_ch;
                            end else begin
                                // Tiled: use tile-local coords (matching Conv2D fix)
                                elem_off = ({8'd0, pool_oh} * pool_sh + {8'd0, pool_fh}) * tile_in_w
                                         + {8'd0, pool_ow} * pool_sw + {8'd0, pool_fw};
                                // Add channel offset for multi-channel pooling
                                elem_off = elem_off * cfg_in_c + pool_ch;
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
                    begin : pool_div_blk
                        reg signed [ACC_W-1:0] rounded;
                        reg signed [ACC_W-1:0] half_count;
                        reg signed [71:0] prod;  // 40-bit * 32-bit = 72-bit
                        half_count = pool_count >> 1;
                        if (pool_acc >= 0)
                            rounded = pool_acc + half_count;
                        else
                            rounded = pool_acc - half_count;
                        // Multiply by reciprocal: q = (rounded * recip) >>> 32
                        // pool_count=1 is special: recip=0x80000000, shift 31
                        prod = rounded * $signed(recip_pool(pool_count[3:0]));
                        if (pool_count == 17'd1)
                            pool_acc <= prod >>> 31;
                        else
                            pool_acc <= prod >>> 32;
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
                add_total_elems <= cfg_out_h * cfg_out_w * cfg_out_c;
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
                // Read-modify-write
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
                elems_per_tile = {17'd0, cfg_tile_h} * {17'd0, cfg_tile_w} * {17'd0, cfg_out_c};
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
