// Open-NPU RTL — Compute Micro-Sequencer
// SPDX-License-Identifier: Apache-2.0
//
// Orchestrates the 64-lane MAC row (Conv2D/FC), DW engine, and vector operators:
//   1. Reads grouped K-major weights from Weight SRAM
//   2. Reads activations from Act SRAM and feeds the MAC row
//   3. Accumulates one partial sum per active row lane
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
    parameter WGT_ADDR_W   = $clog2(`WGT_DEPTH_W),
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
    output wire [15:0]                  wgt_k_start,    // First K index requested for a sliced group
    input  wire                         wgt_reload_done, // Controller loaded next oc_group's weights
    input  wire [31:0]                  cfg_wgt_per_oc,  // Per-oc weight words (0=all weights fit, skip reload)
    input  wire                         cfg_wgt_layout,  // 0=legacy OC-major, 1=K-major (64-lane row)
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
    input  wire [ACT_ADDR_W-1:0]        cfg_out_base,   // CSR output offset inside the OFM bank
    input  wire [ACT_ADDR_W-1:0]        cfg_ofm_base,   // OFM write base, including the DB half
    input  wire [31:0]                  cfg_pool_cfg,   // Pooling config register
    input  wire [31:0]                  cfg_resize_cfg, // Resize config register
    input  wire [31:0]                  cfg_deconv_cfg, // Deconv config: [7:0]=INSERT_H, [15:8]=INSERT_W
    input  wire [31:0]                  cfg_concat_cfg, // Concat config: [15:0]=OFFSET, [31:16]=TOTAL_C
    input  wire                         cfg_2d_load,    // 2D DMA load mode (chain: SRAM row 0 = input row 0, not padding)

    // ─── Weight SRAM Port B (read-only, 256-bit beat) ───
    output reg                          wgt_rd_en,
    output reg  [WGT_ADDR_W-1:0]       wgt_rd_addr,
    input  wire [`SRAM_B_WIDTH-1:0]     wgt_rd_data,

    // ─── Activation SRAM Port B (256-bit IFM read + OFM write) ───
    // act_wr_mask == 0: one 32-bit word in act_wr_data (pool / add / resize).
    // act_wr_mask != 0: byte enables for act_wr_wide, starting at act_wr_addr.
    // Depthwise and row-engine outputs use this wide write port.
    output reg                          act_rd_en,
    output reg  [ACT_ADDR_W-1:0]       act_rd_addr,
    input  wire [`SRAM_B_WIDTH-1:0]     act_rd_data,
    output reg                          act_rd_ofm,    // 1 = RMW read from OFM, 0 = IFM
    output reg                          act_wr_en,
    output reg  [ACT_ADDR_W-1:0]       act_wr_addr,
    output reg  [31:0]                  act_wr_data,
    output reg  [63:0]                  act_wr_mask,
    output reg  [`SRAM_B_WIDTH-1:0]     act_wr_wide,

    // ─── Parameter SRAM Port B (read-only) ───
    output reg                          param_rd_en,
    output reg  [PARAM_ADDR_W-1:0]     param_rd_addr,
    input  wire [31:0]                  param_rd_data,

    // 64-lane row issue, for the performance counter. mac_lanes is the
    // number of products started on a cycle where mac_fire is high.
    output wire                         mac_fire,
    output wire [6:0]                   mac_lanes,

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
    output reg  [ACC_W*`MAC_LANES-1:0] ppu_acc_w,
    output reg  [`MAC_LANES-1:0]       ppu_valid_w,
    output reg  [ACC_W*`MAC_LANES-1:0] ppu_bias_w,
    output reg  [15*`MAC_LANES-1:0]    ppu_mult_w,
    output reg  [6*`MAC_LANES-1:0]     ppu_shift_w,
    output reg  [16*`MAC_LANES-1:0]    ppu_zp_w,
    input  wire [DATA_W*`MAC_LANES-1:0] ppu_out_w,
    input  wire [`MAC_LANES-1:0]       ppu_vout_w
);

    // ─── Derived constants ───
    localparam COL_W = $clog2(ARRAY_SIZE);
    localparam [15:0] ARRAY_SIZE_16 = ARRAY_SIZE;  // 16-bit for comparisons

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
        S_OC_NEXT     = 7'd16,
        S_TILE_NEXT   = 7'd17,
        S_DONE        = 7'd18,
        S_DW_WGT_LOAD = 7'd19,
        S_DW_COMPUTE  = 7'd20,
        S_DW_DRAIN    = 7'd21,
        S_DW_PARAM    = 7'd22,
        S_DW_ACT_STREAM = 7'd24,
        S_DW_PPU_WAIT   = 7'd25,
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
        S_RESIZE_PPU      = 7'd57,
        S_RESIZE_PPU_WAIT = 7'd58,
        S_RESIZE_WB       = 7'd59,
        S_RESIZE_PIX_NEXT = 7'd60,
        S_RESIZE_CH_NEXT  = 7'd61,
        S_TILE_WAIT_DB    = 7'd62, // Wait for DB_EN prefetch before next tile
        S_WAIT_WGT_RELOAD = 7'd63, // Wait for controller to reload next oc_group weights
        S_RESIZE_INTERP1  = 7'd64, // Bilinear interp cycle 1 (4 mults: top/bot)
        S_RESIZE_INTERP2  = 7'd65, // Bilinear interp cycle 2 (2 mults: val64)
        S_ROW_WGT         = 7'd68,
        S_ROW_MAC         = 7'd70,
        S_ROW_PUSH        = 7'd71,
        S_ROW_TAIL        = 7'd72,
        S_ROW_GRP         = 7'd73,
        S_ROW_FILL        = 7'd74;
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

    localparam SRAM_B_W     = `SRAM_B_WIDTH;
    reg [15:0]                   wgt_word_addr;    // current SRAM address

    // Shared activation-read state for pooling, add, and resize.
    reg        act_read_issued;
    reg [4:0]  act_byte_sel;    // byte position within the 256-bit beat

    // Row-engine params are pixel-invariant and cached once per OC group.
    reg [31:0]   param_cache [0:255];  // 64 ch × 4 words
    reg [8:0]    cache_issue, cache_cap;
    reg          param_pend;           // next-OC param fill waits until PPU feed done

    // ─── PPU state ───
    reg [15:0]                  ppu_wait_cnt;   // PPU flush wait counter
    // ─── Param read state ───
    reg [2:0]  param_word_idx;
    reg [31:0] param_buf [0:3];
    reg        param_read_issued;

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
    reg [7:0]  dw_kernel_size;         // kh * kw (cached, up to 16x16=256)
    reg [1:0]  dw_wgt_bsel_base;       // starting byte offset for weight reads
    // Act-read pipe for the OC-major / stream DW engine. SRAM data is
    // visible two cycles after the issue, so iss is the newer request and
    // ret is consumed this cycle. Pads skip the pipe when it is empty.
    reg        dw_iss_vld;
    reg        dw_ret_vld;
    reg [1:0]  dw_iss_bsel;
    reg [1:0]  dw_ret_bsel;
    reg        dw_iss_last;
    reg        dw_ret_last;
    reg        dw_iss_clr;
    reg        dw_ret_clr;
    reg        dw_kdone;              // last tap of this pixel has been issued
    reg [15:0] dw_grp_base;            // stream mode: first channel of resident 16-ch group
    reg [15:0] pool_grp_base;          // pool stream mode: ditto

    // ─── Tile geometry ───
    reg [15:0] sp_oh, sp_ow;                  // Current output pixel coordinates (tile-local)
    reg [15:0] tile_oh_origin, tile_ow_origin; // Global origin of current tile
    reg [15:0] tile_in_w;                      // Input tile width (including halo)
    reg [15:0] kw_eff;                         // Effective kernel width: (kw-1)*dw+1

    reg [15:0] px_remaining;                     // pixels left in this oc_group/tile

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
                         && (dws_total_words > `IFM_DEPTH);

    wire [31:0] k_depth_cfg = {8'd0, cfg_kernel_h} * {8'd0, cfg_kernel_w}
                            * {16'd0, cfg_in_c};
    localparam ROW_N = `MAC_LANES;
    // ─── 64-lane row engine storage ───
    //
    // Weights arrive K-major (see tools/model_packer.py), so one K index's
    // lane weights are contiguous and stream straight out of the 512-bit
    // weight port. wrow is a ring of WROW_K K slots, not the whole layer.
    // Activations are gathered a chunk of spatial taps at a time and racc
    // accumulates across chunks, so nothing here scales with k_depth.
    localparam WROW_K     = 64;    // K slots in the weight ring
    localparam ABUF_WORDS = 2048;  // im2col pool, split into 2*row_pix banks
    localparam ROW_PSH_MAX = 3;    // up to 8 output pixels packed into the lanes
    reg signed [DATA_W-1:0] wrow [0:WROW_K*ROW_N-1];
    reg signed [DATA_W-1:0] abuf [0:ABUF_WORDS-1];
    reg signed [DATA_W-1:0] avec [0:4*ROW_N-1];  // depthwise tap ring
    reg signed [ACC_W-1:0]  racc [0:ROW_N-1];
    reg [6:0]  row_nch;      // G: lanes per pixel = channels in this OC group
    reg [4:0]  row_pix;      // P: output pixels packed across the 64 lanes
    reg [2:0]  row_pix_sh;   // log2(P)
    reg [4:0]  row_pv;       // pixels actually valid in the current group
    reg [12:0] row_bank_sz;  // ABUF_WORDS / (2*P)
    reg [3:0]  row_bank_sh;
    reg        row_dw;
    reg [15:0] row_klen;     // K indices per pixel group
    reg [15:0] row_ks0;      // first K index resident in WGT SRAM
    reg [15:0] row_ksend;    // one past the resident K slice
    reg        row_ks_reload;
    reg [15:0] row_ki;       // K index the MAC is consuming
    reg [15:0] row_kc0;      // first K index of the chunk held in abuf
    reg [15:0] row_tpc;      // spatial taps per chunk
    reg [6:0]  row_beat_k;   // upper bound on K indices spanned by one beat
    reg        row_wfit;     // whole OC group's weights fit the ring
    reg [15:0] row_ntap;
    reg        mac_bank;
    reg        row_ppu;
    reg        row_drain;
    reg        row_last;
    reg [7:0]  ob_n, ob_i;
    // 64 lanes of INT16 is 128 bytes, twice what the 512-bit wide port takes.
    reg [1023:0] obuf;
    reg [31:0] pk;
    reg [1:0]  pk_pos;
    reg [ACT_ADDR_W-1:0] pk_addr;
    reg [15:0] ppu_oh, ppu_ow;
    reg [8:0]  ppu_nlane;   // lanes latched at push; row_pv moves on before capture
    reg [4:0]  ppu_npix;
    reg [6:0]  ppu_nch;
    reg        row_scatter;
    reg [4:0]  scatter_p;

    // ─── weight streamer ───
    // Issues one 64-byte beat per cycle and samples it two cycles later, so
    // the MAC sees a new K index every cycle once the ring has filled.
    reg        wst_on;
    reg [31:0] wst_iss_byte;   // byte offset of the next beat to request
    reg [31:0] wst_nbytes;     // weight bytes in this OC group
    reg [15:0] wst_sk;         // K indices written into the ring so far
    reg [6:0]  wst_sc;         // lane the sampler is at inside wst_sk
    reg [5:0]  wst_skip;       // byte offset of the first beat
    reg [15:0] wst_iss_n;      // beats requested
    reg [15:0] wst_smp_n;      // beats sampled
    wire [1:0] wst_flight = wst_iss_n[1:0] - wst_smp_n[1:0];
    reg [1:0]  wst_f_vld;
    reg [5:0]  wst_f_skip [0:1];
    // DW weights are tap-major over all channels, so an OC group's taps are
    // strided rather than contiguous; they are small enough (kh*kw <= WROW_K)
    // to load once per group with a plain issue/wait/sample walk.
    reg [15:0] dwl_tap;
    reg [1:0]  dwl_ph;
    reg        dwl_on;

    // ─── activation gather ───
    // Same two-stage pipeline: requests go out back to back and land in abuf
    // (conv) or avec (depthwise) two cycles later.
    reg        ga_on;          // gather pipeline enabled for this OC group
    reg [4:0]  ga_p;           // pixel slot being requested
    reg [15:0] ga_t;           // tap within the chunk being requested
    reg [4:0]  gs_p;           // pixel slot being sampled
    reg [15:0] gs_t;
    reg [15:0] gs_e;           // element within the tap
    reg [15:0] gs_left;        // bytes of the tap not yet written
    reg [1:0]  ga_f_vld;       // descriptor pipeline: issue -> +1 -> sample
    reg [1:0]  ga_f_pad;
    reg [5:0]  ga_f_skip [0:1];
    reg [6:0]  ga_f_base [0:1]; // first lane this beat fills
    reg [6:0]  ga_f_nelt [0:1]; // elements in the whole read, not this beat
    reg [1:0]  ga_f_tend;       // this beat completes a depthwise tap
    reg [6:0]  ga_base_r;       // held across the beats of one read
    reg [6:0]  ga_nelt_r;
    reg        ga_tend_r;
    reg [4:0]  ga_step_r;       // pixels this read covers
    reg        ga_bank;        // abuf chunk buffer being filled
    // The P packed pixels are consecutive in raster order, so their outputs
    // are contiguous in NHWC and one origin plus row_pv describes them all.
    reg [15:0] grp_oh, grp_ow;
    reg [15:0] gp_oh [0:7];
    reg [15:0] gp_ow [0:7];
    reg [15:0] ga_fh, ga_fw;   // issuer's spatial tap
    reg [15:0] ga_fh0, ga_fw0; // spatial tap the chunk starts at
    reg [15:0] ga_tap0;        // first spatial tap of the chunk being gathered
    reg [15:0] ga_tpc;         // taps in the chunk being gathered
    // Set when one spatial tap is longer than an abuf bank. A chunk is then
    // a channel slice of one tap, not a run of whole taps.
    reg        row_slice;
    reg [15:0] ga_k0;          // first K index of the slice being gathered
    reg [15:0] ga_ch0;         // channel offset of that slice inside its tap
    reg [15:0] ga_nch;         // channels in the slice
    reg [31:0] ga_addr;        // next beat word address inside the current tap
    reg [15:0] ga_left;        // bytes of the current tap not yet requested
    reg [15:0] mac_kend;       // K index one past the chunk the MAC is on
    reg        ga_busy;        // issuer still inside the chunk
    reg        mac1d_fire;
    reg [6:0]  mac1d_lanes;
    reg signed [DATA_W-1:0] mac1d_act /* verilator public */; // first active lane, test trace
    assign mac_fire  = mac1d_fire;
    assign mac_lanes = mac1d_lanes;

    // The dedicated depthwise bank processes ARRAY_SIZE channels per group.
    wire [4:0] grp_oc = ARRAY_SIZE[4:0];
    // Depthwise keeps one kernel tap per channel, so the row limit is kh*kw,
    // not kh*kw*in_c. K-major conv and depthwise use the 64-lane row.
    // One spatial tap of the im2col vector has to fit an abuf bank, which in
    // the worst case (P=1) is ABUF_WORDS/2.
    wire [31:0] row_k_need = (cfg_op_type == 8'd1)
                           ? ({8'd0, cfg_kernel_h} * {8'd0, cfg_kernel_w})
                           : k_depth_cfg;
    // Pick the widest power-of-two output-channel group whose complete K
    // vector fits the 80KB weight SRAM. Narrow groups use the otherwise idle
    // row lanes for more output pixels.
    wire [31:0] row_one_oc_bytes = k_depth_cfg << (cfg_int16 ? 1 : 0);
    wire [6:0]  row_dense_grp =
        ((row_one_oc_bytes << 6) <= `WEIGHT_BUF_SIZE) ? 7'd64 :
        ((row_one_oc_bytes << 5) <= `WEIGHT_BUF_SIZE) ? 7'd32 :
        ((row_one_oc_bytes << 4) <= `WEIGHT_BUF_SIZE) ? 7'd16 :
        ((row_one_oc_bytes << 3) <= `WEIGHT_BUF_SIZE) ? 7'd8  :
        ((row_one_oc_bytes << 2) <= `WEIGHT_BUF_SIZE) ? 7'd4  :
        ((row_one_oc_bytes << 1) <= `WEIGHT_BUF_SIZE) ? 7'd2  :
        ( row_one_oc_bytes       <= `WEIGHT_BUF_SIZE) ? 7'd1  : 7'd0;
    wire        row_kslice = cfg_wgt_per_oc[31] && cfg_wgt_layout
                           && (cfg_op_type == 8'd0 || cfg_op_type == 8'd2);
    wire [6:0]  row_group_ch = (cfg_op_type == 8'd1 || row_kslice)
                              ? ROW_N[6:0] : row_dense_grp;
    wire [15:0] row_ks_cap = cfg_int16
                           ? (`WGT_DEPTH_W / 32)
                           : (`WGT_DEPTH_W / 16);
    assign wgt_k_start = row_ks0;
    // FC is the same dot product as 1x1 conv. Its input vector may be longer
    // than one abuf bank; the row then slices that vector and accumulates.
    wire        use_row = cfg_wgt_layout && !dw_stream
                       && (cfg_op_type == 8'd0 || cfg_op_type == 8'd1
                           || cfg_op_type == 8'd2)
                       && (row_k_need != 32'd0)
                       && (cfg_op_type == 8'd1 || row_dense_grp != 7'd0)
                       && (cfg_op_type == 8'd1 || cfg_op_type == 8'd2
                           || {16'd0, cfg_in_c} <= (ABUF_WORDS / 2));
    wire [6:0]  param_nch = row_nch;
    wire [8:0]  param_tgt = {param_nch, 2'b00}; // nch * 4 words
    wire        param_feed_busy = row_ppu || row_drain;
    wire        param_ready = !param_pend
                            && (cache_issue >= param_tgt)
                            && (cache_cap + 1 >= param_tgt);
    wire        param_bg_ok = use_row
                            && !param_feed_busy && !param_pend
                            && (state == S_ROW_WGT || state == S_ROW_GRP
                                || state == S_ROW_FILL
                                || state == S_ROW_MAC || state == S_ROW_PUSH);
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
                           && (dws_total_words > `IFM_DEPTH);
    wire        slice_stream = dw_stream | pool_stream;
    wire        resize_mode   = cfg_resize_cfg[0];   // 0=nearest, 1=bilinear
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
    // Combined resize reciprocals, precomputed once per layer.
    reg signed [55:0]      recip_scale_h;      // cfg_in_h * Q40(1/out_h)
    reg signed [55:0]      recip_scale_w;      // cfg_in_w * Q40(1/out_w)
    reg signed [55:0]      recip_scale_h_m1;   // ((cfg_in_h-1)<<8) * Q40(1/(out_h-1))
    reg signed [55:0]      recip_scale_w_m1;   // ((cfg_in_w-1)<<8) * Q40(1/(out_w-1))
    // Tile-origin hoist (avoid 4x redundant division per pixel)
    reg [15:0]             rsz_tile_ih_origin;  // first input row used by this output tile
    reg [15:0]             rsz_tile_iw_origin;  // first input column used by this output tile
    // Bilinear interp pipeline register (for 2-cycle interp split)
    reg signed [63:0]      rsz_top_r, rsz_bot_r;

    integer i;

    // ════════════════════════════════════════════════════════════════════
    // Main FSM
    // ════════════════════════════════════════════════════════════════════

    // Point the gather at a fresh chunk: taps [tap0, tap0+row_tpc) of every
    // packed pixel, landing in abuf bank `bank`.
    task row_start_chunk;
        input [15:0] tap0;
        input        bank;
        reg [15:0] tv;
        begin
            tv = ((tap0 + row_tpc) > row_ntap) ? (row_ntap - tap0) : row_tpc;
            ga_tap0 <= tap0;
            ga_tpc  <= tv;
            ga_bank <= bank;
            ga_p    <= 5'd0;
            ga_t    <= 16'd0;
            gs_p    <= 5'd0;
            gs_t    <= 16'd0;
            gs_e    <= 16'd0;
            gs_left <= 16'd0;
            ga_left <= 16'd0;
            ga_fh   <= tap0 / {8'd0, cfg_kernel_w};
            ga_fw   <= tap0 % {8'd0, cfg_kernel_w};
            ga_fh0  <= tap0 / {8'd0, cfg_kernel_w};
            ga_fw0  <= tap0 % {8'd0, cfg_kernel_w};
            ga_busy <= 1'b1;
            ga_on   <= 1'b1;
        end
    endtask

    // One channel slice of a single tap. Used when in_c is longer than the
    // abuf bank, so a whole tap cannot be gathered at once. k0 is an index
    // in the im2col vector.
    task row_start_slice;
        input [15:0] k0;
        input        bank;
        reg [15:0] tap, ch, room, len, lim;
        begin
            tap  = (cfg_in_c == 16'd0) ? 16'd0 : (k0 / cfg_in_c);
            ch   = (cfg_in_c == 16'd0) ? k0 : (k0 % cfg_in_c);
            room = cfg_in_c - ch;
            len  = ({3'd0, row_bank_sz} < room) ? {3'd0, row_bank_sz} : room;
            lim  = row_kslice ? row_ksend : row_klen;
            if (k0 + len > lim)
                len = lim - k0;
            ga_k0   <= k0;
            ga_ch0  <= ch;
            ga_nch  <= len;
            ga_tap0 <= tap;
            ga_tpc  <= 16'd1;
            ga_bank <= bank;
            ga_p    <= 5'd0;
            ga_t    <= 16'd0;
            gs_p    <= 5'd0;
            gs_t    <= 16'd0;
            gs_e    <= 16'd0;
            gs_left <= 16'd0;
            ga_left <= 16'd0;
            ga_fh   <= tap / {8'd0, cfg_kernel_w};
            ga_fw   <= tap % {8'd0, cfg_kernel_w};
            ga_fh0  <= tap / {8'd0, cfg_kernel_w};
            ga_fw0  <= tap % {8'd0, cfg_kernel_w};
            ga_busy <= 1'b1;
            ga_on   <= 1'b1;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            tile_done_r <= 1'b0;
            oc_group_done <= 1'b0;
            tile_wait_delay <= 1'b0;
            // All outputs idle
            wgt_rd_en   <= 1'b0;
            wgt_rd_addr <= 0;
            act_rd_en   <= 1'b0;
            act_rd_addr <= 0;
            act_rd_ofm  <= 1'b0;
            act_wr_en   <= 1'b0;
            act_wr_addr <= 0;
            act_wr_data <= 32'd0;
            act_wr_mask <= 64'd0;
            act_wr_wide <= {`SRAM_B_WIDTH{1'b0}};
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
            wgt_word_addr <= 0;
            act_read_issued <= 0; act_byte_sel <= 0;
            ppu_wait_cnt <= 0;
            param_word_idx <= 0; param_read_issued <= 0;
            wgt_base <= 0; act_base <= 0; param_base <= 0; out_base <= 0;
            dw_ch_idx <= 0; dw_cnt <= 0; dw_read_issued <= 0; dw_init_phase <= 0;
            dw_oh <= 0; dw_ow <= 0; dw_fh <= 0; dw_fw <= 0;
            dw_kernel_size <= 0;
            dw_wgt_bsel_base <= 0;
            dw_iss_vld <= 1'b0; dw_ret_vld <= 1'b0;
            dw_iss_bsel <= 2'd0; dw_ret_bsel <= 2'd0;
            dw_iss_last <= 1'b0; dw_ret_last <= 1'b0;
            dw_iss_clr <= 1'b0; dw_ret_clr <= 1'b0;
            dw_kdone <= 1'b0;
            dw_grp_base <= 0;
            pool_grp_base <= 0;
            mac1d_fire <= 1'b0;
            row_ppu <= 1'b0;
            row_drain <= 1'b0;
            ga_on <= 1'b0;
            ga_busy <= 1'b0;
            row_slice <= 1'b0;
            ga_f_vld <= 2'd0;
            wst_on <= 1'b0;
            wst_f_vld <= 2'd0;
            dwl_on <= 1'b0;
            ppu_npix <= 5'd1;
            ppu_nch <= 7'd1;
            row_scatter <= 1'b0;
            scatter_p <= 5'd0;
            row_ks0 <= 16'd0;
            row_ksend <= 16'd0;
            row_ks_reload <= 1'b0;
            param_pend <= 1'b0;
            sp_oh <= 0; sp_ow <= 0; tile_oh_origin <= 0; tile_ow_origin <= 0;
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
            for (i = 0; i < 4; i = i + 1)
                param_buf[i] <= 0;
        end else begin
            // ─── Default pulse deassertion ───
            done         <= 1'b0;
            tile_done_r  <= 1'b0;
            wgt_rd_en   <= 1'b0;
            act_rd_en   <= 1'b0;
            act_rd_ofm  <= 1'b0;
            act_wr_en   <= 1'b0;
            act_wr_mask <= 64'd0;
            mac1d_fire  <= 1'b0;
            param_rd_en <= 1'b0;
            ppu_in_valid <= 1'b0;
            ppu_valid_w  <= {`MAC_LANES{1'b0}};
            dw_wgt_valid <= 1'b0;
            dw_in_valid  <= 1'b0;
            dw_acc_clear <= 1'b0;
            dw_wgt_valid_w <= {ARRAY_SIZE{1'b0}};
            dw_in_valid_w  <= {ARRAY_SIZE{1'b0}};

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
                    param_cache[cache_cap[7:0]] <= param_rd_data;
                    cache_cap <= cache_cap + 1;
                end
            end

            // ══════════════════════════════════════════════════════════════
            // Activation gather for the 64-lane row, two stages deep.
            //
            // The issuer puts one read per cycle on the activation port and
            // pushes a descriptor down ga_f_*; two cycles later the sampler
            // pops that descriptor and scatters the beat. Padding taps travel
            // the same pipeline with pad=1 and no read, which keeps the two
            // walkers in lockstep without recomputing the padding test.
            //
            // Conv destination is abuf[(bank*P + p)*bank_sz + t*in_c + e], one
            // chunk of taps for all P packed pixels. Depthwise writes avec as
            // a four-deep tap ring, which is what lets it retire a tap per
            // cycle instead of stalling on every read.
            // ══════════════════════════════════════════════════════════════
            // ══════════════════════════════════════════════════════════════
            // Weight streamer. K-major weights make one K index's lanes
            // contiguous, so the 512-bit port feeds the ring directly and the
            // old full-layer scatter into registers is gone. Same two-stage
            // pipeline as the gather: request every cycle, sample two later.
            // ══════════════════════════════════════════════════════════════
            wst_f_vld[1]  <= wst_f_vld[0];
            wst_f_skip[1] <= wst_f_skip[0];
            wst_f_vld[0]  <= 1'b0;

            if (wst_on) begin : row_wgt_issue
                reg [15:0] ahead;
                ahead = wst_sk - row_ki;
                // Never let a request overwrite a K slot the MAC still needs.
                if (wst_iss_byte < wst_nbytes
                        && (ahead + {9'd0, row_beat_k} * {14'd0, wst_flight + 2'd1})
                            <= WROW_K[15:0]) begin
                    wgt_rd_en     <= 1'b1;
                    wgt_rd_addr   <= wgt_base + wst_iss_byte[WGT_ADDR_W+1:2];
                    wst_f_vld[0]  <= 1'b1;
                    wst_f_skip[0] <= (wst_iss_byte == 32'd0) ? wst_skip : 6'd0;
                    wst_iss_byte  <= (wst_iss_byte == 32'd0)
                                   ? (32'd64 - {26'd0, wst_skip})
                                   : (wst_iss_byte + 32'd64);
                    wst_iss_n     <= wst_iss_n + 16'd1;
                end else
                    wgt_rd_en <= 1'b0;
            end

            if (wst_f_vld[1]) begin : row_wgt_sample
                integer bi;
                reg [15:0] k_i;
                reg [6:0]  c_i;
                reg [31:0] slot;
                k_i = wst_sk;
                c_i = wst_sc;
                for (bi = 0; bi < 64; bi = bi + 1) begin
                    if (bi >= {26'd0, wst_f_skip[1]} && k_i < row_klen) begin
                        slot = ({16'd0, k_i[5:0]} * ROW_N) + {25'd0, c_i};
                        if (cfg_int16) begin
                            if (bi[0] == 1'b0) begin
                                wrow[slot[12:0]] <= $signed(wgt_rd_data[bi*8 +: 16]);
                                if (c_i + 7'd1 >= row_nch) begin
                                    c_i = 7'd0;
                                    k_i = k_i + 16'd1;
                                end else
                                    c_i = c_i + 7'd1;
                            end
                        end else begin
                            wrow[slot[12:0]] <= {{8{wgt_rd_data[bi*8+7]}}, wgt_rd_data[bi*8 +: 8]};
                            if (c_i + 7'd1 >= row_nch) begin
                                c_i = 7'd0;
                                k_i = k_i + 16'd1;
                            end else
                                c_i = c_i + 7'd1;
                        end
                    end
                end
                wst_sk    <= k_i;
                wst_sc    <= c_i;
                wst_smp_n <= wst_smp_n + 16'd1;
            end

            // Depthwise weight load: one beat per spatial tap, G lanes each.
            if (dwl_on) begin : row_dw_wgt
                integer bi;
                reg [31:0] byte_off;
                reg [5:0]  skip;
                byte_off = ({16'd0, dwl_tap} * {16'd0, cfg_in_c}
                          + {16'd0, oc_group} * 32'd64);
                if (cfg_int16) byte_off = byte_off << 1;
                skip = {4'd0, byte_off[1:0]};
                if (dwl_ph == 2'd0) begin
                    wgt_rd_en   <= 1'b1;
                    wgt_rd_addr <= wgt_base + byte_off[WGT_ADDR_W+1:2];
                    wst_skip    <= skip;
                    wst_sc      <= 7'd0;
                    dwl_ph      <= 2'd1;
                end else if (dwl_ph == 2'd1) begin
                    wgt_rd_en <= 1'b0;
                    dwl_ph    <= 2'd2;
                end else begin : dw_wgt_take
                    reg [6:0] c_i;
                    reg [31:0] slot;
                    c_i = wst_sc;
                    for (bi = 0; bi < 64; bi = bi + 1) begin
                        if (bi >= {26'd0, wst_skip} && c_i < row_nch) begin
                            slot = ({16'd0, dwl_tap} * ROW_N) + {25'd0, c_i};
                            if (cfg_int16) begin
                                if (bi[0] == 1'b0) begin
                                    wrow[slot[12:0]] <= $signed(wgt_rd_data[bi*8 +: 16]);
                                    c_i = c_i + 7'd1;
                                end
                            end else begin
                                wrow[slot[12:0]] <= {{8{wgt_rd_data[bi*8+7]}}, wgt_rd_data[bi*8 +: 8]};
                                c_i = c_i + 7'd1;
                            end
                        end
                    end
                    wst_sc <= c_i;
                    if (c_i < row_nch) begin
                        // INT16 group wider than one beat: continue this tap.
                        wgt_rd_en   <= 1'b1;
                        wgt_rd_addr <= wgt_rd_addr + 16;
                        wst_skip    <= 6'd0;
                        dwl_ph      <= 2'd1;
                    end else if (dwl_tap + 16'd1 >= row_klen) begin
                        dwl_on  <= 1'b0;
                        wst_sk  <= row_klen;
                    end else begin
                        dwl_tap <= dwl_tap + 16'd1;
                        dwl_ph  <= 2'd0;
                    end
                end
            end

            ga_f_vld[1]  <= ga_f_vld[0];
            ga_f_pad[1]  <= ga_f_pad[0];
            ga_f_skip[1] <= ga_f_skip[0];
            ga_f_base[1] <= ga_f_base[0];
            ga_f_nelt[1] <= ga_f_nelt[0];
            ga_f_tend[1] <= ga_f_tend[0];
            ga_f_vld[0]  <= 1'b0;
            ga_f_base[0] <= 7'd0;
            ga_f_nelt[0] <= 7'd0;
            ga_f_tend[0] <= 1'b0;

            if (ga_on) begin : row_gather_issue
                reg signed [31:0] ih, iw, remain, run;
                reg [15:0] poh, pow;
                reg        is_pad, last_beat, tap_done;
                reg [15:0] nelt, nbyte;
                reg [31:0] elem, byte_off;
                reg [5:0]  skip;
                reg [6:0]  base;
                reg [4:0]  step;
                reg        go, tend;
                nelt = row_dw ? {9'd0, row_nch}
                     : (row_slice ? ga_nch : cfg_in_c);
                base = 7'd0;
                tend = 1'b1;
                tap_done = 1'b1;
                step = 5'd1;
                // DW walks taps continuously and may run three ahead of the
                // MAC; conv stops at the end of the chunk it is filling.
                go = row_dw ? ((ga_t - row_ki) < 16'd3) : ga_busy;
                if (go) begin
                    last_beat = 1'b0;
                    if (ga_left != 16'd0) begin
                        // Continuation beat of a read wider than 64 bytes.
                        act_rd_en   <= 1'b1;
                        act_rd_ofm  <= 1'b0;
                        act_rd_addr <= ga_addr[ACT_ADDR_W-1:0];
                        ga_addr     <= ga_addr + 32'd16;
                        ga_left     <= (ga_left > 16'd64) ? (ga_left - 16'd64) : 16'd0;
                        ga_f_vld[0]  <= 1'b1;
                        ga_f_pad[0]  <= 1'b0;
                        ga_f_skip[0] <= 6'd0;
                        ga_f_base[0] <= ga_base_r;
                        ga_f_nelt[0] <= ga_nelt_r;
                        last_beat    = (ga_left <= 16'd64);
                        ga_f_tend[0] <= last_beat && ga_tend_r;
                        tap_done     = ga_tend_r;
                        step         = ga_step_r;
                    end else begin
                        // Depthwise pixels of one group sit on one row, so
                        // pixel ga_p is (sp_oh, sp_ow + ga_p). Stride 1 makes
                        // those pixels contiguous in NHWC.
                        if (row_dw) begin
                            poh = sp_oh;
                            pow = sp_ow + {11'd0, ga_p};
                        end else begin
                            poh = gp_oh[ga_p[2:0]];
                            pow = gp_ow[ga_p[2:0]];
                        end
                        ih = $signed({1'b0, tile_oh_origin + poh})
                           * $signed({1'b0, cfg_stride_h})
                           + $signed({16'd0, ga_fh}) - $signed({16'd0, cfg_pad_top});
                        iw = $signed({1'b0, tile_ow_origin + pow})
                           * $signed({1'b0, cfg_stride_w})
                           + $signed({16'd0, ga_fw}) - $signed({16'd0, cfg_pad_left});
                        is_pad = (ih < 0) || (ih >= $signed({1'b0, cfg_in_h}))
                              || (iw < 0) || (iw >= $signed({1'b0, cfg_in_w}));
                        // Depthwise pixels of a group are contiguous in NHWC
                        // when stride is 1, so an in-bounds run is one read and
                        // a padding run is one zero-point fill. A group that
                        // crosses the image edge is those two runs, not one
                        // read per pixel.
                        if (row_dw) begin
                            remain = $signed({27'd0, row_pv}) - $signed({27'd0, ga_p});
                            if (remain < 32'sd1)
                                remain = 32'sd1;
                            if ((ih < 0) || (ih >= $signed({1'b0, cfg_in_h}))
                                    || (iw >= $signed({1'b0, cfg_in_w}))) begin
                                run = remain;
                                is_pad = 1'b1;
                            end else if (iw < 0) begin
                                run = -iw;
                                if (run > remain)
                                    run = remain;
                                is_pad = 1'b1;
                            end else begin
                                run = $signed({1'b0, cfg_in_w}) - iw;
                                if (run > remain)
                                    run = remain;
                                is_pad = 1'b0;
                            end
                            if (run < 32'sd1)
                                run = 32'sd1;
                            step = run[4:0];
                            nelt = {11'd0, step} * {9'd0, row_nch};
                            base = {2'd0, ga_p} * {2'd0, row_nch};
                            tend = ({11'd0, ga_p} + {11'd0, step} >= {11'd0, row_pv});
                        end
                        ga_base_r <= base;
                        ga_nelt_r <= nelt[6:0];
                        ga_tend_r <= tend;
                        ga_step_r <= step;
                        ga_f_vld[0]  <= 1'b1;
                        ga_f_pad[0]  <= is_pad;
                        ga_f_base[0] <= base;
                        ga_f_nelt[0] <= nelt[6:0];
                        if (is_pad) begin
                            act_rd_en    <= 1'b0;
                            ga_f_skip[0] <= 6'd0;
                            ga_f_tend[0] <= tend;
                            last_beat    = 1'b1;
                        end else begin
                            if (cfg_tile_h == 16'd0)
                                elem = (ih[15:0] * cfg_in_w + iw[15:0]) * cfg_in_c;
                            else
                                elem = (({16'd0, poh} * {8'd0, cfg_stride_h} + {16'd0, ga_fh})
                                      * tile_in_w
                                      + {16'd0, pow} * {8'd0, cfg_stride_w} + {16'd0, ga_fw})
                                      * cfg_in_c;
                            if (row_dw)
                                elem = elem + ({16'd0, oc_group} * 32'd64);
                            else if (row_slice)
                                elem = elem + {16'd0, ga_ch0};
                            byte_off = cfg_int16 ? (elem << 1) : elem;
                            skip  = {4'd0, byte_off[1:0]};
                            nbyte = cfg_int16 ? (nelt << 1) : nelt;
                            act_rd_en    <= 1'b1;
                            act_rd_ofm   <= 1'b0;
                            act_rd_addr  <= act_base + byte_off[ACT_ADDR_W+1:2];
                            ga_addr      <= {16'd0, act_base + byte_off[ACT_ADDR_W+1:2]} + 32'd16;
                            ga_f_skip[0] <= skip;
                            // The first beat delivers 64-skip bytes.
                            if ({10'd0, skip} + nbyte > 16'd64) begin
                                ga_left <= nbyte - (16'd64 - {10'd0, skip});
                                ga_f_tend[0] <= 1'b0;
                            end else begin
                                ga_left      <= 16'd0;
                                last_beat    = 1'b1;
                                ga_f_tend[0] <= tend;
                            end
                        end
                        tap_done = tend;
                    end
                    if (last_beat) begin
                        if (row_dw && !tap_done)
                            ga_p <= ga_p + step;
                        else begin
                            if (row_dw)
                                ga_p <= 5'd0;
                            if (ga_fw + 16'd1 >= {8'd0, cfg_kernel_w}) begin
                                ga_fw <= 16'd0;
                                if (ga_fh + 16'd1 >= {8'd0, cfg_kernel_h})
                                    ga_fh <= 16'd0;
                                else
                                    ga_fh <= ga_fh + 16'd1;
                            end else
                                ga_fw <= ga_fw + 16'd1;
                            if (row_dw)
                                ga_t <= ga_t + 16'd1;
                            else if (ga_t + 16'd1 >= ga_tpc) begin
                                ga_t <= 16'd0;
                                // Next pixel restarts at the chunk's first tap.
                                ga_fh <= ga_fh0;
                                ga_fw <= ga_fw0;
                                if (ga_p + 5'd1 >= row_pv)
                                    ga_busy <= 1'b0;
                                else
                                    ga_p <= ga_p + 5'd1;
                            end else
                                ga_t <= ga_t + 16'd1;
                        end
                    end
                end else
                    act_rd_en <= 1'b0;
            end

            if (ga_f_vld[1]) begin : row_gather_sample
                integer bi;
                reg [31:0] left;
                reg [15:0] el, nelt;
                reg [31:0] dbase, dst;
                reg signed [DATA_W-1:0] zpv;
                reg        done;
                nelt = row_dw ? {9'd0, ga_f_nelt[1]}
                     : (row_slice ? ga_nch : cfg_in_c);
                zpv  = cfg_int16 ? cfg_in_zp : {{8{cfg_in_zp[7]}}, cfg_in_zp[7:0]};
                // DW lands in a four-deep tap ring; conv lands in bank
                // (ga_bank, gs_p) at the tap's im2col offset. A packed group
                // writes its pixels contiguously from ga_f_base.
                dbase = row_dw
                      ? ({24'd0, gs_t[1:0], 6'd0} + {25'd0, ga_f_base[1]})
                      : ((({31'd0, ga_bank} << row_pix_sh) + {27'd0, gs_p}) << row_bank_sh)
                        + {16'd0, gs_t} * {16'd0, cfg_in_c};
                done = 1'b0;
                if (ga_f_pad[1]) begin
                    for (bi = 0; bi < 1024; bi = bi + 1)
                        if (bi < nelt) begin
                            dst = dbase + bi[31:0];
                            if (row_dw) avec[dst[7:0]]  <= zpv;
                            else        abuf[dst[10:0]] <= zpv;
                        end
                    done = 1'b1;
                end else begin
                    left = (gs_left != 16'd0) ? {16'd0, gs_left}
                         : (cfg_int16 ? {15'd0, nelt, 1'b0} : {16'd0, nelt});
                    el   = gs_e;
                    for (bi = 0; bi < 64; bi = bi + 1) begin
                        if (bi >= {26'd0, ga_f_skip[1]} && left != 0) begin
                            dst = dbase + {16'd0, el};
                            if (cfg_int16) begin
                                if (bi[0] == 1'b0 && left >= 32'd2) begin
                                    if (row_dw) avec[dst[7:0]]  <= $signed(act_rd_data[bi*8 +: 16]);
                                    else        abuf[dst[10:0]] <= $signed(act_rd_data[bi*8 +: 16]);
                                    el = el + 16'd1;
                                    left = left - 32'd2;
                                end
                            end else begin
                                if (row_dw) avec[dst[7:0]]  <= {{8{act_rd_data[bi*8+7]}}, act_rd_data[bi*8 +: 8]};
                                else        abuf[dst[10:0]] <= {{8{act_rd_data[bi*8+7]}}, act_rd_data[bi*8 +: 8]};
                                el = el + 16'd1;
                                left = left - 32'd1;
                            end
                        end
                    end
                    gs_e    <= el;
                    gs_left <= left[15:0];
                    if (left == 0) done = 1'b1;
                end
                if (done) begin
                    gs_e    <= 16'd0;
                    gs_left <= 16'd0;
                    if (row_dw) begin
                        if (ga_f_tend[1])
                            gs_t <= gs_t + 16'd1;
                    end else if (gs_t + 16'd1 >= ga_tpc) begin
                        gs_t <= 16'd0;
                        gs_p <= gs_p + 5'd1;
                    end else
                        gs_t <= gs_t + 16'd1;
                end
            end



            (* parallel_case, full_case *)
            case (state)

            // ══════════════════════════════════════════════════════════════
            S_IDLE: begin
                if (start) begin
                    if (use_row)
                        oc_groups_total <= (cfg_out_c + {9'd0, row_group_ch} - 1)
                                         / {9'd0, row_group_ch};
                    else if (cfg_op_type == 8'd1)
                        oc_groups_total <= (cfg_out_c + ARRAY_SIZE_16 - 1) / ARRAY_SIZE_16;
                    else if (cfg_op_type == 8'd3 || cfg_op_type == 8'd5)
                        oc_groups_total <= cfg_out_c;
                    else
                        oc_groups_total <= (cfg_out_c + ARRAY_SIZE_16 - 1) / ARRAY_SIZE_16;

                    k_depth <= {8'd0, cfg_kernel_h} * {8'd0, cfg_kernel_w} * cfg_in_c;
                    kw_eff  <= cfg_kernel_w;

                    if (cfg_tile_h == 17'd0) begin
                        out_tile_h <= cfg_out_h;
                        out_tile_w <= cfg_out_w;
                    end else begin
                        out_tile_h <= cfg_tile_h;
                        out_tile_w <= cfg_tile_w;
                    end

                    tile_y <= 0;
                    tile_x <= 0;
                    // A layer with no output channels has nothing to compute;
                    // the OC loop would otherwise still run one group and
                    // stall waiting for a drain that never arrives.
                    state  <= (cfg_out_c == 16'd0) ? S_DONE : S_TILE_SETUP;
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
                            reg [31:0] out_start_h, out_start_w;
                            out_start_h = tile_y * cfg_tile_h;
                            out_start_w = tile_x * cfg_tile_w;
                            if (resize_mode) begin
                                rsz_tile_ih_origin <= (cfg_out_h > 1)
                                    ? (out_start_h * (cfg_in_h - 1)) / (cfg_out_h - 1)
                                    : 0;
                                rsz_tile_iw_origin <= (cfg_out_w > 1)
                                    ? (out_start_w * (cfg_in_w - 1)) / (cfg_out_w - 1)
                                    : 0;
                            end else begin
                                rsz_tile_ih_origin <= (cfg_out_h != 0)
                                    ? (out_start_h * cfg_in_h) / cfg_out_h : 0;
                                rsz_tile_iw_origin <= (cfg_out_w != 0)
                                    ? (out_start_w * cfg_in_w) / cfg_out_w : 0;
                            end
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
                // Compute the input tile width used by activation addressing.
                if (cfg_tile_h == 17'd0) begin
                    tile_in_w <= cfg_in_w;
                end else begin
                    // Pooling uses pool_cfg params, Conv uses cfg_stride/kw_eff
                    if (cfg_op_type == 3) begin  // POOLING
                        tile_in_w <= cfg_tile_w * {12'd0, pool_cfg_sw} + {12'd0, pool_cfg_w} - {12'd0, pool_cfg_sw};
                    end else begin
                        tile_in_w <= cfg_tile_w * cfg_stride_w + kw_eff - cfg_stride_w;
                    end
                end
                // OFM is a separate bank from IFM. cfg_ofm_base already
                // selects the double-buffer half the store DMA will read.
                // Adding cfg_act_base here used the IFM half (3072) while
                // the store reads the OFM half (6144), so every odd tile
                // was stored as zeros.
                // DW stream mode: act SRAM[0] holds the resident slice;
                // the 1x1xC output accumulates in the fixed high region
                // (npu_ctrl stores from DW_STREAM_OUT_BASE for these layers).
                if (slice_stream) begin
                    out_base    <= `DW_STREAM_OUT_BASE;
                    dw_grp_base <= 16'd0;
                end else begin
                    out_base <= cfg_ofm_base;
                end

                oc_group <= 0;
                row_ks0 <= 16'd0;
                row_ks_reload <= 1'b0;

                if (cfg_op_type == 8'd1 && dw_stream) begin
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
                    wgt_base <= 0;
                else if (cfg_int16)
                    wgt_base <= (oc_group * ARRAY_SIZE_16 * k_depth) >> 1;
                else
                    wgt_base <= (oc_group * ARRAY_SIZE_16 * k_depth) >> 2;
                // Param base: 4 words per channel, ARRAY_SIZE channels per group.
                // When per-oc PARAM reload is active (weights reload per oc_group
                // AND out_c exceeds the param SRAM capacity of SPAD_KB*16/4
                // channels), the controller reloads this oc_group's params to
                // param SRAM[0], so param_base must be 0 (not oc_group*64).
                if (cfg_wgt_per_oc != 0 && ({17'd0, cfg_out_c} * 4 > `PARAM_DEPTH_W)) begin
                    param_base <= 16'd0;
                end else
                    // 4 words per channel.
                    param_base <= oc_group * {11'd0, grp_oc} * 16'd4;

                // Reset spatial coords for first pixel
                sp_oh <= 0;
                sp_ow <= 0;

                begin : pixel_count_blk
                    reg [31:0] tpx;
                    tpx = {16'd0, out_tile_h} * {16'd0, out_tile_w};
                    px_remaining <= tpx[15:0];
                end

                if (param_feed_busy) begin
                    param_pend <= 1'b1;
                end else begin
                    cache_issue <= 0;
                    cache_cap   <= 0;
                    param_pend  <= 1'b0;
                end
                if (use_row) begin
                    begin : row_init
                        reg [15:0] rem_oc, klen, ntap, tpc, kslen;
                        reg [6:0]  nch;
                        reg [2:0]  psh;
                        reg [3:0]  bsh;
                        reg [12:0] bsz;
                        integer ps;
                        rem_oc = cfg_out_c - oc_group * {9'd0, row_group_ch};
                        nch = (rem_oc >= {9'd0, row_group_ch})
                            ? row_group_ch : rem_oc[6:0];
                        row_nch <= nch;
                        row_dw  <= (cfg_op_type == 8'd1);
                        ntap = {8'd0, cfg_kernel_h} * {8'd0, cfg_kernel_w};
                        klen = (cfg_op_type == 8'd1) ? ntap : k_depth;
                        kslen = (row_kslice && klen > row_ks_cap)
                              ? row_ks_cap : klen;
                        row_klen <= klen;
                        row_ks0 <= 16'd0;
                        row_ksend <= kslen;
                        row_ks_reload <= 1'b0;
                        row_ntap <= ntap;
                        // Pack P = 64/G output pixels into the lanes when the
                        // group is narrower than the row. The packed pixels
                        // are one contiguous NHWC run, so this is only valid
                        // when the group is the whole channel axis. Conv is
                        // also bounded by what one abuf bank can hold.
                        // Depthwise additionally needs stride 1, so one tap
                        // of those pixels is a single contiguous read.
                        psh = 3'd0;
                        if (((cfg_op_type != 8'd1)
                                && (((cfg_out_c << (cfg_int16 ? 1 : 0)) & 16'd3) == 0)
                                && ((({9'd0, nch} << (cfg_int16 ? 1 : 0)) & 16'd3) == 0))
                                || ((cfg_op_type == 8'd1)
                                    && rem_oc == cfg_out_c
                                    && cfg_stride_w == 8'd1
                                    && cfg_in_c == cfg_out_c)) begin
                            for (ps = 1; ps <= ROW_PSH_MAX; ps = ps + 1)
                                if ((({9'd0, nch} << ps) <= ROW_N)
                                        && (cfg_op_type != 8'd1
                                            || (({16'd0, cfg_in_c} << (ps + 1))
                                                <= ABUF_WORDS)))
                                    psh = ps[2:0];
                        end
                        row_pix_sh  <= psh;
                        row_pix     <= 5'd1 << psh;
                        bsh = 4'd10 - {1'b0, psh};          // ABUF_WORDS/(2*P)
                        bsz = 13'd1 << bsh;
                        row_bank_sh <= bsh;
                        row_bank_sz <= bsz;
                        // A 1x1 FC vector is one tap. Split it when that tap
                        // is longer than this pixel's abuf bank.
                        row_slice <= (cfg_op_type != 8'd1)
                                  && (row_kslice || ({3'd0, bsz} < cfg_in_c));
                        // Chunk the K axis by whole spatial taps so a chunk is
                        // always a contiguous slice of the im2col vector.
                        tpc = (cfg_op_type == 8'd1) ? ntap
                            : ((cfg_in_c == 16'd0) ? 16'd1
                               : ({3'd0, bsz} / cfg_in_c));
                        if (tpc == 16'd0) tpc = 16'd1;
                        if (tpc > ntap)   tpc = ntap;
                        row_tpc <= tpc;
                        // Upper bound on K indices one 64-byte beat can span.
                        begin : beat_k_calc
                            reg [15:0] gb;
                            gb = cfg_int16 ? ({9'd0, nch} << 1) : {9'd0, nch};
                            if      (gb >= 16'd64) row_beat_k <= 7'd1;
                            else if (gb >= 16'd32) row_beat_k <= 7'd2;
                            else if (gb >= 16'd16) row_beat_k <= 7'd4;
                            else if (gb >= 16'd8)  row_beat_k <= 7'd8;
                            else if (gb >= 16'd4)  row_beat_k <= 7'd16;
                            else if (gb >= 16'd2)  row_beat_k <= 7'd32;
                            else                   row_beat_k <= 7'd64;
                        end
                        row_wfit   <= (klen <= WROW_K[15:0]);
                        wst_nbytes <= cfg_int16
                                   ? ({16'd0, nch} * {16'd0, kslen} * 32'd2)
                                   : ({16'd0, nch} * {16'd0, kslen});
                        row_ki     <= 0;
                        row_kc0    <= 0;
                        mac_kend   <= 0;
                        mac_bank   <= 1'b0;
                        ga_on      <= 1'b0;
                        ga_busy    <= 1'b0;
                        ga_bank    <= 1'b0;
                        ga_f_vld   <= 2'd0;
                        wst_on     <= 1'b0;
                        wst_f_vld  <= 2'd0;
                        wst_iss_n  <= 16'd0;
                        wst_smp_n  <= 16'd0;
                        wst_sk     <= 16'd0;
                        wst_sc     <= 7'd0;
                        wst_skip   <= 6'd0;
                        wst_iss_byte <= 32'd0;
                        dwl_on     <= 1'b0;
                        dwl_tap    <= 16'd0;
                        dwl_ph     <= 2'd0;
                        row_ppu    <= 1'b0;
                        row_drain  <= 1'b0;
                        row_last   <= 1'b0;
                        pk         <= 0;
                        pk_pos     <= 0;
                        pk_addr    <= out_base;
                    end
                    if (cfg_wgt_per_oc == 0) begin
                        // Depthwise is tap-major over all channels, so the
                        // group offset is per tap and folded into dwl's own
                        // address. Dense conv groups are whole 64*k_depth
                        // blocks, so the group base is a plain multiple.
                        if (cfg_op_type == 8'd1)
                            wgt_base <= 0;
                        else if (cfg_int16)
                            wgt_base <= (oc_group * {9'd0, row_group_ch} * k_depth) >> 1;
                        else
                            wgt_base <= (oc_group * {9'd0, row_group_ch} * k_depth) >> 2;
                    end
                    // Reloaded params sit at SRAM[0]. Resident params are
                    // addressed by the 64-channel group.
                    if (({17'd0, cfg_out_c} * 16'd4) > `PARAM_DEPTH_W)
                        param_base <= 16'd0;
                    else
                        param_base <= oc_group * {9'd0, row_group_ch} * 16'd4;
                    state <= S_ROW_WGT;
                end else if (cfg_op_type == 8'd1) begin
                    dw_init_phase <= 2'd0;
                    dw_ch_idx <= 16'd0;
                    dw_cnt <= 16'd0;
                    dw_read_issued <= 1'b0;
                    state <= S_DW_WGT_LOAD;
                end else
                    state <= S_DONE;
            end

            // ══════════════════════════════════════════════════════════════
            // 64-lane row engine.
            //
            // Lanes are (p, oc): P output pixels packed side by side when the
            // OC group is narrower than 64. The K axis is walked one index per
            // cycle, weights streaming out of the ring and activations out of
            // the abuf chunk the gather filled one chunk ahead. racc carries
            // the partial sum across chunks, so k_depth is unbounded.
            // ══════════════════════════════════════════════════════════════
            S_ROW_WGT: begin
                if (row_dw) begin
                    // Small strided load, done once per OC group.
                    if (!dwl_on && dwl_tap == 16'd0 && wst_sk == 16'd0)
                        dwl_on <= 1'b1;
                    else if (!dwl_on && wst_sk == row_klen)
                        state <= S_ROW_GRP;
                end else if (row_wfit) begin
                    // Whole group fits the ring: stream it once and reuse it
                    // for every pixel group instead of re-reading per pixel.
                    wst_on <= 1'b1;
                    if (wst_sk >= row_klen) begin
                        wst_on <= 1'b0;
                        state  <= S_ROW_GRP;
                    end
                end else
                    state <= S_ROW_GRP;
            end

            // Start a new group of P output pixels.
            S_ROW_GRP: begin : row_grp
                reg [4:0]  pv;
                reg [15:0] t_oh, t_ow, room;
                integer q;
                pv = (px_remaining >= {11'd0, row_pix}) ? row_pix : px_remaining[4:0];
                // Packed depthwise pixels have to stay on one row, so the
                // group shrinks at the right edge instead of wrapping.
                if (row_dw) begin
                    room = out_tile_w - sp_ow;
                    if (room < {11'd0, pv})
                        pv = room[4:0];
                end
                row_pv <= pv;
                grp_oh <= sp_oh;
                grp_ow <= sp_ow;
                t_oh = sp_oh;
                t_ow = sp_ow;
                for (q = 0; q < 8; q = q + 1) begin
                    gp_oh[q] <= t_oh;
                    gp_ow[q] <= t_ow;
                    if (q < {28'd0, pv}) begin
                        if (t_ow + 16'd1 >= out_tile_w) begin
                            t_ow = 16'd0;
                            t_oh = t_oh + 16'd1;
                        end else
                            t_ow = t_ow + 16'd1;
                    end
                end
                // Depthwise advances sp itself in S_ROW_PUSH, by the whole
                // packed group, and comes back here for the next one.
                if (!row_dw) begin
                    sp_oh <= t_oh;
                    sp_ow <= t_ow;
                    px_remaining <= px_remaining - {11'd0, pv};
                end
                row_ki  <= row_kslice ? row_ks0 : 16'd0;
                row_kc0 <= row_kslice ? row_ks0 : 16'd0;
`ifdef DBG_ROW
                $display("[ROWGRP] t=%0t sp=(%0d,%0d) pv=%0d pix=%0d psh=%0d bsh=%0d tpc=%0d ntap=%0d klen=%0d nch=%0d pxrem=%0d otw=%0d",
                         $time, sp_oh, sp_ow, pv, row_pix, row_pix_sh, row_bank_sh, row_tpc, row_ntap, row_klen, row_nch, px_remaining, out_tile_w);
`endif
                // Conv re-streams the weights per pixel group unless the whole
                // group already fits the ring.
                if (!row_dw && !row_wfit) begin
                    wst_iss_byte <= 32'd0;
                    wst_sk       <= row_kslice ? row_ks0 : 16'd0;
                    wst_sc       <= 7'd0;
                    wst_iss_n    <= 16'd0;
                    wst_smp_n    <= 16'd0;
                    wst_f_vld    <= 2'd0;
                    wst_on       <= 1'b1;
                end
                if (row_dw) begin
                    ga_t  <= 16'd0;
                    gs_t  <= 16'd0;
                    ga_fh <= 16'd0;
                    ga_fw <= 16'd0;
                    gs_e  <= 16'd0;
                    gs_left <= 16'd0;
                    ga_f_vld <= 2'd0;
                    ga_p  <= 5'd0;
                    ga_left <= 16'd0;
                    ga_on <= 1'b1;
                    state <= S_ROW_MAC;
                    mac_kend <= row_klen;
                end else begin
                    if (row_slice)
                        row_start_slice(row_kslice ? row_ks0 : 16'd0, 1'b0);
                    else
                        row_start_chunk(16'd0, 1'b0);
                    state <= S_ROW_FILL;
                end
            end

            // Wait for the chunk the gather is filling, then hand it to the
            // MAC and immediately start prefetching the next one.
            S_ROW_FILL: begin
                if (!ga_busy && ga_f_vld == 2'd0 && gs_p >= row_pv) begin : row_swap
                    reg [15:0] tv, t0, k1;
                    if (row_slice) begin
                        mac_bank <= ga_bank;
                        row_kc0  <= ga_k0;
                        mac_kend <= ga_k0 + ga_nch;
                        k1 = ga_k0 + ga_nch;
                        ga_bank  <= ~ga_bank;
                        if (k1 < (row_kslice ? row_ksend : row_klen))
                            row_start_slice(k1, ~ga_bank);
                        else
                            ga_on <= 1'b0;
                    end else begin
                        t0 = ga_tap0;
                        tv = ((t0 + row_tpc) > row_ntap) ? (row_ntap - t0) : row_tpc;
                        mac_bank <= ga_bank;
                        row_kc0  <= t0 * cfg_in_c;
                        mac_kend <= (t0 + tv) * cfg_in_c;
                        ga_bank  <= ~ga_bank;
                        if (t0 + tv < row_ntap)
                            row_start_chunk(t0 + tv, ~ga_bank);
                        else
                            ga_on <= 1'b0;
                    end
                    state <= S_ROW_MAC;
                end
            end

            S_ROW_MAC: begin
                if (row_dw) begin
                    // One tap per cycle as soon as the gather has landed it.
                    if (gs_t > row_ki) begin : mac_dw
                        integer l;
                        reg [6:0] c;
                        reg [6:0] nz;
                        reg signed [31:0] prod;
                        reg [31:0] wslot, aslot;
                        // Lane l = p * G + c. Every packed pixel of channel c
                        // reuses that channel's tap weight.
                        nz = 7'd0;
                        c  = 7'd0;
                        for (l = 0; l < ROW_N; l = l + 1) begin
                            if (l < ({7'd0, row_pv} * {9'd0, row_nch})) begin
                                aslot = ({30'd0, row_ki[1:0]} << 6) + l[31:0];
                                wslot = ({16'd0, row_ki[5:0]} * ROW_N) + {25'd0, c};
                                if (avec[aslot[7:0]] != 0)
                                    nz = nz + 7'd1;
                                prod = avec[aslot[7:0]] * wrow[wslot[12:0]];
                                if (row_ki == 16'd0)
                                    racc[l] <= prod;
                                else
                                    racc[l] <= racc[l] + prod;
                                if (c + 7'd1 >= row_nch)
                                    c = 7'd0;
                                else
                                    c = c + 7'd1;
                            end
                        end
                        mac1d_fire  <= 1'b1;
                        mac1d_act   <= {DATA_W{1'b0}};
                        mac1d_lanes <= nz;
                        if (row_ki + 16'd1 >= row_klen)
                            state <= S_ROW_PUSH;
                        else
                            row_ki <= row_ki + 16'd1;
                    end
                end else if (row_ki < wst_sk || row_wfit) begin : mac_conv
                    integer l;
                    reg [4:0] p;
                    reg [6:0] c, nz;
                    reg signed [DATA_W-1:0] actv;
                    reg signed [31:0] prod;
                    reg [31:0] aslot, wslot;
                    reg [15:0] koff, next_end;
                    koff = row_ki - row_kc0;
                    nz = 7'd0;
                    p  = 5'd0;
                    c  = 7'd0;
                    for (l = 0; l < ROW_N; l = l + 1) begin
                        if (l < ({7'd0, row_pv} * {9'd0, row_nch})) begin
                            aslot = ((({31'd0, mac_bank} << row_pix_sh) + {27'd0, p})
                                     << row_bank_sh) + {16'd0, koff};
                            actv  = abuf[aslot[10:0]];
                            wslot = ({16'd0, row_ki[5:0]} * ROW_N) + {25'd0, c};
                            prod  = actv * wrow[wslot[12:0]];
                            if (actv != 0)
                                nz = nz + 7'd1;
                            if (l == 0)
                                mac1d_act <= actv;
                            if (row_ki == 16'd0)
                                racc[l] <= prod;
                            else
                                racc[l] <= racc[l] + prod;
                            if (c + 7'd1 >= row_nch) begin
                                c = 7'd0;
                                p = p + 5'd1;
                            end else
                                c = c + 7'd1;
                        end
                    end
                    mac1d_fire  <= 1'b1;
                    mac1d_lanes <= nz;
                    if (row_ki + 16'd1 >= (row_kslice ? row_ksend : row_klen)) begin
                        if (row_kslice && row_ksend < row_klen) begin
                            next_end = row_ksend + row_ks_cap;
                            if (next_end > row_klen)
                                next_end = row_klen;
                            row_ks0 <= row_ksend;
                            row_ksend <= next_end;
                            row_ks_reload <= 1'b1;
                            wst_on <= 1'b0;
                            ga_on <= 1'b0;
                            oc_group_done <= 1'b1;
                            state <= S_WAIT_WGT_RELOAD;
                        end else
                            state <= S_ROW_PUSH;
                    end
                    else begin
                        row_ki <= row_ki + 16'd1;
                        if (row_ki + 16'd1 >= mac_kend)
                            state <= S_ROW_FILL;
                    end
                end
            end

            S_ROW_PUSH: begin
                if (!row_ppu && !row_drain && param_ready) begin : row_push
                    integer l;
                    reg [6:0] c;
                    reg [7:0] cw;
                    c = 7'd0;
                    for (l = 0; l < ROW_N; l = l + 1) begin
                        if (l < ({7'd0, row_pv} * {9'd0, row_nch})) begin
                            cw = {c[5:0], 2'b00};
                            ppu_acc_w[ACC_W*l +: ACC_W] <= racc[l];
                            ppu_mult_w[15*l +: 15] <= param_cache[cw][14:0];
                            ppu_shift_w[6*l +: 6]  <= param_cache[cw][21:16];
                            ppu_zp_w[16*l +: 16]   <= param_cache[cw + 8'd1][15:0];
                            ppu_bias_w[ACC_W*l +: ACC_W]
                                <= $signed({param_cache[cw + 8'd3][15:0],
                                            param_cache[cw + 8'd2],
                                            param_cache[cw + 8'd1][31:16]});
                            ppu_valid_w[l] <= 1'b1;
                            if (l == 0) begin
                                ppu_acc_in     <= racc[0];
                                ppu_mult_m     <= param_cache[cw][14:0];
                                ppu_shift_s    <= param_cache[cw][21:16];
                                ppu_zero_point <= param_cache[cw + 8'd1][15:0];
                                ppu_bias       <= $signed({param_cache[cw + 8'd3][15:0],
                                                           param_cache[cw + 8'd2],
                                                           param_cache[cw + 8'd1][31:16]});
                            end
                            if (c + 7'd1 >= row_nch) c = 7'd0;
                            else                     c = c + 7'd1;
                        end
                    end
                    ppu_in_valid <= 1'b1;
`ifdef DBG_ROW
                    $display("[ROWPUSH] t=%0t grp=(%0d,%0d) pv=%0d nch=%0d racc0=%0d racc1=%0d",
                             $time, grp_oh, grp_ow, row_pv, row_nch, racc[0], racc[1]);
`endif
                    row_ppu <= 1'b1;
                    ppu_oh  <= row_dw ? sp_oh : grp_oh;
                    ppu_ow  <= row_dw ? sp_ow : grp_ow;
                    ppu_nlane <= {4'd0, row_pv} * {2'd0, row_nch};
                    ppu_npix <= row_pv;
                    ppu_nch  <= row_nch;
                    if (row_dw) begin
                        if (px_remaining > {11'd0, row_pv}) begin
                            if (sp_ow + {11'd0, row_pv} >= out_tile_w) begin
                                sp_ow <= 16'd0;
                                sp_oh <= sp_oh + 16'd1;
                            end else
                                sp_ow <= sp_ow + {11'd0, row_pv};
                            px_remaining <= px_remaining - {11'd0, row_pv};
                            if (row_pix == 5'd1) begin
                                // Nothing about the group changes, so stay in
                                // the MAC instead of paying a setup cycle.
                                row_ki   <= 16'd0;
                                ga_t     <= 16'd0;
                                gs_t     <= 16'd0;
                                ga_p     <= 5'd0;
                                ga_left  <= 16'd0;
                                ga_fh    <= 16'd0;
                                ga_fw    <= 16'd0;
                                gs_e     <= 16'd0;
                                gs_left  <= 16'd0;
                                ga_f_vld <= 2'd0;
                                state    <= S_ROW_MAC;
                            end else
                                state <= S_ROW_GRP;
                        end else begin
                            ga_on    <= 1'b0;
                            row_last <= 1'b1;
                            state    <= S_ROW_TAIL;
                        end
                    end else if (px_remaining != 16'd0)
                        state <= S_ROW_GRP;
                    else begin
                        wst_on   <= 1'b0;
                        row_last <= 1'b1;
                        state    <= S_ROW_TAIL;
                    end
                end
            end

            S_ROW_TAIL: begin
                if (!row_ppu && !row_drain)
                    state <= S_OC_NEXT;
            end


            S_OC_NEXT: begin
                if (oc_group + 1 >= oc_groups_total) begin
                    state <= S_TILE_NEXT;
                end else begin
                    oc_group <= oc_group + 1;
                    if (cfg_op_type == 8'd1 && dw_stream) begin
                        dw_ch_idx <= 16'd0;
                        // Stream mode group boundary: oc_group+1 is the next
                        // 16-channel group's base. Request act slice + weight
                        // block reload from the controller.
                        dw_grp_base <= (oc_group + 1) * ARRAY_SIZE_16;
                        oc_group_done <= 1'b1;
                        state <= S_WAIT_WGT_RELOAD;
                    end else if (cfg_op_type == 8'd1) begin
                        state <= S_OC_SETUP;
                    end else if (cfg_op_type == 8'd5) begin
                        rsz_ch <= rsz_ch + 1;
                        param_word_idx <= 0;
                        param_read_issued <= 1'b0;
                        state <= S_RESIZE_CH_SETUP;
                    end else begin
                        // Conv2D/FC: request weight reload if per-oc enabled
                        if (cfg_wgt_per_oc != 0) begin
                            row_ks0 <= 16'd0;
                            row_ks_reload <= 1'b0;
                            oc_group_done <= 1'b1;
                            state <= S_WAIT_WGT_RELOAD;
                        end else if (use_row
                                     && (({17'd0, cfg_out_c} * 16'd4) > `PARAM_DEPTH_W)) begin
                            // Params overflow the parameter SRAM. Ask the
                            // controller for this group's block even though
                            // the weights stayed resident.
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
                    if (row_ks_reload) begin
                        row_ks_reload <= 1'b0;
                        row_ki <= row_ks0;
                        row_kc0 <= row_ks0;
                        wst_iss_byte <= 32'd0;
                        wst_sk <= row_ks0;
                        wst_sc <= 7'd0;
                        wst_iss_n <= 16'd0;
                        wst_smp_n <= 16'd0;
                        wst_f_vld <= 2'd0;
                        wst_on <= 1'b1;
                        wst_nbytes <= cfg_int16
                                   ? ({16'd0, row_nch}
                                      * {16'd0, row_ksend - row_ks0} * 32'd2)
                                   : ({16'd0, row_nch}
                                      * {16'd0, row_ksend - row_ks0});
                        row_start_slice(row_ks0, 1'b0);
                        state <= S_ROW_FILL;
                    end else if (dw_stream) begin
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
            // OC-major / stream DW. K-major depthwise uses the 64-lane row
            // (S_ROW_*); this engine is ARRAY_SIZE npu_dw_conv lanes.
            // Flow: WGT_LOAD → PARAM → COMPUTE → (ACT_STREAM → PPU_WAIT)* → OC_NEXT
            // ACT_STREAM issues one tap per cycle. The wide port returns two
            // cycles later and already holds the whole channel group, so the
            // group is stored as one masked beat instead of a per-channel RMW.
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
                    wgt_rd_addr <= wgt_word_addr;
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
                                    wgt_rd_addr  <= wgt_word_addr + 1'b1;
                                    dw_read_issued <= 1'b0;
                                end
                            end else begin
                                // INT8: 4 elements per word
                                if ((dw_cnt[1:0] + dw_wgt_bsel_base) == 2'd3) begin
                                    wgt_word_addr <= wgt_word_addr + 1;
                                    wgt_rd_en    <= 1'b1;
                                    wgt_rd_addr  <= wgt_word_addr + 1'b1;
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
                cache_issue <= 9'd0;
                state <= S_DW_PARAM;
            end

            S_DW_PARAM: begin
                // dw_nch channels × 4 words, from this group's first channel.
                // The wide PPU multiplies by these. A scale left over from
                // the previous layer (or a single channel at oc_group*4)
                // requantizes the whole group to zeros.
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= (dw_ch_base << 2) + cache_issue[8:0];
                    param_read_issued <= 1'b1;
                    act_read_issued <= 1'b0;
                end else if (!act_read_issued) begin
                    act_read_issued <= 1'b1;
                end else begin
                    param_cache[cache_issue[7:0]] <= param_rd_data;
                    if (cache_issue + 9'd1 >= {2'd0, dw_nch, 2'b00}) begin
                        cache_issue <= 9'd0;
                        state <= S_DW_COMPUTE;
                    end else begin
                        cache_issue <= cache_issue + 9'd1;
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
                dw_kdone <= 1'b0;
                dw_iss_vld <= 1'b0;
                dw_ret_vld <= 1'b0;
                state <= S_DW_ACT_STREAM;
            end

            S_DW_ACT_STREAM: begin
                // One kernel tap per cycle once the pipe is full. dw_fh/dw_fw
                // is the next tap to issue; dw_ret_* is the tap whose SRAM
                // beat is valid this cycle.
                begin : dw_act_stream_blk
                    reg signed [15:0] ih_s, iw_s;
                    reg               is_pad, is_last, is_first, consumed_last;
                    reg [1:0]         bsel;
                    reg signed [DATA_W-1:0] zp;
                    integer ci;
                    reg [SRAM_B_W-1:0] sh;
                    reg [ACT_ADDR_W+15:0] elem_off, byte_off;

                    zp = cfg_int16 ? cfg_in_zp
                                   : {{8{cfg_in_zp[7]}}, cfg_in_zp[7:0]};
                    consumed_last = 1'b0;
                    if (dw_ret_vld) begin
                        sh = act_rd_data >> (dw_ret_bsel * 8);
                        dw_in_valid <= 1'b1;
                        dw_in_data  <= cfg_int16 ? $signed(sh[15:0])
                                                : {{8{sh[7]}}, sh[7:0]};
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
                        if (dw_ret_clr)
                            dw_acc_clear <= 1'b1;
                        if (dw_ret_last)
                            consumed_last = 1'b1;
                    end

                    dw_ret_vld  <= dw_iss_vld;
                    dw_ret_bsel <= dw_iss_bsel;
                    dw_ret_last <= dw_iss_last;
                    dw_ret_clr  <= dw_iss_clr;
                    dw_iss_vld  <= 1'b0;

                    if (consumed_last) begin
                        state <= S_DW_PPU_WAIT;
                        ppu_wait_cnt <= 0;
                    end else if (!dw_kdone) begin
                        ih_s = $signed({1'b0, tile_oh_origin + dw_oh}) * $signed({1'b0, cfg_stride_h[7:0]})
                             - $signed({1'b0, cfg_pad_top[7:0]}) + $signed({1'b0, dw_fh});
                        iw_s = $signed({1'b0, tile_ow_origin + dw_ow}) * $signed({1'b0, cfg_stride_w[7:0]})
                             - $signed({1'b0, cfg_pad_left[7:0]}) + $signed({1'b0, dw_fw});
                        is_pad = (ih_s < 0) || (ih_s >= $signed({1'b0, cfg_in_h}))
                              || (iw_s < 0) || (iw_s >= $signed({1'b0, cfg_in_w}));
                        is_last = (dw_fw + 4'd1 >= cfg_kernel_w[3:0])
                               && (dw_fh + 4'd1 >= cfg_kernel_h[3:0]);
                        is_first = (dw_fh == 4'd0) && (dw_fw == 4'd0);
                        // A pad has to follow the reads already in the pipe,
                        // and the MAC accepts one tap per cycle.
                        if (is_pad && !dw_ret_vld && !dw_iss_vld) begin
                            dw_in_valid <= 1'b1;
                            dw_in_data  <= zp;
                            for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1) begin
                                if (ci[4:0] < dw_nch) begin
                                    dw_in_valid_w[ci] <= 1'b1;
                                    dw_in_data_w[DATA_W*ci +: DATA_W] <= zp;
                                end
                            end
                            if (is_first)
                                dw_acc_clear <= 1'b1;
                            if (is_last) begin
                                dw_kdone <= 1'b1;
                                state <= S_DW_PPU_WAIT;
                                ppu_wait_cnt <= 0;
                            end else if (dw_fw + 4'd1 >= cfg_kernel_w[3:0]) begin
                                dw_fw <= 4'd0;
                                dw_fh <= dw_fh + 4'd1;
                            end else
                                dw_fw <= dw_fw + 4'd1;
                        end else if (!is_pad) begin
                            if (dw_stream) begin
                                elem_off = (ih_s[15:0] * cfg_in_w + iw_s[15:0])
                                         * ARRAY_SIZE + (dw_ch_base - dw_grp_base);
                            end else if (cfg_tile_h == 17'd0) begin
                                elem_off = (ih_s[15:0] * cfg_in_w * cfg_in_c)
                                         + (iw_s[15:0] * cfg_in_c)
                                         + dw_ch_base;
                            end else begin
                                begin : dw_addr_2d_blk
                                    reg [17:0] row_rel, col_rel;
                                    row_rel = {2'd0, dw_oh} * {10'd0, cfg_stride_h} + {14'd0, dw_fh};
                                    col_rel = {2'd0, dw_ow} * {10'd0, cfg_stride_w} + {14'd0, dw_fw};
                                    elem_off = (row_rel[15:0] * tile_in_w + col_rel[15:0])
                                               * cfg_in_c + dw_ch_base;
                                end
                            end
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            bsel = byte_off[1:0];
                            act_rd_en   <= 1'b1;
                            act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                            dw_iss_vld  <= 1'b1;
                            dw_iss_bsel <= bsel;
                            dw_iss_last <= is_last;
                            dw_iss_clr  <= is_first;
                            // No activation is fed on this cycle when the
                            // pipe was empty, so the clear resets the MAC
                            // index before the first product arrives.
                            if (is_first && !dw_ret_vld)
                                dw_acc_clear <= 1'b1;
                            if (is_last)
                                dw_kdone <= 1'b1;
                            else if (dw_fw + 4'd1 >= cfg_kernel_w[3:0]) begin
                                dw_fw <= 4'd0;
                                dw_fh <= dw_fh + 4'd1;
                            end else
                                dw_fw <= dw_fw + 4'd1;
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
                                    ppu_mult_w[15*ci +: 15]
                                        <= param_cache[{ci[3:0], 2'b00}][14:0];
                                    ppu_shift_w[6*ci +: 6]
                                        <= param_cache[{ci[3:0], 2'b00}][21:16];
                                    ppu_zp_w[16*ci +: 16]
                                        <= param_cache[{ci[3:0], 2'b00} + 8'd1][15:0];
                                    ppu_bias_w[ACC_W*ci +: ACC_W]
                                        <= $signed({param_cache[{ci[3:0], 2'b00} + 8'd3][15:0],
                                                    param_cache[{ci[3:0], 2'b00} + 8'd2],
                                                    param_cache[{ci[3:0], 2'b00} + 8'd1][31:16]});
                                    ppu_valid_w[ci] <= 1'b1;
                                end
                            end
                        end
                        ppu_wait_cnt <= 1;
                    end
                end else begin
                    if (ppu_vout_w[0] || ppu_out_valid) begin
                        // The ARRAY_SIZE channels of this pixel are contiguous
                        // in NHWC and fit in one 64-byte beat (8×INT16 = 16 B).
                        begin : dw_wide_wb
                            reg [31:0] elem_off, byte_off;
                            reg [SRAM_B_W-1:0] wrwide;
                            reg [63:0] wrmask;
                            reg signed [DATA_W-1:0] oval;
                            integer bi;
                            elem_off = (dw_oh * out_tile_w + dw_ow)
                                     * cfg_out_c + dw_ch_base;
                            byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                            wrwide = {SRAM_B_W{1'b0}};
                            wrmask = 64'd0;
                            for (bi = 0; bi < ARRAY_SIZE; bi = bi + 1) begin
                                if (bi[4:0] < dw_nch) begin
                                    if (bi == 0 && !ppu_vout_w[0])
                                        oval = ppu_out_data;
                                    else
                                        oval = ppu_out_w[DATA_W*bi +: DATA_W];
                                    if (cfg_int16) begin
                                        wrwide[8*(byte_off[1:0] + 2*bi) +: 16] = oval;
                                        wrmask[byte_off[1:0] + 2*bi +: 2] = 2'b11;
                                    end else begin
                                        wrwide[8*(byte_off[1:0] + bi) +: 8] = oval[7:0];
                                        wrmask[byte_off[1:0] + bi] = 1'b1;
                                    end
                                end
                            end
                            act_wr_en   <= 1'b1;
                            act_wr_addr <= out_base + byte_off[ACT_ADDR_W+1:2];
                            act_wr_wide <= wrwide;
                            act_wr_mask <= wrmask;
                        end
                        dw_kdone   <= 1'b0;
                        dw_iss_vld <= 1'b0;
                        dw_ret_vld <= 1'b0;
                        dw_fh <= 4'd0;
                        dw_fw <= 4'd0;
                        if (dw_ow + 16'd1 >= out_tile_w) begin
                            dw_ow <= 16'd0;
                            if (dw_oh + 16'd1 >= out_tile_h)
                                state <= S_OC_NEXT;
                            else begin
                                dw_oh <= dw_oh + 16'd1;
                                state <= S_DW_ACT_STREAM;
                            end
                        end else begin
                            dw_ow <= dw_ow + 16'd1;
                            state <= S_DW_ACT_STREAM;
                        end
                    end else begin
                        ppu_wait_cnt <= ppu_wait_cnt + 1;
                    end
                end
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
                    out_base <= cfg_ofm_base;
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
                // Read-modify-write for the destination word.
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
                // Precompute combined reciprocals (single multiply per pixel).
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

            // 64-lane PPU result, one output beat. Word-aligned groups take
            // the wide port; a narrow channel count is packed across pixels.
            if (row_ppu && !row_drain && (ppu_vout_w[0] || ppu_out_valid)) begin : row_cap
                integer c;
                reg [8:0] nbytes, nlane;
                reg [31:0] elem, bbase;
                reg [511:0] wide;
                reg [1023:0] bytes;
                reg signed [DATA_W-1:0] lane;
                reg [7:0] nb;
                // Address this beat at its own NHWC offset. A 64-channel
                // group of a wider layer is not contiguous with the next
                // pixel: the other groups' channels sit in between.
                nlane = ppu_nlane;
                nbytes = cfg_int16 ? {nlane[7:0], 1'b0} : nlane;
                elem = ({16'd0, ppu_oh} * {16'd0, out_tile_w} + {16'd0, ppu_ow})
                     * {16'd0, cfg_out_c}
                     + {16'd0, oc_group} * {25'd0, row_group_ch};
                bbase = cfg_int16 ? (elem << 1) : elem;
                if (ppu_npix > 5'd1) begin
                    // Lanes are ordered [pixel][channel]. Preserve the NHWC
                    // gaps between OC groups by writing one pixel per cycle.
                    nb = 8'd0;
                    bytes = 1024'd0;
                    for (c = 0; c < ROW_N; c = c + 1) begin
                        if (c < nlane) begin
                            if (ppu_vout_w[c])
                                lane = ppu_out_w[DATA_W*c +: DATA_W];
                            else if (c == 0 && ppu_out_valid)
                                lane = ppu_out_data;
                            else
                                lane = {DATA_W{1'b0}};
                            if (cfg_int16) begin
                                bytes[16*c +: 16] = lane;
                                nb = nb + 8'd2;
                            end else begin
                                bytes[8*c +: 8] = lane[7:0];
                                nb = nb + 8'd1;
                            end
                        end
                    end
                    obuf <= bytes;
                    ob_n <= nb;
                    scatter_p <= 5'd0;
                    row_scatter <= 1'b1;
                    row_drain <= 1'b1;
                end else if (bbase[1:0] == 2'b00 && nbytes[1:0] == 2'b00
                        && nbytes != 0 && nbytes <= 9'd64) begin
                    wide = 512'd0;
                    if (cfg_int16) begin
                        // A 512-bit beat holds at most 32 INT16 lanes.  Keep
                        // the static part-selects in range as well as gating
                        // them with the runtime lane count.
                        for (c = 0; c < 32; c = c + 1) begin
                            if (c < nlane) begin
                                if (ppu_vout_w[c])
                                    lane = ppu_out_w[DATA_W*c +: DATA_W];
                                else if (c == 0 && ppu_out_valid)
                                    lane = ppu_out_data;
                                else
                                    lane = {DATA_W{1'b0}};
                                wide[16*c +: 16] = lane;
                            end
                        end
                    end else begin
                        for (c = 0; c < ROW_N; c = c + 1) begin
                            if (c < nlane) begin
                                if (ppu_vout_w[c])
                                    lane = ppu_out_w[DATA_W*c +: DATA_W];
                                else if (c == 0 && ppu_out_valid)
                                    lane = ppu_out_data;
                                else
                                    lane = {DATA_W{1'b0}};
                                wide[8*c +: 8] = lane[7:0];
                            end
                        end
                    end
                    act_wr_en   <= 1'b1;
                    act_wr_addr <= out_base + bbase[ACT_ADDR_W+1:2];
                    act_wr_wide <= wide;
                    act_wr_data <= wide[31:0];
                    if (nbytes >= 9'd64)
                        act_wr_mask <= 64'hFFFF_FFFF_FFFF_FFFF;
                    else
                        act_wr_mask <= (64'h1 << nbytes) - 64'h1;
                    // A later group may be narrow enough to need the byte
                    // packer, which continues the same NHWC byte stream.
                    pk      <= 32'd0;
                    pk_pos  <= 2'd0;
                    pk_addr <= out_base + bbase[ACT_ADDR_W+1:2]
                             + {{(ACT_ADDR_W-7){1'b0}}, nbytes[8:2]};
                    row_ppu <= 1'b0;
                end else begin
                    nb = 8'd0;
                    bytes = 1024'd0;
                    for (c = 0; c < ROW_N; c = c + 1) begin
                        if (c < nlane) begin
                            if (ppu_vout_w[c])
                                lane = ppu_out_w[DATA_W*c +: DATA_W];
                            else if (c == 0 && ppu_out_valid)
                                lane = ppu_out_data;
                            else
                                lane = {DATA_W{1'b0}};
                            if (cfg_int16) begin
                                bytes[16*c +: 16] = lane;
                                nb = nb + 8'd2;
                            end else begin
                                bytes[8*c +: 8] = lane[7:0];
                                nb = nb + 8'd1;
                            end
                        end
                    end
                    obuf <= bytes;
                    ob_n <= nb;
                    ob_i <= 8'd0;
                    // Start this pixel at its own channel offset. Carrying
                    // pk_addr across pixels packs the group tightly and
                    // makes the next group overwrite channel 0.
                    pk      <= 32'd0;
                    pk_pos  <= 2'd0;
                    pk_addr <= out_base + bbase[ACT_ADDR_W+1:2];
                    row_drain <= 1'b1;
                end
            end else if (row_drain && row_scatter) begin : row_scatter_pack
                integer sc;
                reg [31:0] pixel, elem, bbase;
                reg [8:0] nbytes;
                reg [511:0] wide;
                pixel = ({16'd0, ppu_oh} * {16'd0, out_tile_w}
                         + {16'd0, ppu_ow}) + {27'd0, scatter_p};
                elem = pixel * {16'd0, cfg_out_c}
                     + {16'd0, oc_group} * {25'd0, row_group_ch};
                bbase = cfg_int16 ? (elem << 1) : elem;
                nbytes = cfg_int16 ? ({2'd0, ppu_nch} << 1)
                                   : {2'd0, ppu_nch};
                wide = 512'd0;
                if (cfg_int16) begin
                    for (sc = 0; sc < 32; sc = sc + 1)
                        if (sc < ppu_nch)
                            wide[16*sc +: 16]
                                = obuf[({27'd0, scatter_p} * {25'd0, ppu_nch}
                                        + sc) * 16 +: 16];
                end else begin
                    for (sc = 0; sc < ROW_N; sc = sc + 1)
                        if (sc < ppu_nch)
                            wide[8*sc +: 8]
                                = obuf[({27'd0, scatter_p} * {25'd0, ppu_nch}
                                        + sc) * 8 +: 8];
                end
                act_wr_en   <= 1'b1;
                act_wr_addr <= out_base + bbase[ACT_ADDR_W+1:2];
                act_wr_wide <= wide;
                act_wr_data <= wide[31:0];
                act_wr_mask <= (nbytes >= 9'd64)
                            ? 64'hFFFF_FFFF_FFFF_FFFF
                            : (64'h1 << nbytes) - 64'h1;
                if (scatter_p + 1 >= ppu_npix) begin
                    scatter_p <= 5'd0;
                    row_scatter <= 1'b0;
                    row_drain <= 1'b0;
                    row_ppu <= 1'b0;
                end else
                    scatter_p <= scatter_p + 1'b1;
            end else if (row_drain) begin : row_pack
                integer t;
                reg [1:0] pos;
                reg [31:0] word;
                reg [7:0] ii;
                reg wrote;
                pos = pk_pos;
                word = pk;
                ii = ob_i;
                wrote = 1'b0;
                for (t = 0; t < 4; t = t + 1) begin
                    if (!wrote && ii < ob_n) begin
                        word[pos*8 +: 8] = obuf[8*ii +: 8];
                        ii = ii + 8'd1;
                        if (pos == 2'd3) begin
                            act_wr_en   <= 1'b1;
                            act_wr_addr <= pk_addr;
                            act_wr_data <= word;
                            act_wr_wide <= {480'd0, word};
                            act_wr_mask <= 64'hF;
                            wrote = 1'b1;
                            pos = 2'd0;
                            word = 32'd0;
                        end else
                            pos = pos + 2'd1;
                    end
                end
                pk    <= word;
                pk_pos <= pos;
                ob_i  <= ii;
                if (wrote)
                    pk_addr <= pk_addr + 1'b1;
                if (ii >= ob_n && (!wrote || pos == 2'd0)) begin
                    row_drain <= 1'b0;
                    row_ppu   <= 1'b0;
                    if (!wrote && row_last && pos != 2'd0) begin
                        act_wr_en   <= 1'b1;
                        act_wr_addr <= pk_addr;
                        act_wr_data <= word;
                        act_wr_wide <= {480'd0, word};
                        // DMA stores this 32-bit word whole. Unused high
                        // bytes are zero in `word`, so the full byte enable
                        // clears stale data left by the previous layer.
                        act_wr_mask <= 64'hF;
                        pk      <= 32'd0;
                        pk_pos  <= 2'd0;
                        pk_addr <= pk_addr + 1'b1;
                    end
                end
            end

        end
    end

endmodule
