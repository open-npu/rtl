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
//   default:     tile_y → tile_x → oc_group
//   group-major: oc_group → tile_y → tile_x
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

`ifdef NPU_COMPUTE_BLACKBOX
(* blackbox *)
`endif
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
    input  wire                         cfg_group_major, // Reuse one resident OC group across all spatial tiles
    input  wire                         db_prefetch_done,  // DB_EN: prefetch complete, safe to start next tile
    // Controller-owned tile mode.  Coordinates are tile indices; actual
    // dimensions are already clipped at the output boundary.  One START
    // executes exactly this tile and produces DONE without tile_done.
    input  wire                         cfg_single_tile_en,
    input  wire [15:0]                  cfg_external_tile_y,
    input  wire [15:0]                  cfg_external_tile_x,
    input  wire [15:0]                  cfg_external_tile_h_actual,
    input  wire [15:0]                  cfg_external_tile_w_actual,
    input  wire                         cfg_external_origin_en,
    input  wire signed [16:0]           cfg_external_origin_h,
    input  wire signed [16:0]           cfg_external_origin_w,
    input  wire [15:0]                  cfg_external_tile_in_w,
    input  wire                         cfg_halo_expand_en,
    // Per-tile store support: actual (clipped) tile output dimensions
    output wire [15:0]                  tile_out_h_actual, // Current tile's actual output height (clipped at border)
    output wire [15:0]                  tile_out_w_actual, // Current tile's actual output width (clipped at border)
    output wire [6:0]                   dbg_state,         // Live FSM state for the cycle histogram

    // ─── Layer configuration (from CSR) ───
    input  wire [7:0]                   cfg_op_type,
    input  wire                         cfg_int16,      // 1=INT16 mode, 0=INT8 mode
    input  wire [15:0]                  cfg_in_c,
    input  wire [15:0]                  cfg_out_h,
    input  wire [15:0]                  cfg_out_w,
    input  wire [15:0]                  cfg_out_c,
    input  wire [7:0]                   cfg_kernel_h,
    input  wire [7:0]                   cfg_kernel_w,
    input  wire [7:0]                   cfg_dilation_h,
    input  wire [7:0]                   cfg_dilation_w,
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
    input  wire [WGT_ADDR_W-1:0]        cfg_wgt_base,   // descriptor-local weight SRAM word base
    input  wire [PARAM_ADDR_W-1:0]      cfg_param_base, // selected parameter ping/pong bank
    input  wire [31:0]                  cfg_pool_cfg,   // Pooling config register
    input  wire [31:0]                  cfg_resize_cfg, // Resize config register
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

    // ─── PPU (scalar lane 0 + PPU_LANES-wide time-multiplexed bank) ───
    output reg  signed [ACC_W-1:0]     ppu_acc_in /* verilator public */,
    output reg                          ppu_in_valid,
    output reg  signed [ACC_W-1:0]     ppu_bias,
    output reg  [14:0]                  ppu_mult_m,
    output reg  [5:0]                   ppu_shift_s,
    output reg  signed [15:0]          ppu_zero_point,
    input  wire signed [DATA_W-1:0]    ppu_out_data,
    input  wire                         ppu_out_valid,
    output reg  [ACC_W*`PPU_LANES-1:0] ppu_acc_w,
    output reg  [`PPU_LANES-1:0]       ppu_valid_w,
    output reg  [ACC_W*`PPU_LANES-1:0] ppu_bias_w,
    output reg  [15*`PPU_LANES-1:0]    ppu_mult_w,
    output reg  [6*`PPU_LANES-1:0]     ppu_shift_w,
    output reg  [16*`PPU_LANES-1:0]    ppu_zp_w,
    input  wire [DATA_W*`PPU_LANES-1:0] ppu_out_w,
    input  wire [`PPU_LANES-1:0]       ppu_vout_w
);

`ifndef NPU_COMPUTE_BLACKBOX
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
        S_RESIZE_RECIP    = 7'd56, // 40-cycle restoring reciprocal precompute
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
        S_ROW_FILL        = 7'd74,
        // Wide elementwise-Add states.  Concat keeps the scalar scatter path.
        S_ADD_VEC_READ_A  = 7'd75,
        S_ADD_VEC_READ_B  = 7'd76,
        S_ADD_VEC_COMPUTE = 7'd77,
        S_ADD_VEC_WAIT    = 7'd78,
        S_ADD_VEC_WB      = 7'd79,
        // Wide Pool states: one 64ch INT8 / 32ch INT16 pixel, or a batch of
        // 4x16ch / 2x32ch INT8 MaxPool pixels per 64-lane operation.
        S_POOL_VEC_PARAM  = 7'd80,
        S_POOL_VEC_READ   = 7'd81,
        S_POOL_VEC_ACC    = 7'd82,
        S_POOL_VEC_PPU    = 7'd83,
        S_POOL_VEC_WAIT   = 7'd84,
        S_POOL_VEC_WB     = 7'd85,
        S_POOL_VEC_NEXT   = 7'd86,
        S_POOL_VEC_DIV    = 7'd87;
    (* fsm_encoding = "one_hot" *)
    reg [6:0] state;

    // ─── Tile iteration ───
    reg [15:0] tile_y, tile_x;
    reg        single_tile_active;
    reg [15:0] single_tile_h_actual, single_tile_w_actual;
    reg        external_origin_active, halo_expand_active;
    reg signed [16:0] external_origin_h, external_origin_w;
    reg [15:0] external_tile_in_w;
    reg        tile_done_r;
    reg        tile_wait_delay;  // DB_EN wait: skip 1 cycle, then wait for prefetch
    assign tile_done = tile_done_r;
    // Pool stream iterates pool_ch (not oc_group) — report the pool group
    // base so the controller reloads the right slice.
    assign oc_group_out = pool_stream ? pool_grp_base : oc_group;
    // Expose actual (border-clipped) tile output dims for per-tile store DMA
    assign tile_out_h_actual = out_tile_h;
    assign tile_out_w_actual = out_tile_w;
    assign dbg_state = state;
    // Group-major owns the spatial loop and is therefore incompatible with
    // an externally owned single-tile invocation.
    wire group_major_active = cfg_group_major && !single_tile_active;
    wire signed [17:0] halo_global_oh = external_origin_h
                                      + $signed({1'b0, ppu_oh});
    wire signed [17:0] halo_global_ow = external_origin_w
                                      + $signed({1'b0, ppu_ow});
    wire halo_output_pad = halo_expand_active
                         && ((halo_global_oh < 0)
                             || (halo_global_oh >= $signed({2'b0, cfg_out_h}))
                             || (halo_global_ow < 0)
                             || (halo_global_ow >= $signed({2'b0, cfg_out_w})));
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

    // Row-engine params are pixel-invariant and cached once per OC group:
    // 64 per-channel registers (81 bits/ch instead of 128 bits/ch, single write port).
    (* mem2reg *) reg [14:0]      pc_mult  [0:63];
    (* mem2reg *) reg [5:0]       pc_shift [0:63];
    (* mem2reg *) reg [15:0]      pc_zp    [0:63];
    (* mem2reg *) reg [15:0]      pc_blo   [0:63];
    (* mem2reg *) reg [ACC_W-17:0] pc_bhi  [0:63];
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
    reg [4:0]  dw_fh, dw_fw;          // Filter position (0..16)
    reg [8:0]  dw_kernel_size;         // kh * kw (cached, up to 16x16=256)
    reg [7:0]  dw_tap_idx;             // Active kernel tap index in S_DW_ACT_STREAM
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
    localparam PPU_N = `PPU_LANES;
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
    // Banked row memories: 64 independent single-write-port banks eliminate
    // dynamic multi-write mux trees while keeping 1-cycle gather/MAC throughput.
    reg signed [DATA_W-1:0] wrow_b [0:ROW_N-1][0:WROW_K-1];
    reg signed [DATA_W-1:0] abuf_b [0:ROW_N-1][0:(ABUF_WORDS/ROW_N)-1];
    reg signed [DATA_W-1:0] avec_b [0:ROW_N-1][0:3];  // depthwise 4-tap ring
    reg signed [ACC_W-1:0]  racc [0:ROW_N-1];
    reg [5:0]  lane_c_r [0:ROW_N-1]; // per-lane channel index within OC group
    reg [2:0]  lane_p_r [0:ROW_N-1]; // per-lane packed pixel index (0..7)
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
    reg [7:0]  scatter_bi;
    reg [2:0]  row_push_sub;  // 8-lane sub-beat index pushed to PPU (0..7)
    reg [2:0]  row_cap_sub;   // 8-lane sub-beat index captured into obuf (0..7)
    reg [3:0]  row_cap_tgt;   // Total 8-lane sub-beats expected (1..8)
    reg [2:0]  vec_ppu_sub;   // Shared Pool/Add vector 8-lane feed beat (0..7)
    reg [2:0]  vec_cap_sub;   // Shared Pool/Add vector 8-lane capture beat (0..7)

    // ─── weight streamer (1 K-index per beat, 2-stage pipeline) ───
    reg        wst_on;
    reg [31:0] wst_iss_byte;   // byte offset of the next beat to request
    reg [15:0] wst_ik;         // K index currently being issued
    reg [6:0]  wst_ic;         // channel offset within wst_ik
    reg [15:0] wst_kend;       // K end bound for current resident slice
    reg [15:0] wst_sk;         // K indices written into the ring so far
    reg [6:0]  wst_sc;         // lane offset used by DW weight loader
    reg [5:0]  wst_skip;       // byte offset used by DW weight loader
    reg [1:0]  wst_f_vld;
    reg [1:0]  wst_f_skip [0:1];
    reg [15:0] wst_f_k    [0:1];
    reg [6:0]  wst_f_c0   [0:1];
    reg [6:0]  wst_f_nch  [0:1];
    reg [1:0]  wst_f_kend;
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
    reg [10:0] ga_f_nelt [0:1]; // elements in the whole read, not this beat
    reg [4:0]  ga_f_step [0:1]; // dense taps / DW pixels covered by this read
    reg [1:0]  ga_f_tend;       // this beat completes a depthwise tap
    reg [6:0]  ga_base_r;       // held across the beats of one read
    reg [10:0] ga_nelt_r;
    reg        ga_tend_r;
    reg        ga_pad_r;        // continuation beat is zero-point pad fill
    reg [4:0]  ga_step_r;       // pixels this read covers
    reg        ga_bank;        // abuf chunk buffer being filled
    // The P packed pixels are consecutive in raster order, so their outputs
    // are contiguous in NHWC and one origin plus row_pv describes them all.
    reg [15:0] grp_oh, grp_ow;
    reg [15:0] gp_oh [0:7];
    reg [15:0] gp_ow [0:7];
    reg [15:0] ga_fh, ga_fw;   // issuer's spatial tap
    reg [15:0] ga_fh0, ga_fw0; // spatial tap the chunk starts at
    reg [15:0] ga_next_fh0, ga_next_fw0; // next chunk's starting (fh, fw)
    reg [15:0] ga_next_tap, ga_next_ch, ga_next_fh, ga_next_fw; // next slice coords
    reg [15:0] ks0_tap_r, ks0_ch_r, ks0_fh_r, ks0_fw_r;         // saved coords at row_ks0
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
    wire [2:0]  row_dense_sh =
        ((row_one_oc_bytes << 6) <= `WEIGHT_BUF_SIZE) ? 3'd6 :
        ((row_one_oc_bytes << 5) <= `WEIGHT_BUF_SIZE) ? 3'd5 :
        ((row_one_oc_bytes << 4) <= `WEIGHT_BUF_SIZE) ? 3'd4 :
        ((row_one_oc_bytes << 3) <= `WEIGHT_BUF_SIZE) ? 3'd3 :
        ((row_one_oc_bytes << 2) <= `WEIGHT_BUF_SIZE) ? 3'd2 :
        ((row_one_oc_bytes << 1) <= `WEIGHT_BUF_SIZE) ? 3'd1 : 3'd0;
    wire        row_kslice = cfg_wgt_per_oc[31] && cfg_wgt_layout
                           && (cfg_op_type == 8'd0 || cfg_op_type == 8'd2);
    wire [6:0]  row_group_ch = (cfg_op_type == 8'd1 || row_kslice)
                              ? ROW_N[6:0] : row_dense_grp;
    wire [2:0]  row_group_sh = (cfg_op_type == 8'd1 || row_kslice)
                              ? 3'd6 : row_dense_sh;
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
    reg [15:0]             pool_vec_ch_base;
    reg [6:0]              pool_vec_div_lane;
    reg [2:0]              pool_vec_pix_idx;
    reg [2:0]              pool_vec_batch_n;
    // Vector Pool reuses racc[0:63] for accumulation and obuf[1023:0] for PPU output.
    // Narrow-channel INT8 max-pool fills the 64-lane vector with adjacent
    // output pixels: 4x16ch or 2x32ch. Each source pixel still uses one wide
    // SRAM read, but comparison, PPU, and writeback are shared by the batch.
    wire [2:0] pool_vec_pix_cap =
        (!cfg_int16 && cfg_out_c == 16'd16) ? 3'd4 :
        (!cfg_int16 && cfg_out_c == 16'd32) ? 3'd2 : 3'd1;
    wire       pool_vec_multi = (pool_vec_pix_cap != 3'd1);

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
    reg                    add_vec_active;  // Current Add chunk uses the wide path
    reg [`SRAM_B_WIDTH-1:0] add_vec_a;
    reg [`SRAM_B_WIDTH-1:0] add_vec_b;

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
    // Combined resize reciprocals, precomputed once per layer via 40-step restoring shift-subtract.
    reg [40:0]             rsz_rcp_h, rsz_rcp_w, rsz_rcp_hm1, rsz_rcp_wm1;
    reg [16:0]             rsz_rem_h, rsz_rem_w, rsz_rem_hm1, rsz_rem_wm1;
    reg [39:0]             rsz_q_h,   rsz_q_w,   rsz_q_hm1,   rsz_q_wm1;
    reg [5:0]              rsz_div_cnt;
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

    // ─── Combinational 64-bit popcount (6-level adder tree, no loop carry) ───
    function [6:0] popcnt64;
        input [63:0] v;
        integer k;
        reg [1:0] s1 [0:31];
        reg [2:0] s2 [0:15];
        reg [3:0] s3 [0:7];
        reg [4:0] s4 [0:3];
        reg [5:0] s5 [0:1];
        begin
            for (k = 0; k < 32; k = k + 1) s1[k] = {1'b0, v[2*k]} + {1'b0, v[2*k+1]};
            for (k = 0; k < 16; k = k + 1) s2[k] = {1'b0, s1[2*k]} + {1'b0, s1[2*k+1]};
            for (k = 0; k < 8;  k = k + 1) s3[k] = {1'b0, s2[2*k]} + {1'b0, s2[2*k+1]};
            for (k = 0; k < 4;  k = k + 1) s4[k] = {1'b0, s3[2*k]} + {1'b0, s3[2*k+1]};
            for (k = 0; k < 2;  k = k + 1) s5[k] = {1'b0, s4[2*k]} + {1'b0, s4[2*k+1]};
            popcnt64 = {1'b0, s5[0]} + {1'b0, s5[1]};
        end
    endfunction

    // ─── Control & banked-memory write-port wires ───
    wire [4:0]  grp_pv_raw_w = (px_remaining >= {11'd0, row_pix})
                             ? row_pix : px_remaining[4:0];
    wire [15:0] grp_room_w   = out_tile_w - sp_ow;
    wire [4:0]  grp_pv_w     = (row_dw && (grp_room_w < {11'd0, grp_pv_raw_w}))
                             ? grp_room_w[4:0] : grp_pv_raw_w;

    wire [8:0]  pv_tgt_w     = cfg_int16 ? 9'd128
                             : (pool_vec_multi ? {cfg_out_c[6:0], 2'b00} : 9'd256);
    wire        pc_wr_bg_w   = rst_n && !(param_pend && !param_feed_busy)
                             && param_bg_ok && (cache_issue >= 9'd2)
                             && (cache_cap < param_tgt);
    wire        pc_wr_dw_w   = rst_n && (state == S_DW_PARAM)
                             && param_read_issued && act_read_issued;
    wire        pc_wr_pv_w   = rst_n && (state == S_POOL_VEC_PARAM)
                             && (cache_issue >= 9'd2) && (cache_cap < pv_tgt_w);
    wire        pc_we_w      = pc_wr_bg_w || pc_wr_dw_w || pc_wr_pv_w;
    wire [7:0]  pc_widx_w    = pc_wr_dw_w ? cache_issue[7:0] : cache_cap[7:0];

    // Single-write-port per-channel parameter register file (1 sequential block, gated by pc_we_w)
    always @(posedge clk) begin
        if (pc_we_w) begin
            case (pc_widx_w[1:0])
                2'd0: begin
                    pc_mult[pc_widx_w[7:2]]  <= param_rd_data[14:0];
                    pc_shift[pc_widx_w[7:2]] <= param_rd_data[21:16];
                end
                2'd1: begin
                    pc_zp[pc_widx_w[7:2]]  <= param_rd_data[15:0];
                    pc_blo[pc_widx_w[7:2]] <= param_rd_data[31:16];
                end
                2'd2: begin
                    pc_bhi[pc_widx_w[7:2]] <= param_rd_data[ACC_W-17:0];
                end
                default: ;
            endcase
        end
    end

    // ─── Banked Row Memories (64 independent single-write-port banks) ───
    wire        dwl_take_en_w = rst_n && dwl_on && (dwl_ph == 2'd2);
    wire [7:0]  dwl_avail_b_w = 8'd64 - {6'd0, wst_skip[1:0]};
    wire [6:0]  dwl_rem_ch_w  = row_nch - wst_sc;
    wire [7:0]  dwl_rem_b_w   = cfg_int16 ? ({1'b0, dwl_rem_ch_w} << 1)
                                          : {1'b0, dwl_rem_ch_w};
    wire [7:0]  dwl_take_b_w  = (dwl_rem_b_w <= dwl_avail_b_w)
                              ? dwl_rem_b_w : dwl_avail_b_w;
    wire [6:0]  dwl_take_ch_w = cfg_int16 ? dwl_take_b_w[7:1] : dwl_take_b_w[6:0];

    wire        wrow_we_any_w = (rst_n && wst_f_vld[1]) || dwl_take_en_w;
    wire [5:0]  wrow_wr_k_w   = dwl_take_en_w ? dwl_tap[5:0]   : wst_f_k[1][5:0];
    wire [6:0]  wrow_wr_c0_w  = dwl_take_en_w ? wst_sc         : wst_f_c0[1];
    wire [6:0]  wrow_wr_nch_w = dwl_take_en_w ? dwl_take_ch_w  : wst_f_nch[1];
    wire [1:0]  wrow_wr_sk_w  = dwl_take_en_w ? wst_skip[1:0]  : wst_f_skip[1];

    // OC-major / stream DW reuses wrow_b[0..31][0..63] (8 channels × up to 256 taps),
    // eliminating the external 8×256×16-bit async-reset npu_dw_bank register array.
    wire        dw_wgt_we_w   = rst_n && (state == S_DW_WGT_LOAD)
                             && (dw_init_phase == 2'd2) && dw_read_issued;
    wire [1:0]  dw_wgt_bs16_w = {dw_cnt[0], 1'b0} + dw_wgt_bsel_base;
    wire [1:0]  dw_wgt_bs8_w  = dw_cnt[1:0] + dw_wgt_bsel_base;
    wire signed [DATA_W-1:0] dw_wgt_val_w = cfg_int16
        ? (dw_wgt_bs16_w[1] ? $signed(wgt_rd_data[31:16]) : $signed(wgt_rd_data[15:0]))
        : ((dw_wgt_bs8_w == 2'd0) ? {{8{wgt_rd_data[7]}},  wgt_rd_data[7:0]}  :
           (dw_wgt_bs8_w == 2'd1) ? {{8{wgt_rd_data[15]}}, wgt_rd_data[15:8]} :
           (dw_wgt_bs8_w == 2'd2) ? {{8{wgt_rd_data[23]}}, wgt_rd_data[23:16]} :
                                    {{8{wgt_rd_data[31]}}, wgt_rd_data[31:24]});

    wire        ga_smp_en_w   = rst_n && ga_f_vld[1];
    wire [15:0] gs_nelt_w     = row_dw ? {5'd0, ga_f_nelt[1]}
                              : (row_slice ? ga_nch : {5'd0, ga_f_nelt[1]});
    wire [31:0] gs_left_w     = (gs_left != 16'd0) ? {16'd0, gs_left}
                              : (cfg_int16 ? {15'd0, gs_nelt_w, 1'b0}
                                           : {16'd0, gs_nelt_w});
    wire [31:0] gs_avail_w    = 32'd64 - {30'd0, ga_f_skip[1][1:0]};
    wire [31:0] gs_take_b_w   = (gs_left_w <= gs_avail_w) ? gs_left_w : gs_avail_w;
    wire [6:0]  gs_take_el_w  = cfg_int16 ? gs_take_b_w[7:1] : gs_take_b_w[6:0];
    wire [31:0] gs_dbase_w    = row_dw
        ? ({24'd0, gs_t[1:0], 6'd0} + {25'd0, ga_f_base[1]})
        : ((({31'd0, ga_bank} << row_pix_sh) + {27'd0, gs_p}) << row_bank_sh)
          + {16'd0, gs_t} * {16'd0, cfg_in_c};
    wire [10:0] gs_dst0_w     = gs_dbase_w[10:0] + gs_e[10:0];

    wire [15:0] mac_koff_w    = row_ki - row_kc0;
    wire        mac_step_go_w = row_dw ? (gs_t > row_ki)
                                       : (row_ki < wst_sk || row_wfit);

    genvar gb;
    generate
        for (gb = 0; gb < ROW_N; gb = gb + 1) begin : g_row_lanes
            always @(posedge clk) begin
                if (dw_wgt_we_w) begin
                    if ((gb < 32) && (gb[2:0] == dw_ch_idx[2:0]) && (gb[4:3] == dw_cnt[7:6]))
                        wrow_b[gb][dw_cnt[5:0]] <= dw_wgt_val_w;
                end else if (wrow_we_any_w) begin
                    if ((gb[6:0] >= wrow_wr_c0_w) && ((gb[6:0] - wrow_wr_c0_w) < wrow_wr_nch_w)) begin
                        wrow_b[gb][wrow_wr_k_w] <= cfg_int16
                            ? $signed(wgt_rd_data[({1'b0, gb[4:0] - wrow_wr_c0_w[4:0], 1'b0} + {5'd0, wrow_wr_sk_w}) * 8 +: 16])
                            : {{8{wgt_rd_data[({1'b0, gb[5:0] - wrow_wr_c0_w[5:0]} + {5'd0, wrow_wr_sk_w}) * 8 + 7]}},
                               wgt_rd_data[({1'b0, gb[5:0] - wrow_wr_c0_w[5:0]} + {5'd0, wrow_wr_sk_w}) * 8 +: 8]};
                    end
                end
            end

            always @(posedge clk) begin
                if (ga_smp_en_w && row_dw) begin
                    if ({1'b0, gb[5:0] - gs_dst0_w[5:0]} < gs_take_el_w) begin
                        avec_b[gb][gs_dst0_w[7:6]] <= ga_f_pad[1]
                            ? (cfg_int16 ? $signed(cfg_in_zp) : {{8{cfg_in_zp[7]}}, cfg_in_zp[7:0]})
                            : (cfg_int16
                                ? $signed(act_rd_data[({1'b0, gb[4:0] - gs_dst0_w[4:0], 1'b0} + {5'd0, ga_f_skip[1][1:0]}) * 8 +: 16])
                                : {{8{act_rd_data[({1'b0, gb[5:0] - gs_dst0_w[5:0]} + {5'd0, ga_f_skip[1][1:0]}) * 8 + 7]}},
                                   act_rd_data[({1'b0, gb[5:0] - gs_dst0_w[5:0]} + {5'd0, ga_f_skip[1][1:0]}) * 8 +: 8]});
                    end
                end
            end

            always @(posedge clk) begin
                if (ga_smp_en_w && !row_dw) begin
                    if ({1'b0, gb[5:0] - gs_dst0_w[5:0]} < gs_take_el_w) begin
                        abuf_b[gb][(gs_dst0_w + {5'd0, gb[5:0] - gs_dst0_w[5:0]}) >> 6] <= ga_f_pad[1]
                            ? (cfg_int16 ? $signed(cfg_in_zp) : {{8{cfg_in_zp[7]}}, cfg_in_zp[7:0]})
                            : (cfg_int16
                                ? $signed(act_rd_data[({1'b0, gb[4:0] - gs_dst0_w[4:0], 1'b0} + {5'd0, ga_f_skip[1][1:0]}) * 8 +: 16])
                                : {{8{act_rd_data[({1'b0, gb[5:0] - gs_dst0_w[5:0]} + {5'd0, ga_f_skip[1][1:0]}) * 8 + 7]}},
                                   act_rd_data[({1'b0, gb[5:0] - gs_dst0_w[5:0]} + {5'd0, ga_f_skip[1][1:0]}) * 8 +: 8]});
                    end
                end
            end
        end
    endgenerate

    // ─── Combinational datapath wires for WGT/ACT issue & output packing ───
    // Keeping these pure wires outside always @(posedge clk) prevents Yosys
    // ProcessGenerator from multiplying CaseRules across sequential blocks.
    wire [15:0] wiss_ahead_w   = wst_ik - row_ki;
    wire        wiss_go_w      = (wst_ik < wst_kend) && (wiss_ahead_w < WROW_K[15:0]);
    wire [1:0]  wiss_boff_w    = wst_iss_byte[1:0];
    wire [6:0]  wiss_rem_ch_w  = row_nch - wst_ic;
    wire [7:0]  wiss_rem_b_w   = cfg_int16 ? ({1'b0, wiss_rem_ch_w} << 1) : {1'b0, wiss_rem_ch_w};
    wire [7:0]  wiss_avail_b_w = 8'd64 - {6'd0, wiss_boff_w};
    wire [7:0]  wiss_take_b_w  = (wiss_rem_b_w <= wiss_avail_b_w) ? wiss_rem_b_w : wiss_avail_b_w;
    wire [6:0]  wiss_take_ch_w = cfg_int16 ? wiss_take_b_w[7:1] : wiss_take_b_w[6:0];
    wire        wiss_kend_w    = (wiss_take_b_w == wiss_rem_b_w);

    wire [31:0] dwl_elem_w     = ({16'd0, dwl_tap} * {16'd0, cfg_in_c}) + ({16'd0, oc_group} * 32'd64);
    wire [31:0] dwl_boff_w     = cfg_int16 ? (dwl_elem_w << 1) : dwl_elem_w;
    wire [5:0]  dwl_skip_w     = {4'd0, dwl_boff_w[1:0]};
    wire [6:0]  dwl_cnext_w    = wst_sc + dwl_take_ch_w;

    wire        giss_go_w      = row_dw ? ((ga_t - row_ki) < 16'd3) : ga_busy;
    wire [15:0] giss_poh_w     = row_dw ? sp_oh : gp_oh[ga_p[2:0]];
    wire [15:0] giss_pow_w     = row_dw ? (sp_ow + {11'd0, ga_p}) : gp_ow[ga_p[2:0]];
    wire signed [31:0] giss_ih_w = external_origin_active
        ? ((external_origin_h + $signed({1'b0, giss_poh_w})) * $signed({1'b0, cfg_stride_h})
           + $signed({16'd0, ga_fh}) * $signed({1'b0, cfg_dilation_h})
           - $signed({16'd0, cfg_pad_top}))
        : ($signed({1'b0, tile_oh_origin + giss_poh_w}) * $signed({1'b0, cfg_stride_h})
           + $signed({16'd0, ga_fh}) * $signed({1'b0, cfg_dilation_h})
           - $signed({16'd0, cfg_pad_top}));
    wire signed [31:0] giss_iw_w = external_origin_active
        ? ((external_origin_w + $signed({1'b0, giss_pow_w})) * $signed({1'b0, cfg_stride_w})
           + $signed({16'd0, ga_fw}) * $signed({1'b0, cfg_dilation_w})
           - $signed({16'd0, cfg_pad_left}))
        : ($signed({1'b0, tile_ow_origin + giss_pow_w}) * $signed({1'b0, cfg_stride_w})
           + $signed({16'd0, ga_fw}) * $signed({1'b0, cfg_dilation_w})
           - $signed({16'd0, cfg_pad_left}));
    wire        giss_h_oob_w   = (giss_ih_w < 0) || (giss_ih_w >= $signed({1'b0, cfg_in_h}));
    wire        giss_w_oob_hi  = (giss_iw_w >= $signed({1'b0, cfg_in_w}));
    wire        giss_w_oob_lo  = (giss_iw_w < 0);
    wire        giss_c1_run_w  = !row_dw && !row_slice && !cfg_int16
                              && (cfg_in_c == 16'd1) && (cfg_dilation_w == 8'd1);
    wire signed [31:0] giss_dw_rem_raw = $signed({27'd0, row_pv}) - $signed({27'd0, ga_p});
    wire signed [31:0] giss_dw_rem_w   = (giss_dw_rem_raw < 32'sd1) ? 32'sd1 : giss_dw_rem_raw;
    wire signed [31:0] giss_c1_kw_rem  = $signed({24'd0, cfg_kernel_w}) - $signed({16'd0, ga_fw});
    wire signed [31:0] giss_c1_tp_rem  = $signed({16'd0, ga_tpc}) - $signed({16'd0, ga_t});
    wire signed [31:0] giss_c1_rem_w   = (giss_c1_kw_rem > giss_c1_tp_rem) ? giss_c1_tp_rem : giss_c1_kw_rem;
    wire signed [31:0] giss_rem_w      = row_dw ? giss_dw_rem_w : giss_c1_rem_w;
    wire signed [31:0] giss_niw_w      = -giss_iw_w;
    wire signed [31:0] giss_inb_w      = $signed({1'b0, cfg_in_w}) - giss_iw_w;
    wire signed [31:0] giss_run_raw    = (giss_h_oob_w || giss_w_oob_hi) ? giss_rem_w
                                       : (giss_w_oob_lo ? ((giss_niw_w > giss_rem_w) ? giss_rem_w : giss_niw_w)
                                                        : ((giss_inb_w > giss_rem_w) ? giss_rem_w : giss_inb_w));
    wire [4:0]  giss_run_step  = (giss_run_raw < 32'sd1) ? 5'd1 : giss_run_raw[4:0];
    wire        giss_is_pad_w  = giss_h_oob_w || giss_w_oob_hi || giss_w_oob_lo;
    wire [4:0]  giss_step_new  = (row_dw || giss_c1_run_w) ? giss_run_step : 5'd1;
    wire [15:0] giss_nelt_new  = row_dw ? ({11'd0, giss_step_new} * {9'd0, row_nch})
                               : (giss_c1_run_w ? {11'd0, giss_step_new}
                                  : (row_slice ? ga_nch : cfg_in_c));
    wire [6:0]  giss_base_new  = row_dw ? ({2'd0, ga_p} * {2'd0, row_nch}) : 7'd0;
    wire        giss_tend_new  = row_dw ? ({11'd0, ga_p} + {11'd0, giss_step_new} >= {11'd0, row_pv}) : 1'b1;
    wire [15:0] giss_nbyte_new = cfg_int16 ? (giss_nelt_new << 1) : giss_nelt_new;
    wire [31:0] giss_elem_base = (cfg_tile_h == 16'd0)
        ? ((giss_ih_w[15:0] * cfg_in_w + giss_iw_w[15:0]) * cfg_in_c)
        : ((({16'd0, giss_poh_w} * {8'd0, cfg_stride_h} + {16'd0, ga_fh} * {8'd0, cfg_dilation_h}) * tile_in_w
            + {16'd0, giss_pow_w} * {8'd0, cfg_stride_w} + {16'd0, ga_fw} * {8'd0, cfg_dilation_w}) * cfg_in_c);
    wire [31:0] giss_elem_w    = row_dw ? (giss_elem_base + ({16'd0, oc_group} * 32'd64))
                               : (row_slice ? (giss_elem_base + {16'd0, ga_ch0}) : giss_elem_base);
    wire [31:0] giss_boff_w    = cfg_int16 ? (giss_elem_w << 1) : giss_elem_w;
    wire [5:0]  giss_skip_w    = {4'd0, giss_boff_w[1:0]};
    wire        giss_multi_new = giss_is_pad_w ? (giss_nbyte_new > 16'd64)
                                               : (({10'd0, giss_skip_w} + giss_nbyte_new) > 16'd64);
    wire [15:0] giss_left_new  = giss_multi_new
        ? (giss_is_pad_w ? (giss_nbyte_new - 16'd64) : (giss_nbyte_new - (16'd64 - {10'd0, giss_skip_w})))
        : 16'd0;
    wire        giss_last_w    = (ga_left != 16'd0) ? (ga_left <= 16'd64) : ~giss_multi_new;
    wire        giss_tdone_w   = (ga_left != 16'd0) ? ga_tend_r : giss_tend_new;
    wire [4:0]  giss_step_eff  = (ga_left != 16'd0) ? ga_step_r : giss_step_new;
    wire [15:0] giss_fw_inc    = ga_fw + (row_dw ? 16'd1 : {11'd0, giss_step_eff});
    wire        giss_fw_wrap   = (giss_fw_inc >= {8'd0, cfg_kernel_w});
    wire [15:0] giss_nfh_w     = giss_fw_wrap
                               ? ((ga_fh + 16'd1 >= {8'd0, cfg_kernel_h}) ? 16'd0 : (ga_fh + 16'd1))
                               : ga_fh;
    wire [15:0] giss_nfw_w     = giss_fw_wrap ? 16'd0 : giss_fw_inc;

    // ════════════════════════════════════════════════════════════════════
    // Main FSM
    // ════════════════════════════════════════════════════════════════════

    // Point the gather at a fresh chunk: taps [tap0, tap0+row_tpc) of every
    // packed pixel, landing in abuf bank `bank`. (fh0, fw0) are tracked
    // incrementally without hardware division.
    task row_start_chunk;
        input [15:0] tap0;
        input [15:0] fh0;
        input [15:0] fw0;
        input        bank;
        begin
            ga_tap0 <= tap0;
            ga_tpc  <= ((tap0 + row_tpc) > row_ntap) ? (row_ntap - tap0) : row_tpc;
            ga_bank <= bank;
            ga_p    <= 5'd0;
            ga_t    <= 16'd0;
            gs_p    <= 5'd0;
            gs_t    <= 16'd0;
            gs_e    <= 16'd0;
            gs_left <= 16'd0;
            ga_left <= 16'd0;
            ga_pad_r<= 1'b0;
            ga_fh   <= fh0;
            ga_fw   <= fw0;
            ga_fh0  <= fh0;
            ga_fw0  <= fw0;
            ga_busy <= 1'b1;
            ga_on   <= 1'b1;
        end
    endtask

    // One channel slice of a single tap. Coordinates (tap, ch, fh, fw) are
    // stepped incrementally across slices without division or modulo.
    task row_start_slice;
        input [15:0] k0;
        input [15:0] tap;
        input [15:0] ch;
        input [15:0] fh;
        input [15:0] fw;
        input        bank;
        reg [15:0] room, len_raw, lim, len;
        begin
            room    = (cfg_in_c == 16'd0) ? 16'd0 : (cfg_in_c - ch);
            len_raw = ({3'd0, row_bank_sz} < room) ? {3'd0, row_bank_sz} : room;
            lim     = row_kslice ? row_ksend : row_klen;
            len     = ((k0 + len_raw) > lim) ? (lim - k0) : len_raw;
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
            ga_pad_r<= 1'b0;
            ga_fh   <= fh;
            ga_fw   <= fw;
            ga_fh0  <= fh;
            ga_fw0  <= fw;
            if (cfg_in_c != 16'd0 && (ch + len >= cfg_in_c)) begin
                ga_next_ch  <= 16'd0;
                ga_next_tap <= tap + 16'd1;
                if (fw + 16'd1 >= {8'd0, cfg_kernel_w}) begin
                    ga_next_fw <= 16'd0;
                    ga_next_fh <= fh + 16'd1;
                end else begin
                    ga_next_fw <= fw + 16'd1;
                    ga_next_fh <= fh;
                end
            end else begin
                ga_next_ch  <= ch + len;
                ga_next_tap <= tap;
                ga_next_fh  <= fh;
                ga_next_fw  <= fw;
            end
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
            // Internal state
            tile_y <= 0; tile_x <= 0; oc_group <= 0;
            single_tile_active <= 1'b0;
            single_tile_h_actual <= 16'd0;
            single_tile_w_actual <= 16'd0;
            external_origin_active <= 1'b0;
            halo_expand_active <= 1'b0;
            external_origin_h <= 17'sd0;
            external_origin_w <= 17'sd0;
            external_tile_in_w <= 16'd0;
            oc_groups_total <= 1; k_depth <= 1;
            out_tile_h <= 1; out_tile_w <= 1;
            wgt_word_addr <= 0;
            act_read_issued <= 0; act_byte_sel <= 0;
            ppu_wait_cnt <= 0;
            param_word_idx <= 0; param_read_issued <= 0;
            wgt_base <= 0; act_base <= 0; param_base <= 0; out_base <= 0;
            dw_ch_idx <= 0; dw_cnt <= 0; dw_read_issued <= 0; dw_init_phase <= 0;
            dw_oh <= 0; dw_ow <= 0; dw_fh <= 0; dw_fw <= 0;
            dw_kernel_size <= 0; dw_tap_idx <= 8'd0;
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
            wst_ik <= 16'd0;
            wst_ic <= 7'd0;
            wst_kend <= 16'd0;
            wst_f_kend <= 2'd0;
            row_ki <= 16'd0;
            row_kc0 <= 16'd0;
            ga_pad_r <= 1'b0;
            ga_next_fh0 <= 16'd0;
            ga_next_fw0 <= 16'd0;
            ga_next_tap <= 16'd0;
            ga_next_ch <= 16'd0;
            ga_next_fh <= 16'd0;
            ga_next_fw <= 16'd0;
            ks0_tap_r <= 16'd0;
            ks0_ch_r <= 16'd0;
            ks0_fh_r <= 16'd0;
            ks0_fw_r <= 16'd0;
            begin : rst_row_map
                integer rl;
                for (rl = 0; rl < ROW_N; rl = rl + 1) begin
                    lane_c_r[rl] <= 6'd0;
                    lane_p_r[rl] <= 3'd0;
                end
            end
            ga_on <= 1'b0;
            ga_busy <= 1'b0;
            row_slice <= 1'b0;
            ga_f_vld <= 2'd0;
            ga_f_step[0] <= 5'd1; ga_f_step[1] <= 5'd1;
            wst_on <= 1'b0;
            wst_f_vld <= 2'd0;
            dwl_on <= 1'b0;
            ppu_npix <= 5'd1;
            ppu_nch <= 7'd1;
            row_scatter <= 1'b0;
            scatter_p <= 5'd0;
            scatter_bi <= 8'd0;
            row_push_sub <= 3'd0;
            row_cap_sub  <= 3'd0;
            row_cap_tgt  <= 4'd1;
            vec_ppu_sub  <= 3'd0;
            vec_cap_sub  <= 3'd0;
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
            pool_vec_ch_base <= 0; pool_vec_div_lane <= 0;
            pool_vec_pix_idx <= 0; pool_vec_batch_n <= 1;
            // Add resets
            add_M_A <= 0; add_M_B <= 0; add_S_A <= 0; add_S_B <= 0;
            add_val_a <= 0; add_val_b <= 0;
            add_elem_cnt <= 0; add_total_elems <= 0; add_tile_elem_cnt <= 0;
            add_rd_phase <= 0; add_param_phase <= 0; add_param_idx <= 0;
            add_wb_phase <= 0; add_wb_addr <= 0; add_wb_bytesel <= 0; add_wb_byte <= 0;
            add_vec_active <= 1'b0; add_vec_a <= 0; add_vec_b <= 0;
            // Resize resets
            rsz_oh <= 0; rsz_ow <= 0; rsz_ch <= 0;
            rsz_v00 <= 0; rsz_v01 <= 0; rsz_v10 <= 0; rsz_v11 <= 0;
            rsz_frac_h <= 0; rsz_frac_w <= 0;
            rsz_ih0 <= 0; rsz_iw0 <= 0; rsz_ih1 <= 0; rsz_iw1 <= 0;
            rsz_rd_phase <= 0; rsz_wb_phase <= 0;
            rsz_wb_addr <= 0; rsz_wb_bytesel <= 0; rsz_wb_byte <= 0;
            rsz_rcp_h <= 41'd0; rsz_rcp_w <= 41'd0; rsz_rcp_hm1 <= 41'd0; rsz_rcp_wm1 <= 41'd0;
            rsz_rem_h <= 17'd0; rsz_rem_w <= 17'd0; rsz_rem_hm1 <= 17'd0; rsz_rem_wm1 <= 17'd0;
            rsz_q_h   <= 40'd0; rsz_q_w   <= 40'd0; rsz_q_hm1   <= 40'd0; rsz_q_wm1   <= 40'd0;
            rsz_div_cnt <= 6'd0;
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
            ppu_valid_w  <= {PPU_N{1'b0}};

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
            wst_f_k[1]    <= wst_f_k[0];
            wst_f_c0[1]   <= wst_f_c0[0];
            wst_f_nch[1]  <= wst_f_nch[0];
            wst_f_kend[1] <= wst_f_kend[0];
            wst_f_vld[0]  <= 1'b0;
            wst_f_kend[0] <= 1'b0;

            if (wst_on) begin : row_wgt_issue
                if (wiss_go_w) begin
                    wgt_rd_en     <= 1'b1;
                    wgt_rd_addr   <= wgt_base + wst_iss_byte[WGT_ADDR_W+1:2];
                    wst_f_vld[0]  <= 1'b1;
                    wst_f_skip[0] <= wiss_boff_w;
                    wst_f_k[0]    <= wst_ik;
                    wst_f_c0[0]   <= wst_ic;
                    wst_f_nch[0]  <= wiss_take_ch_w;
                    wst_f_kend[0] <= wiss_kend_w;
                    wst_iss_byte  <= wst_iss_byte + {24'd0, wiss_take_b_w};
                    if (wiss_kend_w) begin
                        wst_ic <= 7'd0;
                        wst_ik <= wst_ik + 16'd1;
                    end else
                        wst_ic <= wst_ic + wiss_take_ch_w;
                end else
                    wgt_rd_en <= 1'b0;
            end

            if (wst_f_vld[1] && wst_f_kend[1]) begin
                wst_sk <= wst_f_k[1] + 16'd1;
            end

            // Depthwise weight load: one beat per spatial tap, G lanes each.
            // Data writes into wrow_b occur in the standalone generate block.
            if (dwl_on) begin : row_dw_wgt
                if (dwl_ph == 2'd0) begin
                    wgt_rd_en   <= 1'b1;
                    wgt_rd_addr <= wgt_base + dwl_boff_w[WGT_ADDR_W+1:2];
                    wst_skip    <= dwl_skip_w;
                    wst_sc      <= 7'd0;
                    dwl_ph      <= 2'd1;
                end else if (dwl_ph == 2'd1) begin
                    wgt_rd_en <= 1'b0;
                    dwl_ph    <= 2'd2;
                end else begin : dw_wgt_take
                    wst_sc <= dwl_cnext_w;
                    if (dwl_cnext_w < row_nch) begin
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
            ga_f_step[1] <= ga_f_step[0];
            ga_f_tend[1] <= ga_f_tend[0];
            ga_f_vld[0]  <= 1'b0;
            ga_f_base[0] <= 7'd0;
            ga_f_nelt[0] <= 11'd0;
            ga_f_step[0] <= 5'd1;
            ga_f_tend[0] <= 1'b0;

            if (ga_on) begin : row_gather_issue
                if (giss_go_w) begin
                    if (ga_left != 16'd0) begin
                        // Continuation beat of a read (or pad fill) wider than 64 bytes.
                        act_rd_en   <= ~ga_pad_r;
                        act_rd_ofm  <= 1'b0;
                        act_rd_addr <= ga_addr[ACT_ADDR_W-1:0];
                        ga_addr     <= ga_addr + 32'd16;
                        ga_left     <= (ga_left > 16'd64) ? (ga_left - 16'd64) : 16'd0;
                        ga_f_vld[0]  <= 1'b1;
                        ga_f_pad[0]  <= ga_pad_r;
                        ga_f_skip[0] <= 6'd0;
                        ga_f_base[0] <= ga_base_r;
                        ga_f_nelt[0] <= ga_nelt_r;
                        ga_f_step[0] <= ga_step_r;
                        ga_f_tend[0] <= (ga_left <= 16'd64) && ga_tend_r;
                    end else begin
                        ga_base_r    <= giss_base_new;
                        ga_nelt_r    <= giss_nelt_new[10:0];
                        ga_tend_r    <= giss_tend_new;
                        ga_pad_r     <= giss_is_pad_w;
                        ga_step_r    <= giss_step_new;
                        ga_f_vld[0]  <= 1'b1;
                        ga_f_pad[0]  <= giss_is_pad_w;
                        ga_f_base[0] <= giss_base_new;
                        ga_f_nelt[0] <= giss_nelt_new[10:0];
                        ga_f_step[0] <= giss_step_new;
                        ga_left      <= giss_left_new;
                        ga_f_tend[0] <= giss_multi_new ? 1'b0 : giss_tend_new;
                        if (giss_is_pad_w) begin
                            act_rd_en    <= 1'b0;
                            ga_f_skip[0] <= 6'd0;
                        end else begin
                            act_rd_en    <= 1'b1;
                            act_rd_ofm   <= 1'b0;
                            act_rd_addr  <= act_base + giss_boff_w[ACT_ADDR_W+1:2];
                            ga_addr      <= {16'd0, act_base + giss_boff_w[ACT_ADDR_W+1:2]} + 32'd16;
                            ga_f_skip[0] <= giss_skip_w;
                        end
                    end
                    if (giss_last_w) begin
                        if (row_dw && !giss_tdone_w)
                            ga_p <= ga_p + giss_step_eff;
                        else begin
                            if (row_dw)
                                ga_p <= 5'd0;
                            ga_fh <= giss_nfh_w;
                            ga_fw <= giss_nfw_w;
                            if (row_dw)
                                ga_t <= ga_t + 16'd1;
                            else if (ga_t + {11'd0, giss_step_eff} >= ga_tpc) begin
                                if (ga_p == 5'd0) begin
                                    ga_next_fh0 <= giss_nfh_w;
                                    ga_next_fw0 <= giss_nfw_w;
                                end
                                ga_t  <= 16'd0;
                                ga_fh <= ga_fh0;
                                ga_fw <= ga_fw0;
                                if (ga_p + 5'd1 >= row_pv)
                                    ga_busy <= 1'b0;
                                else
                                    ga_p <= ga_p + 5'd1;
                            end else
                                ga_t <= ga_t + {11'd0, giss_step_eff};
                        end
                    end
                end else
                    act_rd_en <= 1'b0;
            end

            // Banked writes into abuf_b / avec_b occur in g_row_lanes above;
            // only the closed-form scalar pointers update here.
            if (ga_f_vld[1]) begin : row_gather_sample
                if (gs_left_w <= gs_avail_w) begin
                    gs_e    <= 16'd0;
                    gs_left <= 16'd0;
                    if (row_dw) begin
                        if (ga_f_tend[1])
                            gs_t <= gs_t + 16'd1;
                    end else if (gs_t + {11'd0, ga_f_step[1]} >= ga_tpc) begin
                        gs_t <= 16'd0;
                        gs_p <= gs_p + 5'd1;
                    end else
                        gs_t <= gs_t + {11'd0, ga_f_step[1]};
                end else begin
                    gs_e    <= gs_e + {9'd0, gs_take_el_w};
                    gs_left <= gs_left_w[15:0] - gs_take_b_w[15:0];
                end
            end



            (* parallel_case, full_case *)
            case (state)

            // ══════════════════════════════════════════════════════════════
            S_IDLE: begin
                if (start) begin
                    if (use_row)
                        oc_groups_total <= (cfg_out_c + {9'd0, row_group_ch} - 16'd1)
                                         >> row_group_sh;
                    else if (cfg_op_type == 8'd3 || cfg_op_type == 8'd5)
                        oc_groups_total <= cfg_out_c;
                    else
                        oc_groups_total <= (cfg_out_c + ARRAY_SIZE_16 - 16'd1) >> COL_W;

                    k_depth <= {8'd0, cfg_kernel_h} * {8'd0, cfg_kernel_w} * cfg_in_c;
                    kw_eff  <= (cfg_kernel_w == 8'd0) ? 16'd0
                             : (({8'd0, cfg_kernel_w} - 16'd1)
                                * {8'd0, cfg_dilation_w} + 16'd1);

                    single_tile_active <= cfg_single_tile_en;
                    single_tile_h_actual <= cfg_external_tile_h_actual;
                    single_tile_w_actual <= cfg_external_tile_w_actual;
                    external_origin_active <= cfg_external_origin_en;
                    external_origin_h <= cfg_external_origin_h;
                    external_origin_w <= cfg_external_origin_w;
                    external_tile_in_w <= cfg_external_tile_in_w;
                    halo_expand_active <= cfg_halo_expand_en;

                    if (cfg_single_tile_en) begin
                        out_tile_h <= cfg_external_tile_h_actual;
                        out_tile_w <= cfg_external_tile_w_actual;
                    end else if (cfg_tile_h == 17'd0) begin
                        out_tile_h <= cfg_out_h;
                        out_tile_w <= cfg_out_w;
                    end else begin
                        out_tile_h <= cfg_tile_h;
                        out_tile_w <= cfg_tile_w;
                    end

                    tile_y <= cfg_single_tile_en ? cfg_external_tile_y : 16'd0;
                    tile_x <= cfg_single_tile_en ? cfg_external_tile_x : 16'd0;
                    oc_group <= 0;
                    // A layer with no output channels has nothing to compute;
                    // the OC loop would otherwise still run one group and
                    // stall waiting for a drain that never arrives.
                    if (cfg_out_c == 16'd0) begin
                        state <= S_DONE;
                    end else if (cfg_op_type == 8'd5) begin
                        rsz_rem_h   <= 17'd0;
                        rsz_rem_w   <= 17'd0;
                        rsz_rem_hm1 <= 17'd0;
                        rsz_rem_wm1 <= 17'd0;
                        rsz_q_h     <= 40'd0;
                        rsz_q_w     <= 40'd0;
                        rsz_q_hm1   <= 40'd0;
                        rsz_q_wm1   <= 40'd0;
                        rsz_div_cnt <= 6'd0;
                        state       <= S_RESIZE_RECIP;
                    end else begin
                        state <= S_TILE_SETUP;
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            S_TILE_SETUP: begin
                // Compute base addresses
                wgt_base <= cfg_wgt_base;
                param_base <= cfg_param_base;
                // Input: always use cfg_act_base; tile offset is folded into
                // conv_ih_base/conv_iw_base via tile_oh_origin/tile_ow_origin
                act_base <= cfg_act_base;
                // Reset tile dims to full before clipping.
                // Without this, border-tile clipping persists into next tile
                // (e.g., tile(0,3) clips out_tile_w to 4, tile(1,0) inherits 4).
                // For non-tiled (cfg_tile_h=0): use full output dims.
                if (single_tile_active) begin
                    out_tile_h <= single_tile_h_actual;
                    out_tile_w <= single_tile_w_actual;
                    tile_oh_origin <= tile_y * cfg_tile_h;
                    tile_ow_origin <= tile_x * cfg_tile_w;
                    if (cfg_op_type == 8'd5) begin
                        begin : ext_rsz_tile_origin_blk
                            reg [31:0] ext_out_start_h, ext_out_start_w;
                            reg [71:0] ext_prod_h, ext_prod_w;
                            ext_out_start_h = tile_y * cfg_tile_h;
                            ext_out_start_w = tile_x * cfg_tile_w;
                            if (resize_mode) begin
                                ext_prod_h = ({40'd0, ext_out_start_h} * {56'd0, cfg_in_h - 16'd1})
                                           * {31'd0, rsz_rcp_hm1};
                                ext_prod_w = ({40'd0, ext_out_start_w} * {56'd0, cfg_in_w - 16'd1})
                                           * {31'd0, rsz_rcp_wm1};
                                rsz_tile_ih_origin <= (cfg_out_h > 16'd1) ? ext_prod_h[56:40] : 17'd0;
                                rsz_tile_iw_origin <= (cfg_out_w > 16'd1) ? ext_prod_w[56:40] : 17'd0;
                            end else begin
                                ext_prod_h = ({40'd0, ext_out_start_h} * {56'd0, cfg_in_h})
                                           * {31'd0, rsz_rcp_h};
                                ext_prod_w = ({40'd0, ext_out_start_w} * {56'd0, cfg_in_w})
                                           * {31'd0, rsz_rcp_w};
                                rsz_tile_ih_origin <= (cfg_out_h != 16'd0) ? ext_prod_h[56:40] : 17'd0;
                                rsz_tile_iw_origin <= (cfg_out_w != 16'd0) ? ext_prod_w[56:40] : 17'd0;
                            end
                        end
                    end
                end else if (cfg_tile_h == 17'd0) begin
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
                    // Hoist tile-input origin for Resize via precomputed Q40 reciprocals
                    if (cfg_op_type == 8'd5) begin
                        begin : rsz_tile_origin_blk
                            reg [31:0] out_start_h, out_start_w;
                            reg [71:0] org_prod_h, org_prod_w;
                            out_start_h = tile_y * cfg_tile_h;
                            out_start_w = tile_x * cfg_tile_w;
                            if (resize_mode) begin
                                org_prod_h = ({40'd0, out_start_h} * {56'd0, cfg_in_h - 16'd1})
                                           * {31'd0, rsz_rcp_hm1};
                                org_prod_w = ({40'd0, out_start_w} * {56'd0, cfg_in_w - 16'd1})
                                           * {31'd0, rsz_rcp_wm1};
                                rsz_tile_ih_origin <= (cfg_out_h > 16'd1) ? org_prod_h[56:40] : 17'd0;
                                rsz_tile_iw_origin <= (cfg_out_w > 16'd1) ? org_prod_w[56:40] : 17'd0;
                            end else begin
                                org_prod_h = ({40'd0, out_start_h} * {56'd0, cfg_in_h})
                                           * {31'd0, rsz_rcp_h};
                                org_prod_w = ({40'd0, out_start_w} * {56'd0, cfg_in_w})
                                           * {31'd0, rsz_rcp_w};
                                rsz_tile_ih_origin <= (cfg_out_h != 16'd0) ? org_prod_h[56:40] : 17'd0;
                                rsz_tile_iw_origin <= (cfg_out_w != 16'd0) ? org_prod_w[56:40] : 17'd0;
                            end
                        end
                    end
                end
                `ifdef NPU_SIM_DEBUG
                if (cfg_tile_h != 0)
                    $display("[CMP_TILE] t=%0t tile(%0d,%0d) act_base=%0d out_base=%0d ow_origin_reg=%0d out_tw=%0d tile_in_w=%0d",
                             $time, tile_y, tile_x, act_base, act_base + cfg_out_base,
                             tile_ow_origin, out_tile_w, tile_in_w);
                `endif
                // Clip tile dimensions to image boundary (after reset + origin)
                if (cfg_tile_h != 17'd0 && !single_tile_active) begin
                    if (cfg_tile_w > cfg_out_w - tile_x * cfg_tile_w)
                        out_tile_w <= cfg_out_w - tile_x * cfg_tile_w;
                    if (cfg_tile_h > cfg_out_h - tile_y * cfg_tile_h)
                        out_tile_h <= cfg_out_h - tile_y * cfg_tile_h;
                end
                // Compute the input tile width used by activation addressing.
                if (cfg_tile_h == 17'd0) begin
                    tile_in_w <= cfg_in_w;
                end else if (single_tile_active && external_tile_in_w != 0) begin
                    tile_in_w <= external_tile_in_w;
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

                // The normal schedule starts each spatial tile at OC group 0.
                // Group-major keeps the resident group while walking all tiles.
                if (!group_major_active)
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
                    // In the normal tile-major schedule, SRAM still contains
                    // the previous tile's final group and group 0 must be
                    // restored. Group-major keeps one group resident across
                    // tiles; group 0 was loaded before compute_start.
                    if (cfg_wgt_per_oc != 0 && !group_major_active) begin
                        oc_group_done <= 1'b1;  // Request reload of oc_group 0
                        state <= S_WAIT_WGT_RELOAD;
                    end else begin
                        state <= S_OC_SETUP;
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            S_OC_SETUP: begin
                // Weight base is descriptor-relative. Per-OC reloads replace
                // the contents at cfg_wgt_base; resident groups add offsets.
                if (cfg_wgt_per_oc != 0)
                    wgt_base <= cfg_wgt_base;
                else if (cfg_int16)
                    wgt_base <= cfg_wgt_base
                              + ((oc_group * ARRAY_SIZE_16 * k_depth) >> 1);
                else
                    wgt_base <= cfg_wgt_base
                              + ((oc_group * ARRAY_SIZE_16 * k_depth) >> 2);
                // Param base: 4 words per channel, ARRAY_SIZE channels per group.
                // When per-oc PARAM reload is active (weights reload per oc_group
                // AND out_c exceeds the param SRAM capacity of SPAD_KB*16/4
                // channels), the controller reloads this oc_group's params to
                // param SRAM[0], so param_base must be 0 (not oc_group*64).
                if (cfg_wgt_per_oc != 0 && ({17'd0, cfg_out_c} * 4 > `PARAM_DEPTH_W)) begin
                    param_base <= cfg_param_base;
                end else
                    // 4 words per channel.
                    param_base <= cfg_param_base
                                + oc_group * {11'd0, grp_oc} * 16'd4;

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
                        reg [31:0] group_byte;
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
                        if (!halo_expand_active && (((cfg_op_type != 8'd1)
                                && (((cfg_out_c << (cfg_int16 ? 1 : 0)) & 16'd3) == 0)
                                && ((({9'd0, nch} << (cfg_int16 ? 1 : 0)) & 16'd3) == 0))
                                || ((cfg_op_type == 8'd1)
                                    && rem_oc == cfg_out_c
                                    && cfg_stride_w == 8'd1
                                    && cfg_in_c == cfg_out_c))) begin
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
                        // Chunk the K axis by whole spatial taps without a
                        // 16-bit hardware divider.
                        if (cfg_op_type == 8'd1)
                            tpc = ntap;
                        else if (cfg_in_c == 16'd0)
                            tpc = 16'd1;
                        else if (({16'd0, ntap} * {16'd0, cfg_in_c}) <= {19'd0, bsz})
                            tpc = ntap;
                        else if (({16'd0, cfg_in_c} * 32'd9) <= {19'd0, bsz} && ntap >= 16'd9)
                            tpc = 16'd9;
                        else if (({16'd0, cfg_in_c} << 3) <= {19'd0, bsz} && ntap >= 16'd8)
                            tpc = 16'd8;
                        else if (({16'd0, cfg_in_c} << 2) <= {19'd0, bsz} && ntap >= 16'd4)
                            tpc = 16'd4;
                        else if (({16'd0, cfg_in_c} * 32'd3) <= {19'd0, bsz} && ntap >= 16'd3)
                            tpc = 16'd3;
                        else if (({16'd0, cfg_in_c} << 1) <= {19'd0, bsz} && ntap >= 16'd2)
                            tpc = 16'd2;
                        else
                            tpc = 16'd1;
                        row_tpc <= tpc;
                        row_wfit   <= (klen <= WROW_K[15:0]);
                        row_ki     <= 0;
                        row_kc0    <= 0;
                        mac_kend   <= 0;
                        mac_bank   <= 1'b0;
                        ga_on      <= 1'b0;
                        ga_busy    <= 1'b0;
                        ga_bank    <= 1'b0;
                        ga_f_vld   <= 2'd0;
                        ks0_tap_r  <= 16'd0;
                        ks0_ch_r   <= 16'd0;
                        ks0_fh_r   <= 16'd0;
                        ks0_fw_r   <= 16'd0;
                        wst_on     <= 1'b0;
                        wst_f_vld  <= 2'd0;
                        wst_f_kend <= 2'd0;
                        wst_ik     <= 16'd0;
                        wst_ic     <= 7'd0;
                        wst_kend   <= kslen;
                        wst_sk     <= 16'd0;
                        wst_sc     <= 7'd0;
                        wst_skip   <= 6'd0;
                        wst_iss_byte <= 32'd0;
                        row_ki     <= 16'd0;
                        row_kc0    <= 16'd0;
                        dwl_on     <= 1'b0;
                        dwl_tap    <= 16'd0;
                        dwl_ph     <= 2'd0;
                        row_ppu    <= 1'b0;
                        row_drain  <= 1'b0;
                        row_scatter <= 1'b0;
                        scatter_bi <= 8'd0;
                        row_push_sub <= 3'd0;
                        row_cap_sub  <= 3'd0;
                        row_cap_tgt  <= 4'd1;
                        row_last   <= 1'b0;
                        pk         <= 0;
                        group_byte = oc_group * {25'd0, row_group_ch};
                        if (cfg_int16)
                            group_byte = group_byte << 1;
                        pk_pos     <= group_byte[1:0];
                        pk_addr    <= out_base
                                   + group_byte[ACT_ADDR_W+1:2];
                    end
                    if (cfg_wgt_per_oc == 0) begin
                        // Depthwise is tap-major over all channels, so the
                        // group offset is per tap and folded into dwl's own
                        // address. Dense conv groups are whole 64*k_depth
                        // blocks, so the group base is a plain multiple.
                        if (cfg_op_type == 8'd1)
                            wgt_base <= cfg_wgt_base;
                        else if (cfg_int16)
                            wgt_base <= cfg_wgt_base
                                      + ((oc_group * {9'd0, row_group_ch}
                                          * k_depth) >> 1);
                        else
                            wgt_base <= cfg_wgt_base
                                      + ((oc_group * {9'd0, row_group_ch}
                                          * k_depth) >> 2);
                    end
                    // Reloaded params sit at SRAM[0]. Resident params are
                    // addressed by the 64-channel group.
                    if (({17'd0, cfg_out_c} * 16'd4) > `PARAM_DEPTH_W)
                        param_base <= cfg_param_base;
                    else
                        param_base <= cfg_param_base
                                    + oc_group * {9'd0, row_group_ch} * 16'd4;
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
            S_ROW_WGT: begin : row_wgt_st
                integer lw;
                reg [9:0] nch4, nch2, nch1, g10, gr2, gr1, gr0;
                reg       pb2, pb1, pb0;
                nch4 = {1'b0, row_nch, 2'b00};
                nch2 = {2'd0, row_nch, 1'b0};
                nch1 = {3'd0, row_nch};
                for (lw = 0; lw < ROW_N; lw = lw + 1) begin
                    g10 = lw[9:0];
                    pb2 = (g10 >= nch4);
                    gr2 = pb2 ? (g10 - nch4) : g10;
                    pb1 = (gr2 >= nch2);
                    gr1 = pb1 ? (gr2 - nch2) : gr2;
                    pb0 = (gr1 >= nch1);
                    gr0 = pb0 ? (gr1 - nch1) : gr1;
                    lane_c_r[lw] <= gr0[5:0];
                    lane_p_r[lw] <= {pb2, pb1, pb0};
                end
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
                integer q;
                reg [4:0]  q_eff;
                reg [19:0] ow_sum, w8, w4, w2, w1, r3, r2, r1, r0;
                reg        b3, b2, b1, b0;
                w8 = {1'b0, out_tile_w, 3'b000};
                w4 = {2'd0, out_tile_w, 2'b00};
                w2 = {3'd0, out_tile_w, 1'b0};
                w1 = {4'd0, out_tile_w};
                row_pv <= grp_pv_w;
                grp_oh <= sp_oh;
                grp_ow <= sp_ow;
                for (q = 0; q < 8; q = q + 1) begin
                    q_eff  = (q == 0) ? 5'd0 : ((grp_pv_w < q[4:0]) ? grp_pv_w : q[4:0]);
                    ow_sum = {4'd0, sp_ow} + {15'd0, q_eff};
                    b3     = (ow_sum >= w8);
                    r3     = b3 ? (ow_sum - w8) : ow_sum;
                    b2     = (r3 >= w4);
                    r2     = b2 ? (r3 - w4) : r3;
                    b1     = (r2 >= w2);
                    r1     = b1 ? (r2 - w2) : r2;
                    b0     = (r1 >= w1);
                    r0     = b0 ? (r1 - w1) : r1;
                    gp_oh[q] <= sp_oh + {12'd0, b3, b2, b1, b0};
                    gp_ow[q] <= r0[15:0];
                end
                // Depthwise advances sp itself in S_ROW_PUSH, by the whole
                // packed group, and comes back here for the next one.
                if (!row_dw) begin
                    ow_sum = {4'd0, sp_ow} + {15'd0, grp_pv_w};
                    b3     = (ow_sum >= w8);
                    r3     = b3 ? (ow_sum - w8) : ow_sum;
                    b2     = (r3 >= w4);
                    r2     = b2 ? (r3 - w4) : r3;
                    b1     = (r2 >= w2);
                    r1     = b1 ? (r2 - w2) : r2;
                    b0     = (r1 >= w1);
                    r0     = b0 ? (r1 - w1) : r1;
                    sp_oh <= sp_oh + {12'd0, b3, b2, b1, b0};
                    sp_ow <= r0[15:0];
                    px_remaining <= px_remaining - {11'd0, grp_pv_w};
                end
                row_ki  <= row_kslice ? row_ks0 : 16'd0;
                row_kc0 <= row_kslice ? row_ks0 : 16'd0;
`ifdef DBG_ROW
                $display("[ROWGRP] t=%0t sp=(%0d,%0d) pv=%0d pix=%0d psh=%0d bsh=%0d tpc=%0d ntap=%0d klen=%0d nch=%0d pxrem=%0d otw=%0d",
                         $time, sp_oh, sp_ow, grp_pv_w, row_pix, row_pix_sh, row_bank_sh, row_tpc, row_ntap, row_klen, row_nch, px_remaining, out_tile_w);
`endif
                // Conv re-streams the weights per pixel group unless the whole
                // group already fits the ring.
                if (!row_dw && !row_wfit) begin
                    wst_iss_byte <= 32'd0;
                    wst_ik       <= row_kslice ? row_ks0 : 16'd0;
                    wst_ic       <= 7'd0;
                    wst_kend     <= row_kslice ? row_ksend : row_klen;
                    wst_sk       <= row_kslice ? row_ks0 : 16'd0;
                    wst_sc       <= 7'd0;
                    wst_f_vld    <= 2'd0;
                    wst_f_kend   <= 2'd0;
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
                        row_start_slice(row_kslice ? row_ks0 : 16'd0,
                                        row_kslice ? ks0_tap_r : 16'd0,
                                        row_kslice ? ks0_ch_r  : 16'd0,
                                        row_kslice ? ks0_fh_r  : 16'd0,
                                        row_kslice ? ks0_fw_r  : 16'd0,
                                        1'b0);
                    else
                        row_start_chunk(16'd0, 16'd0, 16'd0, 1'b0);
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
                            row_start_slice(k1, ga_next_tap, ga_next_ch,
                                            ga_next_fh, ga_next_fw, ~ga_bank);
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
                            row_start_chunk(t0 + tv, ga_next_fh0, ga_next_fw0, ~ga_bank);
                        else
                            ga_on <= 1'b0;
                    end
                    state <= S_ROW_MAC;
                end
            end

            S_ROW_MAC: begin
                if (mac_step_go_w) begin : mac_step_blk
                    integer l;
                    reg signed [DATA_W-1:0] pix_act [0:7];
                    reg signed [DATA_W-1:0] w_lo    [0:31];
                    reg signed [DATA_W-1:0] m_act, m_wgt;
                    reg signed [ACC_W-1:0]  m_prod, m_base;
                    reg [ROW_N-1:0]         nz_bmp;
                    reg [10:0]              nlane_m;
                    reg [15:0]              next_end;
                    nlane_m = {6'd0, row_pv} * {4'd0, row_nch};
                    for (l = 0; l < 8; l = l + 1)
                        pix_act[l] = abuf_b[mac_koff_w[5:0]][(((({10'd0, mac_bank} << row_pix_sh) + l[10:0]) << row_bank_sh) + mac_koff_w[10:0]) >> 6];
                    for (l = 0; l < 32; l = l + 1)
                        w_lo[l] = wrow_b[l][row_ki[5:0]];
                    for (l = 0; l < ROW_N; l = l + 1) begin
                        m_act  = row_dw ? avec_b[l][row_ki[1:0]] : pix_act[lane_p_r[l]];
                        m_wgt  = (row_pix_sh == 3'd0) ? wrow_b[l][row_ki[5:0]] : w_lo[lane_c_r[l][4:0]];
                        m_prod = m_act * m_wgt;
                        m_base = (row_ki == 16'd0) ? $signed({ACC_W{1'b0}}) : racc[l];
                        nz_bmp[l] = (l[10:0] < nlane_m) && (m_act != 0);
                        racc[l] <= (l[10:0] < nlane_m) ? (m_base + m_prod) : racc[l];
                    end
                    mac1d_fire  <= 1'b1;
                    mac1d_act   <= row_dw ? {DATA_W{1'b0}} : pix_act[0];
                    mac1d_lanes <= popcnt64(nz_bmp);
                    if (row_dw) begin
                        if (row_ki + 16'd1 >= row_klen) begin
                            ga_on <= 1'b0;
                            state <= S_ROW_PUSH;
                        end else
                            row_ki <= row_ki + 16'd1;
                    end else if (row_ki + 16'd1 >= (row_kslice ? row_ksend : row_klen)) begin
                        if (row_kslice && row_ksend < row_klen) begin
                            next_end = row_ksend + row_ks_cap;
                            if (next_end > row_klen)
                                next_end = row_klen;
                            row_ks0 <= row_ksend;
                            row_ksend <= next_end;
                            ks0_tap_r <= ga_next_tap;
                            ks0_ch_r  <= ga_next_ch;
                            ks0_fh_r  <= ga_next_fh;
                            ks0_fw_r  <= ga_next_fw;
                            row_ks_reload <= 1'b1;
                            wst_on <= 1'b0;
                            ga_on <= 1'b0;
                            oc_group_done <= 1'b1;
                            state <= S_WAIT_WGT_RELOAD;
                        end else
                            state <= S_ROW_PUSH;
                    end else begin
                        row_ki <= row_ki + 16'd1;
                        if (row_ki + 16'd1 >= mac_kend)
                            state <= S_ROW_FILL;
                    end
                end
            end

            S_ROW_PUSH: begin
                if ((row_push_sub != 3'd0) || (!row_ppu && !row_drain && param_ready)) begin : row_push
                    integer l;
                    reg [10:0] nlane_p;
                    reg [3:0]  nbeats_p;
                    reg [5:0]  gl_idx, ch_idx;
                    reg        gl_act;
                    nlane_p  = {6'd0, row_pv} * {4'd0, row_nch};
                    nbeats_p = (nlane_p == 11'd0) ? 4'd1 : ((nlane_p + 11'd7) >> 3);
                    for (l = 0; l < PPU_N; l = l + 1) begin
                        gl_idx = {row_push_sub, l[2:0]};
                        gl_act = ({5'd0, gl_idx} < nlane_p);
                        ch_idx = (row_pix_sh == 3'd0) ? gl_idx : {1'b0, lane_c_r[gl_idx][4:0]};
                        ppu_acc_w[ACC_W*l +: ACC_W]  <= gl_act ? racc[gl_idx] : {ACC_W{1'b0}};
                        ppu_mult_w[15*l +: 15]       <= gl_act ? pc_mult[ch_idx] : 15'd0;
                        ppu_shift_w[6*l +: 6]        <= gl_act ? pc_shift[ch_idx] : 6'd0;
                        ppu_zp_w[16*l +: 16]         <= gl_act ? pc_zp[ch_idx] : 16'd0;
                        ppu_bias_w[ACC_W*l +: ACC_W] <= gl_act ? {pc_bhi[ch_idx], pc_blo[ch_idx]} : {ACC_W{1'b0}};
                        ppu_valid_w[l]               <= gl_act;
                    end
                    if (row_push_sub == 3'd0) begin
                        ppu_acc_in     <= racc[0];
                        ppu_mult_m     <= pc_mult[0];
                        ppu_shift_s    <= pc_shift[0];
                        ppu_zero_point <= pc_zp[0];
                        ppu_bias       <= $signed({pc_bhi[0], pc_blo[0]});
                        ppu_in_valid   <= 1'b1;
`ifdef DBG_ROW
                        $display("[ROWPUSH] t=%0t grp=(%0d,%0d) pv=%0d nch=%0d racc0=%0d racc1=%0d",
                                 $time, grp_oh, grp_ow, row_pv, row_nch, racc[0], racc[1]);
`endif
                        row_ppu     <= 1'b1;
                        row_cap_sub <= 3'd0;
                        row_cap_tgt <= nbeats_p;
                        ppu_oh      <= row_dw ? sp_oh : grp_oh;
                        ppu_ow      <= row_dw ? sp_ow : grp_ow;
                        ppu_nlane   <= {4'd0, row_pv} * {2'd0, row_nch};
                        ppu_npix    <= row_pv;
                        ppu_nch     <= row_nch;
                    end
                    if ({1'b0, row_push_sub} + 4'd1 >= nbeats_p) begin
                        row_push_sub <= 3'd0;
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
                                    ga_on    <= 1'b1;
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
                    end else begin
                        row_push_sub <= row_push_sub + 3'd1;
                    end
                end
            end

            S_ROW_TAIL: begin
                if (!row_ppu && !row_drain)
                    state <= S_OC_NEXT;
            end


            S_OC_NEXT: begin
                if (group_major_active) begin
                    // One OC group has completed for this tile. Keep its
                    // weights resident and advance spatially. At the last
                    // tile, request the next group only after the controller
                    // has written this group's channel slice to DDR.
                    if (cfg_tile_h == 17'd0) begin
                        state <= S_DONE;
                    end else if (tile_x + 1 < cfg_tile_num_w) begin
                        tile_x <= tile_x + 1'b1;
                        tile_done_r <= 1'b1;
                        state <= S_TILE_WAIT_DB;
                    end else if (tile_y + 1 < cfg_tile_num_h) begin
                        tile_x <= 0;
                        tile_y <= tile_y + 1'b1;
                        tile_done_r <= 1'b1;
                        state <= S_TILE_WAIT_DB;
                    end else if (oc_group + 1 >= oc_groups_total) begin
                        state <= S_DONE;
                    end else begin
                        // oc_group_out changes to the requested group. The
                        // controller derives the just-completed store group as
                        // oc_group_out-1 while oc_group_done is asserted.
                        tile_x <= 0;
                        tile_y <= 0;
                        oc_group <= oc_group + 1'b1;
                        tile_done_r <= 1'b1;
                        oc_group_done <= 1'b1;
                        state <= S_WAIT_WGT_RELOAD;
                    end
                end else if (oc_group + 1 >= oc_groups_total) begin
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
                        wst_ik <= row_ks0;
                        wst_ic <= 7'd0;
                        wst_kend <= row_ksend;
                        wst_sk <= row_ks0;
                        wst_sc <= 7'd0;
                        wst_f_vld <= 2'd0;
                        wst_f_kend <= 2'd0;
                        wst_on <= 1'b1;
                        row_start_slice(row_ks0, ks0_tap_r, ks0_ch_r,
                                        ks0_fh_r, ks0_fw_r, 1'b0);
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
                    end else if (group_major_active) begin
                        // Weight, params, and tile(0,0) activation are all
                        // resident when the controller acknowledges a
                        // group-major reload.
                        state <= S_TILE_SETUP;
                    end else begin
                        state <= S_OC_SETUP;
                    end
                end
            end

            S_TILE_NEXT: begin
                if (single_tile_active) begin
                    state <= S_DONE;
                end else if (cfg_tile_h == 17'd0) begin
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
                            add_elem_cnt <= add_elem_cnt
                                            + (add_vec_active
                                               ? (cfg_int16 ? 16'd32 : 16'd64)
                                               : 16'd1);
                            add_rd_phase <= 0;
                            if (!is_concat
                                && ({16'd0, cfg_tile_h} * {16'd0, cfg_tile_w}
                                    * {16'd0, cfg_out_c})
                                   >= (cfg_int16 ? 32'd32 : 32'd64)) begin
                                add_vec_active <= 1'b1;
                                state <= S_ADD_VEC_READ_A;
                            end else begin
                                add_vec_active <= 1'b0;
                                state <= S_ADD_READ_A;
                            end
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
                // Load kh*kw weights for channel group into wrow_b banks
                case (dw_init_phase)
                2'd0: begin
                    // Phase 0: setup
                    dw_kernel_size <= {4'd0, cfg_kernel_h[4:0]} * {4'd0, cfg_kernel_w[4:0]};
                    begin : dw_wgt_addr_setup
                        reg [15:0] wgt_elem_start;
                        reg [15:0] wgt_byte_start;
                        if (dw_stream)
                            wgt_elem_start = (dw_ch_base + {11'd0, dw_ch_idx[4:0]} - dw_grp_base)
                                           * {11'd0, cfg_kernel_h[4:0]}
                                           * {11'd0, cfg_kernel_w[4:0]};
                        else
                            wgt_elem_start = (dw_ch_base + {11'd0, dw_ch_idx[4:0]})
                                           * {11'd0, cfg_kernel_h[4:0]}
                                           * {11'd0, cfg_kernel_w[4:0]};
                        wgt_byte_start = cfg_int16 ? (wgt_elem_start << 1) : wgt_elem_start;
                        wgt_word_addr <= {7'd0, wgt_base} + wgt_byte_start[15:2];
                        dw_wgt_bsel_base <= wgt_byte_start[1:0];
                    end
                    dw_init_phase <= 2'd1;
                end
                2'd1: begin
                    // Phase 1: issue first SRAM read
                    wgt_rd_en   <= 1'b1;
                    wgt_rd_addr <= wgt_word_addr;
                    dw_read_issued <= 1'b0;
                    dw_init_phase <= 2'd2;
                end
                2'd2: begin
                    // Phase 2+: weight feeding loop (writes to wrow_b via g_row_lanes)
                    if (!dw_read_issued) begin
                        dw_read_issued <= 1'b1;
                    end else begin
                        dw_cnt <= dw_cnt + 16'd1;

                        if (dw_cnt + 16'd1 >= {7'd0, dw_kernel_size}) begin
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
                                if (({dw_cnt[0], 1'b0} + dw_wgt_bsel_base) >= 2'd2) begin
                                    wgt_word_addr <= wgt_word_addr + 1;
                                    wgt_rd_en    <= 1'b1;
                                    wgt_rd_addr  <= wgt_word_addr + 1'b1;
                                    dw_read_issued <= 1'b0;
                                end
                            end else begin
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
                dw_cnt <= 16'd0;
                param_word_idx <= 0;
                param_read_issued <= 1'b0;
                cache_issue <= 9'd0;
                state <= S_DW_PARAM;
            end

            S_DW_PARAM: begin
                // dw_nch channels × 4 words, from this group's first channel.
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= cfg_param_base
                                   + (dw_ch_base << 2) + cache_issue[8:0];
                    param_read_issued <= 1'b1;
                    act_read_issued <= 1'b0;
                end else if (!act_read_issued) begin
                    act_read_issued <= 1'b1;
                end else begin
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
                dw_oh <= 16'd0;
                dw_ow <= 16'd0;
                dw_fh <= 5'd0;
                dw_fw <= 5'd0;
                dw_tap_idx <= 8'd0;
                dw_read_issued <= 1'b0;
                dw_kdone <= 1'b0;
                dw_iss_vld <= 1'b0;
                dw_ret_vld <= 1'b0;
                state <= S_DW_ACT_STREAM;
            end

            S_DW_ACT_STREAM: begin
                // One kernel tap per cycle once the pipe is full. dw_fh/dw_fw
                // is the next tap to issue; dw_ret_* is the tap whose SRAM
                // beat is valid this cycle. Accumulates directly in racc[0..7]
                // using weights cached in wrow_b.
                begin : dw_act_stream_blk
                    reg signed [15:0] ih_s, iw_s;
                    reg               is_pad, is_last, is_first, consumed_last;
                    reg [1:0]         bsel;
                    reg signed [DATA_W-1:0] zp, act_ci, wgt_ci;
                    reg [ARRAY_SIZE-1:0] dw_nz_bmp;
                    integer ci;
                    reg [SRAM_B_W-1:0] sh;
                    reg [ACT_ADDR_W+15:0] elem_off, byte_off;

                    zp = cfg_int16 ? cfg_in_zp
                                   : {{8{cfg_in_zp[7]}}, cfg_in_zp[7:0]};
                    consumed_last = 1'b0;
                    if (dw_ret_vld) begin
                        sh = act_rd_data >> (dw_ret_bsel * 8);
                        dw_nz_bmp = {ARRAY_SIZE{1'b0}};
                        for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1) begin
                            act_ci = cfg_int16 ? $signed(sh[16*ci +: 16])
                                               : {{8{sh[8*ci+7]}}, sh[8*ci +: 8]};
                            case (dw_tap_idx[7:6])
                                2'd0:    wgt_ci = wrow_b[{3'd0, ci[2:0]}][dw_tap_idx[5:0]];
                                2'd1:    wgt_ci = wrow_b[{3'd1, ci[2:0]}][dw_tap_idx[5:0]];
                                2'd2:    wgt_ci = wrow_b[{3'd2, ci[2:0]}][dw_tap_idx[5:0]];
                                default: wgt_ci = wrow_b[{3'd3, ci[2:0]}][dw_tap_idx[5:0]];
                            endcase
                            racc[ci] <= (ci[4:0] < dw_nch)
                                ? ((dw_ret_clr ? $signed({ACC_W{1'b0}}) : racc[ci]) + act_ci * wgt_ci)
                                : $signed({ACC_W{1'b0}});
                            dw_nz_bmp[ci] = (ci[4:0] < dw_nch) && (act_ci != 0);
                        end
                        mac1d_fire  <= 1'b1;
                        mac1d_lanes <= popcnt64({56'd0, dw_nz_bmp});
                        dw_tap_idx  <= dw_ret_last ? 8'd0 : (dw_tap_idx + 8'd1);
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
                             - $signed({1'b0, cfg_pad_top[7:0]})
                             + $signed({1'b0, dw_fh}) * $signed({1'b0, cfg_dilation_h});
                        iw_s = $signed({1'b0, tile_ow_origin + dw_ow}) * $signed({1'b0, cfg_stride_w[7:0]})
                             - $signed({1'b0, cfg_pad_left[7:0]})
                             + $signed({1'b0, dw_fw}) * $signed({1'b0, cfg_dilation_w});
                        is_pad = (ih_s < 0) || (ih_s >= $signed({1'b0, cfg_in_h}))
                              || (iw_s < 0) || (iw_s >= $signed({1'b0, cfg_in_w}));
                        is_last = (dw_fw + 5'd1 >= cfg_kernel_w[4:0])
                               && (dw_fh + 5'd1 >= cfg_kernel_h[4:0]);
                        is_first = (dw_fh == 5'd0) && (dw_fw == 5'd0);
                        // A pad has to follow the reads already in the pipe,
                        // and the MAC accepts one tap per cycle.
                        if (is_pad && !dw_ret_vld && !dw_iss_vld) begin
                            dw_nz_bmp = {ARRAY_SIZE{1'b0}};
                            for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1) begin
                                case (dw_tap_idx[7:6])
                                    2'd0:    wgt_ci = wrow_b[{3'd0, ci[2:0]}][dw_tap_idx[5:0]];
                                    2'd1:    wgt_ci = wrow_b[{3'd1, ci[2:0]}][dw_tap_idx[5:0]];
                                    2'd2:    wgt_ci = wrow_b[{3'd2, ci[2:0]}][dw_tap_idx[5:0]];
                                    default: wgt_ci = wrow_b[{3'd3, ci[2:0]}][dw_tap_idx[5:0]];
                                endcase
                                racc[ci] <= (ci[4:0] < dw_nch)
                                    ? ((is_first ? $signed({ACC_W{1'b0}}) : racc[ci]) + zp * wgt_ci)
                                    : $signed({ACC_W{1'b0}});
                                dw_nz_bmp[ci] = (ci[4:0] < dw_nch) && (zp != 0);
                            end
                            mac1d_fire  <= 1'b1;
                            mac1d_lanes <= popcnt64({56'd0, dw_nz_bmp});
                            dw_tap_idx  <= is_last ? 8'd0 : (dw_tap_idx + 8'd1);
                            if (is_last) begin
                                dw_kdone <= 1'b1;
                                state <= S_DW_PPU_WAIT;
                                ppu_wait_cnt <= 0;
                            end else if (dw_fw + 5'd1 >= cfg_kernel_w[4:0]) begin
                                dw_fw <= 5'd0;
                                dw_fh <= dw_fh + 5'd1;
                            end else
                                dw_fw <= dw_fw + 5'd1;
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
                                    row_rel = {2'd0, dw_oh} * {10'd0, cfg_stride_h}
                                            + {13'd0, dw_fh} * {10'd0, cfg_dilation_h};
                                    col_rel = {2'd0, dw_ow} * {10'd0, cfg_stride_w}
                                            + {13'd0, dw_fw} * {10'd0, cfg_dilation_w};
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
                            if (is_last)
                                dw_kdone <= 1'b1;
                            else if (dw_fw + 5'd1 >= cfg_kernel_w[4:0]) begin
                                dw_fw <= 5'd0;
                                dw_fh <= dw_fh + 5'd1;
                            end else
                                dw_fw <= dw_fw + 5'd1;
                        end
                    end
                end
            end

            S_DW_PPU_WAIT: begin
                // Feed racc[0..7] into 8-lane PPU, then write back PPU output
                if (ppu_wait_cnt == 0) begin : dw_ppu_push_blk
                    integer l;
                    ppu_acc_in     <= racc[0];
                    ppu_mult_m     <= pc_mult[0];
                    ppu_shift_s    <= pc_shift[0];
                    ppu_zero_point <= pc_zp[0];
                    ppu_bias       <= $signed({pc_bhi[0], pc_blo[0]});
                    ppu_in_valid   <= 1'b1;
                    for (l = 0; l < PPU_N; l = l + 1) begin
                        ppu_acc_w[ACC_W*l +: ACC_W]  <= (l[4:0] < dw_nch) ? racc[l]                    : {ACC_W{1'b0}};
                        ppu_mult_w[15*l +: 15]       <= (l[4:0] < dw_nch) ? pc_mult[l]                 : 15'd0;
                        ppu_shift_w[6*l +: 6]        <= (l[4:0] < dw_nch) ? pc_shift[l]                : 6'd0;
                        ppu_zp_w[16*l +: 16]         <= (l[4:0] < dw_nch) ? pc_zp[l]                   : 16'd0;
                        ppu_bias_w[ACC_W*l +: ACC_W] <= (l[4:0] < dw_nch) ? {pc_bhi[l], pc_blo[l]}     : {ACC_W{1'b0}};
                        ppu_valid_w[l]               <= (l[4:0] < dw_nch);
                    end
                    ppu_wait_cnt <= 1;
                end else begin
                    if (ppu_vout_w[0] || ppu_out_valid) begin
                        // The ARRAY_SIZE channels of this pixel are contiguous
                        // in NHWC and fit in one 64-byte beat (8×INT16 = 16 B).
                        begin : dw_wide_wb
                            integer l;
                            reg [31:0]  elem_off, boff;
                            reg [5:0]   nbytes;
                            reg [127:0] b16;
                            reg [63:0]  b8;
                            reg signed [DATA_W-1:0] ov;
                            elem_off = ({16'd0, dw_oh} * {16'd0, out_tile_w} + {16'd0, dw_ow}) * {16'd0, cfg_out_c}
                                     + {16'd0, dw_ch_base};
                            boff     = cfg_int16 ? (elem_off << 1) : elem_off;
                            nbytes   = cfg_int16 ? {dw_nch, 1'b0} : {1'b0, dw_nch};
                            for (l = 0; l < ARRAY_SIZE; l = l + 1) begin
                                ov = (l == 0 && !ppu_vout_w[0]) ? ppu_out_data : ppu_out_w[DATA_W*l +: DATA_W];
                                b16[16*l +: 16] = (l[4:0] < dw_nch) ? ov      : 16'd0;
                                b8[8*l +: 8]    = (l[4:0] < dw_nch) ? ov[7:0] : 8'd0;
                            end
                            act_wr_en   <= 1'b1;
                            act_wr_addr <= out_base + boff[ACT_ADDR_W+1:2];
                            act_wr_wide <= ({384'd0, cfg_int16 ? b16 : {64'd0, b8}}) << {boff[1:0], 3'b000};
                            act_wr_mask <= ((64'h1 << nbytes) - 64'h1) << boff[1:0];
                        end
                        dw_kdone   <= 1'b0;
                        dw_iss_vld <= 1'b0;
                        dw_ret_vld <= 1'b0;
                        dw_fh <= 5'd0;
                        dw_fw <= 5'd0;
                        dw_tap_idx <= 8'd0;
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
                // A complete aligned channel beat can use the wide SRAM and
                // PPU ports. Packed stream slices retain the scalar path.
                // AvgPool shares the vector accumulator and serializes only
                // its final reciprocal multiply to avoid 64 divider units.
                if (!pool_stream
                    && ((cfg_int16 && cfg_out_c >= 16'd32
                         && cfg_out_c[4:0] == 5'd0)
                        || (!cfg_int16 && cfg_out_c >= 16'd64
                            && cfg_out_c[5:0] == 6'd0)
                        || (!cfg_int16 && !pool_mode
                            && ((cfg_out_c == 16'd16
                                 && out_tile_w[1:0] == 2'd0)
                                || (cfg_out_c == 16'd32
                                    && out_tile_w[0] == 1'b0))
                            && pool_cfg_h == 4'd2 && pool_cfg_w == 4'd2
                            && pool_cfg_sh == 4'd2 && pool_cfg_sw == 4'd2
                            && cfg_pad_top == 8'd0 && cfg_pad_left == 8'd0))) begin
                    pool_vec_ch_base <= 0;
                    pool_vec_pix_idx <= 0;
                    pool_vec_batch_n <=
                        (out_tile_w < {13'd0, pool_vec_pix_cap})
                            ? out_tile_w[2:0] : pool_vec_pix_cap;
                    cache_issue <= 0;
                    cache_cap <= 0;
                    state <= S_POOL_VEC_PARAM;
                end else begin
                    state <= S_POOL_CH_SETUP;
                end
            end

            // Fill one channel group's PPU parameters into the shared cache.
            // The read/capture overlap matches the row-engine SRAM pipeline.
            S_POOL_VEC_PARAM: begin : pool_vec_param_blk
                if (cache_issue < pv_tgt_w) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= cfg_param_base
                                     + ({pool_vec_ch_base, 2'b00})
                                     + cache_issue;
                    cache_issue <= cache_issue + 1;
                end
                if (cache_issue >= 9'd2 && cache_cap < pv_tgt_w) begin
                    cache_cap <= cache_cap + 1;
                    if (cache_cap + 1 >= pv_tgt_w) begin
                        pool_oh <= 0;
                        pool_ow <= 0;
                        pool_fh <= 0;
                        pool_fw <= 0;
                        pool_rd_phase <= 0;
                        pool_count <= 0;
                        for (i = 0; i < `MAC_LANES; i = i + 1)
                            racc[i] <= pool_mode ? $signed({ACC_W{1'b0}}) : $signed({1'b1, {(ACC_W-1){1'b0}}});
                        state <= S_POOL_VEC_READ;
                    end
                end
            end

            S_POOL_VEC_READ: begin : pool_vec_read_blk
                reg signed [15:0] ih_s, iw_s;
                reg               is_oob;
                reg [31:0]        elem_off, byte_off;
                ih_s = $signed({1'b0, tile_oh_origin + pool_oh})
                       * $signed({1'b0, pool_sh})
                       - $signed({1'b0, cfg_pad_top[7:0]})
                       + $signed({1'b0, pool_fh});
                iw_s = $signed({1'b0, tile_ow_origin + pool_ow
                                + (pool_vec_multi
                                   ? {13'd0, pool_vec_pix_idx} : 16'd0)})
                       * $signed({1'b0, pool_sw})
                       - $signed({1'b0, cfg_pad_left[7:0]})
                       + $signed({1'b0, pool_fw});
                is_oob = (ih_s < 0) || (ih_s >= $signed({1'b0, cfg_in_h}))
                      || (iw_s < 0) || (iw_s >= $signed({1'b0, cfg_in_w}));
                if (is_oob) begin
                    if ({4'd0, pool_fw} + 1 >= pool_kw) begin
                        pool_fw <= 0;
                        if ({4'd0, pool_fh} + 1 >= pool_kh) begin
                            if (pool_mode && pool_count > 0) begin
                                pool_vec_div_lane <= 0;
                                state <= S_POOL_VEC_DIV;
                            end else begin
                                vec_ppu_sub <= 3'd0;
                                vec_cap_sub <= 3'd0;
                                state <= S_POOL_VEC_PPU;
                            end
                        end else
                            pool_fh <= pool_fh + 1;
                    end else begin
                        pool_fw <= pool_fw + 1;
                    end
                end else if (pool_rd_phase == 0) begin
                    if (cfg_tile_h == 17'd0) begin
                        elem_off = (ih_s[15:0] * cfg_in_w * cfg_in_c)
                                   + (iw_s[15:0] * cfg_in_c)
                                   + pool_vec_ch_base;
                    end else begin : pool_vec_addr_2d_blk
                        reg [17:0] row_rel, col_rel;
                        row_rel = {2'd0, pool_oh} * {10'd0, pool_sh}
                                  + {14'd0, pool_fh};
                        col_rel = ({2'd0, pool_ow}
                                   + (pool_vec_multi
                                      ? {15'd0, pool_vec_pix_idx} : 18'd0))
                                  * {10'd0, pool_sw}
                                  + {14'd0, pool_fw};
                        elem_off = (row_rel[15:0] * tile_in_w
                                    + col_rel[15:0]) * cfg_in_c
                                   + pool_vec_ch_base;
                    end
                    byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                    pool_rd_phase <= 1;
                end else if (pool_rd_phase == 1) begin
                    pool_rd_phase <= 2;
                end else begin
                    pool_rd_phase <= 0;
                    state <= S_POOL_VEC_ACC;
                end
            end

            S_POOL_VEC_ACC: begin : pool_vec_acc_blk
                integer l;
                reg                     pen;
                reg [5:0]               psrc;
                reg signed [ACC_W-1:0]  pv8, pv16, pval;
                for (l = 0; l < `MAC_LANES; l = l + 1) begin
                    pen  = pool_vec_multi
                         ? ((cfg_out_c == 16'd16) ? (pool_vec_pix_idx == {1'b0, l[5:4]})
                                                  : (pool_vec_pix_idx == {2'd0, l[5]}))
                         : (!cfg_int16 || (l < 32));
                    psrc = pool_vec_multi
                         ? ((cfg_out_c == 16'd16) ? {2'd0, l[3:0]} : {1'b0, l[4:0]})
                         : l[5:0];
                    pv8  = $signed({{(ACC_W-8){act_rd_data[{psrc, 3'b000} + 7]}}, act_rd_data[{psrc, 3'b000} +: 8]});
                    pv16 = (l < 32)
                         ? $signed({{(ACC_W-16){act_rd_data[16*l[4:0] + 15]}}, act_rd_data[16*l[4:0] +: 16]})
                         : $signed({ACC_W{1'b0}});
                    pval = (!pool_vec_multi && cfg_int16) ? pv16 : pv8;
                    racc[l] <= pen
                        ? (pool_mode ? (racc[l] + pval)
                                     : ((pval > racc[l]) ? pval : racc[l]))
                        : racc[l];
                end
                pool_count <= pool_count + 1;
                if ({4'd0, pool_fw} + 1 >= pool_kw) begin
                    pool_fw <= 0;
                    if ({4'd0, pool_fh} + 1 >= pool_kh) begin
                        if (pool_vec_multi
                            && pool_vec_pix_idx + 1 < pool_vec_batch_n) begin
                            pool_vec_pix_idx <= pool_vec_pix_idx + 1'b1;
                            pool_fh <= 0;
                            pool_count <= 0;
                            state <= S_POOL_VEC_READ;
                        end else if (pool_mode) begin
                            pool_vec_div_lane <= 0;
                            state <= S_POOL_VEC_DIV;
                        end else begin
                            vec_ppu_sub <= 3'd0;
                            vec_cap_sub <= 3'd0;
                            state <= S_POOL_VEC_PPU;
                        end
                    end else begin
                        pool_fh <= pool_fh + 1;
                        state <= S_POOL_VEC_READ;
                    end
                end else begin
                    pool_fw <= pool_fw + 1;
                    state <= S_POOL_VEC_READ;
                end
            end

            // One reciprocal multiply per cycle keeps AvgPool vectorized at
            // the SRAM boundary without replicating a wide divider datapath.
            S_POOL_VEC_DIV: begin : pool_vec_div_blk
                reg signed [ACC_W-1:0] rounded;
                reg signed [ACC_W-1:0] half_count;
                reg signed [71:0] prod;
                if (pool_count == 17'd1) begin
                    racc[pool_vec_div_lane[5:0]] <= racc[pool_vec_div_lane[5:0]];
                end else begin
                    half_count = pool_count >> 1;
                    if (racc[pool_vec_div_lane[5:0]] >= 0)
                        rounded = racc[pool_vec_div_lane[5:0]] + half_count;
                    else
                        rounded = racc[pool_vec_div_lane[5:0]] - half_count;
                    prod = rounded * $signed(recip_pool(pool_count[6:0]));
                    if (pool_count == 17'd2)
                        racc[pool_vec_div_lane[5:0]] <= prod >>> 31;
                    else
                        racc[pool_vec_div_lane[5:0]] <= prod >>> 32;
                end
                if (pool_vec_div_lane + 1
                    >= (cfg_int16 ? 7'd32 : 7'd64)) begin
                    vec_ppu_sub <= 3'd0;
                    vec_cap_sub <= 3'd0;
                    state <= S_POOL_VEC_PPU;
                end else begin
                    pool_vec_div_lane <= pool_vec_div_lane + 1;
                end
            end

            S_POOL_VEC_PPU: begin : pool_vec_ppu_blk
                integer l;
                reg [15:0] pv_tot, pv_nbeats16;
                reg [3:0]  pv_nbeats;
                reg [5:0]  gl_idx, pch_idx;
                reg        pv_act;
                pv_tot      = pool_vec_multi ? ({13'd0, pool_vec_batch_n} * cfg_out_c)
                                             : (cfg_int16 ? 16'd32 : 16'd64);
                pv_nbeats16 = (pv_tot + 16'd7) >> 3;
                pv_nbeats   = (pv_nbeats16 == 16'd0) ? 4'd1 : pv_nbeats16[3:0];
                for (l = 0; l < PPU_N; l = l + 1) begin
                    gl_idx  = {vec_ppu_sub, l[2:0]};
                    pv_act  = ({10'd0, gl_idx} < pv_tot);
                    pch_idx = (pool_vec_multi && cfg_out_c == 16'd16) ? {2'd0, gl_idx[3:0]}
                            : ((pool_vec_multi && cfg_out_c == 16'd32) ? {1'b0, gl_idx[4:0]} : gl_idx);
                    ppu_acc_w[ACC_W*l +: ACC_W]  <= pv_act ? racc[gl_idx] : {ACC_W{1'b0}};
                    ppu_mult_w[15*l +: 15]       <= pv_act ? pc_mult[pch_idx] : 15'd0;
                    ppu_shift_w[6*l +: 6]        <= pv_act ? pc_shift[pch_idx] : 6'd0;
                    ppu_zp_w[16*l +: 16]         <= pv_act ? pc_zp[pch_idx] : 16'd0;
                    ppu_bias_w[ACC_W*l +: ACC_W] <= pv_act ? {pc_bhi[pch_idx], pc_blo[pch_idx]} : {ACC_W{1'b0}};
                    ppu_valid_w[l]               <= pv_act;
                end
                if (ppu_vout_w[0]) begin
                    if (vec_cap_sub == 3'd0)
                        obuf <= {896'd0, ppu_out_w};
                    else
                        obuf[{vec_cap_sub, 7'd0} +: 128] <= ppu_out_w;
                    vec_cap_sub <= vec_cap_sub + 3'd1;
                end
                if ({1'b0, vec_ppu_sub} + 4'd1 >= pv_nbeats) begin
                    vec_ppu_sub <= 3'd0;
                    state <= S_POOL_VEC_WAIT;
                end else begin
                    vec_ppu_sub <= vec_ppu_sub + 3'd1;
                end
            end

            S_POOL_VEC_WAIT: begin : pool_vec_wait_blk
                reg [15:0] pv_tot, pv_nbeats16;
                reg [3:0]  pv_nbeats;
                pv_tot      = pool_vec_multi ? ({13'd0, pool_vec_batch_n} * cfg_out_c)
                                             : (cfg_int16 ? 16'd32 : 16'd64);
                pv_nbeats16 = (pv_tot + 16'd7) >> 3;
                pv_nbeats   = (pv_nbeats16 == 16'd0) ? 4'd1 : pv_nbeats16[3:0];
                if (ppu_vout_w[0]) begin
                    if (vec_cap_sub == 3'd0)
                        obuf <= {896'd0, ppu_out_w};
                    else
                        obuf[{vec_cap_sub, 7'd0} +: 128] <= ppu_out_w;
                    if ({1'b0, vec_cap_sub} + 4'd1 >= pv_nbeats) begin
                        vec_cap_sub <= 3'd0;
                        state <= S_POOL_VEC_WB;
                    end else begin
                        vec_cap_sub <= vec_cap_sub + 3'd1;
                    end
                end
            end

            S_POOL_VEC_WB: begin : pool_vec_wb_blk
                integer l;
                reg [31:0]  elem_off, byte_off;
                reg [511:0] w16, w8;
                elem_off = (pool_oh * out_tile_w + pool_ow) * cfg_out_c
                           + pool_vec_ch_base;
                byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
                w16 = obuf[511:0];
                for (l = 0; l < 64; l = l + 1)
                    w8[8*l +: 8] = obuf[DATA_W*l +: 8];
                act_wr_en   <= 1'b1;
                act_wr_addr <= out_base + byte_off[ACT_ADDR_W+1:2];
                act_wr_wide <= cfg_int16 ? w16 : w8;
                if (pool_vec_multi
                    && pool_vec_batch_n * cfg_out_c < 64)
                    act_wr_mask <= (64'h1
                                    << (pool_vec_batch_n * cfg_out_c)) - 1'b1;
                else
                    act_wr_mask <= 64'hFFFF_FFFF_FFFF_FFFF;
                state <= S_POOL_VEC_NEXT;
            end

            S_POOL_VEC_NEXT: begin : pool_vec_next_blk
                reg [15:0] pix_step;
                reg [15:0] remain;
                pix_step = pool_vec_multi ? {13'd0, pool_vec_batch_n} : 16'd1;
                pool_fh <= 0;
                pool_fw <= 0;
                pool_vec_pix_idx <= 0;
                pool_rd_phase <= 0;
                pool_count <= 0;
                for (i = 0; i < `MAC_LANES; i = i + 1)
                    racc[i] <= pool_mode ? $signed({ACC_W{1'b0}}) : $signed({1'b1, {(ACC_W-1){1'b0}}});
                if (pool_ow + pix_step >= out_tile_w) begin
                    pool_ow <= 0;
                    pool_vec_batch_n <=
                        (out_tile_w < {13'd0, pool_vec_pix_cap})
                            ? out_tile_w[2:0] : pool_vec_pix_cap;
                    if (pool_oh + 1 >= out_tile_h) begin
                        pool_oh <= 0;
                        if (pool_vec_ch_base
                            + (cfg_int16 ? 16'd32 : 16'd64) >= cfg_out_c) begin
                            state <= S_TILE_NEXT;
                        end else begin
                            pool_vec_ch_base <= pool_vec_ch_base
                                                + (cfg_int16 ? 16'd32 : 16'd64);
                            cache_issue <= 0;
                            cache_cap <= 0;
                            state <= S_POOL_VEC_PARAM;
                        end
                    end else begin
                        pool_oh <= pool_oh + 1;
                        state <= S_POOL_VEC_READ;
                    end
                end else begin
                    pool_ow <= pool_ow + pix_step;
                    remain = out_tile_w - (pool_ow + pix_step);
                    pool_vec_batch_n <=
                        (remain < {13'd0, pool_vec_pix_cap})
                            ? remain[2:0] : pool_vec_pix_cap;
                    state <= S_POOL_VEC_READ;
                end
            end

            S_POOL_CH_SETUP: begin
                // Load 4 PPU param words for current channel (same as DW_PARAM)
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= cfg_param_base
                                   + (pool_ch * 4) + param_word_idx;
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
                            pool_acc <= $signed({1'b1, {(ACC_W-1){1'b0}}});
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
                    `ifdef NPU_SIM_DEBUG
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
                    pool_acc <= $signed({1'b1, {(ACC_W-1){1'b0}}});
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
                end else if (pool_stream && (pool_ch[COL_W-1:0] == {COL_W{1'b1}})) begin
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
                add_vec_active <= 1'b0;
                tile_x <= single_tile_active ? tile_x : 16'd0;
                tile_y <= single_tile_active ? tile_y : 16'd0;
                // Set tile dimensions for actual tile size calculation.
                // Clip border tiles to image boundary (same as S_TILE_SETUP).
                if (single_tile_active) begin
                    out_tile_h <= single_tile_h_actual;
                    out_tile_w <= single_tile_w_actual;
                end else if (cfg_tile_h == 17'd0) begin
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
                    param_rd_addr <= cfg_param_base + add_param_idx;
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
                        if ((cfg_tile_h == 17'd0
                             && {16'd0, add_total_elems}
                                >= (cfg_int16 ? 32'd32 : 32'd64))
                            || (cfg_tile_h != 17'd0
                                && ({16'd0, cfg_tile_h} * {16'd0, cfg_tile_w}
                                    * {16'd0, cfg_out_c})
                                   >= (cfg_int16 ? 32'd32 : 32'd64))) begin
                            add_vec_active <= 1'b1;
                            state <= S_ADD_VEC_READ_A;
                        end else begin
                            add_vec_active <= 1'b0;
                            state <= S_ADD_READ_A;
                        end
                    end
                end
            end

            // Add operands are flat, contiguous NHWC tensors.  Process one
            // complete 512-bit beat at a time (64 INT8 or 32 INT16 values).
            // A short final beat deliberately falls back to the scalar path,
            // avoiding a wide SRAM read beyond the allocated tensor.
            S_ADD_VEC_READ_A: begin : add_vec_read_a_blk
                reg [31:0] byte_off;
                byte_off = cfg_int16 ? ({17'd0, add_tile_elem_cnt} << 1)
                                     : {17'd0, add_tile_elem_cnt};
                if (add_rd_phase == 0) begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                    add_rd_phase <= 1;
                end else if (add_rd_phase == 1) begin
                    add_rd_phase <= 2;
                end else begin
                    add_vec_a <= act_rd_data;
                    add_rd_phase <= 0;
                    state <= S_ADD_VEC_READ_B;
                end
            end

            S_ADD_VEC_READ_B: begin : add_vec_read_b_blk
                reg [31:0] byte_off;
                byte_off = cfg_int16 ? ({17'd0, add_tile_elem_cnt} << 1)
                                     : {17'd0, add_tile_elem_cnt};
                if (add_rd_phase == 0) begin
                    act_rd_en   <= 1'b1;
                    act_rd_addr <= act_base + cfg_out_base
                                   + byte_off[ACT_ADDR_W+1:2];
                    add_rd_phase <= 1;
                end else if (add_rd_phase == 1) begin
                    add_rd_phase <= 2;
                end else begin
                    add_vec_b <= act_rd_data;
                    add_rd_phase <= 0;
                    vec_ppu_sub <= 3'd0;
                    vec_cap_sub <= 3'd0;
                    state <= S_ADD_VEC_COMPUTE;
                end
            end

            S_ADD_VEC_COMPUTE: begin : add_vec_compute_blk
                integer l;
                reg [3:0]              av_nbeats;
                reg [5:0]              gl_idx;
                reg signed [ACC_W-1:0] va, vb, pa, pb, ra, rb, rnd_a, rnd_b;
                av_nbeats = cfg_int16 ? 4'd4 : 4'd8;
                rnd_a = $signed({{(ACC_W-1){1'b0}}, 1'b1} << (add_S_A - 1));
                rnd_b = $signed({{(ACC_W-1){1'b0}}, 1'b1} << (add_S_B - 1));
                for (l = 0; l < PPU_N; l = l + 1) begin
                    gl_idx = {vec_ppu_sub, l[2:0]};
                    va = cfg_int16
                       ? $signed({{(ACC_W-16){add_vec_a[16*gl_idx[4:0] + 15]}}, add_vec_a[16*gl_idx[4:0] +: 16]})
                       : $signed({{(ACC_W-8){add_vec_a[8*gl_idx + 7]}},         add_vec_a[8*gl_idx +: 8]});
                    vb = cfg_int16
                       ? $signed({{(ACC_W-16){add_vec_b[16*gl_idx[4:0] + 15]}}, add_vec_b[16*gl_idx[4:0] +: 16]})
                       : $signed({{(ACC_W-8){add_vec_b[8*gl_idx + 7]}},         add_vec_b[8*gl_idx +: 8]});
                    pa = va * $signed({1'b0, add_M_A});
                    pb = vb * $signed({1'b0, add_M_B});
                    ra = (add_S_A > 0) ? ((pa + rnd_a) >>> add_S_A) : pa;
                    rb = (add_S_B > 0) ? ((pb + rnd_b) >>> add_S_B) : pb;
                    ppu_acc_w[ACC_W*l +: ACC_W] <= ra + rb;
                    ppu_valid_w[l]              <= 1'b1;
                end
                if (ppu_vout_w[0]) begin
                    if (vec_cap_sub == 3'd0)
                        obuf <= {896'd0, ppu_out_w};
                    else
                        obuf[{vec_cap_sub, 7'd0} +: 128] <= ppu_out_w;
                    vec_cap_sub <= vec_cap_sub + 3'd1;
                end
                if ({1'b0, vec_ppu_sub} + 4'd1 >= av_nbeats) begin
                    vec_ppu_sub <= 3'd0;
                    state <= S_ADD_VEC_WAIT;
                end else begin
                    vec_ppu_sub <= vec_ppu_sub + 3'd1;
                end
            end

            S_ADD_VEC_WAIT: begin : add_vec_wait_blk
                reg [3:0] av_nbeats;
                av_nbeats = cfg_int16 ? 4'd4 : 4'd8;
                if (ppu_vout_w[0]) begin
                    if (vec_cap_sub == 3'd0)
                        obuf <= {896'd0, ppu_out_w};
                    else
                        obuf[{vec_cap_sub, 7'd0} +: 128] <= ppu_out_w;
                    if ({1'b0, vec_cap_sub} + 4'd1 >= av_nbeats) begin
                        vec_cap_sub <= 3'd0;
                        state <= S_ADD_VEC_WB;
                    end else begin
                        vec_cap_sub <= vec_cap_sub + 3'd1;
                    end
                end
            end

            S_ADD_VEC_WB: begin : add_vec_wb_blk
                integer l;
                reg [31:0]  byte_off;
                reg [511:0] w16, w8;
                byte_off = cfg_int16 ? ({17'd0, add_tile_elem_cnt} << 1)
                                     : {17'd0, add_tile_elem_cnt};
                w16 = obuf[511:0];
                for (l = 0; l < 64; l = l + 1)
                    w8[8*l +: 8] = obuf[DATA_W*l +: 8];
                act_wr_en   <= 1'b1;
                act_wr_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
                act_wr_wide <= cfg_int16 ? w16 : w8;
                act_wr_mask <= 64'hFFFF_FFFF_FFFF_FFFF;
                state <= S_ADD_NEXT;
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
                        `ifdef NPU_SIM_DEBUG
                        if (tile_x == 0 && tile_y == 0 && add_tile_elem_cnt >= 384 && add_tile_elem_cnt <= 390)
                            $display("[ADD_WB] t=%0t elem=%0d tile_elem=%0d wb_addr=%0d rd=0x%08x merged=0x%08x byte=0x%04x bytesel=%0d",
                                     $time, add_elem_cnt, add_tile_elem_cnt, add_wb_addr,
                                     act_rd_data, merged, add_wb_byte, add_wb_bytesel);
                        `endif
                    end
                    state <= S_ADD_NEXT;
                end
                default: add_wb_phase <= 2'd0;
                endcase
            end

            S_ADD_NEXT: begin : add_next_blk
                reg [31:0] elems_per_tile;
                reg [31:0] step;
                act_wr_en <= 1'b0;  // Clear write enable
                ppu_in_valid <= 1'b0;  // Clear PPU input valid
                step = add_vec_active ? (cfg_int16 ? 32'd32 : 32'd64) : 32'd1;
                // Use padded tile size for SRAM layout (DMA skips padding via src_row_len)
                // Concat iterates input elements (in_c per pixel), not out_c.
                elems_per_tile = {17'd0, (single_tile_active
                                           ? single_tile_h_actual : cfg_tile_h)}
                                 * {17'd0, (single_tile_active
                                           ? single_tile_w_actual : cfg_tile_w)}
                                 * {17'd0, (is_concat ? cfg_in_c : cfg_out_c)};
                if (cfg_tile_h == 17'd0 || elems_per_tile == 0) begin
                    // Non-tiled: original logic
                    if ({17'd0, add_elem_cnt} + step >= {17'd0, add_total_elems}) begin
                        state <= S_DONE;
                    end else begin
                        add_elem_cnt <= add_elem_cnt + step[15:0];
                        add_tile_elem_cnt <= add_tile_elem_cnt + step[15:0];
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
                        if (!is_concat
                            && ({17'd0, add_elem_cnt} + step
                                + (cfg_int16 ? 32'd32 : 32'd64)
                                <= {17'd0, add_total_elems})) begin
                            add_vec_active <= 1'b1;
                            state <= S_ADD_VEC_READ_A;
                        end else begin
                            add_vec_active <= 1'b0;
                            state <= S_ADD_READ_A;
                        end
                    end
                end else if ({17'd0, add_tile_elem_cnt} + step >= elems_per_tile) begin
                    // Current tile done — use tile-local counter for boundary check
                    if (single_tile_active) begin
                        state <= S_DONE;
                    end else if (tile_x + 1 >= cfg_tile_num_w) begin
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
                    add_elem_cnt <= add_elem_cnt + step[15:0];
                    add_tile_elem_cnt <= add_tile_elem_cnt + step[15:0];
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
                    if (!is_concat
                        && ({17'd0, add_tile_elem_cnt} + step
                            + (cfg_int16 ? 32'd32 : 32'd64)
                            <= elems_per_tile)) begin
                        add_vec_active <= 1'b1;
                        state <= S_ADD_VEC_READ_A;
                    end else begin
                        add_vec_active <= 1'b0;
                        state <= S_ADD_READ_A;
                    end
                end
            end

            // ══════════════════════════════════════════════════════════════
            // Resize Path (op_type=5)
            // Flow: RECIP (once/layer) → TILE_SETUP → SETUP → CH_SETUP → COORD → READ0[/1/2/3] → INTERP → PPU → WB
            // ══════════════════════════════════════════════════════════════
            S_RESIZE_RECIP: begin : resize_recip_step_blk
                reg [16:0] rn_h, rn_w, rn_hm1, rn_wm1;
                reg [15:0] d_hm1, d_wm1;
                reg        ge_h, ge_w, ge_hm1, ge_wm1;
                reg [39:0] qn_h, qn_w, qn_hm1, qn_wm1;
                reg [40:0] rcp_h_fin, rcp_w_fin, rcp_hm1_fin, rcp_wm1_fin;
                reg [24:0] in_h_m1_sh, in_w_m1_sh;
                d_hm1  = cfg_out_h - 16'd1;
                d_wm1  = cfg_out_w - 16'd1;
                rn_h   = {rsz_rem_h[15:0],   1'b1};
                rn_w   = {rsz_rem_w[15:0],   1'b1};
                rn_hm1 = {rsz_rem_hm1[15:0], 1'b1};
                rn_wm1 = {rsz_rem_wm1[15:0], 1'b1};
                ge_h   = (rn_h   >= {1'b0, cfg_out_h});
                ge_w   = (rn_w   >= {1'b0, cfg_out_w});
                ge_hm1 = (rn_hm1 >= {1'b0, d_hm1});
                ge_wm1 = (rn_wm1 >= {1'b0, d_wm1});
                qn_h   = {rsz_q_h[38:0],   ge_h};
                qn_w   = {rsz_q_w[38:0],   ge_w};
                qn_hm1 = {rsz_q_hm1[38:0], ge_hm1};
                qn_wm1 = {rsz_q_wm1[38:0], ge_wm1};
                rsz_rem_h   <= ge_h   ? (rn_h   - {1'b0, cfg_out_h}) : rn_h;
                rsz_rem_w   <= ge_w   ? (rn_w   - {1'b0, cfg_out_w}) : rn_w;
                rsz_rem_hm1 <= ge_hm1 ? (rn_hm1 - {1'b0, d_hm1})     : rn_hm1;
                rsz_rem_wm1 <= ge_wm1 ? (rn_wm1 - {1'b0, d_wm1})     : rn_wm1;
                rsz_q_h     <= qn_h;
                rsz_q_w     <= qn_w;
                rsz_q_hm1   <= qn_hm1;
                rsz_q_wm1   <= qn_wm1;
                if (rsz_div_cnt == 6'd39) begin
                    rcp_h_fin   = (cfg_out_h != 16'd0) ? ({1'b0, qn_h}   + 41'd1) : 41'd0;
                    rcp_w_fin   = (cfg_out_w != 16'd0) ? ({1'b0, qn_w}   + 41'd1) : 41'd0;
                    rcp_hm1_fin = (cfg_out_h >  16'd1) ? ({1'b0, qn_hm1} + 41'd1) : 41'd0;
                    rcp_wm1_fin = (cfg_out_w >  16'd1) ? ({1'b0, qn_wm1} + 41'd1) : 41'd0;
                    rsz_rcp_h   <= rcp_h_fin;
                    rsz_rcp_w   <= rcp_w_fin;
                    rsz_rcp_hm1 <= rcp_hm1_fin;
                    rsz_rcp_wm1 <= rcp_wm1_fin;
                    recip_scale_h <= {40'd0, cfg_in_h} * {15'd0, rcp_h_fin};
                    recip_scale_w <= {40'd0, cfg_in_w} * {15'd0, rcp_w_fin};
                    in_h_m1_sh = ({9'd0, cfg_in_h} - 25'd1) << 8;
                    in_w_m1_sh = ({9'd0, cfg_in_w} - 25'd1) << 8;
                    recip_scale_h_m1 <= {31'd0, in_h_m1_sh} * {15'd0, rcp_hm1_fin};
                    recip_scale_w_m1 <= {31'd0, in_w_m1_sh} * {15'd0, rcp_wm1_fin};
                    state <= S_TILE_SETUP;
                end else begin
                    rsz_div_cnt <= rsz_div_cnt + 6'd1;
                end
            end

            S_RESIZE_SETUP: begin
                rsz_ch <= 0;
                rsz_oh <= 0;
                rsz_ow <= 0;
                param_word_idx <= 0;
                param_read_issued <= 1'b0;
                rsz_rd_phase <= 0;
                state <= S_RESIZE_CH_SETUP;
            end

            S_RESIZE_CH_SETUP: begin
                if (!param_read_issued) begin
                    param_rd_en   <= 1'b1;
                    param_rd_addr <= cfg_param_base
                                   + (rsz_ch * 4) + param_word_idx;
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

            // 8-lane PPU result, streamed over 1..8 beats into obuf.
            // Word-aligned groups take the wide port; narrow groups pack across pixels.
            if (row_ppu && !row_drain && (ppu_vout_w[0] || ppu_out_valid)) begin : row_cap
                integer cl;
                reg [5:0]               gl_c, ch_c;
                reg signed [DATA_W-1:0] cl_v;
                reg [127:0]             chunk16;
                reg [63:0]              chunk8;
                reg [511:0]             cwide;
                reg [1023:0]            cobuf;
                reg [8:0]               cnbytes;
                reg [31:0]              celem, cbbase;
                for (cl = 0; cl < PPU_N; cl = cl + 1) begin
                    gl_c = {row_cap_sub, cl[2:0]};
                    ch_c = (row_pix_sh == 3'd0) ? gl_c : {1'b0, lane_c_r[gl_c][4:0]};
                    cl_v = halo_output_pad
                        ? $signed(pc_zp[ch_c])
                        : (ppu_vout_w[cl] ? ppu_out_w[DATA_W*cl +: DATA_W]
                           : ((cl == 0 && ppu_out_valid) ? ppu_out_data : {DATA_W{1'b0}}));
                    chunk16[16*cl +: 16] = ({3'd0, gl_c} < ppu_nlane) ? cl_v      : 16'd0;
                    chunk8[8*cl +: 8]    = ({3'd0, gl_c} < ppu_nlane) ? cl_v[7:0] : 8'd0;
                end
                if ({1'b0, row_cap_sub} + 4'd1 < row_cap_tgt) begin
                    if (row_cap_sub == 3'd0)
                        obuf <= cfg_int16 ? {896'd0, chunk16} : {960'd0, chunk8};
                    else if (cfg_int16)
                        obuf[{row_cap_sub, 7'd0} +: 128] <= chunk16;
                    else
                        obuf[{1'b0, row_cap_sub, 6'd0} +: 64] <= chunk8;
                    row_cap_sub <= row_cap_sub + 3'd1;
                end else begin
                    cobuf = (row_cap_sub == 3'd0) ? 1024'd0 : obuf;
                    if (cfg_int16)
                        cobuf[{row_cap_sub, 7'd0} +: 128] = chunk16;
                    else
                        cobuf[{1'b0, row_cap_sub, 6'd0} +: 64] = chunk8;
                    cwide       = cobuf[511:0];
                    row_cap_sub <= 3'd0;
                    cnbytes     = cfg_int16 ? {ppu_nlane[7:0], 1'b0} : ppu_nlane;
                    celem       = ({16'd0, ppu_oh} * {16'd0, out_tile_w} + {16'd0, ppu_ow}) * {16'd0, cfg_out_c}
                                + {16'd0, oc_group} * {25'd0, row_group_ch};
                    cbbase      = cfg_int16 ? (celem << 1) : celem;
                    if (ppu_npix > 5'd1 || cfg_out_c > {9'd0, row_group_ch}) begin
                        obuf <= cobuf;
                        ob_n <= cnbytes[7:0];
                        scatter_p <= 5'd0;
                        scatter_bi <= 8'd0;
                        row_scatter <= 1'b1;
                        row_drain <= 1'b1;
                    end else if (cbbase[1:0] == 2'b00 && cnbytes[1:0] == 2'b00
                            && cnbytes != 0 && cnbytes <= 9'd64) begin
                        act_wr_en   <= 1'b1;
                        act_wr_addr <= out_base + cbbase[ACT_ADDR_W+1:2];
                        act_wr_wide <= cwide;
                        act_wr_data <= cwide[31:0];
                        if (cnbytes >= 9'd64)
                            act_wr_mask <= 64'hFFFF_FFFF_FFFF_FFFF;
                        else
                            act_wr_mask <= (64'h1 << cnbytes) - 64'h1;
                        pk      <= 32'd0;
                        pk_pos  <= 2'd0;
                        pk_addr <= out_base + cbbase[ACT_ADDR_W+1:2]
                                 + {{(ACT_ADDR_W-7){1'b0}}, cnbytes[8:2]};
                        row_ppu <= 1'b0;
                    end else begin
                        obuf <= cobuf;
                        ob_n <= cnbytes[7:0];
                        ob_i <= 8'd0;
                        row_drain <= 1'b1;
                    end
                end
            end else if (row_drain && row_scatter) begin : row_scatter_pack
                reg [31:0]   spix, selem, sbbase, sbaddr;
                reg [8:0]    snbytes, sleft, schunk, smaskn;
                reg [7:0]    sboff;
                reg          spad_last;
                reg [15:0]   ssrcb;
                reg [1023:0] ssh, scmask, sal;
                spix      = ({16'd0, ppu_oh} * {16'd0, out_tile_w} + {16'd0, ppu_ow}) + {27'd0, scatter_p};
                selem     = spix * {16'd0, cfg_out_c} + {16'd0, oc_group} * {25'd0, row_group_ch};
                sbbase    = cfg_int16 ? (selem << 1) : selem;
                snbytes   = cfg_int16 ? ({2'd0, ppu_nch} << 1) : {2'd0, ppu_nch};
                sbaddr    = sbbase + {24'd0, scatter_bi};
                sboff     = {6'd0, sbaddr[1:0]};
                sleft     = snbytes - {1'b0, scatter_bi};
                schunk    = ((sleft + {1'b0, sboff}) > 9'd64) ? (9'd64 - {1'b0, sboff}) : sleft;
                spad_last = row_last && (oc_group + 1 >= oc_groups_total)
                         && (scatter_p + 1 >= ppu_npix)
                         && ({1'b0, scatter_bi} + schunk >= snbytes);
                smaskn    = spad_last
                          ? (schunk + ((9'd4 - (({1'b0, sboff} + schunk) & 9'd3)) & 9'd3))
                          : schunk;
                ssrcb     = ({11'd0, scatter_p} * snbytes) + {8'd0, scatter_bi};
                ssh       = obuf >> {ssrcb[6:0], 3'b000};
                scmask    = (schunk >= 9'd64) ? {512'd0, {512{1'b1}}}
                                              : ((1024'h1 << {schunk[6:0], 3'b000}) - 1024'h1);
                sal       = (ssh & scmask) << {sboff[1:0], 3'b000};
                act_wr_en   <= 1'b1;
                act_wr_addr <= out_base + sbaddr[ACT_ADDR_W+1:2];
                act_wr_wide <= sal[511:0];
                act_wr_data <= sal[31:0];
                act_wr_mask <= ((64'h1 << smaskn) - 64'h1) << sboff;
                if ({1'b0, scatter_bi} + schunk >= snbytes) begin
                    scatter_bi <= 8'd0;
                    if (scatter_p + 1 >= ppu_npix) begin
                        scatter_p <= 5'd0;
                        row_scatter <= 1'b0;
                        row_drain <= 1'b0;
                        row_ppu <= 1'b0;
                    end else
                        scatter_p <= scatter_p + 1'b1;
                end else
                    scatter_bi <= scatter_bi + schunk[7:0];
            end else if (row_drain) begin : row_pack
                reg [7:0]    prem, pnii;
                reg [2:0]    proom, ptake;
                reg          pwrote;
                reg [1023:0] psh;
                reg [31:0]   psrc32, pkeep, psmask, pmword, pnword;
                reg [1:0]    pnpos;
                prem   = (ob_i < ob_n) ? (ob_n - ob_i) : 8'd0;
                proom  = 3'd4 - {1'b0, pk_pos};
                pwrote = (ob_i < ob_n) && (prem >= {5'd0, proom});
                ptake  = (ob_i >= ob_n) ? 3'd0 : (pwrote ? proom : prem[2:0]);
                psh    = obuf >> {ob_i[6:0], 3'b000};
                psrc32 = psh[31:0];
                pkeep  = (32'h1 << {pk_pos, 3'b000}) - 32'h1;
                psmask = pwrote ? 32'hFFFF_FFFF : ((32'h1 << {ptake[1:0], 3'b000}) - 32'h1);
                pmword = (pk & pkeep) | ((psrc32 & psmask) << {pk_pos, 3'b000});
                pnword = pwrote ? 32'd0 : pmword;
                pnpos  = pwrote ? 2'd0  : (pk_pos + ptake[1:0]);
                pnii   = ob_i + {5'd0, ptake};
                pk     <= pnword;
                pk_pos <= pnpos;
                ob_i   <= pnii;
                if (pwrote) begin
                    act_wr_en   <= 1'b1;
                    act_wr_addr <= pk_addr;
                    act_wr_data <= pmword;
                    act_wr_wide <= {480'd0, pmword};
                    act_wr_mask <= 64'hF;
                    pk_addr     <= pk_addr + 1'b1;
                end
                if (pnii >= ob_n && (!pwrote || pnpos == 2'd0)) begin
                    row_drain <= 1'b0;
                    row_ppu   <= 1'b0;
                    if (!pwrote && row_last && pnpos != 2'd0) begin
                        act_wr_en   <= 1'b1;
                        act_wr_addr <= pk_addr;
                        act_wr_data <= pnword;
                        act_wr_wide <= {480'd0, pnword};
                        act_wr_mask <= 64'hF;
                        pk          <= 32'd0;
                        pk_pos      <= 2'd0;
                        pk_addr     <= pk_addr + 1'b1;
                    end
                end
            end

        end
    end

`endif
endmodule
