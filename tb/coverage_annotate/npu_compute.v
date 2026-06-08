//      // verilator_coverage annotation
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
            parameter ACT_ADDR_W   = 13,
            parameter WGT_ADDR_W   = 14,
            parameter PARAM_ADDR_W = 11,
            parameter DATA_W       = `DATA_WIDTH,
            parameter ACC_W        = `ACC_WIDTH
        )(
 004959     input  wire                         clk,
 000011     input  wire                         rst_n,
        
            // ─── Controller handshake ───
%000004     input  wire                         start,
%000004     output reg                          done,
        
            // ─── Layer configuration (from CSR) ───
%000005     input  wire [7:0]                   cfg_op_type,
%000000     input  wire                         cfg_int16,      // 1=INT16 mode, 0=INT8 mode
%000004     input  wire [15:0]                  cfg_in_c,
%000005     input  wire [15:0]                  cfg_out_h,
%000005     input  wire [15:0]                  cfg_out_w,
%000004     input  wire [15:0]                  cfg_out_c,
%000005     input  wire [7:0]                   cfg_kernel_h,
%000003     input  wire [7:0]                   cfg_kernel_w,
%000005     input  wire [7:0]                   cfg_stride_h,
%000003     input  wire [7:0]                   cfg_stride_w,
%000002     input  wire [7:0]                   cfg_pad_top,
%000000     input  wire [7:0]                   cfg_pad_left,
%000000     input  wire [15:0]                  cfg_tile_h,
%000000     input  wire [15:0]                  cfg_tile_w,
%000000     input  wire [15:0]                  cfg_tile_num_h,
%000000     input  wire [15:0]                  cfg_tile_num_w,
%000005     input  wire [15:0]                  cfg_in_w,
%000004     input  wire [15:0]                  cfg_in_h,
%000000     input  wire [ACT_ADDR_W-1:0]        cfg_act_base,   // Input activation base (word addr)
%000000     input  wire [ACT_ADDR_W-1:0]        cfg_out_base,   // Output base (word addr in act SRAM)
%000000     input  wire [31:0]                  cfg_pool_cfg,   // Pooling config register
%000000     input  wire [31:0]                  cfg_resize_cfg, // Resize config register
%000000     input  wire [31:0]                  cfg_deconv_cfg, // Deconv config: [7:0]=INSERT_H, [15:8]=INSERT_W
%000000     input  wire [31:0]                  cfg_concat_cfg, // Concat config: [15:0]=OFFSET, [31:16]=TOTAL_C
        
            // ─── Weight SRAM Port B (read-only) ───
 000064     output reg                          wgt_rd_en,
%000007     output reg  [WGT_ADDR_W-1:0]       wgt_rd_addr,
%000000     input  wire [31:0]                  wgt_rd_data,
        
            // ─── Activation SRAM Port B (read + write) ───
 000128     output reg                          act_rd_en,
%000007     output reg  [ACT_ADDR_W-1:0]       act_rd_addr,
%000008     input  wire [31:0]                  act_rd_data,
 000064     output reg                          act_wr_en,
%000007     output reg  [ACT_ADDR_W-1:0]       act_wr_addr,
~000016     output reg  [31:0]                  act_wr_data,
        
            // ─── Parameter SRAM Port B (read-only) ───
 000256     output reg                          param_rd_en,
~000127     output reg  [PARAM_ADDR_W-1:0]     param_rd_addr,
%000000     input  wire [31:0]                  param_rd_data,
        
            // ─── Systolic Array ───
%000000     output reg  [1:0]                   sa_cmd,
%000000     output reg                          sa_cmd_valid,
%000000     output wire [DATA_W*ARRAY_SIZE-1:0] sa_wgt_data_flat,
%000000     output reg                          sa_wgt_valid,
%000000     output wire [DATA_W*ARRAY_SIZE-1:0] sa_act_data_flat,
%000000     output reg                          sa_act_valid,
%000000     output reg  [$clog2(ARRAY_SIZE)-1:0] sa_drain_col_sel,
            input  wire [ACC_W*ARRAY_SIZE-1:0]  sa_acc_out_flat,
%000000     input  wire                         sa_acc_out_valid,
%000000     input  wire                         sa_busy,
%000000     input  wire                         sa_ready,
        
            // ─── DW Conv ───
 000064     output reg                          dw_wgt_load,
 000064     output reg                          dw_wgt_valid,
%000000     output reg  signed [DATA_W-1:0]    dw_wgt_data,
 000064     output reg                          dw_in_valid,
~000015     output reg  signed [DATA_W-1:0]    dw_in_data,
 000128     output reg                          dw_acc_clear,
%000000     input  wire signed [ACC_W-1:0]     dw_acc_out,
 000064     input  wire                         dw_out_valid,
        
            // ─── PPU ───
%000000     output reg  signed [ACC_W-1:0]     ppu_acc_in,
 000064     output reg                          ppu_in_valid,
%000000     output reg  signed [ACC_W-1:0]     ppu_bias,
%000000     output reg  [14:0]                  ppu_mult_m,
%000000     output reg  [5:0]                   ppu_shift_s,
%000000     output reg  signed [15:0]          ppu_zero_point,
%000000     input  wire signed [DATA_W-1:0]    ppu_out_data,
 000064     input  wire                         ppu_out_valid
        );
        
            // ─── Internal unpacked arrays for systolic interface ───
%000000     reg  signed [DATA_W-1:0] sa_wgt_data [0:ARRAY_SIZE-1];
%000000     reg  signed [DATA_W-1:0] sa_act_data [0:ARRAY_SIZE-1];
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
        
            // ─── FSM States ───
            localparam [5:0]
                S_IDLE        = 6'd0,
                S_TILE_SETUP  = 6'd1,
                S_OC_SETUP    = 6'd2,
                S_WGT_CMD     = 6'd3,
                S_WGT_LOAD    = 6'd4,   // Read+fill wgt_data for one column
                S_WGT_EMIT    = 6'd5,   // Pulse wgt_valid
                S_ACT_CMD     = 6'd6,
                S_ACT_LOAD    = 6'd7,   // Read activation word from SRAM
                S_ACT_EMIT    = 6'd8,   // Pulse act_valid with one byte
                S_ACT_FLUSH   = 6'd9,
                S_DRAIN_CMD   = 6'd10,
                S_DRAIN_WAIT  = 6'd11,
                S_PARAM_LOAD  = 6'd12,
                S_PPU_FEED    = 6'd13,
                S_PPU_WAIT    = 6'd14,
                S_WRITEBACK   = 6'd15,
                S_OC_NEXT     = 6'd16,
                S_TILE_NEXT   = 6'd17,
                S_DONE        = 6'd18,
                S_DW_WGT_LOAD = 6'd19,
                S_DW_COMPUTE  = 6'd20,
                S_DW_DRAIN    = 6'd21,
                S_DW_PARAM    = 6'd22,
                S_DW_PPU      = 6'd23,
                S_DW_ACT_STREAM = 6'd24,
                S_DW_PPU_WAIT   = 6'd25,
                S_DW_WB         = 6'd26,
                S_SPATIAL_SETUP = 6'd27,
                S_REDUCE        = 6'd28,
                S_PIXEL_NEXT    = 6'd29,
                // Pooling states
                S_POOL_SETUP    = 6'd30,
                S_POOL_CH_SETUP = 6'd31,
                S_POOL_READ     = 6'd32,
                S_POOL_ACC      = 6'd33,
                S_POOL_DIV      = 6'd34,
                S_POOL_PPU      = 6'd35,
                S_POOL_PPU_WAIT = 6'd36,
                S_POOL_WB       = 6'd37,
                S_POOL_PIX_NEXT = 6'd38,
                S_POOL_CH_NEXT  = 6'd39,
                // Eltwise Add states
                S_ADD_SETUP     = 6'd40,
                S_ADD_PARAM     = 6'd41,
                S_ADD_READ_A    = 6'd42,
                S_ADD_READ_B    = 6'd43,
                S_ADD_COMPUTE   = 6'd44,
                S_ADD_PPU       = 6'd45,
                S_ADD_PPU_WAIT  = 6'd46,
                S_ADD_WB        = 6'd47,
                S_ADD_NEXT      = 6'd48,
                // Resize states
                S_RESIZE_SETUP    = 6'd49,
                S_RESIZE_CH_SETUP = 6'd50,
                S_RESIZE_COORD    = 6'd51,
                S_RESIZE_READ0    = 6'd52,
                S_RESIZE_READ1    = 6'd53,
                S_RESIZE_READ2    = 6'd54,
                S_RESIZE_READ3    = 6'd55,
                S_RESIZE_INTERP   = 6'd56,
                S_RESIZE_PPU      = 6'd57,
                S_RESIZE_PPU_WAIT = 6'd58,
                S_RESIZE_WB       = 6'd59,
                S_RESIZE_PIX_NEXT = 6'd60,
                S_RESIZE_CH_NEXT  = 6'd61;
        
~000196     reg [5:0] state;
        
            // ─── Tile iteration ───
%000000     reg [15:0] tile_y, tile_x;
~000031     reg [15:0] oc_group;
        
            // ─── Latched config ───
%000004     reg [15:0] oc_groups_total;
%000004     reg [15:0] k_depth;         // kh * kw * in_c
%000001     reg [15:0] out_tile_h, out_tile_w;
        
            // ─── Weight load state ───
            // Load one column at a time: read ceil(k_pass_remain/4) words, fill sa_wgt_data
%000000     reg [$clog2(ARRAY_SIZE)-1:0] wgt_col_idx;     // current column (0..ARRAY_SIZE-1)
%000000     reg [$clog2(ARRAY_SIZE):0]   wgt_byte_idx;    // byte index within column (0..ARRAY_SIZE-1)
%000007     reg [15:0]                   wgt_word_addr;    // current SRAM address
%000000     reg                          wgt_read_issued;  // 1-cycle read latency tracker
%000000     reg                          wgt_data_ready;   // 2nd cycle: SRAM data available
%000000     reg [1:0]                    wgt_bsel;         // byte offset within first word
        
            // ─── Activation stream state ───
%000000     reg [15:0] act_cnt;         // activation byte counter (0..k_depth-1)
%000000     reg [15:0] act_word_addr;   // current SRAM address
 000319     reg        act_read_issued;
%000000     reg        act_data_ready;  // 2nd cycle: SRAM data available
%000000     reg [31:0] act_buf;         // buffered SRAM word
 000031     reg [1:0]  act_byte_sel;    // byte position within word
        
            // ─── Drain state ───
%000000     reg [$clog2(ARRAY_SIZE)-1:0] drain_col;
%000001     reg [$clog2(ARRAY_SIZE)-1:0] col_last;  // last valid drain column for current oc_group
        
            // ─── PPU state ───
%000000     reg [$clog2(ARRAY_SIZE):0] ppu_feed_cnt;  // how many acc values fed to PPU
~000191     reg [15:0]                  ppu_wait_cnt;   // PPU flush wait counter
            reg signed [ACC_W-1:0]     acc_buf [0:ARRAY_SIZE-1];
        
            // ─── Param read state ───
~000127     reg [2:0]  param_word_idx;
%000000     reg [31:0] param_buf [0:3];
 000255     reg        param_read_issued;
%000000     reg        param_data_ready;   // 2nd cycle: SRAM data available
        
            // ─── Writeback state ───
%000000     reg [$clog2(ARRAY_SIZE):0] wb_cnt;   // output bytes collected
%000000     reg [31:0] wb_pack;                   // pack buffer
%000000     reg [ACT_ADDR_W-1:0] wb_addr;
        
            // ─── Address bases ───
%000000     reg [WGT_ADDR_W-1:0]   wgt_base;
%000000     reg [ACT_ADDR_W-1:0]   act_base;
%000000     reg [PARAM_ADDR_W-1:0] param_base;
%000000     reg [ACT_ADDR_W-1:0]   out_base;
        
            // ─── DW state ───
%000000     reg [15:0] dw_ch_idx;
~000064     reg [5:0]  dw_cnt;
 000128     reg        dw_read_issued;
 000064     reg [1:0]  dw_init_phase;          // 0=setup, 1=acc_clear, 2=feeding
%000000     reg [15:0] dw_oh, dw_ow;          // Output pixel coordinates
%000000     reg [3:0]  dw_fh, dw_fw;          // Filter position (0..6)
%000000     reg signed [ACC_W-1:0] dw_acc_buf; // Captured DW output accumulator
%000003     reg [5:0]  dw_kernel_size;         // kh * kw (cached)
 000064     reg [1:0]  dw_wb_phase;            // 0=issue read, 1=wait, 2=merge+write
%000000     reg [15:0] dw_wb_byte;             // PPU output element to write (up to 16-bit)
 000031     reg [1:0]  dw_wb_bytesel;          // byte position within word
%000007     reg [ACT_ADDR_W-1:0] dw_wb_addr;  // target word address
 000031     reg [1:0]  dw_wgt_bsel_base;       // starting byte offset for weight reads
        
            // ─── Spatial pixel loop (Conv2D Plan A) ───
%000000     reg [15:0] sp_oh, sp_ow;                  // Current output pixel coordinates (tile-local)
%000000     reg [15:0] tile_oh_origin, tile_ow_origin; // Global origin of current tile
%000000     reg signed [ACC_W-1:0] dot_acc;           // Reduction accumulator
%000000     reg [$clog2(ARRAY_SIZE):0] reduce_cnt;    // Reduction counter
%000000     reg [ACT_ADDR_W-1:0] pixel_act_base;     // Per-pixel activation base address
            reg signed [ACC_W-1:0] dot_buf [0:ARRAY_SIZE-1]; // Reduced dot products per column
        
            // ─── Multi-pass (k_depth > ARRAY_SIZE) ───
%000000     reg [15:0] k_pass;           // Current pass index (0-based)
%000000     reg [15:0] k_pass_max;       // Total passes - 1
%000000     reg [15:0] k_pass_remain;    // Elements in current pass
        
            // ─── Conv2D kernel window iteration ───
%000000     reg [7:0]  conv_fh, conv_fw;             // Current filter position
%000000     reg [15:0] conv_ch_cnt;                  // Channel counter within (fh, fw)
%000000     reg signed [15:0] conv_ih_base;          // Input row origin: sp_oh*stride_h - pad_top
%000000     reg signed [15:0] conv_iw_base;          // Input col origin: sp_ow*stride_w - pad_left
%000000     reg        conv_is_pad;                  // Current (fh, fw) is padding
%000000     reg [15:0] conv_elem_cnt;                // Total elements emitted in this pass
        
            // ─── Flush counter (reused) ───
%000000     reg [15:0] flush_cnt;
        
            // ─── Pooling state ───
%000000     wire        pool_mode     = cfg_pool_cfg[0];      // 0=Max, 1=Avg
%000000     wire [3:0]  pool_cfg_h    = cfg_pool_cfg[7:4];
%000000     wire [3:0]  pool_cfg_w    = cfg_pool_cfg[11:8];
%000000     wire [3:0]  pool_cfg_sh   = cfg_pool_cfg[15:12];
%000000     wire [3:0]  pool_cfg_sw   = cfg_pool_cfg[19:16];
%000000     wire        global_pool   = cfg_pool_cfg[20];
%000000     wire        resize_mode   = cfg_resize_cfg[0];   // 0=nearest, 1=bilinear
            // ─── Deconv state ───
%000000     wire [7:0]  cfg_insert_h  = cfg_deconv_cfg[7:0];
%000000     wire [7:0]  cfg_insert_w  = cfg_deconv_cfg[15:8];
%000000     wire        is_deconv     = (cfg_op_type == 8'd6);
%000004     wire [15:0] deconv_exp_h  = cfg_in_h + (cfg_in_h - 16'd1) * {8'd0, cfg_insert_h};
%000005     wire [15:0] deconv_exp_w  = cfg_in_w + (cfg_in_w - 16'd1) * {8'd0, cfg_insert_w};
%000001     wire [8:0]  deconv_step_h = {1'b0, cfg_insert_h} + 9'd1; // ins_h + 1
%000001     wire [8:0]  deconv_step_w = {1'b0, cfg_insert_w} + 9'd1; // ins_w + 1
%000000     reg         deconv_skip;  // 1 = current (fh, fw) maps to zero-inserted position
            // ─── Concat state ───
%000000     wire [15:0] concat_offset  = cfg_concat_cfg[15:0];
%000000     wire [15:0] concat_total_c = cfg_concat_cfg[31:16];
%000000     wire        is_concat      = (cfg_op_type == 8'd7);
%000000     reg signed [ACC_W-1:0] pool_acc;     // Running sum or max
%000000     reg [15:0]             pool_count;   // Valid element count (AvgPool)
%000000     reg [3:0]              pool_fh, pool_fw; // Window position
%000000     reg [15:0]             pool_oh, pool_ow; // Output pixel coords
%000000     reg [15:0]             pool_ch;          // Current channel
%000000     reg [7:0]              pool_kh, pool_kw; // Effective kernel size
%000000     reg [7:0]              pool_sh, pool_sw; // Effective stride
%000000     reg [1:0]              pool_rd_phase;    // SRAM read phasing
%000000     reg [1:0]              pool_wb_phase;    // Writeback phasing
%000000     reg [ACT_ADDR_W-1:0]  pool_wb_addr;     // Writeback word address
%000000     reg [1:0]              pool_wb_bytesel;  // Writeback byte select
%000000     reg [15:0]             pool_wb_byte;     // PPU output to write
        
            // ─── Eltwise Add state ───
%000000     reg [14:0] add_M_A, add_M_B;
%000000     reg [5:0]  add_S_A, add_S_B;
%000000     reg signed [ACC_W-1:0] add_val_a, add_val_b;
%000000     reg [15:0] add_elem_cnt;      // Current element index (flat)
%000000     reg [15:0] add_total_elems;   // H * W * C
%000000     reg [1:0]  add_rd_phase;      // SRAM read phasing
%000000     reg [1:0]  add_param_phase;   // Param read phasing
%000000     reg [1:0]  add_param_idx;     // Which param word (0 or 1)
%000000     reg [1:0]  add_wb_phase;      // Writeback phasing
%000000     reg [ACT_ADDR_W-1:0]  add_wb_addr;     // Writeback word address
%000000     reg [1:0]             add_wb_bytesel;  // Writeback byte select
%000000     reg [15:0]            add_wb_byte;     // PPU output to write
        
            // ─── Resize state ───
%000000     reg [15:0]             rsz_oh, rsz_ow;
%000000     reg [15:0]             rsz_ch;
%000000     reg signed [ACC_W-1:0] rsz_v00, rsz_v01, rsz_v10, rsz_v11;
%000000     reg [7:0]              rsz_frac_h, rsz_frac_w;
%000000     reg [15:0]             rsz_ih0, rsz_iw0, rsz_ih1, rsz_iw1;
%000000     reg [1:0]              rsz_rd_phase;
%000000     reg [1:0]              rsz_wb_phase;
%000000     reg [ACT_ADDR_W-1:0]   rsz_wb_addr;
%000000     reg [1:0]              rsz_wb_bytesel;
%000000     reg [15:0]             rsz_wb_byte;
        
            integer i;
        
            // ════════════════════════════════════════════════════════════════════
            // Main FSM
            // ════════════════════════════════════════════════════════════════════
        
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n) begin
 000035             state <= S_IDLE;
 000035             done  <= 1'b0;
                    // All outputs idle
 000035             sa_cmd       <= MODE_IDLE;
 000035             sa_cmd_valid <= 1'b0;
 000035             sa_wgt_valid <= 1'b0;
 000035             sa_act_valid <= 1'b0;
 000035             sa_drain_col_sel <= 0;
 000035             wgt_rd_en   <= 1'b0;
 000035             wgt_rd_addr <= 0;
 000035             act_rd_en   <= 1'b0;
 000035             act_rd_addr <= 0;
 000035             act_wr_en   <= 1'b0;
 000035             act_wr_addr <= 0;
 000035             act_wr_data <= 32'd0;
 000035             param_rd_en <= 1'b0;
 000035             param_rd_addr <= 0;
 000035             ppu_acc_in   <= 0;
 000035             ppu_in_valid <= 1'b0;
 000035             ppu_bias     <= 0;
 000035             ppu_mult_m   <= 0;
 000035             ppu_shift_s  <= 0;
 000035             ppu_zero_point <= 0;
 000035             dw_wgt_load  <= 1'b0;
 000035             dw_wgt_valid <= 1'b0;
 000035             dw_wgt_data  <= 0;
 000035             dw_in_valid  <= 1'b0;
 000035             dw_in_data   <= 0;
 000035             dw_acc_clear <= 1'b0;
                    // Internal state
 000035             tile_y <= 0; tile_x <= 0; oc_group <= 0;
 000035             oc_groups_total <= 1; k_depth <= 1;
 000035             out_tile_h <= 1; out_tile_w <= 1;
 000035             wgt_col_idx <= 0; wgt_byte_idx <= 0;
 000035             wgt_word_addr <= 0; wgt_read_issued <= 0; wgt_data_ready <= 0;
 000035             act_cnt <= 0; act_word_addr <= 0;
 000035             act_read_issued <= 0; act_data_ready <= 0; act_buf <= 0; act_byte_sel <= 0;
 000035             drain_col <= 0; col_last <= COL_MAX;
 000035             ppu_feed_cnt <= 0; ppu_wait_cnt <= 0;
 000035             param_word_idx <= 0; param_read_issued <= 0; param_data_ready <= 0;
 000035             wb_cnt <= 0; wb_pack <= 0; wb_addr <= 0;
 000035             wgt_base <= 0; act_base <= 0; param_base <= 0; out_base <= 0;
 000035             dw_ch_idx <= 0; dw_cnt <= 0; dw_read_issued <= 0; dw_init_phase <= 0;
 000035             dw_oh <= 0; dw_ow <= 0; dw_fh <= 0; dw_fw <= 0;
 000035             dw_acc_buf <= 0; dw_kernel_size <= 0;
 000035             dw_wb_phase <= 0; dw_wb_byte <= 0; dw_wb_bytesel <= 0; dw_wb_addr <= 0;
 000035             dw_wgt_bsel_base <= 0;
 000035             flush_cnt <= 0;
 000035             sp_oh <= 0; sp_ow <= 0; tile_oh_origin <= 0; tile_ow_origin <= 0;
 000035             dot_acc <= 0; reduce_cnt <= 0; pixel_act_base <= 0;
 000035             k_pass <= 0; k_pass_max <= 0; k_pass_remain <= 0;
 000035             conv_fh <= 0; conv_fw <= 0; conv_ch_cnt <= 0;
 000035             conv_ih_base <= 0; conv_iw_base <= 0; conv_is_pad <= 0;
 000035             conv_elem_cnt <= 0;
                    // Pooling resets
 000035             pool_acc <= 0; pool_count <= 0;
 000035             pool_fh <= 0; pool_fw <= 0; pool_oh <= 0; pool_ow <= 0;
 000035             pool_ch <= 0; pool_kh <= 0; pool_kw <= 0; pool_sh <= 0; pool_sw <= 0;
 000035             pool_rd_phase <= 0; pool_wb_phase <= 0;
 000035             pool_wb_addr <= 0; pool_wb_bytesel <= 0; pool_wb_byte <= 0;
                    // Add resets
 000035             add_M_A <= 0; add_M_B <= 0; add_S_A <= 0; add_S_B <= 0;
 000035             add_val_a <= 0; add_val_b <= 0;
 000035             add_elem_cnt <= 0; add_total_elems <= 0;
 000035             add_rd_phase <= 0; add_param_phase <= 0; add_param_idx <= 0;
 000035             add_wb_phase <= 0; add_wb_addr <= 0; add_wb_bytesel <= 0; add_wb_byte <= 0;
                    // Resize resets
 000035             rsz_oh <= 0; rsz_ow <= 0; rsz_ch <= 0;
 000035             rsz_v00 <= 0; rsz_v01 <= 0; rsz_v10 <= 0; rsz_v11 <= 0;
 000035             rsz_frac_h <= 0; rsz_frac_w <= 0;
 000035             rsz_ih0 <= 0; rsz_iw0 <= 0; rsz_ih1 <= 0; rsz_iw1 <= 0;
 000035             rsz_rd_phase <= 0; rsz_wb_phase <= 0;
 000035             rsz_wb_addr <= 0; rsz_wb_bytesel <= 0; rsz_wb_byte <= 0;
 000560             for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
 000560                 sa_wgt_data[i] <= 0;
 000560                 sa_act_data[i] <= 0;
 000560                 acc_buf[i] <= 0;
 000560                 dot_buf[i] <= 0;
                    end
 000140             for (i = 0; i < 4; i = i + 1)
 000140                 param_buf[i] <= 0;
 002450         end else begin
                    // ─── Default pulse deassertion ───
 002450             done         <= 1'b0;
 002450             sa_cmd_valid <= 1'b0;
 002450             sa_wgt_valid <= 1'b0;
 002450             sa_act_valid <= 1'b0;
 002450             wgt_rd_en   <= 1'b0;
 002450             act_rd_en   <= 1'b0;
 002450             act_wr_en   <= 1'b0;
 002450             param_rd_en <= 1'b0;
 002450             ppu_in_valid <= 1'b0;
 002450             dw_wgt_valid <= 1'b0;
 002450             dw_in_valid  <= 1'b0;
 002450             dw_acc_clear <= 1'b0;
        
 002450             case (state)
        
                    // ══════════════════════════════════════════════════════════════
 001420             S_IDLE: begin
~001418                 if (start) begin
%000002                     if (cfg_op_type == 8'd1 || cfg_op_type == 8'd3 || cfg_op_type == 8'd5)
%000002                         oc_groups_total <= cfg_out_c;
                            else
%000000                         oc_groups_total <= (cfg_out_c + ARRAY_SIZE_16 - 1) / ARRAY_SIZE_16;
        
%000002                     k_depth <= {8'd0, cfg_kernel_h} * {8'd0, cfg_kernel_w} * cfg_in_c;
        
%000002                     if (cfg_tile_h == 16'd0) begin
%000002                         out_tile_h <= cfg_out_h;
%000002                         out_tile_w <= cfg_out_w;
%000000                     end else begin
%000000                         out_tile_h <= cfg_tile_h;
%000000                         out_tile_w <= cfg_tile_w;
                            end
        
%000002                     tile_y <= 0;
%000002                     tile_x <= 0;
%000002                     state  <= S_TILE_SETUP;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
%000002             S_TILE_SETUP: begin
                        // Compute base addresses
%000002                 wgt_base <= 0;  // Weight base fixed (all OC weights from start)
%000002                 param_base <= 0;
                        // Input: always use cfg_act_base; tile offset is folded into
                        // conv_ih_base/conv_iw_base via tile_oh_origin/tile_ow_origin
                        // so the padding check and address calc use global coordinates.
%000002                 act_base <= cfg_act_base;
%000002                 tile_oh_origin <= tile_y * out_tile_h;
%000002                 tile_ow_origin <= tile_x * out_tile_w;
                        // Output base includes tile offset
%000002                 if (cfg_int16) begin
%000000                     out_base <= cfg_out_base
%000000                                + ((tile_y * out_tile_h * cfg_out_w
%000000                                   * cfg_out_c * 16'd2
%000000                                  + tile_x * out_tile_w * cfg_out_c * 16'd2) >> 2);
%000002                 end else begin
%000002                     out_base <= cfg_out_base
%000002                                + ((tile_y * out_tile_h * cfg_out_w
%000002                                   * cfg_out_c
%000002                                  + tile_x * out_tile_w * cfg_out_c) >> 2);
                        end
        
%000002                 oc_group <= 0;
        
%000002                 if (cfg_op_type == 8'd1) begin
%000002                     dw_cnt <= 0;
%000002                     dw_read_issued <= 1'b0;
%000002                     dw_init_phase <= 2'd0;
%000002                     state <= S_DW_WGT_LOAD;
%000000                 end else if (cfg_op_type == 8'd3) begin
%000000                     state <= S_POOL_SETUP;
%000000                 end else if (cfg_op_type == 8'd4 || cfg_op_type == 8'd7) begin
%000000                     state <= S_ADD_SETUP;
%000000                 end else if (cfg_op_type == 8'd5) begin
%000000                     state <= S_RESIZE_SETUP;
%000000                 end else begin
%000000                     state <= S_OC_SETUP;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
%000000             S_OC_SETUP: begin
                        // Weight base for this OC group:
                        //   INT8: oc_group * ARRAY_SIZE * k_depth / 4
                        //   INT16: oc_group * ARRAY_SIZE * k_depth * 2 / 4
%000000                 if (cfg_int16)
%000000                     wgt_base <= (oc_group * ARRAY_SIZE_16 * k_depth) >> 1;
                        else
%000000                     wgt_base <= (oc_group * ARRAY_SIZE_16 * k_depth) >> 2;
                        // Param base: 4 words per channel, ARRAY_SIZE channels per group
%000000                 param_base <= oc_group * ARRAY_SIZE_16 * 4;
        
                        // Multi-pass setup
%000000                 k_pass <= 0;
%000000                 k_pass_max <= (k_depth - 1) / ARRAY_SIZE_16;
        
                        // Last valid drain column for this oc_group
%000000                 begin : col_last_blk
                            reg [15:0] remaining_oc;
%000000                     remaining_oc = cfg_out_c - oc_group * ARRAY_SIZE_16;
%000000                     if (remaining_oc >= ARRAY_SIZE_16)
%000000                         col_last <= COL_MAX;
                            else
%000000                         col_last <= remaining_oc[$clog2(ARRAY_SIZE)-1:0] - 1;
                        end
        
                        // Reset spatial coords for first pixel
%000000                 sp_oh <= 0;
%000000                 sp_ow <= 0;
        
%000000                 wgt_col_idx <= 0;
%000000                 state <= S_WGT_CMD;
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // WEIGHT LOAD: load ARRAY_SIZE columns, one per wgt_valid pulse
                    // ══════════════════════════════════════════════════════════════
%000000             S_WGT_CMD: begin
%000000                 sa_cmd       <= MODE_WGT_LOAD;
%000000                 sa_cmd_valid <= 1'b1;
                        // Begin loading column 0
%000000                 wgt_byte_idx    <= 0;
%000000                 wgt_read_issued <= 1'b0;
%000000                 wgt_data_ready  <= 1'b0;
                        // Compute k_pass_remain for this pass
~002450                 k_pass_remain <= (k_pass == k_pass_max)
%000000                     ? (k_depth - k_pass * ARRAY_SIZE_16)
                            : ARRAY_SIZE_16;
                        // Address for column wgt_col_idx, starting at k_pass offset:
                        //   INT8: byte_offset = col * k_depth + k_pass * ARRAY_SIZE, word=byte_off/4
                        //   INT16: byte_offset = (col * k_depth + k_pass * ARRAY_SIZE) * 2, word=byte_off/4
%000000                 begin : wgt_cmd_blk
                            reg [15:0] elem_off;
                            reg [15:0] byte_off;
%000000                     elem_off = wgt_col_idx * k_depth[15:0] + k_pass * ARRAY_SIZE_16;
~002450                     byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                     wgt_word_addr <= wgt_base + byte_off[15:2];
%000000                     wgt_bsel <= byte_off[1:0];
                        end
%000000                 state <= S_WGT_LOAD;
                    end
        
%000000             S_WGT_LOAD: begin
                        // Fill sa_wgt_data[0..k_pass_remain-1] for current column, zero-pad rest
                        // INT8: Read SRAM words (4 bytes each) and unpack with byte offset (wgt_bsel)
                        // INT16: Read SRAM words (2 half-words each) and unpack
~002450                 if (!wgt_read_issued) begin
                            // Phase 0: Issue SRAM read
%000000                     wgt_rd_en   <= 1'b1;
%000000                     wgt_rd_addr <= wgt_word_addr[WGT_ADDR_W-1:0];
%000000                     wgt_read_issued <= 1'b1;
%000000                     wgt_data_ready  <= 1'b0;
%000000                 end else if (!wgt_data_ready) begin
                            // Phase 1: Wait for SRAM read latency
%000000                     wgt_data_ready <= 1'b1;
%000000                 end else begin
                            // Phase 2: Data available from wgt_rd_data
%000000                     if (cfg_int16) begin : wgt_unpack_int16_blk
                                // INT16: extract 2 half-words per SRAM word
                                reg [31:0] shifted;
%000000                         shifted = wgt_rd_data >> (wgt_bsel * 8);
%000000                         if (wgt_byte_idx < k_pass_remain)
%000000                             sa_wgt_data[wgt_byte_idx[COL_W-1:0]] <= $signed(shifted[15:0]);
%000000                         if (wgt_byte_idx + 1 < k_pass_remain && wgt_bsel == 2'd0)
%000000                             sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 1] <= $signed(shifted[31:16]);
%000000                     end else begin : wgt_unpack_int8_blk
                                // INT8: extract 4 bytes, sign-extend each to 16-bit
                                reg [31:0] shifted;
%000000                         shifted = wgt_rd_data >> (wgt_bsel * 8);
%000000                         if (wgt_byte_idx < k_pass_remain)
%000000                             sa_wgt_data[wgt_byte_idx[COL_W-1:0]] <= {{8{shifted[7]}}, shifted[7:0]};
%000000                         if (wgt_byte_idx + 1 < k_pass_remain && wgt_bsel < 2'd3)
%000000                             sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 1] <= {{8{shifted[15]}}, shifted[15:8]};
%000000                         if (wgt_byte_idx + 2 < k_pass_remain && wgt_bsel < 2'd2)
%000000                             sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 2] <= {{8{shifted[23]}}, shifted[23:16]};
%000000                         if (wgt_byte_idx + 3 < k_pass_remain && wgt_bsel < 2'd1)
%000000                             sa_wgt_data[wgt_byte_idx[COL_W-1:0] + 3] <= {{8{shifted[31]}}, shifted[31:24]};
                            end
        
%000000                     begin : wgt_advance_blk
                                reg [COL_W:0] elems_this_word;
%000000                         if (cfg_int16)
%000000                             elems_this_word = (wgt_bsel == 2'd0) ? 2 : 1;
                                else
%000000                             elems_this_word = {2'd0, 3'd4 - {1'b0, wgt_bsel}};
%000000                         wgt_byte_idx <= wgt_byte_idx + elems_this_word;
        
%000000                         if (wgt_byte_idx + elems_this_word >= k_pass_remain) begin
                                    // All k_pass_remain elements loaded; zero-pad remaining rows
%000000                             if (k_pass_remain < ARRAY_SIZE_16) begin
%000000                                 for (i = 0; i < ARRAY_SIZE; i = i + 1)
%000000                                     if (i[COL_W-1:0] >= k_pass_remain[COL_W-1:0])
%000000                                         sa_wgt_data[i[COL_W-1:0]] <= 0;
                                    end
                                    // Emit
%000000                             state <= S_WGT_EMIT;
%000000                         end else begin
                                    // Need more words (next word starts at offset 0)
%000000                             wgt_word_addr <= wgt_word_addr + 1;
%000000                             wgt_bsel <= 2'd0;
%000000                             wgt_read_issued <= 1'b0;
                                end
                            end
                        end
                    end
        
%000000             S_WGT_EMIT: begin
                        // Pulse wgt_valid for this column
%000000                 sa_wgt_valid <= 1'b1;
        
%000000                 if (wgt_col_idx == COL_MAX) begin
                            // All columns loaded → go to spatial setup (compute act addr)
%000000                     state <= S_SPATIAL_SETUP;
%000000                 end else begin
                            // Next column
%000000                     wgt_col_idx <= wgt_col_idx + 1;
%000000                     wgt_byte_idx <= 0;
%000000                     wgt_read_issued <= 1'b0;
%000000                     wgt_data_ready  <= 1'b0;
%000000                     begin : wgt_emit_next_blk
                                reg [15:0] elem_off;
                                reg [15:0] byte_off;
%000000                         elem_off = (wgt_col_idx + 1) * k_depth[15:0]
%000000                                  + k_pass * ARRAY_SIZE_16;
%000000                         byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                         wgt_word_addr <= wgt_base + byte_off[15:2];
%000000                         wgt_bsel <= byte_off[1:0];
                            end
%000000                     state <= S_WGT_LOAD;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // ACTIVATION STREAM: send k_depth values, one per target row
                    // ══════════════════════════════════════════════════════════════
%000000             S_ACT_CMD: begin
                        // Wait for systolic to be ready before issuing COMPUTE
%000000                 if (sa_ready) begin
%000000                     sa_cmd       <= MODE_COMPUTE;
%000000                     sa_cmd_valid <= 1'b1;
%000000                     act_cnt      <= 0;
%000000                     act_byte_sel <= 2'd0;
%000000                     act_read_issued <= 1'b0;
%000000                     act_data_ready  <= 1'b0;
        
                            // Compute activation address for current (conv_fh, conv_fw)
%000000                     begin : act_addr_blk
                                reg signed [15:0] ih, iw;
                                reg signed [15:0] eh, ew;
                                reg [15:0] elem_off;
                                reg [15:0] byte_off;
%000000                         if (is_deconv) begin
                                    // Deconv: eh = oh + pad - fh, then check modulo
%000000                             eh = conv_ih_base - $signed({8'd0, conv_fh});
%000000                             ew = conv_iw_base - $signed({8'd0, conv_fw});
%000000                             if ((eh < 0) || (eh >= $signed({1'b0, deconv_exp_h}))
                                        || (ew < 0) || (ew >= $signed({1'b0, deconv_exp_w}))
                                        || (eh[15:0] % deconv_step_h[8:0] != 0)
%000000                                 || (ew[15:0] % deconv_step_w[8:0] != 0)) begin
%000000                                 conv_is_pad  <= 1'b0;
%000000                                 deconv_skip  <= 1'b1;
%000000                             end else begin
%000000                                 ih = $signed(eh[15:0] / deconv_step_h[8:0]);
%000000                                 iw = $signed(ew[15:0] / deconv_step_w[8:0]);
%000000                                 conv_is_pad  <= (ih < 0) || (ih >= $signed({1'b0, cfg_in_h}))
%000000                                              || (iw < 0) || (iw >= $signed({1'b0, cfg_in_w}));
%000000                                 deconv_skip  <= 1'b0;
%000000                                 elem_off = (ih[15:0] * cfg_in_w + iw[15:0]) * cfg_in_c
%000000                                          + conv_ch_cnt;
%000000                                 byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                                 act_word_addr <= {2'd0, act_base} + byte_off[15:2];
%000000                                 act_byte_sel <= byte_off[1:0];
                                    end
%000000                         end else begin
%000000                             ih = conv_ih_base + $signed({8'd0, conv_fh});
%000000                             iw = conv_iw_base + $signed({8'd0, conv_fw});
%000000                             conv_is_pad <= (ih < 0) || (ih >= $signed({1'b0, cfg_in_h}))
%000000                                         || (iw < 0) || (iw >= $signed({1'b0, cfg_in_w}));
%000000                             deconv_skip <= 1'b0;
%000000                             elem_off = (ih[15:0] * cfg_in_w + iw[15:0]) * cfg_in_c
%000000                                      + conv_ch_cnt;
%000000                             byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                             act_word_addr <= {2'd0, act_base} + byte_off[15:2];
%000000                             act_byte_sel <= byte_off[1:0];
                                end
                            end
%000000                     state <= S_ACT_LOAD;
                        end
                    end
        
%000000             S_ACT_LOAD: begin
                        // Read one SRAM word (4 activation bytes)
                        // If padding or deconv_skip, skip read and go directly to emit zeros
~002450                 if (conv_is_pad || deconv_skip) begin
%000000                     act_buf <= 32'd0;
%000000                     state <= S_ACT_EMIT;
%000000                 end else if (!act_read_issued) begin
%000000                     act_rd_en   <= 1'b1;
%000000                     act_rd_addr <= act_word_addr[ACT_ADDR_W-1:0];
%000000                     act_read_issued <= 1'b1;
%000000                     act_data_ready  <= 1'b0;
%000000                 end else if (!act_data_ready) begin
                            // Wait for SRAM read latency
%000000                     act_data_ready <= 1'b1;
%000000                 end else begin
                            // Data available
%000000                     act_buf <= act_rd_data;
%000000                     state <= S_ACT_EMIT;
                        end
                    end
        
%000000             S_ACT_EMIT: begin
                        // Send activation to row act_cnt ONLY (per-row targeting)
%000000                 begin : act_emit_blk
                            reg signed [DATA_W-1:0] aval;
%000000                     if (cfg_int16) begin
                                // INT16: extract half-word (2 elements per 32-bit word)
%000000                         case (act_byte_sel[1])
%000000                             1'b0: aval = $signed(act_buf[15:0]);
%000000                             1'b1: aval = $signed(act_buf[31:16]);
%000000                             default: aval = 0;
                                endcase
%000000                     end else begin
                                // INT8: extract byte, sign-extend to 16-bit
%000000                         case (act_byte_sel)
%000000                             2'd0: aval = {{8{act_buf[7]}},  act_buf[7:0]};
%000000                             2'd1: aval = {{8{act_buf[15]}}, act_buf[15:8]};
%000000                             2'd2: aval = {{8{act_buf[23]}}, act_buf[23:16]};
%000000                             2'd3: aval = {{8{act_buf[31]}}, act_buf[31:24]};
%000000                             default: aval = 0;
                                endcase
                            end
%000000                     for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
%000000                         if (i[COL_W-1:0] == act_cnt[COL_W-1:0])
%000000                             sa_act_data[i] <= aval;
                                else
%000000                             sa_act_data[i] <= 0;
                            end
                        end
%000000                 sa_act_valid <= 1'b1;
%000000                 act_cnt <= act_cnt + 1;
%000000                 conv_ch_cnt <= conv_ch_cnt + 1;
        
%000000                 if (act_cnt + 1 >= k_pass_remain) begin
                            // All K elements for this pass streamed → flush
%000000                     flush_cnt <= 0;
%000000                     state <= S_ACT_FLUSH;
%000000                 end else if (conv_ch_cnt + 1 >= cfg_in_c) begin
                            // Reached end of channels for current (fh, fw)
                            // Advance to next filter position
%000000                     conv_ch_cnt <= 0;
%000000                     if (conv_fw + 1 >= {8'd0, cfg_kernel_w}) begin
%000000                         conv_fw <= 0;
%000000                         conv_fh <= conv_fh + 1;
%000000                     end else begin
%000000                         conv_fw <= conv_fw + 1;
                            end
                            // Need to recompute address for new (fh, fw)
%000000                     act_read_issued <= 1'b0;
%000000                     act_data_ready  <= 1'b0;
%000000                     state <= S_ACT_LOAD;
%000000                     begin : next_pos_blk
                                reg signed [15:0] nih, niw;
                                reg signed [15:0] neh, new_;
                                reg [15:0] elem_off_n;
                                reg [15:0] byte_off_n;
                                reg [7:0] next_fw, next_fh;
%000000                         if (conv_fw + 1 >= {8'd0, cfg_kernel_w}) begin
%000000                             next_fw = 0;
%000000                             next_fh = conv_fh + 1;
%000000                         end else begin
%000000                             next_fw = conv_fw + 1;
%000000                             next_fh = conv_fh;
                                end
%000000                         if (is_deconv) begin
%000000                             neh = conv_ih_base - $signed({8'd0, next_fh});
%000000                             new_ = conv_iw_base - $signed({8'd0, next_fw});
%000000                             if ((neh < 0) || (neh >= $signed({1'b0, deconv_exp_h}))
                                        || (new_ < 0) || (new_ >= $signed({1'b0, deconv_exp_w}))
                                        || (neh[15:0] % deconv_step_h[8:0] != 0)
%000000                                 || (new_[15:0] % deconv_step_w[8:0] != 0)) begin
%000000                                 conv_is_pad  <= 1'b0;
%000000                                 deconv_skip  <= 1'b1;
%000000                             end else begin
%000000                                 nih = $signed(neh[15:0] / deconv_step_h[8:0]);
%000000                                 niw = $signed(new_[15:0] / deconv_step_w[8:0]);
%000000                                 conv_is_pad <= (nih < 0) || (nih >= $signed({1'b0, cfg_in_h}))
%000000                                             || (niw < 0) || (niw >= $signed({1'b0, cfg_in_w}));
%000000                                 deconv_skip <= 1'b0;
%000000                                 elem_off_n = (nih[15:0] * cfg_in_w + niw[15:0]) * cfg_in_c;
%000000                                 byte_off_n = cfg_int16 ? (elem_off_n << 1) : elem_off_n;
%000000                                 act_word_addr <= {2'd0, act_base} + byte_off_n[15:2];
%000000                                 act_byte_sel <= byte_off_n[1:0];
                                    end
%000000                         end else begin
%000000                             nih = conv_ih_base + $signed({8'd0, next_fh});
%000000                             niw = conv_iw_base + $signed({8'd0, next_fw});
%000000                             conv_is_pad <= (nih < 0) || (nih >= $signed({1'b0, cfg_in_h}))
%000000                                         || (niw < 0) || (niw >= $signed({1'b0, cfg_in_w}));
%000000                             deconv_skip <= 1'b0;
%000000                             elem_off_n = (nih[15:0] * cfg_in_w + niw[15:0]) * cfg_in_c;
%000000                             byte_off_n = cfg_int16 ? (elem_off_n << 1) : elem_off_n;
%000000                             act_word_addr <= {2'd0, act_base} + byte_off_n[15:2];
%000000                             act_byte_sel <= byte_off_n[1:0];
                                end
                            end
%000000                 end else if (cfg_int16 ? (act_byte_sel[1] == 1'b1) : (act_byte_sel == 2'd3)) begin
                            // Need next SRAM word
%000000                     act_byte_sel <= 2'd0;
%000000                     act_word_addr <= act_word_addr + 1;
%000000                     act_read_issued <= 1'b0;
%000000                     state <= S_ACT_LOAD;
%000000                 end else begin
                            // Next element in same word
%000000                     act_byte_sel <= cfg_int16 ? (act_byte_sel + 2'd2) : (act_byte_sel + 2'd1);
                        end
                    end
        
%000000             S_ACT_FLUSH: begin
                        // Wait ARRAY_SIZE-1 cycles for pipeline propagation
%000000                 flush_cnt <= flush_cnt + 1;
%000000                 if (flush_cnt >= ARRAY_SIZE_16 - 1) begin
%000000                     drain_col <= 0;
%000000                     state <= S_DRAIN_CMD;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // DRAIN: iterate columns, each takes 2 cycles (DRAIN + DRAIN_OUT)
                    // ══════════════════════════════════════════════════════════════
%000000             S_DRAIN_CMD: begin
                        // Systolic accepts DRAIN from S_COMPUTE or S_READY
%000000                 sa_cmd           <= MODE_DRAIN;
%000000                 sa_cmd_valid     <= 1'b1;
%000000                 sa_drain_col_sel <= drain_col;
%000000                 state            <= S_DRAIN_WAIT;
                    end
        
%000000             S_DRAIN_WAIT: begin
%000000                 if (sa_acc_out_valid) begin
                            // Capture per-row results for this column
%000000                     for (i = 0; i < ARRAY_SIZE; i = i + 1)
%000000                         acc_buf[i] <= sa_acc_out[i];
                            // Reduce: sum rows 0..k_depth-1 into dot product
%000000                     reduce_cnt <= 0;
%000000                     dot_acc <= 0;
%000000                     state <= S_REDUCE;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // REDUCE: sum k_pass_remain partial products into one dot product
                    // Accumulate across passes in dot_buf[drain_col]
                    // ══════════════════════════════════════════════════════════════
%000000             S_REDUCE: begin
%000000                 dot_acc <= dot_acc + acc_buf[reduce_cnt[COL_W-1:0]];
%000000                 reduce_cnt <= reduce_cnt + 1;
%000000                 if (reduce_cnt + 1 >= k_pass_remain) begin
                            // Accumulate reduced dot product into dot_buf[drain_col]
%000000                     dot_buf[drain_col] <= dot_buf[drain_col]
%000000                         + dot_acc + acc_buf[reduce_cnt[COL_W-1:0]];
                            // Next column or decide next step
%000000                     if (drain_col == COL_MAX) begin
                                // All columns drained for this pass
%000000                         if (k_pass >= k_pass_max) begin
                                    // Final pass — proceed to PPU
%000000                             param_word_idx <= 0;
%000000                             param_read_issued <= 1'b0;
%000000                             param_data_ready  <= 1'b0;
%000000                             drain_col <= 0;
%000000                             state <= S_PARAM_LOAD;
%000000                         end else begin
                                    // More passes needed — reload weights for next slice
%000000                             k_pass <= k_pass + 1;
%000000                             wgt_col_idx <= 0;
%000000                             state <= S_WGT_CMD;
                                end
%000000                     end else begin
%000000                         drain_col <= drain_col + 1;
%000000                         state <= S_DRAIN_CMD;
                            end
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // PARAM READ: 4 words per channel — now iterates drain_col
                    // as the output channel index (0..ARRAY_SIZE-1)
                    // ══════════════════════════════════════════════════════════════
%000000             S_PARAM_LOAD: begin
~001552                 if (!param_read_issued) begin
%000000                     param_rd_en   <= 1'b1;
%000000                     param_rd_addr <= param_base + drain_col * 4 + param_word_idx;
%000000                     param_read_issued <= 1'b1;
%000000                     param_data_ready  <= 1'b0;
%000000                 end else if (!param_data_ready) begin
                            // Wait for SRAM read latency
%000000                     param_data_ready <= 1'b1;
%000000                 end else begin
%000000                     param_buf[param_word_idx] <= param_rd_data;
%000000                     if (param_word_idx == 3'd3) begin
%000000                         ppu_mult_m     <= param_buf[0][14:0];
%000000                         ppu_shift_s    <= param_buf[0][21:16];
%000000                         ppu_zero_point <= $signed(param_buf[1][15:0]);
%000000                         ppu_bias       <= $signed({param_buf[3][15:0], param_buf[2],
%000000                                                    param_buf[1][31:16]});
%000000                         state <= S_PPU_FEED;
%000000                     end else begin
%000000                         param_word_idx <= param_word_idx + 1;
%000000                         param_read_issued <= 1'b0;
                            end
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // PPU FEED: send ONE dot product (dot_buf[drain_col]) to PPU
                    // ══════════════════════════════════════════════════════════════
%000000             S_PPU_FEED: begin
%000000                 ppu_acc_in   <= dot_buf[drain_col];
%000000                 ppu_in_valid <= 1'b1;
%000000                 state        <= S_PPU_WAIT;
                    end
        
%000000             S_PPU_WAIT: begin
                        // Wait for PPU output (4-cycle pipeline)
%000000                 if (ppu_out_valid) begin
%000000                     state <= S_WRITEBACK;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // WRITEBACK: pack output bytes and write SRAM words
                    // Output NHWC layout: byte at (sp_oh*out_tile_w + sp_ow)*out_c + oc
                    // Since oc_group*ARRAY_SIZE and ARRAY_SIZE are multiples of 4,
                    // drain_col[1:0] directly gives byte position within SRAM word.
                    // ══════════════════════════════════════════════════════════════
%000000             S_WRITEBACK: begin
                        // Pack output elements into wb_pack and write SRAM words
                        // INT8: 4 channels per word; INT16: 2 channels per word
%000000                 if (cfg_int16) begin
%000000                     case (drain_col[0])
%000000                         1'b0: wb_pack[15:0] <= ppu_out_data;
%000000                         1'b1: begin
                                    // Write full word (2 output channels)
%000000                             act_wr_en   <= 1'b1;
%000000                             act_wr_addr <= out_base +
%000000                                 (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
%000000                                   + oc_group * ARRAY_SIZE_16
%000000                                   + ({12'd0, drain_col} & ~16'd1)) >> 1);
%000000                             act_wr_data <= {ppu_out_data, wb_pack[15:0]};
                                end
                            endcase
%000000                 end else begin
%000000                     case (drain_col[1:0])
%000000                         2'd0: wb_pack[7:0]   <= ppu_out_data[7:0];
%000000                         2'd1: wb_pack[15:8]  <= ppu_out_data[7:0];
%000000                         2'd2: wb_pack[23:16] <= ppu_out_data[7:0];
%000000                         2'd3: begin
                                    // Write full word (4 output channels)
%000000                             act_wr_en   <= 1'b1;
%000000                             act_wr_addr <= out_base +
%000000                                 (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
%000000                                   + oc_group * ARRAY_SIZE_16
%000000                                   + ({12'd0, drain_col} & ~16'd3)) >> 2);
%000000                             act_wr_data <= {ppu_out_data[7:0], wb_pack[23:0]};
                                end
                            endcase
                        end
        
                        // Advance to next channel or finish pixel
%000000                 if (drain_col == col_last) begin
                            // Flush remaining partial word
%000000                     if (cfg_int16) begin
%000000                         if (drain_col[0] != 1'b1) begin
%000000                             act_wr_en   <= 1'b1;
%000000                             act_wr_addr <= out_base +
%000000                                 (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
%000000                                   + oc_group * ARRAY_SIZE_16
%000000                                   + ({12'd0, drain_col} & ~16'd1)) >> 1);
%000000                             act_wr_data <= wb_pack;
                                end
%000000                     end else begin
%000000                         if (drain_col[1:0] != 2'd3) begin
%000000                             act_wr_en   <= 1'b1;
%000000                             act_wr_addr <= out_base +
%000000                                 (((sp_oh * out_tile_w + sp_ow) * cfg_out_c
%000000                                   + oc_group * ARRAY_SIZE_16
%000000                                   + ({12'd0, drain_col} & ~16'd3)) >> 2);
%000000                             act_wr_data <= wb_pack;
                                end
                            end
%000000                     state <= S_PIXEL_NEXT;
%000000                 end else begin
%000000                     drain_col <= drain_col + 1;
%000000                     param_word_idx <= 0;
%000000                     param_read_issued <= 1'b0;
%000000                     param_data_ready  <= 1'b0;
%000000                     state <= S_PARAM_LOAD;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // SPATIAL SETUP: compute per-pixel activation address
                    // ══════════════════════════════════════════════════════════════
%000000             S_SPATIAL_SETUP: begin
                        // Compute input window origin for output pixel (sp_oh, sp_ow)
                        // Use global output coordinates for input address & padding check
%000000                 if (is_deconv) begin
                            // Deconv: ih_base = oh + pad_top (fh subtracted later in S_ACT_CMD)
%000000                     conv_ih_base <= $signed({1'b0, tile_oh_origin + sp_oh})
%000000                                   + $signed({1'b0, cfg_pad_top[7:0]});
%000000                     conv_iw_base <= $signed({1'b0, tile_ow_origin + sp_ow})
%000000                                   + $signed({1'b0, cfg_pad_left[7:0]});
%000000                 end else begin
%000000                     conv_ih_base <= $signed({1'b0, tile_oh_origin + sp_oh}) * $signed({1'b0, cfg_stride_h[7:0]})
%000000                                   - $signed({1'b0, cfg_pad_top[7:0]});
%000000                     conv_iw_base <= $signed({1'b0, tile_ow_origin + sp_ow}) * $signed({1'b0, cfg_stride_w[7:0]})
%000000                                   - $signed({1'b0, cfg_pad_left[7:0]});
                        end
        
                        // Compute starting (fh, fw, ch) from k_pass offset
                        // flat_start = k_pass * ARRAY_SIZE
                        // fh = flat_start / (kw * in_c)
                        // fw = (flat_start / in_c) % kw
                        // ch = flat_start % in_c
%000000                 begin : spatial_setup_blk
                            reg [15:0] flat_start;
                            reg [15:0] kw_x_inc;
%000000                     flat_start = k_pass * ARRAY_SIZE_16;
%000000                     kw_x_inc = {8'd0, cfg_kernel_w} * cfg_in_c;
~002289                     if (cfg_kernel_h == 8'd1 && cfg_kernel_w == 8'd1) begin
                                // 1×1 conv: simple contiguous addressing
%000000                         conv_fh <= 0;
%000000                         conv_fw <= 0;
%000000                         conv_ch_cnt <= flat_start;
%000000                     end else begin
                                // General conv: decompose flat_start into (fh, fw, ch)
%000000                         conv_fh <= flat_start / kw_x_inc;
%000000                         conv_fw <= (flat_start % kw_x_inc) / cfg_in_c;
%000000                         conv_ch_cnt <= flat_start % cfg_in_c;
                            end
                        end
        
%000000                 conv_elem_cnt <= 0;
%000000                 drain_col <= 0;
%000000                 wb_pack <= 0;
                        // Clear partial_sum on first pass of new pixel
%000000                 if (k_pass == 0) begin
%000000                     for (i = 0; i < ARRAY_SIZE; i = i + 1)
%000000                         dot_buf[i] <= 0;
                        end
%000000                 state <= S_ACT_CMD;
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // PIXEL NEXT: advance spatial pixel or go to OC_NEXT
                    // ══════════════════════════════════════════════════════════════
%000000             S_PIXEL_NEXT: begin
                        // Reset k_pass for next pixel
%000000                 k_pass <= 0;
%000000                 if (sp_ow + 1 >= out_tile_w) begin
%000000                     if (sp_oh + 1 >= out_tile_h) begin
                                // All pixels done for this OC group
%000000                         state <= S_OC_NEXT;
%000000                     end else begin
%000000                         sp_ow <= 0;
%000000                         sp_oh <= sp_oh + 1;
%000000                         if (k_pass_max > 0) begin
%000000                             wgt_col_idx <= 0;
%000000                             state <= S_WGT_CMD;
                                end else
%000000                             state <= S_SPATIAL_SETUP;
                            end
%000000                 end else begin
%000000                     sp_ow <= sp_ow + 1;
%000000                     if (k_pass_max > 0) begin
%000000                         wgt_col_idx <= 0;
%000000                         state <= S_WGT_CMD;
                            end else
%000000                         state <= S_SPATIAL_SETUP;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
 000032             S_OC_NEXT: begin
~000030                 if (oc_group + 1 >= oc_groups_total) begin
%000002                     state <= S_TILE_NEXT;
 000030                 end else begin
 000030                     oc_group <= oc_group + 1;
 000030                     if (cfg_op_type == 8'd1) begin
                                // DW Conv: next channel
 000030                         dw_cnt <= 0;
 000030                         dw_read_issued <= 1'b0;
 000030                         dw_init_phase <= 2'd0;
 000030                         state <= S_DW_WGT_LOAD;
%000000                     end else if (cfg_op_type == 8'd5) begin
%000000                         rsz_ch <= rsz_ch + 1;
%000000                         param_word_idx <= 0;
%000000                         param_read_issued <= 1'b0;
%000000                         state <= S_RESIZE_CH_SETUP;
%000000                     end else begin
%000000                         state <= S_OC_SETUP;
                            end
                        end
                    end
        
%000002             S_TILE_NEXT: begin
%000002                 if (cfg_tile_h == 16'd0) begin
%000002                     state <= S_DONE;
%000000                 end else if (tile_x + 1 >= cfg_tile_num_w) begin
%000000                     if (tile_y + 1 >= cfg_tile_num_h) begin
%000000                         state <= S_DONE;
%000000                     end else begin
%000000                         tile_x <= 0;
%000000                         tile_y <= tile_y + 1;
%000000                         state <= S_TILE_SETUP;
                            end
%000000                 end else begin
%000000                     tile_x <= tile_x + 1;
%000000                     state <= S_TILE_SETUP;
                        end
                    end
        
%000002             S_DONE: begin
%000002                 done  <= 1'b1;
%000002                 state <= S_IDLE;
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // DW Conv Path — Full Implementation
                    // Flow: WGT_LOAD → PARAM → COMPUTE → (ACT_STREAM → PPU_WAIT)* → PPU → OC_NEXT
                    // ══════════════════════════════════════════════════════════════
 000128             S_DW_WGT_LOAD: begin
                        // Load kh*kw weights for channel oc_group from Weight SRAM
 000128                 dw_wgt_load <= 1'b1;
        
 000128                 case (dw_init_phase)
 000032                 2'd0: begin
                            // Phase 0: setup
 000032                     dw_kernel_size <= cfg_kernel_h[3:0] * cfg_kernel_w[3:0];
 000032                     begin : dw_wgt_addr_setup
                                reg [15:0] wgt_elem_start;
                                reg [15:0] wgt_byte_start;
 000032                         wgt_elem_start = oc_group * {2'd0, cfg_kernel_h[3:0]}
 000032                                        * {2'd0, cfg_kernel_w[3:0]};
~002450                         wgt_byte_start = cfg_int16 ? (wgt_elem_start << 1) : wgt_elem_start;
 000032                         wgt_word_addr <= {6'd0, wgt_base} + wgt_byte_start[15:2];
 000032                         dw_wgt_bsel_base <= wgt_byte_start[1:0];
                            end
 000032                     dw_init_phase <= 2'd1;
                        end
 000032                 2'd1: begin
                            // Phase 1: send acc_clear, issue first SRAM read
 000032                     dw_acc_clear <= 1'b1;
 000032                     wgt_rd_en   <= 1'b1;
 000032                     wgt_rd_addr <= wgt_word_addr[WGT_ADDR_W-1:0];
 000032                     dw_read_issued <= 1'b0;
 000032                     dw_init_phase <= 2'd2;
                        end
 000064                 2'd2: begin
                            // Phase 2+: weight feeding loop
 001906                     if (!dw_read_issued) begin
 000032                         dw_read_issued <= 1'b1;
 000032                     end else begin
                                // SRAM data available — extract element and feed
~000032                         if (cfg_int16) begin : dw_wgt_extract_int16
                                    reg [1:0] bsel;
%000000                             bsel = {dw_cnt[0], 1'b0} + dw_wgt_bsel_base;
%000000                             case (bsel[1])
%000000                                 1'b0: dw_wgt_data <= $signed(wgt_rd_data[15:0]);
%000000                                 1'b1: dw_wgt_data <= $signed(wgt_rd_data[31:16]);
                                    endcase
 000032                         end else begin : dw_wgt_extract_int8
                                    reg [1:0] bsel;
 000032                             bsel = dw_cnt[1:0] + dw_wgt_bsel_base;
 000032                             case (bsel)
%000008                                 2'd0: dw_wgt_data <= {{8{wgt_rd_data[7]}},  wgt_rd_data[7:0]};
%000008                                 2'd1: dw_wgt_data <= {{8{wgt_rd_data[15]}}, wgt_rd_data[15:8]};
%000008                                 2'd2: dw_wgt_data <= {{8{wgt_rd_data[23]}}, wgt_rd_data[23:16]};
%000008                                 2'd3: dw_wgt_data <= {{8{wgt_rd_data[31]}}, wgt_rd_data[31:24]};
                                    endcase
                                end
 000032                         dw_wgt_valid <= 1'b1;
 000032                         dw_cnt <= dw_cnt + 1;
        
~000032                         if (dw_cnt + 1 >= dw_kernel_size) begin
                                    // All weights loaded
 000032                             state <= S_DW_DRAIN;
%000000                         end else begin
                                    // Check if need next SRAM word
%000000                             if (cfg_int16) begin
                                        // INT16: 2 elements per word; need next word when bsel[1]==1
%000000                                 if (({dw_cnt[0], 1'b0} + dw_wgt_bsel_base) >= 2'd2) begin
%000000                                     wgt_word_addr <= wgt_word_addr + 1;
%000000                                     wgt_rd_en    <= 1'b1;
%000000                                     wgt_rd_addr  <= wgt_word_addr[WGT_ADDR_W-1:0] + 1;
%000000                                     dw_read_issued <= 1'b0;
                                        end
%000000                             end else begin
                                        // INT8: 4 elements per word
%000000                                 if ((dw_cnt[1:0] + dw_wgt_bsel_base) == 2'd3) begin
%000000                                     wgt_word_addr <= wgt_word_addr + 1;
%000000                                     wgt_rd_en    <= 1'b1;
%000000                                     wgt_rd_addr  <= wgt_word_addr[WGT_ADDR_W-1:0] + 1;
%000000                                     dw_read_issued <= 1'b0;
                                        end
                                    end
                                end
                            end
                        end
%000000                 default: dw_init_phase <= 2'd0;
                        endcase
                    end
        
 000032             S_DW_DRAIN: begin
                        // Transition state: deassert wgt_load, go to param
 000032                 dw_wgt_load <= 1'b0;
 000032                 dw_cnt <= 0;
 000032                 param_word_idx <= 0;
 000032                 param_read_issued <= 1'b0;
 000032                 state <= S_DW_PARAM;
                    end
        
 000384             S_DW_PARAM: begin
                        // Load 4 PPU param words for current channel
                        // 3-phase per word: issue → wait → capture
 001552                 if (!param_read_issued) begin
 000128                     param_rd_en   <= 1'b1;
 000128                     param_rd_addr <= (oc_group * 4) + param_word_idx;
 000128                     param_read_issued <= 1'b1;
 000128                     act_read_issued <= 1'b0;  // reuse as wait flag
 000128                 end else if (!act_read_issued) begin
                            // Wait for SRAM read latency
 000128                     act_read_issued <= 1'b1;
 000128                 end else begin
 000128                     param_buf[param_word_idx] <= param_rd_data;
 000096                     if (param_word_idx == 3'd3) begin
                                // Extract params
 000032                         ppu_mult_m     <= param_buf[0][14:0];
 000032                         ppu_shift_s    <= param_buf[0][21:16];
 000032                         ppu_zero_point <= $signed(param_buf[1][15:0]);
 000032                         ppu_bias       <= $signed({param_buf[3][15:0], param_buf[2],
 000032                                                    param_buf[1][31:16]});
 000032                         state <= S_DW_COMPUTE;
 000096                     end else begin
 000096                         param_word_idx <= param_word_idx + 1;
 000096                         param_read_issued <= 1'b0;
                            end
                        end
                    end
        
 000032             S_DW_COMPUTE: begin
                        // Initialize pixel loop
 000032                 dw_oh <= 0;
 000032                 dw_ow <= 0;
 000032                 dw_fh <= 0;
 000032                 dw_fw <= 0;
 000032                 dw_read_issued <= 1'b0;
 000032                 state <= S_DW_ACT_STREAM;
                    end
        
 000096             S_DW_ACT_STREAM: begin
                        // Stream window elements for pixel (dw_oh, dw_ow)
 000096                 begin : dw_act_stream_blk
                            reg signed [15:0] ih_s, iw_s;
                            reg               is_pad;
 000096                     ih_s = $signed({1'b0, tile_oh_origin + dw_oh}) * $signed({1'b0, cfg_stride_h[7:0]})
 000096                          - $signed({1'b0, cfg_pad_top[7:0]}) + $signed({1'b0, dw_fh});
 000096                     iw_s = $signed({1'b0, tile_ow_origin + dw_ow}) * $signed({1'b0, cfg_stride_w[7:0]})
 000096                          - $signed({1'b0, cfg_pad_left[7:0]}) + $signed({1'b0, dw_fw});
 000096                     is_pad = (ih_s < 0) || (ih_s >= $signed({1'b0, cfg_in_h}))
~002388                           || (iw_s < 0) || (iw_s >= $signed({1'b0, cfg_in_w}));
        
%000000                     if (is_pad) begin
                                // Padding: feed zero
%000000                         dw_in_valid <= 1'b1;
%000000                         dw_in_data  <= {DATA_W{1'b0}};
%000000                         if (dw_fh == 0 && dw_fw == 0)
%000000                             dw_acc_clear <= 1'b1;
        
                                // Advance filter position
%000000                         if (dw_fw + 1 >= cfg_kernel_w[3:0]) begin
%000000                             dw_fw <= 0;
%000000                             if (dw_fh + 1 >= cfg_kernel_h[3:0]) begin
%000000                                 state <= S_DW_PPU_WAIT;
%000000                                 ppu_wait_cnt <= 0;
%000000                             end else begin
%000000                                 dw_fh <= dw_fh + 1;
                                    end
%000000                         end else begin
%000000                             dw_fw <= dw_fw + 1;
                                end
%000000                         dw_read_issued <= 1'b0;
 000064                     end else if (!dw_read_issued) begin
                                // In-bounds: issue SRAM read
 000032                         begin : dw_addr_calc
                                    reg [ACT_ADDR_W+15:0] elem_off;
                                    reg [ACT_ADDR_W+15:0] byte_off;
 000032                             elem_off = (ih_s[15:0] * cfg_in_w * cfg_in_c)
 000032                                      + (iw_s[15:0] * cfg_in_c)
 000032                                      + oc_group;
~000032                             byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
 000032                             act_rd_en   <= 1'b1;
 000032                             act_rd_addr <= byte_off[ACT_ADDR_W+1:2];
 000032                             act_byte_sel <= byte_off[1:0];
                                end
 000032                         dw_read_issued <= 1'b1;
 000032                         act_read_issued <= 1'b0;
~000032                         if (dw_fh == 0 && dw_fw == 0)
 000032                             dw_acc_clear <= 1'b1;
 000032                     end else if (!act_read_issued) begin
 000032                         act_read_issued <= 1'b1;
 000032                     end else begin
                                // SRAM data available — extract element and feed
~000032                         if (cfg_int16) begin : dw_act_extract_int16
%000000                             case (act_byte_sel[1])
%000000                                 1'b0: dw_in_data <= $signed(act_rd_data[15:0]);
%000000                                 1'b1: dw_in_data <= $signed(act_rd_data[31:16]);
                                    endcase
 000032                         end else begin : dw_act_extract_int8
 000032                             case (act_byte_sel)
%000008                                 2'd0: dw_in_data <= {{8{act_rd_data[7]}},  act_rd_data[7:0]};
%000008                                 2'd1: dw_in_data <= {{8{act_rd_data[15]}}, act_rd_data[15:8]};
%000008                                 2'd2: dw_in_data <= {{8{act_rd_data[23]}}, act_rd_data[23:16]};
%000008                                 2'd3: dw_in_data <= {{8{act_rd_data[31]}}, act_rd_data[31:24]};
                                    endcase
                                end
 000032                         dw_in_valid <= 1'b1;
 000032                         dw_read_issued <= 1'b0;
        
                                // Advance filter position
~000032                         if (dw_fw + 1 >= cfg_kernel_w[3:0]) begin
 000032                             dw_fw <= 0;
~000032                             if (dw_fh + 1 >= cfg_kernel_h[3:0]) begin
 000032                                 state <= S_DW_PPU_WAIT;
 000032                                 ppu_wait_cnt <= 0;
%000000                             end else begin
%000000                                 dw_fh <= dw_fh + 1;
                                    end
%000000                         end else begin
%000000                             dw_fw <= dw_fw + 1;
                                end
                            end
                        end
                    end
        
 000224             S_DW_PPU_WAIT: begin
                        // Wait for dw_out_valid, then feed PPU, wait for PPU output
 000160                 if (ppu_wait_cnt == 0) begin
 000032                     if (dw_out_valid) begin
 000032                         ppu_acc_in   <= dw_acc_out;
 000032                         ppu_in_valid <= 1'b1;
 000032                         ppu_wait_cnt <= 1;
                            end
 000160                 end else begin
 000128                     if (ppu_out_valid) begin
                                // Compute NHWC address for this pixel/channel
 000032                         begin : dw_wb_addr_calc
                                    reg [31:0] elem_off;
                                    reg [31:0] byte_off;
 000032                             elem_off = (dw_oh * out_tile_w + dw_ow)
 000032                                      * cfg_out_c + oc_group;
~000032                             byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
 000032                             dw_wb_addr    <= out_base + byte_off[ACT_ADDR_W+1:2];
 000032                             dw_wb_bytesel <= byte_off[1:0];
                                end
 000032                         dw_wb_byte  <= ppu_out_data;
 000032                         dw_wb_phase <= 2'd0;
 000032                         state <= S_DW_WB;
 000128                     end else begin
 000128                         ppu_wait_cnt <= ppu_wait_cnt + 1;
                            end
                        end
                    end
        
 000096             S_DW_WB: begin
                        // Read-modify-write: place output element at correct NHWC position
 000096                 case (dw_wb_phase)
 000032                 2'd0: begin
 000032                     act_rd_en   <= 1'b1;
 000032                     act_rd_addr <= dw_wb_addr;
 000032                     dw_wb_phase <= 2'd1;
                        end
 000032                 2'd1: begin
 000032                     dw_wb_phase <= 2'd2;
                        end
 000032                 2'd2: begin
                            // Merge element into read word and write back
 000032                     begin : dw_wb_merge
                                reg [31:0] merged;
 000032                         merged = act_rd_data;
~000032                         if (cfg_int16) begin
                                    // INT16: merge half-word
%000000                             case (dw_wb_bytesel[1])
%000000                                 1'b0: merged[15:0]  = dw_wb_byte;
%000000                                 1'b1: merged[31:16] = dw_wb_byte;
                                    endcase
 000032                         end else begin
                                    // INT8: merge byte
 000032                             case (dw_wb_bytesel)
%000008                                 2'd0: merged[7:0]   = dw_wb_byte[7:0];
%000008                                 2'd1: merged[15:8]  = dw_wb_byte[7:0];
%000008                                 2'd2: merged[23:16] = dw_wb_byte[7:0];
%000008                                 2'd3: merged[31:24] = dw_wb_byte[7:0];
                                    endcase
                                end
 000032                         act_wr_en   <= 1'b1;
 000032                         act_wr_addr <= dw_wb_addr;
 000032                         act_wr_data <= merged;
                            end
        
                            // Advance to next pixel
~000032                     if (dw_ow + 1 >= out_tile_w) begin
 000032                         dw_ow <= 0;
~000032                         if (dw_oh + 1 >= out_tile_h) begin
 000032                             state <= S_OC_NEXT;
%000000                         end else begin
%000000                             dw_oh <= dw_oh + 1;
%000000                             dw_fh <= 0;
%000000                             dw_fw <= 0;
%000000                             dw_read_issued <= 1'b0;
%000000                             ppu_wait_cnt <= 0;
%000000                             state <= S_DW_ACT_STREAM;
                                end
%000000                     end else begin
%000000                         dw_ow <= dw_ow + 1;
%000000                         dw_fh <= 0;
%000000                         dw_fw <= 0;
%000000                         dw_read_issued <= 1'b0;
%000000                         ppu_wait_cnt <= 0;
%000000                         state <= S_DW_ACT_STREAM;
                            end
                        end
%000000                 default: dw_wb_phase <= 2'd0;
                        endcase
                    end
        
%000000             S_DW_PPU: begin
                        // No longer used (writeback handled in S_DW_WB)
%000000                 state <= S_OC_NEXT;
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // POOLING Path (op_type=3)
                    // Flow: SETUP → CH_SETUP → [READ → ACC]* → DIV → PPU → PPU_WAIT → WB → PIX_NEXT → CH_NEXT
                    // ══════════════════════════════════════════════════════════════
%000000             S_POOL_SETUP: begin
                        // Latch effective pool kernel/stride (global override)
%000000                 if (global_pool) begin
%000000                     pool_kh <= cfg_in_h[7:0];
%000000                     pool_kw <= cfg_in_w[7:0];
%000000                     pool_sh <= cfg_in_h[7:0];
%000000                     pool_sw <= cfg_in_w[7:0];
%000000                 end else begin
%000000                     pool_kh <= {4'd0, pool_cfg_h};
%000000                     pool_kw <= {4'd0, pool_cfg_w};
%000000                     pool_sh <= {4'd0, pool_cfg_sh};
%000000                     pool_sw <= {4'd0, pool_cfg_sw};
                        end
%000000                 pool_ch <= 0;
%000000                 param_word_idx <= 0;
%000000                 param_read_issued <= 1'b0;
%000000                 state <= S_POOL_CH_SETUP;
                    end
        
%000000             S_POOL_CH_SETUP: begin
                        // Load 4 PPU param words for current channel (same as DW_PARAM)
~001552                 if (!param_read_issued) begin
%000000                     param_rd_en   <= 1'b1;
%000000                     param_rd_addr <= (pool_ch * 4) + param_word_idx;
%000000                     param_read_issued <= 1'b1;
%000000                     act_read_issued <= 1'b0;
%000000                 end else if (!act_read_issued) begin
%000000                     act_read_issued <= 1'b1;
%000000                 end else begin
%000000                     param_buf[param_word_idx] <= param_rd_data;
%000000                     if (param_word_idx == 3'd3) begin
%000000                         ppu_mult_m     <= param_buf[0][14:0];
%000000                         ppu_shift_s    <= param_buf[0][21:16];
%000000                         ppu_zero_point <= $signed(param_buf[1][15:0]);
%000000                         ppu_bias       <= $signed({param_buf[3][15:0], param_buf[2],
%000000                                                    param_buf[1][31:16]});
                                // Start spatial loop for this channel
%000000                         pool_oh <= 0;
%000000                         pool_ow <= 0;
%000000                         pool_fh <= 0;
%000000                         pool_fw <= 0;
%000000                         pool_rd_phase <= 0;
                                // Initialize accumulator
%000000                         if (pool_mode) begin
%000000                             pool_acc <= 0;  // AvgPool: sum=0
%000000                         end else begin
%000000                             pool_acc <= {1'b1, {(ACC_W-1){1'b0}}};  // MaxPool: -2^(ACC_W-1)
                                end
%000000                         pool_count <= 0;
%000000                         state <= S_POOL_READ;
%000000                     end else begin
%000000                         param_word_idx <= param_word_idx + 1;
%000000                         param_read_issued <= 1'b0;
                            end
                        end
                    end
        
%000000             S_POOL_READ: begin
                        // Compute input coords, bounds-check, issue SRAM read
%000000                 begin : pool_read_blk
                            reg signed [15:0] ih_s, iw_s;
                            reg               is_oob;
%000000                     ih_s = $signed({1'b0, tile_oh_origin + pool_oh}) * $signed({1'b0, pool_sh})
%000000                          - $signed({1'b0, cfg_pad_top[7:0]}) + $signed({1'b0, pool_fh});
%000000                     iw_s = $signed({1'b0, tile_ow_origin + pool_ow}) * $signed({1'b0, pool_sw})
%000000                          - $signed({1'b0, cfg_pad_left[7:0]}) + $signed({1'b0, pool_fw});
%000000                     is_oob = (ih_s < 0) || (ih_s >= $signed({1'b0, cfg_in_h}))
~002388                           || (iw_s < 0) || (iw_s >= $signed({1'b0, cfg_in_w}));
        
%000000                     if (is_oob) begin
                                // Out-of-bounds: skip, advance window position
%000000                         if ({4'd0, pool_fw} + 1 >= pool_kw) begin
%000000                             pool_fw <= 0;
%000000                             if ({4'd0, pool_fh} + 1 >= pool_kh) begin
                                        // Window complete
%000000                                 state <= S_POOL_DIV;
%000000                             end else begin
%000000                                 pool_fh <= pool_fh + 1;
                                    end
%000000                         end else begin
%000000                             pool_fw <= pool_fw + 1;
                                end
%000000                     end else if (pool_rd_phase == 0) begin
                                // Issue SRAM read
%000000                         begin : pool_addr_calc
                                    reg [31:0] elem_off;
                                    reg [31:0] byte_off;
%000000                             elem_off = (ih_s[15:0] * cfg_in_w * cfg_in_c)
%000000                                      + (iw_s[15:0] * cfg_in_c)
%000000                                      + pool_ch;
%000000                             byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                             act_rd_en   <= 1'b1;
%000000                             act_rd_addr <= cfg_act_base + byte_off[ACT_ADDR_W+1:2];
%000000                             act_byte_sel <= byte_off[1:0];
                                end
%000000                         pool_rd_phase <= 1;
%000000                     end else if (pool_rd_phase == 1) begin
                                // Wait for SRAM latency
%000000                         pool_rd_phase <= 2;
%000000                     end else begin
                                // Data available — go to ACC
%000000                         pool_rd_phase <= 0;
%000000                         state <= S_POOL_ACC;
                            end
                        end
                    end
        
%000000             S_POOL_ACC: begin
                        // Extract element and accumulate
%000000                 begin : pool_acc_blk
                            reg signed [ACC_W-1:0] val;
%000000                     if (cfg_int16) begin
%000000                         case (act_byte_sel[1])
%000000                             1'b0: val = $signed(act_rd_data[15:0]);
%000000                             1'b1: val = $signed(act_rd_data[31:16]);
                                endcase
%000000                     end else begin
%000000                         case (act_byte_sel)
%000000                             2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
%000000                             2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
%000000                             2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
%000000                             2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                                endcase
                            end
        
%000000                     if (pool_mode) begin
                                // AvgPool: accumulate sum
%000000                         pool_acc <= pool_acc + val;
%000000                     end else begin
                                // MaxPool: keep max
%000000                         if (val > pool_acc)
%000000                             pool_acc <= val;
                            end
%000000                     pool_count <= pool_count + 1;
                        end
        
                        // Advance window position
%000000                 if ({4'd0, pool_fw} + 1 >= pool_kw) begin
%000000                     pool_fw <= 0;
%000000                     if ({4'd0, pool_fh} + 1 >= pool_kh) begin
                                // Window complete
%000000                         state <= S_POOL_DIV;
%000000                     end else begin
%000000                         pool_fh <= pool_fh + 1;
%000000                         state <= S_POOL_READ;
                            end
%000000                 end else begin
%000000                     pool_fw <= pool_fw + 1;
%000000                     state <= S_POOL_READ;
                        end
                    end
        
%000000             S_POOL_DIV: begin
                        // AvgPool: symmetric rounding division
                        // MaxPool: pass through
~002450                 if (pool_mode && pool_count > 0) begin
%000000                     begin : pool_div_blk
                                reg signed [ACC_W-1:0] rounded;
                                reg signed [ACC_W-1:0] half_count;
%000000                         half_count = pool_count >> 1;
%000000                         if (pool_acc >= 0)
%000000                             rounded = pool_acc + half_count;
                                else
%000000                             rounded = pool_acc - half_count;
%000000                         pool_acc <= rounded / $signed({1'b0, pool_count});
                            end
                        end
%000000                 state <= S_POOL_PPU;
                    end
        
%000000             S_POOL_PPU: begin
                        // Feed pool_acc to PPU (CONV_REQ mode)
%000000                 ppu_acc_in   <= pool_acc;
%000000                 ppu_in_valid <= 1'b1;
%000000                 state <= S_POOL_PPU_WAIT;
                    end
        
%000000             S_POOL_PPU_WAIT: begin
%000000                 if (ppu_out_valid) begin
                            // Compute writeback address
%000000                     begin : pool_wb_addr_calc
                                reg [31:0] elem_off;
                                reg [31:0] byte_off;
%000000                         elem_off = (pool_oh * out_tile_w + pool_ow)
%000000                                  * cfg_out_c + pool_ch;
%000000                         byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                         pool_wb_addr    <= out_base + byte_off[ACT_ADDR_W+1:2];
%000000                         pool_wb_bytesel <= byte_off[1:0];
                            end
%000000                     pool_wb_byte <= ppu_out_data;
%000000                     pool_wb_phase <= 2'd0;
%000000                     state <= S_POOL_WB;
                        end
                    end
        
%000000             S_POOL_WB: begin
                        // Read-modify-write (same pattern as S_DW_WB)
%000000                 case (pool_wb_phase)
%000000                 2'd0: begin
%000000                     act_rd_en   <= 1'b1;
%000000                     act_rd_addr <= pool_wb_addr;
%000000                     pool_wb_phase <= 2'd1;
                        end
%000000                 2'd1: begin
%000000                     pool_wb_phase <= 2'd2;
                        end
%000000                 2'd2: begin
%000000                     begin : pool_wb_merge
                                reg [31:0] merged;
%000000                         merged = act_rd_data;
%000000                         if (cfg_int16) begin
%000000                             case (pool_wb_bytesel[1])
%000000                                 1'b0: merged[15:0]  = pool_wb_byte;
%000000                                 1'b1: merged[31:16] = pool_wb_byte;
                                    endcase
%000000                         end else begin
%000000                             case (pool_wb_bytesel)
%000000                                 2'd0: merged[7:0]   = pool_wb_byte[7:0];
%000000                                 2'd1: merged[15:8]  = pool_wb_byte[7:0];
%000000                                 2'd2: merged[23:16] = pool_wb_byte[7:0];
%000000                                 2'd3: merged[31:24] = pool_wb_byte[7:0];
                                    endcase
                                end
%000000                         act_wr_en   <= 1'b1;
%000000                         act_wr_addr <= pool_wb_addr;
%000000                         act_wr_data <= merged;
                            end
%000000                     state <= S_POOL_PIX_NEXT;
                        end
%000000                 default: pool_wb_phase <= 2'd0;
                        endcase
                    end
        
%000000             S_POOL_PIX_NEXT: begin
                        // Advance output pixel, reset window for next pixel
%000000                 pool_fh <= 0;
%000000                 pool_fw <= 0;
%000000                 pool_rd_phase <= 0;
%000000                 if (pool_mode) begin
%000000                     pool_acc <= 0;
%000000                 end else begin
%000000                     pool_acc <= {1'b1, {(ACC_W-1){1'b0}}};
                        end
%000000                 pool_count <= 0;
        
%000000                 if (pool_ow + 1 >= out_tile_w) begin
%000000                     pool_ow <= 0;
%000000                     if (pool_oh + 1 >= out_tile_h) begin
%000000                         state <= S_POOL_CH_NEXT;
%000000                     end else begin
%000000                         pool_oh <= pool_oh + 1;
%000000                         state <= S_POOL_READ;
                            end
%000000                 end else begin
%000000                     pool_ow <= pool_ow + 1;
%000000                     state <= S_POOL_READ;
                        end
                    end
        
%000000             S_POOL_CH_NEXT: begin
%000000                 if (pool_ch + 1 >= cfg_out_c) begin
%000000                     state <= S_TILE_NEXT;
%000000                 end else begin
%000000                     pool_ch <= pool_ch + 1;
%000000                     param_word_idx <= 0;
%000000                     param_read_issued <= 1'b0;
%000000                     state <= S_POOL_CH_SETUP;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // Eltwise Add Path (op_type=4)
                    // Flow: SETUP → PARAM → [READ_A → READ_B → COMPUTE → PPU → PPU_WAIT → WB → NEXT]*
                    // ══════════════════════════════════════════════════════════════
%000000             S_ADD_SETUP: begin
%000000                 add_total_elems <= cfg_out_h * cfg_out_w * cfg_out_c;
%000000                 add_elem_cnt <= 0;
%000000                 add_param_idx <= 0;
%000000                 add_param_phase <= 0;
%000000                 state <= S_ADD_PARAM;
                    end
        
%000000             S_ADD_PARAM: begin
                        // Read 2 words from Param SRAM (global Add rescale params)
%000000                 if (add_param_phase == 0) begin
%000000                     param_rd_en   <= 1'b1;
%000000                     param_rd_addr <= add_param_idx;
%000000                     add_param_phase <= 1;
%000000                 end else if (add_param_phase == 1) begin
                            // Wait SRAM latency
%000000                     add_param_phase <= 2;
%000000                 end else begin
                            // Capture
%000000                     if (add_param_idx == 0) begin
%000000                         add_M_A <= param_rd_data[14:0];
%000000                         add_S_A <= param_rd_data[21:16];
%000000                         if (is_concat) begin
                                    // Concat: only 1 param word needed
%000000                             add_rd_phase <= 0;
%000000                             state <= S_ADD_READ_A;
%000000                         end else begin
%000000                             add_param_idx <= 1;
%000000                             add_param_phase <= 0;
                                end
%000000                     end else begin
%000000                         add_M_B <= param_rd_data[14:0];
%000000                         add_S_B <= param_rd_data[21:16];
%000000                         add_rd_phase <= 0;
%000000                         state <= S_ADD_READ_A;
                            end
                        end
                    end
        
%000000             S_ADD_READ_A: begin
                        // Read element from Branch A (act_base region)
%000000                 if (add_rd_phase == 0) begin
%000000                     begin : add_a_addr_calc
                                reg [31:0] byte_off;
%000000                         byte_off = cfg_int16 ? ({16'd0, add_elem_cnt} << 1) : {16'd0, add_elem_cnt};
%000000                         act_rd_en   <= 1'b1;
%000000                         act_rd_addr <= cfg_act_base + byte_off[ACT_ADDR_W+1:2];
%000000                         act_byte_sel <= byte_off[1:0];
                            end
%000000                     add_rd_phase <= 1;
%000000                 end else if (add_rd_phase == 1) begin
%000000                     add_rd_phase <= 2;
%000000                 end else begin
                            // Extract element
%000000                     if (cfg_int16) begin
%000000                         case (act_byte_sel[1])
%000000                             1'b0: add_val_a <= $signed(act_rd_data[15:0]);
%000000                             1'b1: add_val_a <= $signed(act_rd_data[31:16]);
                                endcase
%000000                     end else begin
%000000                         case (act_byte_sel)
%000000                             2'd0: add_val_a <= {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
%000000                             2'd1: add_val_a <= {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
%000000                             2'd2: add_val_a <= {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
%000000                             2'd3: add_val_a <= {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                                endcase
                            end
%000000                     add_rd_phase <= 0;
%000000                     state <= is_concat ? S_ADD_COMPUTE : S_ADD_READ_B;
                        end
                    end
        
%000000             S_ADD_READ_B: begin
                        // Read element from Branch B (out_base region)
%000000                 if (add_rd_phase == 0) begin
%000000                     begin : add_b_addr_calc
                                reg [31:0] byte_off;
%000000                         byte_off = cfg_int16 ? ({16'd0, add_elem_cnt} << 1) : {16'd0, add_elem_cnt};
%000000                         act_rd_en   <= 1'b1;
%000000                         act_rd_addr <= cfg_out_base + byte_off[ACT_ADDR_W+1:2];
%000000                         act_byte_sel <= byte_off[1:0];
                            end
%000000                     add_rd_phase <= 1;
%000000                 end else if (add_rd_phase == 1) begin
%000000                     add_rd_phase <= 2;
%000000                 end else begin
                            // Extract element
%000000                     if (cfg_int16) begin
%000000                         case (act_byte_sel[1])
%000000                             1'b0: add_val_b <= $signed(act_rd_data[15:0]);
%000000                             1'b1: add_val_b <= $signed(act_rd_data[31:16]);
                                endcase
%000000                     end else begin
%000000                         case (act_byte_sel)
%000000                             2'd0: add_val_b <= {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
%000000                             2'd1: add_val_b <= {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
%000000                             2'd2: add_val_b <= {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
%000000                             2'd3: add_val_b <= {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                                endcase
                            end
%000000                     add_rd_phase <= 0;
%000000                     state <= S_ADD_COMPUTE;
                        end
                    end
        
%000000             S_ADD_COMPUTE: begin
                        // Dual rescale: rescaled_A = (val_A * M_A + round) >> S_A
%000000                 begin : add_compute_blk
                            reg signed [ACC_W-1:0] prod_a, prod_b;
                            reg signed [ACC_W-1:0] rescaled_a, rescaled_b;
%000000                     prod_a = add_val_a * $signed({1'b0, add_M_A});
%000000                     if (add_S_A > 0)
%000000                         rescaled_a = (prod_a + (1 <<< (add_S_A - 1))) >>> add_S_A;
                            else
%000000                         rescaled_a = prod_a;
%000000                     if (is_concat) begin
                                // Concat: single branch rescale only
%000000                         ppu_acc_in <= rescaled_a;
%000000                     end else begin
%000000                         prod_b = add_val_b * $signed({1'b0, add_M_B});
%000000                         if (add_S_B > 0)
%000000                             rescaled_b = (prod_b + (1 <<< (add_S_B - 1))) >>> add_S_B;
                                else
%000000                             rescaled_b = prod_b;
%000000                         ppu_acc_in <= rescaled_a + rescaled_b;
                            end
                        end
%000000                 state <= S_ADD_PPU;
                    end
        
%000000             S_ADD_PPU: begin
%000000                 ppu_in_valid <= 1'b1;
%000000                 state <= S_ADD_PPU_WAIT;
                    end
        
%000000             S_ADD_PPU_WAIT: begin
%000000                 if (ppu_out_valid) begin
                            // Compute writeback address
%000000                     begin : add_wb_addr_calc
                                reg [31:0] byte_off;
%000000                         if (is_concat) begin
                                    // Concat: output[pixel * total_c + offset + ch]
                                    reg [31:0] pixel;
                                    reg [31:0] ch;
%000000                             pixel = {16'd0, add_elem_cnt} / {16'd0, cfg_out_c};
%000000                             ch    = {16'd0, add_elem_cnt} % {16'd0, cfg_out_c};
%000000                             byte_off = (pixel * {16'd0, concat_total_c} + {16'd0, concat_offset} + ch)
%000000                                        << (cfg_int16 ? 1 : 0);
%000000                         end else begin
                                    // Add: flat overwrite at out_base
%000000                             byte_off = cfg_int16 ? ({16'd0, add_elem_cnt} << 1) : {16'd0, add_elem_cnt};
                                end
%000000                         add_wb_addr    <= cfg_out_base + byte_off[ACT_ADDR_W+1:2];
%000000                         add_wb_bytesel <= byte_off[1:0];
                            end
%000000                     add_wb_byte <= ppu_out_data;
%000000                     add_wb_phase <= 2'd0;
%000000                     state <= S_ADD_WB;
                        end
                    end
        
%000000             S_ADD_WB: begin
                        // Read-modify-write
%000000                 case (add_wb_phase)
%000000                 2'd0: begin
%000000                     act_rd_en   <= 1'b1;
%000000                     act_rd_addr <= add_wb_addr;
%000000                     add_wb_phase <= 2'd1;
                        end
%000000                 2'd1: begin
%000000                     add_wb_phase <= 2'd2;
                        end
%000000                 2'd2: begin
%000000                     begin : add_wb_merge
                                reg [31:0] merged;
%000000                         merged = act_rd_data;
%000000                         if (cfg_int16) begin
%000000                             case (add_wb_bytesel[1])
%000000                                 1'b0: merged[15:0]  = add_wb_byte;
%000000                                 1'b1: merged[31:16] = add_wb_byte;
                                    endcase
%000000                         end else begin
%000000                             case (add_wb_bytesel)
%000000                                 2'd0: merged[7:0]   = add_wb_byte[7:0];
%000000                                 2'd1: merged[15:8]  = add_wb_byte[7:0];
%000000                                 2'd2: merged[23:16] = add_wb_byte[7:0];
%000000                                 2'd3: merged[31:24] = add_wb_byte[7:0];
                                    endcase
                                end
%000000                         act_wr_en   <= 1'b1;
%000000                         act_wr_addr <= add_wb_addr;
%000000                         act_wr_data <= merged;
                            end
%000000                     state <= S_ADD_NEXT;
                        end
%000000                 default: add_wb_phase <= 2'd0;
                        endcase
                    end
        
%000000             S_ADD_NEXT: begin
%000000                 if (add_elem_cnt + 1 >= add_total_elems) begin
%000000                     state <= S_DONE;
%000000                 end else begin
%000000                     add_elem_cnt <= add_elem_cnt + 1;
%000000                     add_rd_phase <= 0;
%000000                     state <= S_ADD_READ_A;
                        end
                    end
        
                    // ══════════════════════════════════════════════════════════════
                    // Resize Path (op_type=5)
                    // Flow: SETUP → CH_SETUP → COORD → READ0[/1/2/3] → INTERP → PPU → WB → PIX_NEXT → CH_NEXT
                    // ══════════════════════════════════════════════════════════════
%000000             S_RESIZE_SETUP: begin
%000000                 rsz_ch <= 0;
%000000                 rsz_oh <= 0;
%000000                 rsz_ow <= 0;
%000000                 param_word_idx <= 0;
%000000                 param_read_issued <= 1'b0;
%000000                 rsz_rd_phase <= 0;
%000000                 state <= S_RESIZE_CH_SETUP;
                    end
        
%000000             S_RESIZE_CH_SETUP: begin
~001552                 if (!param_read_issued) begin
%000000                     param_rd_en   <= 1'b1;
%000000                     param_rd_addr <= (rsz_ch * 4) + param_word_idx;
%000000                     param_read_issued <= 1'b1;
%000000                     act_read_issued <= 1'b0;
%000000                 end else if (!act_read_issued) begin
%000000                     act_read_issued <= 1'b1;
%000000                 end else begin
%000000                     param_buf[param_word_idx] <= param_rd_data;
%000000                     if (param_word_idx == 3'd3) begin
%000000                         ppu_mult_m     <= param_buf[0][14:0];
%000000                         ppu_shift_s    <= param_buf[0][21:16];
%000000                         ppu_zero_point <= $signed(param_buf[1][15:0]);
%000000                         ppu_bias       <= $signed({param_buf[3][15:0], param_buf[2],
%000000                                                    param_buf[1][31:16]});
%000000                         rsz_oh <= 0;
%000000                         rsz_ow <= 0;
%000000                         rsz_rd_phase <= 0;
%000000                         state <= S_RESIZE_COORD;
%000000                     end else begin
%000000                         param_word_idx <= param_word_idx + 1;
%000000                         param_read_issued <= 1'b0;
                            end
                        end
                    end
        
%000000             S_RESIZE_COORD: begin
%000000                 begin : rsz_coord_blk
                            reg [31:0] src_h_q8, src_w_q8;
                            reg [31:0] oh_global, ow_global;
                            reg [15:0] ih_nearest, iw_nearest;
%000000                     oh_global = tile_oh_origin + rsz_oh;
%000000                     ow_global = tile_ow_origin + rsz_ow;
        
~002450                     if (!resize_mode) begin
%000000                         ih_nearest = (oh_global * cfg_in_h) / cfg_out_h;
%000000                         iw_nearest = (ow_global * cfg_in_w) / cfg_out_w;
%000000                         rsz_ih0 <= ih_nearest;
%000000                         rsz_iw0 <= iw_nearest;
%000000                         rsz_ih1 <= ih_nearest;
%000000                         rsz_iw1 <= iw_nearest;
%000000                         rsz_frac_h <= 0;
%000000                         rsz_frac_w <= 0;
%000000                     end else begin
%000000                         if (cfg_out_h > 1)
%000000                             src_h_q8 = (oh_global * ((cfg_in_h - 1) << 8)) / (cfg_out_h - 1);
                                else
%000000                             src_h_q8 = 0;
%000000                         if (cfg_out_w > 1)
%000000                             src_w_q8 = (ow_global * ((cfg_in_w - 1) << 8)) / (cfg_out_w - 1);
                                else
%000000                             src_w_q8 = 0;
        
%000000                         rsz_ih0 <= src_h_q8[31:8];
%000000                         rsz_iw0 <= src_w_q8[31:8];
%000000                         rsz_frac_h <= src_h_q8[7:0];
%000000                         rsz_frac_w <= src_w_q8[7:0];
        
%000000                         if (src_h_q8[31:8] + 1 >= cfg_in_h)
%000000                             rsz_ih1 <= cfg_in_h - 1;
                                else
%000000                             rsz_ih1 <= src_h_q8[31:8] + 1;
%000000                         if (src_w_q8[31:8] + 1 >= cfg_in_w)
%000000                             rsz_iw1 <= cfg_in_w - 1;
                                else
%000000                             rsz_iw1 <= src_w_q8[31:8] + 1;
                            end
                        end
%000000                 rsz_rd_phase <= 0;
%000000                 state <= S_RESIZE_READ0;
                    end
        
%000000             S_RESIZE_READ0: begin
%000000                 case (rsz_rd_phase)
%000000                 2'd0: begin
%000000                     begin : rsz_read0_addr
                                reg [31:0] elem_off;
                                reg [31:0] byte_off;
%000000                         elem_off = (rsz_ih0 * cfg_in_w * cfg_in_c)
%000000                                  + (rsz_iw0 * cfg_in_c)
%000000                                  + rsz_ch;
~002450                         byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                         act_rd_en   <= 1'b1;
%000000                         act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
%000000                         act_byte_sel <= byte_off[1:0];
                            end
%000000                     rsz_rd_phase <= 2'd1;
                        end
%000000                 2'd1: begin
%000000                     rsz_rd_phase <= 2'd2;
                        end
%000000                 2'd2: begin
%000000                     begin : rsz_read0_cap
                                reg signed [ACC_W-1:0] val;
%000000                         if (cfg_int16) begin
%000000                             case (act_byte_sel[1])
%000000                                 1'b0: val = $signed(act_rd_data[15:0]);
%000000                                 1'b1: val = $signed(act_rd_data[31:16]);
                                    endcase
%000000                         end else begin
%000000                             case (act_byte_sel)
%000000                                 2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
%000000                                 2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
%000000                                 2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
%000000                                 2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                                    endcase
                                end
%000000                         rsz_v00 <= val;
                            end
%000000                     rsz_rd_phase <= 0;
%000000                     if (resize_mode)
%000000                         state <= S_RESIZE_READ1;
                            else
%000000                         state <= S_RESIZE_PPU;
                        end
%000000                 default: rsz_rd_phase <= 0;
                        endcase
                    end
        
%000000             S_RESIZE_READ1: begin
%000000                 case (rsz_rd_phase)
%000000                 2'd0: begin
%000000                     begin : rsz_read1_addr
                                reg [31:0] elem_off;
                                reg [31:0] byte_off;
%000000                         elem_off = (rsz_ih0 * cfg_in_w * cfg_in_c)
%000000                                  + (rsz_iw1 * cfg_in_c)
%000000                                  + rsz_ch;
~002450                         byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                         act_rd_en   <= 1'b1;
%000000                         act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
%000000                         act_byte_sel <= byte_off[1:0];
                            end
%000000                     rsz_rd_phase <= 2'd1;
                        end
%000000                 2'd1: begin
%000000                     rsz_rd_phase <= 2'd2;
                        end
%000000                 2'd2: begin
%000000                     begin : rsz_read1_cap
                                reg signed [ACC_W-1:0] val;
%000000                         if (cfg_int16) begin
%000000                             case (act_byte_sel[1])
%000000                                 1'b0: val = $signed(act_rd_data[15:0]);
%000000                                 1'b1: val = $signed(act_rd_data[31:16]);
                                    endcase
%000000                         end else begin
%000000                             case (act_byte_sel)
%000000                                 2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
%000000                                 2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
%000000                                 2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
%000000                                 2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                                    endcase
                                end
%000000                         rsz_v01 <= val;
                            end
%000000                     rsz_rd_phase <= 0;
%000000                     state <= S_RESIZE_READ2;
                        end
%000000                 default: rsz_rd_phase <= 0;
                        endcase
                    end
        
%000000             S_RESIZE_READ2: begin
%000000                 case (rsz_rd_phase)
%000000                 2'd0: begin
%000000                     begin : rsz_read2_addr
                                reg [31:0] elem_off;
                                reg [31:0] byte_off;
%000000                         elem_off = (rsz_ih1 * cfg_in_w * cfg_in_c)
%000000                                  + (rsz_iw0 * cfg_in_c)
%000000                                  + rsz_ch;
~002450                         byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                         act_rd_en   <= 1'b1;
%000000                         act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
%000000                         act_byte_sel <= byte_off[1:0];
                            end
%000000                     rsz_rd_phase <= 2'd1;
                        end
%000000                 2'd1: begin
%000000                     rsz_rd_phase <= 2'd2;
                        end
%000000                 2'd2: begin
%000000                     begin : rsz_read2_cap
                                reg signed [ACC_W-1:0] val;
%000000                         if (cfg_int16) begin
%000000                             case (act_byte_sel[1])
%000000                                 1'b0: val = $signed(act_rd_data[15:0]);
%000000                                 1'b1: val = $signed(act_rd_data[31:16]);
                                    endcase
%000000                         end else begin
%000000                             case (act_byte_sel)
%000000                                 2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
%000000                                 2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
%000000                                 2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
%000000                                 2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                                    endcase
                                end
%000000                         rsz_v10 <= val;
                            end
%000000                     rsz_rd_phase <= 0;
%000000                     state <= S_RESIZE_READ3;
                        end
%000000                 default: rsz_rd_phase <= 0;
                        endcase
                    end
        
%000000             S_RESIZE_READ3: begin
%000000                 case (rsz_rd_phase)
%000000                 2'd0: begin
%000000                     begin : rsz_read3_addr
                                reg [31:0] elem_off;
                                reg [31:0] byte_off;
%000000                         elem_off = (rsz_ih1 * cfg_in_w * cfg_in_c)
%000000                                  + (rsz_iw1 * cfg_in_c)
%000000                                  + rsz_ch;
~002450                         byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                         act_rd_en   <= 1'b1;
%000000                         act_rd_addr <= act_base + byte_off[ACT_ADDR_W+1:2];
%000000                         act_byte_sel <= byte_off[1:0];
                            end
%000000                     rsz_rd_phase <= 2'd1;
                        end
%000000                 2'd1: begin
%000000                     rsz_rd_phase <= 2'd2;
                        end
%000000                 2'd2: begin
%000000                     begin : rsz_read3_cap
                                reg signed [ACC_W-1:0] val;
%000000                         if (cfg_int16) begin
%000000                             case (act_byte_sel[1])
%000000                                 1'b0: val = $signed(act_rd_data[15:0]);
%000000                                 1'b1: val = $signed(act_rd_data[31:16]);
                                    endcase
%000000                         end else begin
%000000                             case (act_byte_sel)
%000000                                 2'd0: val = {{(ACC_W-8){act_rd_data[7]}},  act_rd_data[7:0]};
%000000                                 2'd1: val = {{(ACC_W-8){act_rd_data[15]}}, act_rd_data[15:8]};
%000000                                 2'd2: val = {{(ACC_W-8){act_rd_data[23]}}, act_rd_data[23:16]};
%000000                                 2'd3: val = {{(ACC_W-8){act_rd_data[31]}}, act_rd_data[31:24]};
                                    endcase
                                end
%000000                         rsz_v11 <= val;
                            end
%000000                     rsz_rd_phase <= 0;
%000000                     state <= S_RESIZE_INTERP;
                        end
%000000                 default: rsz_rd_phase <= 0;
                        endcase
                    end
        
%000000             S_RESIZE_INTERP: begin
%000000                 begin : rsz_interp_blk
                            reg [8:0] one_minus_h, one_minus_w;
                            reg signed [63:0] top, bot, val64;
%000000                     one_minus_h = 9'd256 - {1'b0, rsz_frac_h};
%000000                     one_minus_w = 9'd256 - {1'b0, rsz_frac_w};
%000000                     top = rsz_v00 * $signed({1'b0, one_minus_w})
%000000                         + rsz_v01 * $signed({1'b0, rsz_frac_w});
%000000                     bot = rsz_v10 * $signed({1'b0, one_minus_w})
%000000                         + rsz_v11 * $signed({1'b0, rsz_frac_w});
%000000                     val64 = top * $signed({1'b0, one_minus_h})
%000000                           + bot * $signed({1'b0, rsz_frac_h});
%000000                     ppu_acc_in <= $signed((val64 + 64'sd32768) >>> 16);
                        end
%000000                 state <= S_RESIZE_PPU;
                    end
        
%000000             S_RESIZE_PPU: begin
~002450                 if (!resize_mode)
%000000                     ppu_acc_in <= rsz_v00;
%000000                 ppu_in_valid <= 1'b1;
%000000                 state <= S_RESIZE_PPU_WAIT;
                    end
        
%000000             S_RESIZE_PPU_WAIT: begin
%000000                 if (ppu_out_valid) begin
%000000                     begin : rsz_wb_addr_calc
                                reg [31:0] elem_off;
                                reg [31:0] byte_off;
%000000                         elem_off = (rsz_oh * out_tile_w + rsz_ow)
%000000                                  * cfg_out_c + rsz_ch;
%000000                         byte_off = cfg_int16 ? (elem_off << 1) : elem_off;
%000000                         rsz_wb_addr    <= out_base + byte_off[ACT_ADDR_W+1:2];
%000000                         rsz_wb_bytesel <= byte_off[1:0];
                            end
%000000                     rsz_wb_byte  <= ppu_out_data;
%000000                     rsz_wb_phase <= 2'd0;
%000000                     state <= S_RESIZE_WB;
                        end
                    end
        
%000000             S_RESIZE_WB: begin
%000000                 case (rsz_wb_phase)
%000000                 2'd0: begin
%000000                     act_rd_en   <= 1'b1;
%000000                     act_rd_addr <= rsz_wb_addr;
%000000                     rsz_wb_phase <= 2'd1;
                        end
%000000                 2'd1: begin
%000000                     rsz_wb_phase <= 2'd2;
                        end
%000000                 2'd2: begin
%000000                     begin : rsz_wb_merge
                                reg [31:0] merged;
%000000                         merged = act_rd_data;
%000000                         if (cfg_int16) begin
%000000                             case (rsz_wb_bytesel[1])
%000000                                 1'b0: merged[15:0]  = rsz_wb_byte;
%000000                                 1'b1: merged[31:16] = rsz_wb_byte;
                                    endcase
%000000                         end else begin
%000000                             case (rsz_wb_bytesel)
%000000                                 2'd0: merged[7:0]   = rsz_wb_byte[7:0];
%000000                                 2'd1: merged[15:8]  = rsz_wb_byte[7:0];
%000000                                 2'd2: merged[23:16] = rsz_wb_byte[7:0];
%000000                                 2'd3: merged[31:24] = rsz_wb_byte[7:0];
                                    endcase
                                end
%000000                         act_wr_en   <= 1'b1;
%000000                         act_wr_addr <= rsz_wb_addr;
%000000                         act_wr_data <= merged;
                            end
%000000                     state <= S_RESIZE_PIX_NEXT;
                        end
%000000                 default: rsz_wb_phase <= 2'd0;
                        endcase
                    end
        
%000000             S_RESIZE_PIX_NEXT: begin
%000000                 rsz_rd_phase <= 0;
%000000                 if (rsz_ow + 1 >= out_tile_w) begin
%000000                     rsz_ow <= 0;
%000000                     if (rsz_oh + 1 >= out_tile_h) begin
%000000                         state <= S_RESIZE_CH_NEXT;
%000000                     end else begin
%000000                         rsz_oh <= rsz_oh + 1;
%000000                         state <= S_RESIZE_COORD;
                            end
%000000                 end else begin
%000000                     rsz_ow <= rsz_ow + 1;
%000000                     state <= S_RESIZE_COORD;
                        end
                    end
        
%000000             S_RESIZE_CH_NEXT: begin
%000000                 if (rsz_ch + 1 >= cfg_out_c) begin
%000000                     state <= S_TILE_NEXT;
%000000                 end else begin
%000000                     state <= S_OC_NEXT;
                        end
                    end
        
%000000             default: state <= S_IDLE;
        
                    endcase
                end
            end
        
        endmodule
        
