// Testbench wrapper for npu_compute — instantiates compute + systolic + PPU + SRAMs
// Provides a self-contained datapath for functional testing.

`include "npu_defines.vh"

module npu_compute_tb #(
    parameter ARRAY_SIZE   = `ARRAY_SIZE,
    parameter ACT_DEPTH    = 1024,
    parameter WGT_DEPTH    = 1024,
    parameter PARAM_DEPTH  = 256
)(
    input  wire clk,
    input  wire rst_n,

    // Controller interface
    input  wire        start,
    output wire        done,

    // Configuration
    input  wire [7:0]  cfg_op_type,
    input  wire [15:0] cfg_in_c,
    input  wire [15:0] cfg_out_h,
    input  wire [15:0] cfg_out_w,
    input  wire [15:0] cfg_out_c,
    input  wire [7:0]  cfg_kernel_h,
    input  wire [7:0]  cfg_kernel_w,
    input  wire [7:0]  cfg_stride_h,
    input  wire [7:0]  cfg_stride_w,
    input  wire [7:0]  cfg_pad_top,
    input  wire [7:0]  cfg_pad_left,
    input  wire [15:0] cfg_tile_h,
    input  wire [15:0] cfg_tile_w,
    input  wire [15:0] cfg_tile_num_h,
    input  wire [15:0] cfg_tile_num_w,
    input  wire [15:0] cfg_in_w,
    input  wire [15:0] cfg_in_h,
    input  wire [$clog2(ACT_DEPTH)-1:0] cfg_act_base,
    input  wire [$clog2(ACT_DEPTH)-1:0] cfg_out_base,

    // PPU configuration (directly from post_ctrl register)
    input  wire [1:0]  ppu_mode,
    input  wire        ppu_relu_en,
    input  wire        ppu_bias_en,
    input  wire        ppu_zp_en,
    input  wire        cfg_int16
);

    localparam DATA_W       = `DATA_WIDTH;
    localparam ACC_W        = `ACC_WIDTH;
    localparam ACT_ADDR_W   = $clog2(ACT_DEPTH);
    localparam WGT_ADDR_W   = $clog2(WGT_DEPTH);
    localparam PARAM_ADDR_W = $clog2(PARAM_DEPTH);

    // ─── SRAM Port B wires (compute engine side) ───
    wire                    wgt_rd_en;
    wire [WGT_ADDR_W-1:0]  wgt_rd_addr;
    wire [31:0]             wgt_rd_data;

    wire                    act_rd_en;
    wire [ACT_ADDR_W-1:0]  act_rd_addr;
    wire [31:0]             act_rd_data;
    wire                    act_wr_en;
    wire [ACT_ADDR_W-1:0]  act_wr_addr;
    wire [31:0]             act_wr_data;

    wire                    param_rd_en;
    wire [PARAM_ADDR_W-1:0] param_rd_addr;
    wire [31:0]             param_rd_data;

    // ─── Systolic wires ───
    wire [1:0]                          sa_cmd;
    wire                                sa_cmd_valid;
    wire [DATA_W*ARRAY_SIZE-1:0]       sa_wgt_data_flat;
    wire                                sa_wgt_valid;
    wire [DATA_W*ARRAY_SIZE-1:0]       sa_act_data_flat;
    wire                                sa_act_valid;
    wire [$clog2(ARRAY_SIZE)-1:0]      sa_drain_col_sel;
    wire [ACC_W*ARRAY_SIZE-1:0]        sa_acc_out_flat;
    wire                                sa_acc_out_valid;
    wire                                sa_busy, sa_ready;

    // ─── DW Conv wires ───
    wire                                dw_wgt_load, dw_wgt_valid, dw_in_valid, dw_acc_clear;
    wire signed [DATA_W-1:0]           dw_wgt_data, dw_in_data;
    wire signed [ACC_W-1:0]            dw_acc_out;
    wire                                dw_out_valid;

    // ─── PPU wires ───
    wire signed [ACC_W-1:0]            ppu_acc_in;
    wire                                ppu_in_valid;
    wire signed [ACC_W-1:0]            ppu_bias;
    wire [14:0]                         ppu_mult_m;
    wire [5:0]                          ppu_shift_s;
    wire signed [15:0]                 ppu_zero_point;
    wire signed [DATA_W-1:0]           ppu_out_data;
    wire                                ppu_out_valid;

    // ═══════════════════════════════════════════════════════════════════
    // SRAMs (Port A unused here — would be DMA in full system)
    // We pre-load via Port A from the testbench using initial blocks or force
    // For cocotb: use hierarchical access to write SRAM contents directly
    // ═══════════════════════════════════════════════════════════════════

    npu_sram #(.DATA_W(32), .DEPTH(WGT_DEPTH)) u_sram_wgt (
        .clk(clk),
        .a_en(1'b0), .a_we(1'b0), .a_addr({WGT_ADDR_W{1'b0}}),
        .a_wdata(32'd0), .a_rdata(),
        .b_en(wgt_rd_en), .b_we(1'b0), .b_addr(wgt_rd_addr),
        .b_wdata(32'd0), .b_rdata(wgt_rd_data)
    );

    npu_sram #(.DATA_W(32), .DEPTH(ACT_DEPTH)) u_sram_act (
        .clk(clk),
        .a_en(1'b0), .a_we(1'b0), .a_addr({ACT_ADDR_W{1'b0}}),
        .a_wdata(32'd0), .a_rdata(),
        .b_en(act_rd_en | act_wr_en), .b_we(act_wr_en), .b_addr(act_wr_en ? act_wr_addr : act_rd_addr),
        .b_wdata(act_wr_data), .b_rdata(act_rd_data)
    );

    npu_sram #(.DATA_W(32), .DEPTH(PARAM_DEPTH)) u_sram_param (
        .clk(clk),
        .a_en(1'b0), .a_we(1'b0), .a_addr({PARAM_ADDR_W{1'b0}}),
        .a_wdata(32'd0), .a_rdata(),
        .b_en(param_rd_en), .b_we(1'b0), .b_addr(param_rd_addr),
        .b_wdata(32'd0), .b_rdata(param_rd_data)
    );

    // ═══════════════════════════════════════════════════════════════════
    // Compute Micro-Sequencer
    // ═══════════════════════════════════════════════════════════════════

    npu_compute #(
        .ARRAY_SIZE   (ARRAY_SIZE),
        .ACT_ADDR_W   (ACT_ADDR_W),
        .WGT_ADDR_W   (WGT_ADDR_W),
        .PARAM_ADDR_W (PARAM_ADDR_W),
        .DATA_W       (DATA_W),
        .ACC_W        (ACC_W)
    ) u_compute (
        .clk(clk), .rst_n(rst_n),
        .start(start), .done(done),
        .cfg_op_type(cfg_op_type),
        .cfg_int16(cfg_int16),
        .cfg_in_c(cfg_in_c), .cfg_out_h(cfg_out_h), .cfg_out_w(cfg_out_w),
        .cfg_out_c(cfg_out_c), .cfg_kernel_h(cfg_kernel_h), .cfg_kernel_w(cfg_kernel_w),
        .cfg_stride_h(cfg_stride_h), .cfg_stride_w(cfg_stride_w),
        .cfg_pad_top(cfg_pad_top), .cfg_pad_left(cfg_pad_left),
        .cfg_tile_h(cfg_tile_h), .cfg_tile_w(cfg_tile_w),
        .cfg_tile_num_h(cfg_tile_num_h), .cfg_tile_num_w(cfg_tile_num_w),
        .cfg_in_w(cfg_in_w), .cfg_in_h(cfg_in_h),
        .cfg_act_base(cfg_act_base), .cfg_out_base(cfg_out_base),
        // SRAM
        .wgt_rd_en(wgt_rd_en), .wgt_rd_addr(wgt_rd_addr), .wgt_rd_data(wgt_rd_data),
        .act_rd_en(act_rd_en), .act_rd_addr(act_rd_addr), .act_rd_data(act_rd_data),
        .act_wr_en(act_wr_en), .act_wr_addr(act_wr_addr), .act_wr_data(act_wr_data),
        .param_rd_en(param_rd_en), .param_rd_addr(param_rd_addr), .param_rd_data(param_rd_data),
        // Systolic
        .sa_cmd(sa_cmd), .sa_cmd_valid(sa_cmd_valid),
        .sa_wgt_data_flat(sa_wgt_data_flat), .sa_wgt_valid(sa_wgt_valid),
        .sa_act_data_flat(sa_act_data_flat), .sa_act_valid(sa_act_valid),
        .sa_drain_col_sel(sa_drain_col_sel),
        .sa_acc_out_flat(sa_acc_out_flat), .sa_acc_out_valid(sa_acc_out_valid),
        .sa_busy(sa_busy), .sa_ready(sa_ready),
        // DW Conv
        .dw_wgt_load(dw_wgt_load), .dw_wgt_valid(dw_wgt_valid),
        .dw_wgt_data(dw_wgt_data), .dw_in_valid(dw_in_valid),
        .dw_in_data(dw_in_data), .dw_acc_clear(dw_acc_clear),
        .dw_acc_out(dw_acc_out), .dw_out_valid(dw_out_valid),
        // PPU
        .ppu_acc_in(ppu_acc_in), .ppu_in_valid(ppu_in_valid),
        .ppu_bias(ppu_bias), .ppu_mult_m(ppu_mult_m),
        .ppu_shift_s(ppu_shift_s), .ppu_zero_point(ppu_zero_point),
        .ppu_out_data(ppu_out_data), .ppu_out_valid(ppu_out_valid)
    );

    // ═══════════════════════════════════════════════════════════════════
    // Systolic Array
    // ═══════════════════════════════════════════════════════════════════

    npu_systolic #(
        .ROWS(ARRAY_SIZE), .COLS(ARRAY_SIZE),
        .DATA_W(DATA_W), .ACC_W(ACC_W)
    ) u_systolic (
        .clk(clk), .rst_n(rst_n),
        .cmd(sa_cmd), .cmd_valid(sa_cmd_valid),
        .wgt_data_flat(sa_wgt_data_flat), .wgt_valid(sa_wgt_valid),
        .act_data_flat(sa_act_data_flat), .act_valid(sa_act_valid),
        .drain_col_sel(sa_drain_col_sel),
        .acc_out_flat(sa_acc_out_flat), .acc_out_valid(sa_acc_out_valid),
        .busy(sa_busy), .ready(sa_ready)
    );

    // ═══════════════════════════════════════════════════════════════════
    // DW Conv Engine
    // ═══════════════════════════════════════════════════════════════════

    npu_dw_conv #(
        .DATA_W(DATA_W), .ACC_W(ACC_W), .MAX_KSZ(7)
    ) u_dw_conv (
        .clk(clk), .rst_n(rst_n),
        .kernel_h(cfg_kernel_h[3:0]), .kernel_w(cfg_kernel_w[3:0]),
        .wgt_load(dw_wgt_load), .wgt_valid(dw_wgt_valid), .wgt_data(dw_wgt_data),
        .in_valid(dw_in_valid), .in_data(dw_in_data), .acc_clear(dw_acc_clear),
        .acc_out(dw_acc_out), .out_valid(dw_out_valid)
    );

    // ═══════════════════════════════════════════════════════════════════
    // PPU
    // ═══════════════════════════════════════════════════════════════════

    npu_ppu #(
        .ACC_W(ACC_W), .DATA_W(DATA_W),
        .BIAS_W(`BIAS_WIDTH), .MULT_W(`PARAM_M_BITS),
        .SHIFT_W(`PARAM_S_BITS), .ZP_W(`PARAM_ZP_BITS)
    ) u_ppu (
        .clk(clk), .rst_n(rst_n),
        .mode(ppu_mode),
        .relu_en(ppu_relu_en),
        .bias_en(ppu_bias_en),
        .zp_en(ppu_zp_en),
        .int16_mode(cfg_int16),
        .acc_in(ppu_acc_in), .in_valid(ppu_in_valid),
        .bias(ppu_bias), .mult_m(ppu_mult_m),
        .shift_s(ppu_shift_s), .zero_point(ppu_zero_point),
        .out_data(ppu_out_data), .out_valid(ppu_out_valid)
    );

endmodule
