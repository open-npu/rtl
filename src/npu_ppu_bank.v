// Open-NPU RTL — ARRAY_SIZE-wide PPU bank
// SPDX-License-Identifier: Apache-2.0
//
// Instantiates N single-lane npu_ppu units. Lane 0 also accepts the legacy
// scalar ports so DW / Pool / Add / Resize keep their existing 1-wide feed.

`include "npu_defines.vh"

module npu_ppu_bank #(
    parameter N       = `ARRAY_SIZE,
    parameter ACC_W   = `ACC_WIDTH,
    parameter DATA_W  = `DATA_WIDTH,
    parameter BIAS_W  = `BIAS_WIDTH,
    parameter MULT_W  = `PARAM_M_BITS,
    parameter SHIFT_W = `PARAM_S_BITS,
    parameter ZP_W    = `PARAM_ZP_BITS
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire [1:0]                   mode,
    input  wire                         relu_en,
    input  wire                         relu6_en,
    input  wire                         bias_en,
    input  wire                         zp_en,
    input  wire                         int16_mode,
    input  wire signed [ZP_W-1:0]       clamp_max,

    // Wide feed (Conv S_PPU_STREAM)
    input  wire [ACC_W*N-1:0]           acc_wide,
    input  wire [N-1:0]                 valid_wide,
    input  wire [ACC_W*N-1:0]           bias_wide,
    input  wire [15*N-1:0]              mult_wide,
    input  wire [6*N-1:0]               shift_wide,
    input  wire [16*N-1:0]              zp_wide,

    // Legacy scalar feed (lane 0), used when valid_wide == 0
    input  wire signed [ACC_W-1:0]      acc_s,
    input  wire                         valid_s,
    input  wire signed [ACC_W-1:0]      bias_s,
    input  wire [14:0]                  mult_s,
    input  wire [5:0]                   shift_s,
    input  wire signed [15:0]           zp_s,

    output wire [DATA_W*N-1:0]          out_wide,
    output wire [N-1:0]                 vout_wide,
    output wire signed [DATA_W-1:0]     out_s,
    output wire                         vout_s
);

    assign out_s  = out_wide[DATA_W-1:0];
    assign vout_s = vout_wide[0];

    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : lanes
            wire signed [ACC_W-1:0] acc_i;
            wire                    val_i;
            wire signed [ACC_W-1:0] bias_i;
            wire [14:0]             m_i;
            wire [5:0]              s_i;
            wire signed [15:0]      zp_i;

            if (gi == 0) begin : lane0
                assign val_i  = valid_wide[0] | valid_s;
                assign acc_i  = valid_wide[0] ? acc_wide[ACC_W-1:0] : acc_s;
                assign bias_i = valid_wide[0] ? bias_wide[ACC_W-1:0] : bias_s;
                assign m_i    = valid_wide[0] ? mult_wide[14:0] : mult_s;
                assign s_i    = valid_wide[0] ? shift_wide[5:0] : shift_s;
                assign zp_i   = valid_wide[0] ? zp_wide[15:0] : zp_s;
            end else begin : lanen
                assign val_i  = valid_wide[gi];
                assign acc_i  = acc_wide[ACC_W*gi +: ACC_W];
                assign bias_i = bias_wide[ACC_W*gi +: ACC_W];
                assign m_i    = mult_wide[15*gi +: 15];
                assign s_i    = shift_wide[6*gi +: 6];
                assign zp_i   = zp_wide[16*gi +: 16];
            end

            npu_ppu #(
                .ACC_W(ACC_W), .DATA_W(DATA_W), .BIAS_W(BIAS_W),
                .MULT_W(MULT_W), .SHIFT_W(SHIFT_W), .ZP_W(ZP_W)
            ) u_ppu (
                .clk(clk), .rst_n(rst_n),
                .mode(mode), .relu_en(relu_en), .relu6_en(relu6_en),
                .bias_en(bias_en), .zp_en(zp_en),
                .int16_mode(int16_mode), .clamp_max(clamp_max),
                .acc_in(acc_i), .in_valid(val_i),
                .bias(bias_i), .mult_m(m_i), .shift_s(s_i), .zero_point(zp_i),
                .out_data(out_wide[DATA_W*gi +: DATA_W]),
                .out_valid(vout_wide[gi])
            );
        end
    endgenerate

endmodule
