// Open-NPU RTL — ARRAY_SIZE-wide depthwise bank
// SPDX-License-Identifier: Apache-2.0

`include "npu_defines.vh"

module npu_dw_bank #(
    parameter N      = `ARRAY_SIZE,
    parameter DATA_W = `DATA_WIDTH,
    parameter ACC_W  = `ACC_WIDTH,
    parameter MAX_KSZ = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [3:0]               kernel_h,
    input  wire [3:0]               kernel_w,

    input  wire                     wgt_load,
    input  wire [N-1:0]             wgt_valid,
    input  wire [DATA_W*N-1:0]      wgt_data_flat,

    input  wire [N-1:0]             in_valid,
    input  wire [DATA_W*N-1:0]      in_data_flat,
    input  wire                     acc_clear,

    output wire [ACC_W*N-1:0]       acc_flat,
    output wire [N-1:0]             out_valid
);

    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : lanes
            npu_dw_conv #(
                .DATA_W(DATA_W), .ACC_W(ACC_W), .MAX_KSZ(MAX_KSZ)
            ) u_dw (
                .clk(clk), .rst_n(rst_n),
                .kernel_h(kernel_h), .kernel_w(kernel_w),
                .wgt_load(wgt_load),
                .wgt_valid(wgt_valid[gi]),
                .wgt_data(wgt_data_flat[DATA_W*gi +: DATA_W]),
                .in_valid(in_valid[gi]),
                .in_data(in_data_flat[DATA_W*gi +: DATA_W]),
                .acc_clear(acc_clear),
                .acc_out(acc_flat[ACC_W*gi +: ACC_W]),
                .out_valid(out_valid[gi])
            );
        end
    endgenerate

endmodule
