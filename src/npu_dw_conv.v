// Open-NPU RTL — Depthwise Convolution Engine
// SPDX-License-Identifier: Apache-2.0
//
// Single-channel DW convolution unit. Computes one output pixel per activation.
// The controller instantiates N of these in parallel for N-channel parallelism.
//
// Operation:
//   1. Load kernel weights (k_h × k_w values, up to 7×7=49 max)
//   2. Stream input activation window elements one per cycle
//   3. After all kernel elements are accumulated, output is valid
//
// Interface:
//   - Weight loading: wgt_valid + wgt_data loads weights sequentially
//   - Compute: in_valid + in_data streams window elements
//   - Output: acc_out valid after k_h*k_w input cycles

`include "npu_defines.vh"

module npu_dw_conv #(
    parameter DATA_W    = `DATA_WIDTH,     // 8-bit input
    parameter ACC_W     = `ACC_WIDTH,      // 40-bit accumulator
    parameter MAX_KSZ   = 7               // Max kernel dimension (7×7)
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // ─── Configuration ───
    input  wire [3:0]              kernel_h,      // Kernel height (1-7)
    input  wire [3:0]              kernel_w,      // Kernel width (1-7)

    // ─── Weight Loading ───
    input  wire                     wgt_load,      // 1=loading weights mode
    input  wire                     wgt_valid,     // Weight data valid
    input  wire signed [DATA_W-1:0] wgt_data,     // Weight value (INT8)

    // ─── Compute Interface ───
    input  wire                     in_valid,      // Input activation valid
    input  wire signed [DATA_W-1:0] in_data,      // Input activation value (INT8)
    input  wire                     acc_clear,     // Clear accumulator (new output pixel)

    // ─── Output ───
    output reg signed [ACC_W-1:0]   acc_out,       // Accumulated result
    output reg                      out_valid      // Output valid pulse
);

    // ─── Weight storage ───
    reg signed [DATA_W-1:0] weights [0:MAX_KSZ*MAX_KSZ-1];
    reg [5:0] wgt_idx;   // Weight load index (0 to k_h*k_w-1)

    // ─── Compute state ───
    reg [5:0] compute_idx;  // Current MAC index within kernel window
    wire [5:0] kernel_size = kernel_h * kernel_w;

    // ─── MAC computation ───
    wire signed [2*DATA_W-1:0] product = in_data * weights[compute_idx];

    // ─── Weight loading integer ───
    integer wi;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out     <= {ACC_W{1'b0}};
            out_valid   <= 1'b0;
            wgt_idx     <= 6'd0;
            compute_idx <= 6'd0;
            for (wi = 0; wi < MAX_KSZ*MAX_KSZ; wi = wi + 1)
                weights[wi] <= {DATA_W{1'b0}};
        end else begin
            out_valid <= 1'b0;  // default: no output

            if (wgt_load) begin
                // ─── Weight loading mode ───
                if (wgt_valid) begin
                    weights[wgt_idx] <= wgt_data;
                    wgt_idx <= wgt_idx + 1;
                end
                if (acc_clear) begin
                    wgt_idx <= 6'd0;
                end
            end else begin
                // ─── Compute mode ───
                if (acc_clear) begin
                    acc_out     <= {ACC_W{1'b0}};
                    compute_idx <= 6'd0;
                end

                if (in_valid) begin
                    if (acc_clear)
                        acc_out <= {{(ACC_W-2*DATA_W){product[2*DATA_W-1]}}, product};
                    else
                        acc_out <= acc_out + {{(ACC_W-2*DATA_W){product[2*DATA_W-1]}}, product};

                    if (compute_idx + 1 >= kernel_size) begin
                        out_valid   <= 1'b1;
                        compute_idx <= 6'd0;
                    end else begin
                        compute_idx <= compute_idx + 1;
                    end
                end
            end
        end
    end

endmodule
