// Open-NPU RTL — Post-Processing Unit (PPU) — Single Lane
// SPDX-License-Identifier: Apache-2.0
//
// Pipelined per-channel requantization:
//   acc(40b) → +bias_q[ch] → ×M[ch] → >>S[ch](+round) → +zp[ch] → clamp → ReLU → out
//
// Pipeline: 4 register stages, 4-cycle latency, 1 result/cycle throughput.
//
// Modes:
//   CONV_REQ (2'b00): Full pipeline — bias, mul, shift, zp, clamp, relu
//   RELU_ONLY (2'b10): Bypass bias/mul/shift/zp, apply clamp+relu only
//   PASSTHROUGH (2'b11): Direct passthrough (acc truncated to output width)
//
// The controller instantiates ARRAY_SIZE PPU lanes in parallel.

`include "npu_defines.vh"

module npu_ppu #(
    parameter ACC_W   = `ACC_WIDTH,    // 40-bit accumulator input
    parameter DATA_W  = `DATA_WIDTH,   // 8-bit output
    parameter BIAS_W  = `BIAS_WIDTH,   // 64-bit bias (use lower ACC_W bits)
    parameter MULT_W  = `PARAM_M_BITS, // 15-bit M
    parameter SHIFT_W = `PARAM_S_BITS, // 6-bit S
    parameter ZP_W    = `PARAM_ZP_BITS // 16-bit zero_point
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // ─── Control ───
    input  wire [1:0]                   mode,       // 2'b00=CONV_REQ, 2'b10=RELU_ONLY, 2'b11=PASS
    input  wire                         relu_en,    // Enable ReLU
    input  wire                         bias_en,    // Enable bias addition
    input  wire                         zp_en,      // Enable zero_point addition

    // ─── Input ───
    input  wire signed [ACC_W-1:0]      acc_in,     // From systolic array
    input  wire                         in_valid,

    // ─── Per-channel parameters (provided by controller) ───
    input  wire signed [ACC_W-1:0]      bias,       // bias_q (sign-extended to ACC_W)
    input  wire [MULT_W-1:0]            mult_m,     // M (unsigned 15-bit)
    input  wire [SHIFT_W-1:0]           shift_s,    // S (unsigned 6-bit)
    input  wire signed [ZP_W-1:0]       zero_point, // zp (signed 16-bit)

    // ─── Output ───
    output reg  signed [DATA_W-1:0]     out_data,
    output reg                          out_valid
);

    // ─── Mode encoding ───
    localparam MODE_CONV_REQ    = 2'b00;
    localparam MODE_RELU_ONLY   = 2'b10;
    localparam MODE_PASSTHROUGH = 2'b11;

    // ─── Product width: ACC_W + MULT_W ───
    localparam PROD_W = ACC_W + MULT_W;  // 40 + 15 = 55

    // ─── Pipeline Stage Registers ───

    // Stage 1: Bias addition result
    reg signed [ACC_W-1:0]      s1_biased;
    reg                         s1_valid;
    reg [1:0]                   s1_mode;
    reg [MULT_W-1:0]            s1_mult_m;
    reg [SHIFT_W-1:0]           s1_shift_s;
    reg signed [ZP_W-1:0]       s1_zp;
    reg                         s1_relu_en;
    reg                         s1_zp_en;

    // Stage 2: Multiply result
    reg signed [PROD_W-1:0]     s2_product;
    reg                         s2_valid;
    reg [1:0]                   s2_mode;
    reg [SHIFT_W-1:0]           s2_shift_s;
    reg signed [ZP_W-1:0]       s2_zp;
    reg                         s2_relu_en;
    reg                         s2_zp_en;

    // Stage 3: Shift result
    reg signed [ZP_W:0]         s3_shifted;  // 17-bit
    reg                         s3_valid;
    reg [1:0]                   s3_mode;
    reg signed [ZP_W-1:0]       s3_zp;
    reg                         s3_relu_en;
    reg                         s3_zp_en;

    // ─── Combinational intermediate signals ───
    wire signed [ACC_W-1:0]  biased_val;
    wire signed [PROD_W-1:0] product_val;
    wire signed [PROD_W-1:0] rounded_product;
    wire signed [ZP_W:0]     shifted_val;

    // ═══════════════════════════════════════════════════════════════════
    // Combinational logic between stages (continuous assigns)
    // ═══════════════════════════════════════════════════════════════════

    // Stage 1 input: Bias addition
    assign biased_val = (bias_en && mode == MODE_CONV_REQ) ? (acc_in + bias) : acc_in;

    // Stage 2 input: Multiply by M
    assign product_val = (s1_mode == MODE_CONV_REQ) ?
                         s1_biased * $signed({1'b0, s1_mult_m}) :
                         {{MULT_W{s1_biased[ACC_W-1]}}, s1_biased};

    // Stage 3 input: Rounding right shift
    wire [SHIFT_W-1:0] s2_shift = s2_shift_s;
    wire signed [PROD_W-1:0] round_bit = (s2_shift > 0) ?
                                          ($signed({{(PROD_W-1){1'b0}}, 1'b1}) << (s2_shift - 1)) :
                                          0;
    assign rounded_product = s2_product + round_bit;
    wire signed [PROD_W-1:0] arith_shifted_full = rounded_product >>> s2_shift;
    assign shifted_val = (s2_mode == MODE_CONV_REQ) ?
                         arith_shifted_full[ZP_W:0] :
                         s2_product[ZP_W:0];

    // ═══════════════════════════════════════════════════════════════════
    // Single always block for ALL pipeline registers
    // All combinational logic computed inline to avoid simulation
    // timing issues with continuous assignments.
    // ═══════════════════════════════════════════════════════════════════
    always @(posedge clk or negedge rst_n) begin : pipeline
        // Local variables for combinational computation
        reg signed [ZP_W:0] zp_result;

        if (!rst_n) begin
            s1_biased  <= 0;
            s1_valid   <= 1'b0;
            s1_mode    <= MODE_PASSTHROUGH;
            s1_mult_m  <= 0;
            s1_shift_s <= 0;
            s1_zp      <= 0;
            s1_relu_en <= 1'b0;
            s1_zp_en   <= 1'b0;

            s2_product <= 0;
            s2_valid   <= 1'b0;
            s2_mode    <= MODE_PASSTHROUGH;
            s2_shift_s <= 0;
            s2_zp      <= 0;
            s2_relu_en <= 1'b0;
            s2_zp_en   <= 1'b0;

            s3_shifted <= 0;
            s3_valid   <= 1'b0;
            s3_mode    <= MODE_PASSTHROUGH;
            s3_zp      <= 0;
            s3_relu_en <= 1'b0;
            s3_zp_en   <= 1'b0;

            out_data   <= 0;
            out_valid  <= 1'b0;
        end else begin
            // ─── Stage 4 (output): ZP add + Clamp + ReLU ───
            // MUST come first to read s3 values BEFORE they get NBA-updated
            if (s3_zp_en && s3_mode == MODE_CONV_REQ)
                zp_result = s3_shifted + {{1{s3_zp[ZP_W-1]}}, s3_zp};
            else
                zp_result = s3_shifted;

            out_valid  <= s3_valid;

            if (s3_mode == MODE_PASSTHROUGH) begin
                out_data <= zp_result[DATA_W-1:0];
            end else if (s3_mode == MODE_RELU_ONLY) begin
                if (s3_relu_en && zp_result < 0)
                    out_data <= 8'sd0;
                else if (zp_result < -128)
                    out_data <= -8'sd128;
                else if (zp_result > 127)
                    out_data <= 8'sd127;
                else
                    out_data <= zp_result[DATA_W-1:0];
            end else begin
                // MODE_CONV_REQ: Clamp then ReLU
                if (zp_result < -128) begin
                    out_data <= (s3_relu_en) ? 8'sd0 : -8'sd128;
                end else if (zp_result > 127) begin
                    out_data <= 8'sd127;
                end else begin
                    if (s3_relu_en && zp_result < 0)
                        out_data <= 8'sd0;
                    else
                        out_data <= zp_result[DATA_W-1:0];
                end
            end

            // ─── Stage 1: Bias ───
            s1_valid   <= in_valid;
            s1_mode    <= mode;
            s1_biased  <= biased_val;
            s1_mult_m  <= mult_m;
            s1_shift_s <= shift_s;
            s1_zp      <= zero_point;
            s1_relu_en <= relu_en;
            s1_zp_en   <= zp_en;

            // ─── Stage 2: Multiply ───
            s2_valid   <= s1_valid;
            s2_mode    <= s1_mode;
            s2_product <= product_val;
            s2_shift_s <= s1_shift_s;
            s2_zp      <= s1_zp;
            s2_relu_en <= s1_relu_en;
            s2_zp_en   <= s1_zp_en;

            // ─── Stage 3: Shift ───
            s3_valid   <= s2_valid;
            s3_mode    <= s2_mode;
            s3_shifted <= shifted_val;
            s3_zp      <= s2_zp;
            s3_relu_en <= s2_relu_en;
            s3_zp_en   <= s2_zp_en;
        end
    end

endmodule
