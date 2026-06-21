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
    input  wire                         int16_mode, // 1=INT16 clamp, 0=INT8 clamp

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

    // ─── Product width: ACC_W + MULT_W + 1 (sign bit from signed×extended) ───
    localparam PROD_W = ACC_W + MULT_W + 1'b1;  // 40 + 15 + 1 = 56

    // ─── Pipeline Stage Registers ───

    // Stage 1: after bias addition
    reg signed [ACC_W-1:0]      s1_biased;
    reg                         s1_valid;
    reg [1:0]                   s1_mode;
    reg [MULT_W-1:0]            s1_mult_m;
    reg [SHIFT_W-1:0]           s1_shift_s;
    reg signed [ZP_W-1:0]       s1_zp;
    reg                         s1_relu_en;
    reg                         s1_zp_en;
    reg                         s1_int16;

    // Stage 2: after multiply
    reg signed [PROD_W-1:0]     s2_product;
    reg                         s2_valid;
    reg [1:0]                   s2_mode;
    reg [SHIFT_W-1:0]           s2_shift_s;
    reg signed [ZP_W-1:0]       s2_zp;
    reg                         s2_relu_en;
    reg                         s2_zp_en;
    reg                         s2_int16;

    // Stage 3: after shift
    reg signed [ZP_W:0]         s3_shifted;  // 17-bit
    reg                         s3_valid;
    reg [1:0]                   s3_mode;
    reg signed [ZP_W-1:0]       s3_zp;
    reg                         s3_relu_en;
    reg                         s3_zp_en;
    reg                         s3_int16;

    // ═══════════════════════════════════════════════════════════════════
    // Pipeline — single always block with all computations inline
    // ═══════════════════════════════════════════════════════════════════
    always @(posedge clk or negedge rst_n) begin : pipeline
        // Blocking-assignment intermediates (not registers)
        reg signed [ACC_W-1:0]  biased_v;
        reg signed [PROD_W-1:0] product_v;
        reg signed [PROD_W-1:0] rounded_v;
        reg signed [PROD_W-1:0] shifted_full;
        reg signed [ZP_W:0]     shifted_v;
        reg signed [ZP_W:0]     zp_result;
        reg [SHIFT_W-1:0]       shift_amt;

        if (!rst_n) begin
            s1_biased  <= 0;
            s1_valid   <= 1'b0;
            s1_mode    <= MODE_PASSTHROUGH;
            s1_mult_m  <= 0;
            s1_shift_s <= 0;
            s1_zp      <= 0;
            s1_relu_en <= 1'b0;
            s1_zp_en   <= 1'b0;
            s1_int16   <= 1'b0;

            s2_product <= 0;
            s2_valid   <= 1'b0;
            s2_mode    <= MODE_PASSTHROUGH;
            s2_shift_s <= 0;
            s2_zp      <= 0;
            s2_relu_en <= 1'b0;
            s2_zp_en   <= 1'b0;
            s2_int16   <= 1'b0;

            s3_shifted <= 0;
            s3_valid   <= 1'b0;
            s3_mode    <= MODE_PASSTHROUGH;
            s3_zp      <= 0;
            s3_relu_en <= 1'b0;
            s3_zp_en   <= 1'b0;
            s3_int16   <= 1'b0;

            out_data   <= 0;
            out_valid  <= 1'b0;
        end else begin
            // ─── Compute Stage 4 output FIRST (reads current s3 regs) ───
            if (s3_zp_en && s3_mode == MODE_CONV_REQ)
                zp_result = s3_shifted + $signed({{1{s3_zp[ZP_W-1]}}, s3_zp});
            else
                zp_result = s3_shifted;

            out_valid <= s3_valid;

            if (s3_mode == MODE_PASSTHROUGH) begin
                out_data <= zp_result[DATA_W-1:0];
            end else if (s3_mode == MODE_RELU_ONLY) begin
                if (s3_int16) begin
                    // INT16 clamp+relu
                    if (s3_relu_en && zp_result < $signed(17'sd0))
                        out_data <= 16'sd0;
                    else if (zp_result < -$signed(17'sd32768))
                        out_data <= -16'sd32768;
                    else if (zp_result > $signed(17'sd32767))
                        out_data <= 16'sd32767;
                    else
                        out_data <= zp_result[DATA_W-1:0];
                end else begin
                    // INT8 clamp+relu (sign-extend to 16-bit output)
                    if (s3_relu_en && zp_result < $signed(17'sd0))
                        out_data <= 16'sd0;
                    else if (zp_result < -$signed(17'sd128))
                        out_data <= {{8{1'b1}}, 8'h80};  // -128 sign-extended
                    else if (zp_result > $signed(17'sd127))
                        out_data <= 16'sd127;
                    else
                        out_data <= {{8{zp_result[7]}}, zp_result[7:0]};
                end
            end else begin
                // MODE_CONV_REQ: Clamp then ReLU
                if (s3_int16) begin
                    // INT16: clamp to [-32768, 32767]
                    if (zp_result < -$signed(17'sd32768)) begin
                        out_data <= (s3_relu_en) ? 16'sd0 : -16'sd32768;
                    end else if (zp_result > $signed(17'sd32767)) begin
                        out_data <= 16'sd32767;
                    end else begin
                        if (s3_relu_en && zp_result < $signed(17'sd0))
                            out_data <= 16'sd0;
                        else
                            out_data <= zp_result[DATA_W-1:0];
                    end
                end else begin
                    // INT8: clamp to [-128, 127], sign-extend to 16-bit
                    if (zp_result < -$signed(17'sd128)) begin
                        out_data <= (s3_relu_en) ? 16'sd0 : {{8{1'b1}}, 8'h80};
                    end else if (zp_result > $signed(17'sd127)) begin
                        out_data <= 16'sd127;
                    end else begin
                        if (s3_relu_en && zp_result < $signed(17'sd0))
                            out_data <= 16'sd0;
                        else
                            out_data <= {{8{zp_result[7]}}, zp_result[7:0]};
                    end
                end
            end

            // ─── Compute Stage 3: rounding right shift (reads current s2 regs) ───
            shift_amt = s2_shift_s;
            if (s2_mode == MODE_CONV_REQ) begin
                if (shift_amt > 0)
                    rounded_v = s2_product + ($signed({{(PROD_W-1){1'b0}}, 1'b1}) << (shift_amt - 1));
                else
                    rounded_v = s2_product;
                shifted_full = rounded_v >>> shift_amt;
                // Saturate to 17-bit signed range to avoid truncation overflow
                // 17-bit signed: [-65536, +65535]
                if (shifted_full > 56'sd65535)
                    shifted_v = 17'sd65535;
                else if (shifted_full < -56'sd65536)
                    shifted_v = -17'sd65536;
                else
                    shifted_v = shifted_full[ZP_W:0];
            end else begin
                shifted_v = s2_product[ZP_W:0];
            end

            s3_valid   <= s2_valid;
            s3_mode    <= s2_mode;
            s3_shifted <= shifted_v;
            s3_zp      <= s2_zp;
            s3_relu_en <= s2_relu_en;
            s3_zp_en   <= s2_zp_en;
            s3_int16   <= s2_int16;

            // ─── Compute Stage 2: multiply (reads current s1 regs) ───
            if (s1_mode == MODE_CONV_REQ)
                product_v = s1_biased * $signed({1'b0, s1_mult_m});
            else
                product_v = {{MULT_W{s1_biased[ACC_W-1]}}, s1_biased};

            s2_valid   <= s1_valid;
            s2_mode    <= s1_mode;
            s2_product <= product_v;
            s2_shift_s <= s1_shift_s;
            s2_zp      <= s1_zp;
            s2_relu_en <= s1_relu_en;
            s2_zp_en   <= s1_zp_en;
            s2_int16   <= s1_int16;

            // ─── Compute Stage 1: bias addition (reads input ports) ───
            if (bias_en && mode == MODE_CONV_REQ)
                biased_v = acc_in + bias;
            else
                biased_v = acc_in;

            s1_valid   <= in_valid;
            s1_mode    <= mode;
            s1_biased  <= biased_v;
            s1_mult_m  <= mult_m;
            s1_shift_s <= shift_s;
            s1_zp      <= zero_point;
            s1_relu_en <= relu_en;
            s1_zp_en   <= zp_en;
            s1_int16   <= int16_mode;
        end
    end

endmodule
