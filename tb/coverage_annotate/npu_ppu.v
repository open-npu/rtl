//      // verilator_coverage annotation
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
 004959     input  wire                         clk,
 000011     input  wire                         rst_n,
        
            // ─── Control ───
%000000     input  wire [1:0]                   mode,       // 2'b00=CONV_REQ, 2'b10=RELU_ONLY, 2'b11=PASS
%000000     input  wire                         relu_en,    // Enable ReLU
%000000     input  wire                         bias_en,    // Enable bias addition
%000000     input  wire                         zp_en,      // Enable zero_point addition
%000000     input  wire                         int16_mode, // 1=INT16 clamp, 0=INT8 clamp
        
            // ─── Input ───
%000000     input  wire signed [ACC_W-1:0]      acc_in,     // From systolic array
 000064     input  wire                         in_valid,
        
            // ─── Per-channel parameters (provided by controller) ───
%000000     input  wire signed [ACC_W-1:0]      bias,       // bias_q (sign-extended to ACC_W)
%000000     input  wire [MULT_W-1:0]            mult_m,     // M (unsigned 15-bit)
%000000     input  wire [SHIFT_W-1:0]           shift_s,    // S (unsigned 6-bit)
%000000     input  wire signed [ZP_W-1:0]       zero_point, // zp (signed 16-bit)
        
            // ─── Output ───
%000000     output reg  signed [DATA_W-1:0]     out_data,
 000064     output reg                          out_valid
        );
        
            // ─── Mode encoding ───
            localparam MODE_CONV_REQ    = 2'b00;
            localparam MODE_RELU_ONLY   = 2'b10;
            localparam MODE_PASSTHROUGH = 2'b11;
        
            // ─── Product width: ACC_W + MULT_W ───
            localparam PROD_W = ACC_W + MULT_W;  // 40 + 15 = 55
        
            // ─── Pipeline Stage Registers ───
        
            // Stage 1: after bias addition
%000000     reg signed [ACC_W-1:0]      s1_biased;
 000064     reg                         s1_valid;
 000012     reg [1:0]                   s1_mode;
%000000     reg [MULT_W-1:0]            s1_mult_m;
%000000     reg [SHIFT_W-1:0]           s1_shift_s;
%000000     reg signed [ZP_W-1:0]       s1_zp;
%000000     reg                         s1_relu_en;
%000000     reg                         s1_zp_en;
%000000     reg                         s1_int16;
        
            // Stage 2: after multiply
%000000     reg signed [PROD_W-1:0]     s2_product;
 000064     reg                         s2_valid;
 000012     reg [1:0]                   s2_mode;
%000000     reg [SHIFT_W-1:0]           s2_shift_s;
%000000     reg signed [ZP_W-1:0]       s2_zp;
%000000     reg                         s2_relu_en;
%000000     reg                         s2_zp_en;
%000000     reg                         s2_int16;
        
            // Stage 3: after shift
%000000     reg signed [ZP_W:0]         s3_shifted;  // 17-bit
 000064     reg                         s3_valid;
 000012     reg [1:0]                   s3_mode;
%000000     reg signed [ZP_W-1:0]       s3_zp;
%000000     reg                         s3_relu_en;
%000000     reg                         s3_zp_en;
%000000     reg                         s3_int16;
        
            // ═══════════════════════════════════════════════════════════════════
            // Pipeline — single always block with all computations inline
            // ═══════════════════════════════════════════════════════════════════
 002485     always @(posedge clk or negedge rst_n) begin : pipeline
                // Blocking-assignment intermediates (not registers)
                reg signed [ACC_W-1:0]  biased_v;
                reg signed [PROD_W-1:0] product_v;
                reg signed [PROD_W-1:0] rounded_v;
                reg signed [PROD_W-1:0] shifted_full;
                reg signed [ZP_W:0]     shifted_v;
                reg signed [ZP_W:0]     zp_result;
                reg [SHIFT_W-1:0]       shift_amt;
        
 002450         if (!rst_n) begin
 000035             s1_biased  <= 0;
 000035             s1_valid   <= 1'b0;
 000035             s1_mode    <= MODE_PASSTHROUGH;
 000035             s1_mult_m  <= 0;
 000035             s1_shift_s <= 0;
 000035             s1_zp      <= 0;
 000035             s1_relu_en <= 1'b0;
 000035             s1_zp_en   <= 1'b0;
 000035             s1_int16   <= 1'b0;
        
 000035             s2_product <= 0;
 000035             s2_valid   <= 1'b0;
 000035             s2_mode    <= MODE_PASSTHROUGH;
 000035             s2_shift_s <= 0;
 000035             s2_zp      <= 0;
 000035             s2_relu_en <= 1'b0;
 000035             s2_zp_en   <= 1'b0;
 000035             s2_int16   <= 1'b0;
        
 000035             s3_shifted <= 0;
 000035             s3_valid   <= 1'b0;
 000035             s3_mode    <= MODE_PASSTHROUGH;
 000035             s3_zp      <= 0;
 000035             s3_relu_en <= 1'b0;
 000035             s3_zp_en   <= 1'b0;
 000035             s3_int16   <= 1'b0;
        
 000035             out_data   <= 0;
 000035             out_valid  <= 1'b0;
 002450         end else begin
                    // ─── Compute Stage 4 output FIRST (reads current s3 regs) ───
~002450             if (s3_zp_en && s3_mode == MODE_CONV_REQ)
%000000                 zp_result = s3_shifted + $signed({{1{s3_zp[ZP_W-1]}}, s3_zp});
                    else
 002450                 zp_result = s3_shifted;
        
 002450             out_valid <= s3_valid;
        
 000018             if (s3_mode == MODE_PASSTHROUGH) begin
 000018                 out_data <= zp_result[DATA_W-1:0];
~002432             end else if (s3_mode == MODE_RELU_ONLY) begin
%000000                 if (s3_int16) begin
                            // INT16 clamp+relu
%000000                     if (s3_relu_en && zp_result < $signed(17'sd0))
%000000                         out_data <= 16'sd0;
%000000                     else if (zp_result < -$signed(17'sd32768))
%000000                         out_data <= -16'sd32768;
%000000                     else if (zp_result > $signed(17'sd32767))
%000000                         out_data <= 16'sd32767;
                            else
%000000                         out_data <= zp_result[DATA_W-1:0];
%000000                 end else begin
                            // INT8 clamp+relu (sign-extend to 16-bit output)
%000000                     if (s3_relu_en && zp_result < $signed(17'sd0))
%000000                         out_data <= 16'sd0;
%000000                     else if (zp_result < -$signed(17'sd128))
%000000                         out_data <= {{8{1'b1}}, 8'h80};  // -128 sign-extended
%000000                     else if (zp_result > $signed(17'sd127))
%000000                         out_data <= 16'sd127;
                            else
%000000                         out_data <= {{8{zp_result[7]}}, zp_result[7:0]};
                        end
 002432             end else begin
                        // MODE_CONV_REQ: Clamp then ReLU
~002432                 if (s3_int16) begin
                            // INT16: clamp to [-32768, 32767]
%000000                     if (zp_result < -$signed(17'sd32768)) begin
%000000                         out_data <= (s3_relu_en) ? 16'sd0 : -16'sd32768;
%000000                     end else if (zp_result > $signed(17'sd32767)) begin
%000000                         out_data <= 16'sd32767;
%000000                     end else begin
%000000                         if (s3_relu_en && zp_result < $signed(17'sd0))
%000000                             out_data <= 16'sd0;
                                else
%000000                             out_data <= zp_result[DATA_W-1:0];
                            end
 002432                 end else begin
                            // INT8: clamp to [-128, 127], sign-extend to 16-bit
%000000                     if (zp_result < -$signed(17'sd128)) begin
%000000                         out_data <= (s3_relu_en) ? 16'sd0 : {{8{1'b1}}, 8'h80};
~002432                     end else if (zp_result > $signed(17'sd127)) begin
%000000                         out_data <= 16'sd127;
 002432                     end else begin
~002432                         if (s3_relu_en && zp_result < $signed(17'sd0))
%000000                             out_data <= 16'sd0;
                                else
 002432                             out_data <= {{8{zp_result[7]}}, zp_result[7:0]};
                            end
                        end
                    end
        
                    // ─── Compute Stage 3: rounding right shift (reads current s2 regs) ───
 002450             shift_amt = s2_shift_s;
 002438             if (s2_mode == MODE_CONV_REQ) begin
~002438                 if (shift_amt > 0)
%000000                     rounded_v = s2_product + ($signed({{(PROD_W-1){1'b0}}, 1'b1}) << (shift_amt - 1));
                        else
 002438                     rounded_v = s2_product;
 002438                 shifted_full = rounded_v >>> shift_amt;
                        // Saturate to 17-bit signed range to avoid truncation overflow
                        // 17-bit signed: [-65536, +65535]
%000000                 if (shifted_full > 55'sd65535)
%000000                     shifted_v = 17'sd65535;
~002438                 else if (shifted_full < -55'sd65536)
%000000                     shifted_v = -17'sd65536;
                        else
 002438                     shifted_v = shifted_full[ZP_W:0];
 000012             end else begin
 000012                 shifted_v = s2_product[ZP_W:0];
                    end
        
 002450             s3_valid   <= s2_valid;
 002450             s3_mode    <= s2_mode;
 002450             s3_shifted <= shifted_v;
 002450             s3_zp      <= s2_zp;
 002450             s3_relu_en <= s2_relu_en;
 002450             s3_zp_en   <= s2_zp_en;
 002450             s3_int16   <= s2_int16;
        
                    // ─── Compute Stage 2: multiply (reads current s1 regs) ───
~002444             if (s1_mode == MODE_CONV_REQ)
 002444                 product_v = s1_biased * $signed({1'b0, s1_mult_m});
                    else
%000006                 product_v = {{MULT_W{s1_biased[ACC_W-1]}}, s1_biased};
        
 002450             s2_valid   <= s1_valid;
 002450             s2_mode    <= s1_mode;
 002450             s2_product <= product_v;
 002450             s2_shift_s <= s1_shift_s;
 002450             s2_zp      <= s1_zp;
 002450             s2_relu_en <= s1_relu_en;
 002450             s2_zp_en   <= s1_zp_en;
 002450             s2_int16   <= s1_int16;
        
                    // ─── Compute Stage 1: bias addition (reads input ports) ───
~002450             if (bias_en && mode == MODE_CONV_REQ)
%000000                 biased_v = acc_in + bias;
                    else
 002450                 biased_v = acc_in;
        
 002450             s1_valid   <= in_valid;
 002450             s1_mode    <= mode;
 002450             s1_biased  <= biased_v;
 002450             s1_mult_m  <= mult_m;
 002450             s1_shift_s <= shift_s;
 002450             s1_zp      <= zero_point;
 002450             s1_relu_en <= relu_en;
 002450             s1_zp_en   <= zp_en;
 002450             s1_int16   <= int16_mode;
                end
            end
        
        endmodule
        
