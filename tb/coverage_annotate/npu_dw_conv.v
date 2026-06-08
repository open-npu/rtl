//      // verilator_coverage annotation
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
 004959     input  wire                     clk,
 000011     input  wire                     rst_n,
        
            // ─── Configuration ───
%000005     input  wire [3:0]              kernel_h,      // Kernel height (1-7)
%000003     input  wire [3:0]              kernel_w,      // Kernel width (1-7)
        
            // ─── Weight Loading ───
 000064     input  wire                     wgt_load,      // 1=loading weights mode
 000064     input  wire                     wgt_valid,     // Weight data valid
%000000     input  wire signed [DATA_W-1:0] wgt_data,     // Weight value (INT8)
        
            // ─── Compute Interface ───
 000064     input  wire                     in_valid,      // Input activation valid
~000015     input  wire signed [DATA_W-1:0] in_data,      // Input activation value (INT8)
 000128     input  wire                     acc_clear,     // Clear accumulator (new output pixel)
        
            // ─── Output ───
%000000     output reg signed [ACC_W-1:0]   acc_out,       // Accumulated result
 000064     output reg                      out_valid      // Output valid pulse
        );
        
            // ─── Weight storage ───
            reg signed [DATA_W-1:0] weights [0:MAX_KSZ*MAX_KSZ-1];
~000063     reg [5:0] wgt_idx;   // Weight load index (0 to k_h*k_w-1)
        
            // ─── Compute state ───
%000000     reg [5:0] compute_idx;  // Current MAC index within kernel window
%000003     wire [5:0] kernel_size = kernel_h * kernel_w;
        
            // ─── MAC computation ───
%000000     wire signed [2*DATA_W-1:0] product = in_data * weights[compute_idx];
        
            // ─── Weight loading integer ───
            integer wi;
        
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n) begin
 000035             acc_out     <= {ACC_W{1'b0}};
 000035             out_valid   <= 1'b0;
 000035             wgt_idx     <= 6'd0;
 000035             compute_idx <= 6'd0;
 001715             for (wi = 0; wi < MAX_KSZ*MAX_KSZ; wi = wi + 1)
 001715                 weights[wi] <= {DATA_W{1'b0}};
 002450         end else begin
 002450             out_valid <= 1'b0;  // default: no output
        
 002322             if (wgt_load) begin
                        // ─── Weight loading mode ───
 000096                 if (wgt_valid) begin
 000032                     weights[wgt_idx] <= wgt_data;
 000032                     wgt_idx <= wgt_idx + 1;
                        end
 000096                 if (acc_clear) begin
 000032                     wgt_idx <= 6'd0;
                        end
 002322             end else begin
                        // ─── Compute mode ───
 002290                 if (acc_clear) begin
 000032                     acc_out     <= {ACC_W{1'b0}};
 000032                     compute_idx <= 6'd0;
                        end
        
 002290                 if (in_valid) begin
~000032                     if (acc_clear)
%000000                         acc_out <= {{(ACC_W-2*DATA_W){product[2*DATA_W-1]}}, product};
                            else
 000032                         acc_out <= acc_out + {{(ACC_W-2*DATA_W){product[2*DATA_W-1]}}, product};
        
~000032                     if (compute_idx + 1 >= kernel_size) begin
 000032                         out_valid   <= 1'b1;
 000032                         compute_idx <= 6'd0;
%000000                     end else begin
%000000                         compute_idx <= compute_idx + 1;
                            end
                        end
                    end
                end
            end
        
        endmodule
        
