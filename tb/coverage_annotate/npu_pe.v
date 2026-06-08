//      // verilator_coverage annotation
        // Open-NPU RTL — Processing Element (PE)
        // SPDX-License-Identifier: Apache-2.0
        //
        // Single PE of the systolic array.
        // Modes:
        //   IDLE       — waiting for command
        //   WGT_LOAD   — load weight register from weight_in
        //   COMPUTE    — MAC: acc += act_in * weight, pass act_in to act_out
        //   DRAIN      — output accumulator value, then clear
        
        `include "npu_defines.vh"
        
        module npu_pe (
 1269504     input  wire                      clk,
 002816     input  wire                      rst_n,
        
            // Control
%000000     input  wire [1:0]                mode,       // 2'b00=IDLE, 2'b01=WGT_LOAD, 2'b10=COMPUTE, 2'b11=DRAIN
%000000     input  wire                      valid_in,   // Input data valid
        
            // Data path — activation (flows left-to-right)
%000000     input  wire signed [`DATA_WIDTH-1:0]  act_in,
%000000     output reg  signed [`DATA_WIDTH-1:0]  act_out,
%000000     output reg                       act_valid_out,
        
            // Data path — weight (loaded from top)
%000000     input  wire signed [`DATA_WIDTH-1:0]  weight_in,
        
            // Data path — accumulator drain (flows top-to-bottom)
%000000     output reg  signed [`ACC_WIDTH-1:0]   acc_out,
%000000     output reg                       acc_valid
        );
        
            // ─── Mode encoding ───
            localparam MODE_IDLE     = 2'b00;
            localparam MODE_WGT_LOAD = 2'b01;
            localparam MODE_COMPUTE  = 2'b10;
            localparam MODE_DRAIN    = 2'b11;
        
            // ─── Internal registers ───
%000000     reg signed [`DATA_WIDTH-1:0]  weight_reg;
%000000     reg signed [`ACC_WIDTH-1:0]   acc_reg;
        
            // ─── Main logic ───
 636160     always @(posedge clk or negedge rst_n) begin
 627200         if (!rst_n) begin
 008960             weight_reg    <= {`DATA_WIDTH{1'b0}};
 008960             acc_reg       <= {`ACC_WIDTH{1'b0}};
 008960             act_out       <= {`DATA_WIDTH{1'b0}};
 008960             act_valid_out <= 1'b0;
 008960             acc_out       <= {`ACC_WIDTH{1'b0}};
 008960             acc_valid     <= 1'b0;
 627200         end else begin
                    // Defaults
 627200             act_valid_out <= 1'b0;
 627200             acc_valid     <= 1'b0;
        
 627200             case (mode)
%000000                 MODE_WGT_LOAD: begin
%000000                     if (valid_in) begin
%000000                         weight_reg <= weight_in;
                            end
                        end
        
%000000                 MODE_COMPUTE: begin
%000000                     if (valid_in) begin
                                // MAC operation
%000000                         acc_reg <= acc_reg + (act_in * weight_reg);
                                // Pass activation to next PE (systolic flow)
%000000                         act_out       <= act_in;
%000000                         act_valid_out <= 1'b1;
                            end
                        end
        
%000000                 MODE_DRAIN: begin
%000000                     if (valid_in) begin
%000000                         acc_out   <= acc_reg;
%000000                         acc_valid <= 1'b1;
%000000                         acc_reg   <= {`ACC_WIDTH{1'b0}};  // Clear after drain
                            end
                        end
        
 627200                 default: begin // MODE_IDLE
                            // Do nothing
                        end
                    endcase
                end
            end
        
        endmodule
        
