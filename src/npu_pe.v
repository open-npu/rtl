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
    input  wire                      clk,
    input  wire                      rst_n,

    // Control
    input  wire [1:0]                mode,       // 2'b00=IDLE, 2'b01=WGT_LOAD, 2'b10=COMPUTE, 2'b11=DRAIN
    input  wire                      valid_in,   // Input data valid

    // Data path — activation (flows left-to-right)
    input  wire signed [`DATA_WIDTH-1:0]  act_in,
    output reg  signed [`DATA_WIDTH-1:0]  act_out,
    output reg                       act_valid_out,

    // Data path — weight (loaded from top)
    input  wire signed [`DATA_WIDTH-1:0]  weight_in,

    // Data path — accumulator drain (flows top-to-bottom)
    output reg  signed [`ACC_WIDTH-1:0]   acc_out,
    output reg                       acc_valid
);

    // ─── Mode encoding ───
    localparam MODE_IDLE     = 2'b00;
    localparam MODE_WGT_LOAD = 2'b01;
    localparam MODE_COMPUTE  = 2'b10;
    localparam MODE_DRAIN    = 2'b11;

    // ─── Internal registers ───
    reg signed [`DATA_WIDTH-1:0]  weight_reg;
    reg signed [`ACC_WIDTH-1:0]   acc_reg;

    // ─── Main logic ───
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg    <= {`DATA_WIDTH{1'b0}};
            acc_reg       <= {`ACC_WIDTH{1'b0}};
            act_out       <= {`DATA_WIDTH{1'b0}};
            act_valid_out <= 1'b0;
            acc_out       <= {`ACC_WIDTH{1'b0}};
            acc_valid     <= 1'b0;
        end else begin
            // Defaults
            act_valid_out <= 1'b0;
            acc_valid     <= 1'b0;

            case (mode)
                MODE_WGT_LOAD: begin
                    if (valid_in) begin
                        weight_reg <= weight_in;
                    end
                end

                MODE_COMPUTE: begin
                    if (valid_in) begin
                        // MAC operation
                        acc_reg <= acc_reg + (act_in * weight_reg);
                        // Debug: print PE[0][0] MAC for first few cycles
                        // Pass activation to next PE (systolic flow)
                        act_out       <= act_in;
                        act_valid_out <= 1'b1;
                    end
                end

                MODE_DRAIN: begin
                    if (valid_in) begin
                        acc_out   <= acc_reg;
                        acc_valid <= 1'b1;
                        acc_reg   <= {`ACC_WIDTH{1'b0}};  // Clear after drain
                    end
                end

                default: begin // MODE_IDLE
                    // Do nothing
                end
            endcase
        end
    end

endmodule
