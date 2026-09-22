// Open-NPU RTL — Processing Element (PE)
// SPDX-License-Identifier: Apache-2.0
//
// One PE of the weight-stationary systolic array.
//
// PE[r][c] holds W[k=r][oc=c]. Activations are broadcast across a row (all
// columns of row r see the same act_in in the same cycle); partial sums flow
// down a column:
//
//     psum_out = psum_in + act_in * weight_reg
//
// so the bottom row of column c emits the full dot product over the ROWS
// k-slots. The reduction therefore happens inside the array, one result per
// cycle per column, instead of being drained column-by-column into an external
// adder tree.
//
// The caller must skew the activation feed: act[k=r] for a given output pixel
// has to arrive at row r one cycle after act[k=r-1] arrived at row r-1, so that
// the descending partial sum meets the matching activation. npu_systolic owns
// that skew buffer.
//
// Modes:
//   IDLE     — hold
//   WGT_LOAD — latch weight_in into weight_reg
//   COMPUTE  — one MAC into the psum chain

`include "npu_defines.vh"

module npu_pe (
    input  wire                      clk,
    input  wire                      rst_n,

    // Control
    input  wire [1:0]                mode,       // 2'b00=IDLE, 2'b01=WGT_LOAD, 2'b10=COMPUTE
    input  wire                      valid_in,   // Activation valid for this row

    // Activation (broadcast within a row)
    input  wire signed [`DATA_WIDTH-1:0] act_in,

    // Weight (from the column-select broadcast bus)
    input  wire signed [`DATA_WIDTH-1:0] weight_in,
    // Next-pass weight: filled while COMPUTE still uses weight_reg.
    // swap_wgt commits it after the in-flight psum chain has consumed
    // the old weight (same cycle is safe: the MAC reads weight_reg first).
    input  wire                      load_nxt,
    input  wire                      swap_wgt,

    // Partial sum (flows top-to-bottom)
    input  wire signed [`ACC_WIDTH-1:0]  psum_in,
    output reg  signed [`ACC_WIDTH-1:0]  psum_out,
    output reg                       psum_valid_out
);

    localparam MODE_IDLE     = 2'b00;
    localparam MODE_WGT_LOAD = 2'b01;
    localparam MODE_COMPUTE  = 2'b10;

    reg signed [`DATA_WIDTH-1:0] weight_reg;
    reg signed [`DATA_WIDTH-1:0] weight_nxt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= {`DATA_WIDTH{1'b0}};
            weight_nxt     <= {`DATA_WIDTH{1'b0}};
            psum_out       <= {`ACC_WIDTH{1'b0}};
            psum_valid_out <= 1'b0;
        end else begin
            psum_valid_out <= 1'b0;

            case (mode)
                MODE_WGT_LOAD: begin
                    if (valid_in)
                        weight_reg <= weight_in;
                end

                MODE_COMPUTE: begin
                    if (valid_in) begin
                        psum_out       <= psum_in + (act_in * weight_reg);
                        psum_valid_out <= 1'b1;
                    end
                    if (load_nxt)
                        weight_nxt <= weight_in;
                    if (swap_wgt)
                        weight_reg <= weight_nxt;
                end

                default: begin // MODE_IDLE
                    if (swap_wgt)
                        weight_reg <= weight_nxt;
                end
            endcase
        end
    end

endmodule
