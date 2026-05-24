// Open-NPU RTL — Systolic Array (N×N)
// SPDX-License-Identifier: Apache-2.0
//
// Weight-stationary systolic array.
//
// Phases:
//   1. WGT_LOAD: Load weights column-by-column (COLS cycles, wgt_valid=1 each)
//   2. COMPUTE:  Stream activations into col 0 (K cycles with act_valid=1).
//                Systolic propagation adds 1-cycle delay per column.
//                Wait K + COLS - 1 total cycles for all PEs to finish.
//   3. DRAIN:    Assert drain for 1 cycle. PE outputs are registered,
//                so acc_out_valid pulses 1 cycle after drain is triggered.
//                All COLS columns drain simultaneously.
//
// Output: acc_out[row] is from column selected by `drain_col_sel` input.
// The controller iterates drain_col_sel = 0..COLS-1 to read all columns,
// issuing DRAIN once per column. (Each drain clears that column's accumulators.)
//
// Simplified first version:
//   - No double weight buffer
//   - No partial sum
//   - Single drain column per DRAIN phase (re-enter DRAIN for next column)

`include "npu_defines.vh"

module npu_systolic #(
    parameter ROWS   = `ARRAY_SIZE,
    parameter COLS   = `ARRAY_SIZE,
    parameter DATA_W = `DATA_WIDTH,
    parameter ACC_W  = `ACC_WIDTH
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // ─── Control ───
    input  wire [1:0]                   cmd,
    input  wire                         cmd_valid,

    // ─── Weight Load ───
    input  wire signed [DATA_W-1:0]     wgt_data [0:ROWS-1],
    input  wire                         wgt_valid,

    // ─── Activation Input ───
    input  wire signed [DATA_W-1:0]     act_data [0:ROWS-1],
    input  wire                         act_valid,

    // ─── Drain Control ───
    input  wire [$clog2(COLS)-1:0]      drain_col_sel,  // Which column to drain

    // ─── Accumulator Output ───
    output wire signed [ACC_W-1:0]      acc_out [0:ROWS-1],
    output reg                          acc_out_valid,

    // ─── Status ───
    output wire                         busy,
    output wire                         ready
);

    // ─── Mode encoding ───
    localparam MODE_IDLE     = 2'b00;
    localparam MODE_WGT_LOAD = 2'b01;
    localparam MODE_COMPUTE  = 2'b10;
    localparam MODE_DRAIN    = 2'b11;

    // ─── FSM States ───
    localparam S_IDLE     = 3'd0;
    localparam S_WGT_LOAD = 3'd1;
    localparam S_READY    = 3'd2;
    localparam S_COMPUTE  = 3'd3;
    localparam S_DRAIN    = 3'd4;
    localparam S_DRAIN_OUT= 3'd5;  // 1-cycle output phase

    reg [2:0] state, state_next;
    reg [$clog2(COLS)-1:0] wgt_col_cnt;
    reg wgt_load_done;

    // ─── PE interconnect ───
    wire [1:0]               pe_mode   [0:ROWS-1][0:COLS-1];
    wire                     pe_valid  [0:ROWS-1][0:COLS-1];
    wire signed [DATA_W-1:0] pe_act_in  [0:ROWS-1][0:COLS-1];
    wire signed [DATA_W-1:0] pe_act_out [0:ROWS-1][0:COLS-1];
    wire                     pe_act_valid_out [0:ROWS-1][0:COLS-1];
    wire signed [DATA_W-1:0] pe_wgt_in [0:ROWS-1][0:COLS-1];
    wire signed [ACC_W-1:0]  pe_acc_out [0:ROWS-1][0:COLS-1];
    wire                     pe_acc_valid [0:ROWS-1][0:COLS-1];

    // Maximum counter value (COLS-1)
    localparam [$clog2(COLS)-1:0] COL_MAX = COLS - 1;

    // ─── wgt_load_done: fires 1 cycle after last column loaded ───
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            wgt_load_done <= 1'b0;
        else
            wgt_load_done <= (state == S_WGT_LOAD && wgt_valid &&
                             wgt_col_cnt == COL_MAX);
    end

    // ─── FSM ───
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= state_next;
    end

    always @(*) begin
        state_next = state;
        case (state)
            S_IDLE: begin
                if (cmd_valid && cmd == MODE_WGT_LOAD)
                    state_next = S_WGT_LOAD;
            end
            S_WGT_LOAD: begin
                if (wgt_load_done)
                    state_next = S_READY;
            end
            S_READY: begin
                if (cmd_valid && cmd == MODE_COMPUTE)
                    state_next = S_COMPUTE;
                else if (cmd_valid && cmd == MODE_DRAIN)
                    state_next = S_DRAIN;
                else if (cmd_valid && cmd == MODE_WGT_LOAD)
                    state_next = S_WGT_LOAD;
            end
            S_COMPUTE: begin
                if (cmd_valid && cmd == MODE_DRAIN)
                    state_next = S_DRAIN;
                else if (cmd_valid && cmd == MODE_IDLE)
                    state_next = S_READY;
            end
            S_DRAIN: begin
                // PE processes drain in this cycle; output available next cycle
                state_next = S_DRAIN_OUT;
            end
            S_DRAIN_OUT: begin
                // Output is valid this cycle; return to READY for next drain/compute
                state_next = S_READY;
            end
            default: state_next = S_IDLE;
        endcase
    end

    // ─── Weight column counter ───
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            wgt_col_cnt <= 0;
        else if (state == S_WGT_LOAD && wgt_valid)
            wgt_col_cnt <= wgt_col_cnt + 1;
        else if (state != S_WGT_LOAD)
            wgt_col_cnt <= 0;
    end

    // ─── acc_out_valid: high during S_DRAIN_OUT ───
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_out_valid <= 1'b0;
        else
            acc_out_valid <= (state == S_DRAIN);
    end

    // ─── Status ───
    assign busy  = (state == S_WGT_LOAD) || (state == S_DRAIN) || (state == S_DRAIN_OUT);
    assign ready = (state == S_READY);

    // ─── PE Grid ───
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : gen_row
            for (c = 0; c < COLS; c = c + 1) begin : gen_col

                // Activation chain
                if (c == 0) begin : act_col0
                    assign pe_act_in[r][c] = act_data[r];
                end else begin : act_chain
                    assign pe_act_in[r][c] = pe_act_out[r][c-1];
                end

                // Weight input
                assign pe_wgt_in[r][c] = wgt_data[r];

                // PE mode
                assign pe_mode[r][c] =
                    (state == S_WGT_LOAD) ? MODE_WGT_LOAD :
                    (state == S_COMPUTE)  ? MODE_COMPUTE  :
                    (state == S_DRAIN && drain_col_sel == c[$clog2(COLS)-1:0]) ? MODE_DRAIN :
                    MODE_IDLE;

                // PE valid_in
                if (c == 0) begin : valid_col0
                    assign pe_valid[r][c] =
                        (state == S_WGT_LOAD && wgt_valid && wgt_col_cnt == 0) ? 1'b1 :
                        (state == S_COMPUTE)  ? act_valid :
                        (state == S_DRAIN && drain_col_sel == 0) ? 1'b1 :
                        1'b0;
                end else begin : valid_colN
                    assign pe_valid[r][c] =
                        (state == S_WGT_LOAD && wgt_valid && wgt_col_cnt == c[$clog2(COLS)-1:0]) ? 1'b1 :
                        (state == S_COMPUTE)  ? pe_act_valid_out[r][c-1] :
                        (state == S_DRAIN && drain_col_sel == c[$clog2(COLS)-1:0]) ? 1'b1 :
                        1'b0;
                end

                npu_pe u_pe (
                    .clk            (clk),
                    .rst_n          (rst_n),
                    .mode           (pe_mode[r][c]),
                    .valid_in       (pe_valid[r][c]),
                    .act_in         (pe_act_in[r][c]),
                    .act_out        (pe_act_out[r][c]),
                    .act_valid_out  (pe_act_valid_out[r][c]),
                    .weight_in      (pe_wgt_in[r][c]),
                    .acc_out        (pe_acc_out[r][c]),
                    .acc_valid      (pe_acc_valid[r][c])
                );

            end
        end
    endgenerate

    // ─── Output MUX: select drain column ───
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : gen_drain_mux
            reg signed [ACC_W-1:0] acc_mux;
            integer ci;
            always @(*) begin
                acc_mux = {ACC_W{1'b0}};
                for (ci = 0; ci < COLS; ci = ci + 1) begin
                    if (drain_col_sel == ci[$clog2(COLS)-1:0])
                        acc_mux = pe_acc_out[r][ci];
                end
            end
            assign acc_out[r] = acc_mux;
        end
    endgenerate

endmodule
