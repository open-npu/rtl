// Open-NPU RTL — Systolic Array (ROWS x COLS)
// SPDX-License-Identifier: Apache-2.0
//
// Weight-stationary array with in-array partial-sum reduction.
//
// Mapping (unchanged from the previous revision):
//   row r    -> one k-slot of the flattened kernel*in_c dimension
//   column c -> one output channel of the current oc_group
//   PE[r][c] -> W[k=r][oc=c]
//
// Dataflow (this is what changed):
//   - Activations are BROADCAST across a row. There is no left-to-right chain,
//     because nothing travels sideways: the reduction is vertical.
//   - Partial sums flow DOWN a column. PE[r][c] computes
//         psum_out = psum_in + act[r] * W[r][c]
//     with row 0 seeded at zero, so the bottom row of column c emits the
//     complete dot product over all ROWS k-slots.
//   - One full result vector (COLS dot products) leaves the array per cycle,
//     pipelined, instead of the previous drain-one-column-per-3-cycles plus an
//     external adder tree.
//
// Skew: for the descending partial sum to meet the right activation, act[k=r]
// must enter row r exactly r cycles after act[k=0] entered row 0. The caller
// presents an unskewed ROWS-wide vector; the triangular buffer below applies
// the delay.
//
// Bit-exactness: the chain accumulates in row order,
//     ((((0 + a0*w0) + a1*w1) + ...) + a15*w15)
// which is the same left-to-right order, at the same ACC_W width, as the
// external adder tree it replaces. Results are bit-identical.
//
// Phases:
//   1. WGT_LOAD: one column per cycle (broadcast bus + column select), COLS cycles.
//   2. COMPUTE:  stream one ROWS-wide activation vector per cycle. Results
//                appear on psum_out_flat ROWS cycles later, one per cycle.

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
    input  wire [DATA_W*ROWS-1:0]       wgt_data_flat,
    input  wire                         wgt_valid,
    // While COMPUTE, wgt_valid fills weight_nxt one column per cycle.
    // swap_wgt commits every column after the old psum chain has drained.
    input  wire                         swap_wgt,

    // ─── Activation Input (unskewed; one vector per cycle) ───
    input  wire [DATA_W*ROWS-1:0]       act_data_flat,
    input  wire                         act_valid,

    // ─── Result Output (one vector per cycle, COLS dot products) ───
    output wire [ACC_W*COLS-1:0]        psum_out_flat,
    output wire                         psum_out_valid,

    // ─── Status ───
    output wire                         busy,
    output wire                         ready
);

    // ─── Unpack flattened ports ───
    wire signed [DATA_W-1:0] wgt_data [0:ROWS-1];
    wire signed [DATA_W-1:0] act_data [0:ROWS-1];
    genvar gi;
    generate
        for (gi = 0; gi < ROWS; gi = gi + 1) begin : unpack_ports
            assign wgt_data[gi] = wgt_data_flat[DATA_W*gi +: DATA_W];
            assign act_data[gi] = act_data_flat[DATA_W*gi +: DATA_W];
        end
    endgenerate

    // ─── Mode encoding ───
    localparam MODE_IDLE     = 2'b00;
    localparam MODE_WGT_LOAD = 2'b01;
    localparam MODE_COMPUTE  = 2'b10;

    // ─── FSM States ───
    localparam S_IDLE     = 3'd0;
    localparam S_WGT_LOAD = 3'd1;
    localparam S_READY    = 3'd2;
    localparam S_COMPUTE  = 3'd3;

    reg [2:0] state, state_next;
    reg [$clog2(COLS)-1:0] wgt_col_cnt;
    reg [$clog2(COLS)-1:0] nxt_col;
    reg wgt_load_done;

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
                else if (cmd_valid && cmd == MODE_WGT_LOAD)
                    state_next = S_WGT_LOAD;
            end
            S_COMPUTE: begin
                if (cmd_valid && cmd == MODE_IDLE)
                    state_next = S_READY;
                else if (cmd_valid && cmd == MODE_WGT_LOAD)
                    state_next = S_WGT_LOAD;
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

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            nxt_col <= 0;
        else if (swap_wgt || state != S_COMPUTE)
            nxt_col <= 0;
        else if (wgt_valid)
            nxt_col <= nxt_col + 1;
    end

    // ─── Status ───
    assign busy  = (state == S_WGT_LOAD);
    // The array accepts a new activation vector every cycle once it is
    // computing, so S_COMPUTE is "ready" too — otherwise the caller would have
    // to bounce through S_READY between pixels and lose the pipelining.
    assign ready = (state == S_READY) || (state == S_COMPUTE);

    // ═══════════════════════════════════════════════════════════════════
    // Input skew buffer
    //
    // Row r needs its activation r cycles after row 0. skew_data[r] is
    // act_data[r] delayed by r registers; row 0 passes straight through.
    // Total cost is ROWS*(ROWS-1)/2 registers of DATA_W bits.
    // ═══════════════════════════════════════════════════════════════════
    wire signed [DATA_W-1:0] skew_data  [0:ROWS-1];
    wire                     skew_valid [0:ROWS-1];

    genvar sr, sd;
    generate
        for (sr = 0; sr < ROWS; sr = sr + 1) begin : gen_skew_row
            if (sr == 0) begin : skew_passthrough
                assign skew_data[sr]  = act_data[sr];
                assign skew_valid[sr] = act_valid;
            end else begin : skew_delay
                reg signed [DATA_W-1:0] d [0:sr-1];
                reg                     v [0:sr-1];
                integer si;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (si = 0; si < sr; si = si + 1) begin
                            d[si] <= {DATA_W{1'b0}};
                            v[si] <= 1'b0;
                        end
                    end else begin
                        d[0] <= act_data[sr];
                        v[0] <= act_valid;
                        for (si = 1; si < sr; si = si + 1) begin
                            d[si] <= d[si-1];
                            v[si] <= v[si-1];
                        end
                    end
                end
                assign skew_data[sr]  = d[sr-1];
                assign skew_valid[sr] = v[sr-1];
            end
        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════════
    // PE Grid
    // ═══════════════════════════════════════════════════════════════════
    wire [1:0]               pe_mode    [0:ROWS-1][0:COLS-1];
    wire                     pe_valid   [0:ROWS-1][0:COLS-1];
    wire signed [DATA_W-1:0] pe_act_in  [0:ROWS-1][0:COLS-1];
    wire signed [DATA_W-1:0] pe_wgt_in  [0:ROWS-1][0:COLS-1];
    wire signed [ACC_W-1:0]  pe_psum_in [0:ROWS-1][0:COLS-1];
    wire signed [ACC_W-1:0]  pe_psum_out[0:ROWS-1][0:COLS-1];
    wire                     pe_psum_val[0:ROWS-1][0:COLS-1];

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : gen_row
            for (c = 0; c < COLS; c = c + 1) begin : gen_col

                // Activation: broadcast across the row (skewed per row).
                assign pe_act_in[r][c] = skew_data[r];

                // Weight: broadcast bus, one column enabled per load cycle.
                assign pe_wgt_in[r][c] = wgt_data[r];

                // Partial sum: seeded at zero on the top row, chained downward.
                if (r == 0) begin : psum_top
                    assign pe_psum_in[r][c] = {ACC_W{1'b0}};
                end else begin : psum_chain
                    assign pe_psum_in[r][c] = pe_psum_out[r-1][c];
                end

                assign pe_mode[r][c] =
                    (state == S_WGT_LOAD) ? MODE_WGT_LOAD :
                    (state == S_COMPUTE)  ? MODE_COMPUTE  :
                    MODE_IDLE;

                assign pe_valid[r][c] =
                    (state == S_WGT_LOAD) ? (wgt_valid &&
                        wgt_col_cnt == c[$clog2(COLS)-1:0]) :
                    (state == S_COMPUTE)  ? skew_valid[r] :
                    1'b0;

                npu_pe u_pe (
                    .clk            (clk),
                    .rst_n          (rst_n),
                    .mode           (pe_mode[r][c]),
                    .valid_in       (pe_valid[r][c]),
                    .act_in         (pe_act_in[r][c]),
                    .weight_in      (pe_wgt_in[r][c]),
                    .load_nxt       ((state == S_COMPUTE) && wgt_valid
                                     && (nxt_col == c[$clog2(COLS)-1:0])),
                    .swap_wgt       (swap_wgt),
                    .psum_in        (pe_psum_in[r][c]),
                    .psum_out       (pe_psum_out[r][c]),
                    .psum_valid_out (pe_psum_val[r][c])
                );

            end
        end
    endgenerate

    // ─── Output: bottom row carries the completed dot products ───
    generate
        for (c = 0; c < COLS; c = c + 1) begin : gen_out
            assign psum_out_flat[ACC_W*c +: ACC_W] = pe_psum_out[ROWS-1][c];
        end
    endgenerate

    assign psum_out_valid = pe_psum_val[ROWS-1][0];

endmodule
