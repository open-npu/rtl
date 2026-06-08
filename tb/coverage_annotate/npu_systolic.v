//      // verilator_coverage annotation
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
 004959     input  wire                         clk,
 000011     input  wire                         rst_n,
        
            // ─── Control ───
%000000     input  wire [1:0]                   cmd,
%000000     input  wire                         cmd_valid,
        
            // ─── Weight Load ───
%000000     input  wire [DATA_W*ROWS-1:0]       wgt_data_flat,
%000000     input  wire                         wgt_valid,
        
            // ─── Activation Input ───
%000000     input  wire [DATA_W*ROWS-1:0]       act_data_flat,
%000000     input  wire                         act_valid,
        
            // ─── Drain Control ───
%000000     input  wire [$clog2(COLS)-1:0]      drain_col_sel,  // Which column to drain
        
            // ─── Accumulator Output ───
            output wire [ACC_W*ROWS-1:0]        acc_out_flat,
%000000     output reg                          acc_out_valid,
        
            // ─── Status ───
%000000     output wire                         busy,
%000000     output wire                         ready
        );
        
            // ─── Unpack flattened ports to internal arrays ───
%000000     wire signed [DATA_W-1:0] wgt_data [0:ROWS-1];
%000000     wire signed [DATA_W-1:0] act_data [0:ROWS-1];
            wire signed [ACC_W-1:0]  acc_out  [0:ROWS-1];
            genvar gi;
            generate
                for (gi = 0; gi < ROWS; gi = gi + 1) begin : unpack_ports
                    assign wgt_data[gi] = wgt_data_flat[DATA_W*gi +: DATA_W];
                    assign act_data[gi] = act_data_flat[DATA_W*gi +: DATA_W];
                    assign acc_out_flat[ACC_W*gi +: ACC_W] = acc_out[gi];
                end
            endgenerate
        
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
        
%000000     reg [2:0] state, state_next;
%000000     reg [$clog2(COLS)-1:0] wgt_col_cnt;
%000000     reg wgt_load_done;
        
            // ─── PE interconnect ───
            wire [1:0]               pe_mode   [0:ROWS-1][0:COLS-1];
%000000     wire                     pe_valid  [0:ROWS-1][0:COLS-1];
            wire signed [DATA_W-1:0] pe_act_in  [0:ROWS-1][0:COLS-1];
            wire signed [DATA_W-1:0] pe_act_out [0:ROWS-1][0:COLS-1];
%000000     wire                     pe_act_valid_out [0:ROWS-1][0:COLS-1];
            wire signed [DATA_W-1:0] pe_wgt_in [0:ROWS-1][0:COLS-1];
            wire signed [ACC_W-1:0]  pe_acc_out [0:ROWS-1][0:COLS-1];
%000000     wire                     pe_acc_valid [0:ROWS-1][0:COLS-1];
        
            // Maximum counter value (COLS-1)
            localparam [$clog2(COLS)-1:0] COL_MAX = COLS - 1;
        
            // ─── wgt_load_done: fires 1 cycle after last column loaded ───
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n)
 000035             wgt_load_done <= 1'b0;
                else
~002450             wgt_load_done <= (state == S_WGT_LOAD && wgt_valid &&
 002450                              wgt_col_cnt == COL_MAX);
            end
        
            // ─── FSM ───
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n)
 000035             state <= S_IDLE;
                else
 002450             state <= state_next;
            end
        
 014839     always @(*) begin
 014839         state_next = state;
 014839         case (state)
 014839             S_IDLE: begin
~014839                 if (cmd_valid && cmd == MODE_WGT_LOAD)
%000000                     state_next = S_WGT_LOAD;
                    end
%000000             S_WGT_LOAD: begin
%000000                 if (wgt_load_done)
%000000                     state_next = S_READY;
                    end
%000000             S_READY: begin
~014839                 if (cmd_valid && cmd == MODE_COMPUTE)
%000000                     state_next = S_COMPUTE;
%000000                 else if (cmd_valid && cmd == MODE_DRAIN)
%000000                     state_next = S_DRAIN;
%000000                 else if (cmd_valid && cmd == MODE_WGT_LOAD)
%000000                     state_next = S_WGT_LOAD;
                    end
%000000             S_COMPUTE: begin
~014839                 if (cmd_valid && cmd == MODE_DRAIN)
%000000                     state_next = S_DRAIN;
%000000                 else if (cmd_valid && cmd == MODE_IDLE)
%000000                     state_next = S_READY;
                    end
%000000             S_DRAIN: begin
                        // PE processes drain in this cycle; output available next cycle
%000000                 state_next = S_DRAIN_OUT;
                    end
%000000             S_DRAIN_OUT: begin
                        // Output is valid this cycle; return to READY for next drain/compute
%000000                 state_next = S_READY;
                    end
%000000             default: state_next = S_IDLE;
                endcase
            end
        
            // ─── Weight column counter ───
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n)
 000035             wgt_col_cnt <= 0;
~002450         else if (state == S_WGT_LOAD && wgt_valid)
%000000             wgt_col_cnt <= wgt_col_cnt + 1;
~002450         else if (state != S_WGT_LOAD)
 002450             wgt_col_cnt <= 0;
            end
        
            // ─── acc_out_valid: high during S_DRAIN_OUT ───
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n)
 000035             acc_out_valid <= 1'b0;
                else
 002450             acc_out_valid <= (state == S_DRAIN);
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
 237424         for (r = 0; r < ROWS; r = r + 1) begin : gen_drain_mux
                    reg signed [ACC_W-1:0] acc_mux;
                    integer ci;
 237424             always @(*) begin
 237424                 acc_mux = {ACC_W{1'b0}};
 3798784                 for (ci = 0; ci < COLS; ci = ci + 1) begin
 3561360                     if (drain_col_sel == ci[$clog2(COLS)-1:0])
 237424                         acc_mux = pe_acc_out[r][ci];
                        end
                    end
                    assign acc_out[r] = acc_mux;
                end
            endgenerate
        
        endmodule
        
