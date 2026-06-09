// Open-NPU RTL — DMA Engine
// SPDX-License-Identifier: Apache-2.0
//
// Simple DMA controller for transferring data between external memory
// (Wishbone master interface) and internal SRAMs (direct SRAM interface).
//
// Features:
//   - Configurable source address, destination address, transfer length
//   - Configurable external address stride (load/store)
//   - Auto-increment SRAM addressing
//   - Configurable burst length (4/8/16/32 words)
//   - Direction: MEM→SRAM (load) or SRAM→MEM (store)
//   - Busy/done status signals
//   - Abort capability
//
// The DMA operates on 32-bit words. For byte-level transfers,
// the controller module handles packing/unpacking.
//
// Stride behavior:
//   in_stride (load path):  external address advances by in_stride bytes per word
//   out_stride (store path): external address advances by out_stride bytes per word
//   If stride == 0, defaults to 4 (word-aligned linear access).

`include "npu_defines.vh"

module npu_dma #(
    parameter ADDR_W = 32,           // External memory address width
    parameter DATA_W = 32,           // Data bus width
    parameter SRAM_ADDR_W = 16       // Internal SRAM address width
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ─── Control Interface (from CSR/Controller) ───
    input  wire                 start,          // Pulse to begin transfer
    input  wire                 abort,          // Pulse to abort
    input  wire                 dir,            // 0=MEM→SRAM (load), 1=SRAM→MEM (store)
    input  wire [ADDR_W-1:0]   ext_addr,       // External memory start address
    input  wire [SRAM_ADDR_W-1:0] sram_addr,   // Internal SRAM start address
    input  wire [15:0]         xfer_len,       // Transfer length (words)
    input  wire [1:0]          burst_cfg,      // Burst: 0=4, 1=8, 2=16, 3=32
    input  wire [31:0]         cfg_in_stride,  // External address stride for load (bytes)
    input  wire [31:0]         cfg_out_stride, // External address stride for store (bytes)

    // ─── Status Outputs ───
    output reg                  busy,
    output reg                  done_pulse,     // 1-cycle pulse on completion
    output reg  [15:0]         xfer_count,     // Words transferred so far

    // ─── Wishbone Master Interface (to external memory) ───
    output reg                  wb_cyc_o,
    output reg                  wb_stb_o,
    output reg                  wb_we_o,
    output reg  [ADDR_W-1:0]   wb_adr_o,
    output reg  [DATA_W-1:0]   wb_dat_o,
    output wire [3:0]          wb_sel_o,
    input  wire                 wb_ack_i,
    input  wire [DATA_W-1:0]   wb_dat_i,

    // ─── SRAM Interface (to internal SRAM) ───
    output reg                  sram_en,
    output reg                  sram_we,
    output reg  [SRAM_ADDR_W-1:0] sram_addr_o,
    output reg  [DATA_W-1:0]   sram_wdata,
    input  wire [DATA_W-1:0]   sram_rdata
);

    // Always select all bytes
    assign wb_sel_o = 4'hF;

    // ─── FSM States ───
    localparam S_IDLE    = 3'd0;
    localparam S_LOAD    = 3'd1;  // MEM→SRAM: issue WB read
    localparam S_LOAD_WR = 3'd2;  // MEM→SRAM: write to SRAM
    localparam S_STORE_RD= 3'd3;  // SRAM→MEM: read from SRAM
    localparam S_STORE   = 3'd4;  // SRAM→MEM: issue WB write
    localparam S_DONE    = 3'd5;

    reg [2:0] state;
    reg [ADDR_W-1:0]      r_ext_addr;
    reg [SRAM_ADDR_W-1:0] r_sram_addr;
    reg [15:0]            r_xfer_len;
    reg                   r_dir;
    reg [DATA_W-1:0]      r_data_buf;  // temporary data buffer
    reg [ADDR_W-1:0]      r_stride;    // latched stride value for current direction

    // Burst length decode (not used for now — single-beat transfers)
    // wire [5:0] burst_len = (burst_cfg == 2'd0) ? 6'd4 :
    //                        (burst_cfg == 2'd1) ? 6'd8 :
    //                        (burst_cfg == 2'd2) ? 6'd16 : 6'd32;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done_pulse <= 1'b0;
            xfer_count <= 16'd0;
            wb_cyc_o   <= 1'b0;
            wb_stb_o   <= 1'b0;
            wb_we_o    <= 1'b0;
            wb_adr_o   <= {ADDR_W{1'b0}};
            wb_dat_o   <= {DATA_W{1'b0}};
            sram_en    <= 1'b0;
            sram_we    <= 1'b0;
            sram_addr_o<= {SRAM_ADDR_W{1'b0}};
            sram_wdata <= {DATA_W{1'b0}};
            r_ext_addr <= {ADDR_W{1'b0}};
            r_sram_addr<= {SRAM_ADDR_W{1'b0}};
            r_xfer_len <= 16'd0;
            r_dir      <= 1'b0;
            r_data_buf <= {DATA_W{1'b0}};
            r_stride   <= {ADDR_W{1'b0}};
        end else begin
            done_pulse <= 1'b0;  // default: clear pulse
            sram_en    <= 1'b0;  // default: SRAM idle
            sram_we    <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start && xfer_len != 0) begin
                        busy       <= 1'b1;
                        r_ext_addr <= ext_addr;
                        r_sram_addr<= sram_addr;
                        r_xfer_len <= xfer_len;
                        r_dir      <= dir;
                        // Latch stride for this transfer direction
                        r_stride   <= dir ? cfg_out_stride : cfg_in_stride;
                        xfer_count <= 16'd0;
                        if (!dir)
                            state <= S_LOAD;
                        else
                            state <= S_STORE_RD;
                    end
                end

                // ─── LOAD path: MEM → SRAM ───
                S_LOAD: begin
                    if (abort) begin
                        wb_cyc_o <= 1'b0;
                        wb_stb_o <= 1'b0;
                        state    <= S_DONE;
                    end else begin
                        // Issue Wishbone read
                        wb_cyc_o <= 1'b1;
                        wb_stb_o <= 1'b1;
                        wb_we_o  <= 1'b0;
                        wb_adr_o <= r_ext_addr;
                        if (wb_ack_i) begin
                            r_data_buf <= wb_dat_i;
                            wb_cyc_o   <= 1'b0;
                            wb_stb_o   <= 1'b0;
                            state      <= S_LOAD_WR;
                        end
                    end
                end

                S_LOAD_WR: begin
                    // Write fetched data to SRAM
                    sram_en     <= 1'b1;
                    sram_we     <= 1'b1;
                    sram_addr_o <= r_sram_addr;
                    sram_wdata  <= r_data_buf;
                    // Advance pointers
                    //   external: advance by stride (or 4 if stride==0 for linear)
                    r_ext_addr  <= r_ext_addr + (r_stride != 0 ? r_stride : 32'd4);
                    r_sram_addr <= r_sram_addr + 1;
                    xfer_count  <= xfer_count + 1;
                    if (xfer_count + 1 >= r_xfer_len)
                        state <= S_DONE;
                    else
                        state <= S_LOAD;
                end

                // ─── STORE path: SRAM → MEM ───
                S_STORE_RD: begin
                    if (abort) begin
                        state <= S_DONE;
                    end else begin
                        // Read from SRAM (address presented this cycle, data next)
                        sram_en     <= 1'b1;
                        sram_we     <= 1'b0;
                        sram_addr_o <= r_sram_addr;
                        state       <= S_STORE;
                    end
                end

                S_STORE: begin
                    if (abort) begin
                        wb_cyc_o <= 1'b0;
                        wb_stb_o <= 1'b0;
                        state    <= S_DONE;
                    end else begin
                        // Issue Wishbone write with SRAM read data
                        wb_cyc_o <= 1'b1;
                        wb_stb_o <= 1'b1;
                        wb_we_o  <= 1'b1;
                        wb_adr_o <= r_ext_addr;
                        wb_dat_o <= sram_rdata;
                        if (wb_ack_i) begin
                            wb_cyc_o    <= 1'b0;
                            wb_stb_o    <= 1'b0;
                            // Advance pointers
                            //   external: advance by stride (or 4 if stride==0 for linear)
                            r_ext_addr  <= r_ext_addr + (r_stride != 0 ? r_stride : 32'd4);
                            r_sram_addr <= r_sram_addr + 1;
                            xfer_count  <= xfer_count + 1;
                            if (xfer_count + 1 >= r_xfer_len)
                                state <= S_DONE;
                            else
                                state <= S_STORE_RD;
                        end
                    end
                end

                S_DONE: begin
                    busy       <= 1'b0;
                    done_pulse <= 1'b1;
                    wb_cyc_o   <= 1'b0;
                    wb_stb_o   <= 1'b0;
                    state      <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
