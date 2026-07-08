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
//   - 2D transfer mode (row_len × row_count with row_stride) for NHWC tiled access
//
// The DMA operates on 32-bit words. For byte-level transfers,
// the controller module handles packing/unpacking.
//
// Stride behavior:
//   1D mode (row_len==0 or row_count==0):
//     in_stride/out_stride: external address advances by stride bytes per word
//     If stride == 0, defaults to 4 (word-aligned linear access).
//   2D mode (row_len!=0 && row_count!=0):
//     Transfers row_count rows, each row_len words.
//     Within a row: external address advances by stride (or 4) per word.
//     Between rows: external address jumps by row_stride (= stride, reinterpreted).
//     SRAM side is always contiguous (r_sram_addr++ every word).
//     For NHWC: row_len = tile_w * out_c * eb / 4, row_stride = out_w * out_c * eb.
//     Set stride = row_stride (bytes between rows), row_len = words per row.

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
    input  wire [15:0]         cfg_row_len,    // 2D: words per row written to DDR (0 = 1D mode)
    input  wire [15:0]         cfg_row_count,  // 2D: number of rows (0 = 1D mode)
    input  wire [15:0]         cfg_src_row_len,// 2D: words per row read from SRAM (0 = same as row_len)

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
    reg [ADDR_W-1:0]      r_stride;    // latched stride value (row stride in 2D, per-word in 1D)
    // 2D mode: r_row_len = words per row, r_row_count = rows remaining
    // r_row_word_count = words transferred in current row
    // r_row_base = external address of current row start
    // When r_row_len==0 or r_row_count==0, behaves as 1D (backward compatible)
    // In 2D: within row, ext_addr advances by 4 (contiguous). At row end,
    //   ext_addr = r_row_base + r_stride (jump to next row in NHWC layout).
    //   SRAM is always contiguous (r_sram_addr++ every word).
    reg [15:0]            r_row_len;
    reg [15:0]            r_row_count;
    reg [15:0]            r_row_word_count;
    reg [15:0]            r_src_row_len;  // SRAM words per row (>= r_row_len; extra = padding skip)
    reg [ADDR_W-1:0]      r_row_base;
    wire                  mode_2d = (r_row_len != 16'd0) && (r_row_count != 16'd0);
    // In 2D store: words [0..r_row_len-1] written to DDR, words [r_row_len..r_src_row_len-1]
    //   read from SRAM but skipped (padding). SRAM always r_sram_addr++.
    wire [15:0]           effective_src_len = (r_src_row_len != 16'd0) ? r_src_row_len : r_row_len;

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
            r_row_len  <= 16'd0;
            r_row_count<= 16'd0;
            r_row_word_count <= 16'd0;
            r_src_row_len <= 16'd0;
            r_row_base <= {ADDR_W{1'b0}};
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
                        // 2D mode parameters (latched at start)
                        r_row_len  <= cfg_row_len;
                        r_row_count<= cfg_row_count;
                        r_src_row_len <= cfg_src_row_len;
                        r_row_word_count <= 16'd0;
                        r_row_base <= ext_addr;
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
                        `ifndef SYNTHESIS
                        if (xfer_count < 3 && r_sram_addr >= 16'd6144)
                            $display("[DMA_LD] t=%0t xfer=%0d addr=0x%08x sram=%0d", $time, xfer_count, r_ext_addr, r_sram_addr);
                        `endif
                        if (wb_ack_i) begin
                            r_data_buf <= wb_dat_i;
                            `ifndef SYNTHESIS
                            if (xfer_count < 3 && r_sram_addr >= 16'd6144)
                                $display("[DMA_LD_DATA] t=%0t xfer=%0d data=0x%08x", $time, xfer_count, wb_dat_i);
                            `endif
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
                    // Advance SRAM pointer (always contiguous)
                    r_sram_addr <= r_sram_addr + 1;
                    xfer_count  <= xfer_count + 1;
                    r_row_word_count <= r_row_word_count + 1;
                    // Advance external address
                    if (mode_2d && (r_row_word_count + 1 >= r_row_len)) begin
                        // End of 2D row: jump to next row in NHWC layout
                        // r_stride = row stride (bytes between row starts)
                        r_ext_addr <= r_row_base + r_stride;
                        r_row_base <= r_row_base + r_stride;
                        r_row_count <= r_row_count - 1;
                        r_row_word_count <= 16'd0;
                    end else if (mode_2d) begin
                        // Within 2D row: contiguous (4 bytes per word)
                        r_ext_addr  <= r_ext_addr + 32'd4;
                    end else begin
                        // 1D mode: advance by stride (or 4 if stride==0)
                        r_ext_addr  <= r_ext_addr + (r_stride != 0 ? r_stride : 32'd4);
                    end
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
                        `ifndef SYNTHESIS
                        if (mode_2d && (xfer_count < 8 || r_row_word_count == 16'd0 || r_row_word_count == 16'd1))
                            $display("[DMA_2D_WR] t=%0t xfer=%0d addr=0x%08x sram=%0d data=0x%08x row_word=%0d/%0d row_cnt=%0d",
                                     $time, xfer_count, r_ext_addr, r_sram_addr, sram_rdata,
                                     r_row_word_count, r_row_len, r_row_count);
                        `endif
                        wb_dat_o <= sram_rdata;
                        if (wb_ack_i) begin
                            wb_cyc_o    <= 1'b0;
                            wb_stb_o    <= 1'b0;
                            // Advance SRAM pointer (always contiguous)
                            r_sram_addr <= r_sram_addr + 1;
                            xfer_count  <= xfer_count + 1;
                            r_row_word_count <= r_row_word_count + 1;
                            // Advance external address
                            if (mode_2d && (r_row_word_count + 1 >= r_row_len)) begin
                                // End of DDR row: jump to next row in NHWC layout.
                                // Skip SRAM padding: advance sram_addr by (src_len - row_len)
                                // so next row starts at correct SRAM offset.
                                if (effective_src_len > r_row_len)
                                    r_sram_addr <= r_sram_addr + 1 + (effective_src_len - r_row_len);
                                r_ext_addr <= r_row_base + r_stride;
                                r_row_base <= r_row_base + r_stride;
                                r_row_count <= r_row_count - 1;
                                r_row_word_count <= 16'd0;
                            end else if (mode_2d) begin
                                // Within 2D row: contiguous (4 bytes per word)
                                r_ext_addr  <= r_ext_addr + 32'd4;
                            end else begin
                                // 1D mode: advance by stride (or 4 if stride==0)
                                r_ext_addr  <= r_ext_addr + (r_stride != 0 ? r_stride : 32'd4);
                            end
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
