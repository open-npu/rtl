// Open-NPU RTL — DMA Engine (Configurable 32 / 64 / 128-bit Master)
// SPDX-License-Identifier: Apache-2.0
//
// Transfers data between external memory (Wishbone-B4 master, or AXI4 via
// npu_wb_to_axi4) and internal scratchpad SRAMs.
//
// Features:
//   - Parameterized external bus width DATA_W in {32, 64, 128} bits
//   - Arbitrary 4-byte-aligned external start addresses and 2D row strides:
//     automatically aligns bus address and applies head/tail lane masks
//   - 1D contiguous / strided and 2D (row_len x row_count) NHWC tile modes
//   - Incrementing-burst loads and 1-beat-prefetched wide-SRAM stores
//   - Exports 4KB-safe contiguous beat count (wb_burst_beats_o) for AXI4 ARLEN

`include "npu_defines.vh"

module npu_dma #(
    parameter ADDR_W      = 32,              // External memory byte address width
    parameter DATA_W      = `DMA_BUS_WIDTH,  // External data bus width: 32, 64, or 128
    parameter SRAM_ADDR_W = 16               // Internal SRAM 32-bit word address width
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // ─── Control Interface (from CSR/Controller) ───
    input  wire                     start,          // Pulse to begin transfer
    input  wire                     abort,          // Pulse to abort
    input  wire                     dir,            // 0=MEM->SRAM (load), 1=SRAM->MEM (store)
    input  wire [ADDR_W-1:0]        ext_addr,       // External memory start byte address (4B-aligned)
    input  wire [SRAM_ADDR_W-1:0]   sram_addr,      // Internal SRAM start word address
    input  wire [15:0]              xfer_len,       // Transfer length (32-bit words)
    input  wire [1:0]               burst_cfg,      // Burst hint (reserved)
    input  wire [31:0]              cfg_in_stride,  // Unused in 1D (kept for port compatibility)
    input  wire [31:0]              cfg_out_stride, // 2D external row pitch (bytes)
    input  wire [15:0]              cfg_row_len,    // 2D words per row in DDR (0 = 1D mode)
    input  wire [15:0]              cfg_row_count,  // 2D number of rows (0 = 1D mode)
    input  wire [15:0]              cfg_src_row_len,// 2D words per row in SRAM (0 = same as row_len)

    // ─── Status Outputs ───
    output reg                      busy,
    output reg                      done_pulse,     // 1-cycle pulse on completion
    output reg  [15:0]              xfer_count,     // 32-bit words transferred so far

    // ─── Wishbone-B4 Master Interface (to external memory or AXI4 bridge) ───
    output reg                      wb_cyc_o,
    output reg                      wb_stb_o,
    output reg                      wb_we_o,
    output reg  [ADDR_W-1:0]        wb_adr_o,
    output reg  [DATA_W-1:0]        wb_dat_o,
    output reg  [(DATA_W/8)-1:0]    wb_sel_o,
    output wire [2:0]               wb_cti_o,
    output wire [4:0]               wb_burst_beats_o,
    input  wire                     wb_ack_i,
    input  wire [DATA_W-1:0]        wb_dat_i,

    // ─── SRAM Port A Interface (wide write for loads) ───
    output reg                      sram_en,
    output reg  [(DATA_W/32)-1:0]   sram_we,
    output reg  [SRAM_ADDR_W-1:0]   sram_addr_o,
    output reg  [DATA_W-1:0]        sram_wdata,
    input  wire [31:0]              sram_rdata,

    // ─── Wide OFM Port B Read Interface (512-bit read for stores) ───
    output reg                      sram_wide_en,
    output reg  [SRAM_ADDR_W-1:0]   sram_wide_addr,
    input  wire [`SRAM_B_WIDTH-1:0] sram_rdata_wide,
    input  wire                     sram_wide_wait
);

    localparam BUS_WORDS = DATA_W / 32;  // 1, 2, or 4
    localparam BUS_BYTES = DATA_W / 8;   // 4, 8, or 16
    localparam SEL_W     = DATA_W / 8;

    wire _unused_ok = &{1'b0, burst_cfg, cfg_in_stride, sram_rdata};

    // ─── Helper functions for bus alignment & lane steering ───
    function [ADDR_W-1:0] align_ext_addr;
        input [ADDR_W-1:0] a;
        begin
            if (BUS_WORDS == 4)
                align_ext_addr = {a[ADDR_W-1:4], 4'b0000};
            else if (BUS_WORDS == 2)
                align_ext_addr = {a[ADDR_W-1:3], 3'b000};
            else
                align_ext_addr = a;
        end
    endfunction

    function [2:0] addr_lane_off;
        input [1:0] word_lsbs;
        begin
            if (BUS_WORDS == 4)
                addr_lane_off = {1'b0, word_lsbs};
            else if (BUS_WORDS == 2)
                addr_lane_off = {2'b00, word_lsbs[0]};
            else
                addr_lane_off = 3'd0;
        end
    endfunction

    function [SEL_W-1:0] make_write_sel;
        input [2:0] loff;
        input [2:0] nwords;
        integer ln;
        begin
            make_write_sel = {SEL_W{1'b0}};
            for (ln = 0; ln < BUS_WORDS; ln = ln + 1) begin
                if ((ln[2:0] >= loff) && (ln[2:0] < (loff + nwords)))
                    make_write_sel[4*ln +: 4] = 4'hF;
            end
        end
    endfunction

    function [BUS_WORDS-1:0] make_sram_we;
        input [2:0] nwords;
        integer ln;
        begin
            make_sram_we = {BUS_WORDS{1'b0}};
            for (ln = 0; ln < BUS_WORDS; ln = ln + 1) begin
                if (ln[2:0] < nwords)
                    make_sram_we[ln] = 1'b1;
            end
        end
    endfunction

    function [DATA_W-1:0] unpack_load_beat;
        input [DATA_W-1:0] raw;
        input [2:0]        loff;
        integer dst_ln;
        reg [2:0] src_ln;
        begin
            unpack_load_beat = {DATA_W{1'b0}};
            for (dst_ln = 0; dst_ln < BUS_WORDS; dst_ln = dst_ln + 1) begin
                src_ln = dst_ln[2:0] + loff;
                if (src_ln < BUS_WORDS[2:0])
                    unpack_load_beat[32*dst_ln +: 32] = raw[32*src_ln +: 32];
            end
        end
    endfunction

    function [DATA_W-1:0] pack_store_beat;
        input [`SRAM_B_WIDTH+96-1:0] wide_ext;
        input [4:0]                  base_idx;
        input [2:0]                  loff;
        integer bus_ln;
        reg [4:0] word_idx;
        begin
            pack_store_beat = {DATA_W{1'b0}};
            for (bus_ln = 0; bus_ln < BUS_WORDS; bus_ln = bus_ln + 1) begin
                if (bus_ln[2:0] >= loff) begin
                    word_idx = base_idx + {2'd0, (bus_ln[2:0] - loff)};
                    if (word_idx < 5'd19)
                        pack_store_beat[32*bus_ln +: 32] = wide_ext[32*word_idx +: 32];
                end
            end
        end
    endfunction

    // ─── FSM States ───
    localparam S_IDLE        = 3'd0;
    localparam S_LOAD        = 3'd1;
    localparam S_STORE_LATCH = 3'd2;
    localparam S_STORE_RD    = 3'd3;
    localparam S_STORE       = 3'd4;
    localparam S_DONE        = 3'd5;
    localparam S_STORE_REQ   = 3'd6;

    reg [2:0]               state;
    reg [ADDR_W-1:0]        r_ext_addr;
    reg [SRAM_ADDR_W-1:0]   r_sram_addr;
    reg [15:0]              r_xfer_len;
    reg                     r_dir;
    reg [ADDR_W-1:0]        r_stride;
    reg [15:0]              r_row_len;
    reg [15:0]              r_row_count;
    reg [15:0]              r_row_word_count;
    reg [15:0]              r_src_row_len;
    reg [ADDR_W-1:0]        r_row_base;
    reg [`SRAM_B_WIDTH-1:0] store_wide_buf;
    reg [`SRAM_B_WIDTH-1:0] store_pf_buf;
    reg                     store_pf_valid;
    reg [1:0]               pf_state;
    reg [4:0]               store_wide_idx;
    reg [2:0]               r_cur_nwords;

    wire        mode_2d        = (r_row_len != 16'd0) && (r_row_count != 16'd0);
    wire        non_contig_1d  = (!mode_2d) && (r_stride != 32'd0) && (r_stride != 32'd4);
    wire [15:0] effective_src_len = (r_src_row_len != 16'd0) ? r_src_row_len : r_row_len;
    wire [5:0]  rem_in_beat    = 6'd16 - {1'b0, store_wide_idx};

    // ─── Combinational LOAD beat sizing ───
    wire [2:0]  ld_loff     = addr_lane_off(r_ext_addr[3:2]);
    wire [2:0]  ld_max_bw   = non_contig_1d ? 3'd1 : (BUS_WORDS[2:0] - ld_loff);
    wire [15:0] ld_rem_tot  = r_xfer_len - xfer_count;
    wire [15:0] ld_rem_row  = r_row_len - r_row_word_count;
    wire [15:0] ld_rem_req  = (mode_2d && (ld_rem_row < ld_rem_tot)) ? ld_rem_row : ld_rem_tot;
    wire [2:0]  ld_nwords   = (ld_rem_req < {13'd0, ld_max_bw}) ? ld_rem_req[2:0] : ld_max_bw;
    wire        ld_xfer_end = ({1'b0, xfer_count} + {14'd0, ld_nwords}) >= {1'b0, r_xfer_len};
    wire        ld_row_end  = mode_2d && (({1'b0, r_row_word_count} + {14'd0, ld_nwords}) >= {1'b0, r_row_len});
    wire        ld_burst_end = ld_xfer_end || ld_row_end || non_contig_1d;
    wire [BUS_WORDS-1:0] ld_sram_we_comb = make_sram_we(ld_nwords);
    wire [DATA_W-1:0]    ld_wdata_comb   = unpack_load_beat(wb_dat_i, ld_loff);

    // ─── Combinational AXI4-friendly contiguous burst beat count (1..16, 4KB-safe) ───
    wire [15:0] ld_rem_after_first = ld_rem_req - {13'd0, ld_nwords};
    wire [15:0] ld_extra_beats = (BUS_WORDS == 4) ? ((ld_rem_after_first + 16'd3) >> 2) :
                                 (BUS_WORDS == 2) ? ((ld_rem_after_first + 16'd1) >> 1) :
                                                    ld_rem_after_first;
    wire [16:0] ld_total_beats = non_contig_1d ? 17'd1 : (17'd1 + {1'b0, ld_extra_beats});
    wire [12:0] bytes_to_4k    = 13'h1000 - {1'b0, wb_adr_o[11:0]};
    wire [12:0] beats_to_4k    = (BUS_BYTES == 16) ? (bytes_to_4k >> 4) :
                                 (BUS_BYTES == 8)  ? (bytes_to_4k >> 3) :
                                                     (bytes_to_4k >> 2);
    wire [16:0] ld_clamped_4k  = (ld_total_beats < {4'd0, beats_to_4k})
                               ? ld_total_beats : {4'd0, beats_to_4k};
    assign wb_burst_beats_o    = (ld_clamped_4k > 17'd16) ? 5'd16 :
                                 (ld_clamped_4k == 17'd0) ? 5'd1  : ld_clamped_4k[4:0];

    // ─── Combinational STORE beat / next-beat sizing ───
    wire [ADDR_W-1:0] st_step_bytes = non_contig_1d ? r_stride : ({27'd0, r_cur_nwords, 2'b00});
    wire [ADDR_W-1:0] st_next_ext_contig = r_ext_addr + st_step_bytes;
    wire        st_xfer_end = ({1'b0, xfer_count} + {14'd0, r_cur_nwords}) >= {1'b0, r_xfer_len};
    wire        st_row_end  = mode_2d && (({1'b0, r_row_word_count} + {14'd0, r_cur_nwords}) >= {1'b0, r_row_len});
    wire [5:0]  st_next_wide_raw = {1'b0, store_wide_idx} + {3'd0, r_cur_nwords};
    wire        st_wide_wrap = (st_next_wide_raw >= 6'd16);
    wire        st_burst_end = st_xfer_end || st_row_end || non_contig_1d
                            || (st_wide_wrap && (!store_pf_valid || (addr_lane_off(st_next_ext_contig[3:2]) != 3'd0)));

    // Initial beat setup in S_STORE_REQ
    wire [2:0]  req_loff    = addr_lane_off(r_ext_addr[3:2]);
    wire [2:0]  req_max_bw  = non_contig_1d ? 3'd1 : (BUS_WORDS[2:0] - req_loff);
    wire [15:0] req_rem     = (mode_2d && (r_row_len < r_xfer_len)) ? r_row_len : r_xfer_len;
    wire [2:0]  req_nwords  = (req_rem < {13'd0, req_max_bw}) ? req_rem[2:0] : req_max_bw;
    wire [DATA_W-1:0] req_dat_comb = pack_store_beat({96'd0, sram_rdata_wide}, 5'd0, req_loff);
    wire [SEL_W-1:0]  req_sel_comb = make_write_sel(req_loff, req_nwords);

    // Next-beat setup in S_STORE when continuing burst (!st_burst_end)
    wire [2:0]  nxt_loff    = addr_lane_off(st_next_ext_contig[3:2]);
    wire [2:0]  nxt_max_bw  = BUS_WORDS[2:0] - nxt_loff;
    wire [15:0] nxt_rem_tot = r_xfer_len - (xfer_count + {13'd0, r_cur_nwords});
    wire [15:0] nxt_rem_row = r_row_len - (r_row_word_count + {13'd0, r_cur_nwords});
    wire [15:0] nxt_rem_req = (mode_2d && (nxt_rem_row < nxt_rem_tot)) ? nxt_rem_row : nxt_rem_tot;
    wire [4:0]  nxt_wide_idx = st_wide_wrap ? (st_next_wide_raw[4:0] - 5'd16) : st_next_wide_raw[4:0];
    // Note: if st_wide_wrap is true here, store_pf_valid was 1 and gets consumed, so
    // the newly promoted buffer has (16 - nxt_wide_idx) words available without pf.
    wire [5:0]  nxt_avail_wide = (!st_wide_wrap && store_pf_valid)
                               ? 6'd16
                               : (6'd16 - {1'b0, nxt_wide_idx});
    wire [2:0]  nxt_cap     = (nxt_avail_wide < {3'd0, nxt_max_bw}) ? nxt_avail_wide[2:0] : nxt_max_bw;
    wire [2:0]  nxt_nwords  = (nxt_rem_req < {13'd0, nxt_cap}) ? nxt_rem_req[2:0] : nxt_cap;
    wire [`SRAM_B_WIDTH+96-1:0] cur_wide_concat = {
        store_pf_valid ? store_pf_buf[95:0] : 96'd0,
        store_wide_buf
    };
    wire [`SRAM_B_WIDTH+96-1:0] nxt_wide_concat = st_wide_wrap
        ? {96'd0, store_pf_buf}
        : cur_wide_concat;
    wire [DATA_W-1:0] nxt_dat_comb = pack_store_beat(nxt_wide_concat, nxt_wide_idx, nxt_loff);
    wire [SEL_W-1:0]  nxt_sel_comb = make_write_sel(nxt_loff, nxt_nwords);

    assign wb_cti_o = (wb_cyc_o && wb_stb_o)
                    ? ((r_dir ? st_burst_end : ld_burst_end) ? 3'b111 : 3'b010)
                    : 3'b000;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            busy             <= 1'b0;
            done_pulse       <= 1'b0;
            xfer_count       <= 16'd0;
            wb_cyc_o         <= 1'b0;
            wb_stb_o         <= 1'b0;
            wb_we_o          <= 1'b0;
            wb_adr_o         <= {ADDR_W{1'b0}};
            wb_dat_o         <= {DATA_W{1'b0}};
            wb_sel_o         <= {SEL_W{1'b0}};
            sram_en          <= 1'b0;
            sram_we          <= {BUS_WORDS{1'b0}};
            sram_addr_o      <= {SRAM_ADDR_W{1'b0}};
            sram_wdata       <= {DATA_W{1'b0}};
            sram_wide_en     <= 1'b0;
            sram_wide_addr   <= {SRAM_ADDR_W{1'b0}};
            r_ext_addr       <= {ADDR_W{1'b0}};
            r_sram_addr      <= {SRAM_ADDR_W{1'b0}};
            r_xfer_len       <= 16'd0;
            r_dir            <= 1'b0;
            r_stride         <= {ADDR_W{1'b0}};
            r_row_len        <= 16'd0;
            r_row_count      <= 16'd0;
            r_row_word_count <= 16'd0;
            r_src_row_len    <= 16'd0;
            r_row_base       <= {ADDR_W{1'b0}};
            store_wide_buf   <= {`SRAM_B_WIDTH{1'b0}};
            store_pf_buf     <= {`SRAM_B_WIDTH{1'b0}};
            store_pf_valid   <= 1'b0;
            pf_state         <= 2'd0;
            store_wide_idx   <= 5'd0;
            r_cur_nwords     <= 3'd1;
        end else begin
            done_pulse   <= 1'b0;
            sram_en      <= 1'b0;
            sram_we      <= {BUS_WORDS{1'b0}};
            sram_wide_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    store_pf_valid <= 1'b0;
                    pf_state       <= 2'd0;
                    if (start && xfer_len != 16'd0) begin
                        busy             <= 1'b1;
                        r_ext_addr       <= ext_addr;
                        r_sram_addr      <= sram_addr;
                        r_xfer_len       <= xfer_len;
                        r_dir            <= dir;
                        r_stride         <= (cfg_row_len != 16'd0) ? cfg_out_stride : 32'd0;
                        r_row_len        <= cfg_row_len;
                        r_row_count      <= cfg_row_count;
                        r_src_row_len    <= cfg_src_row_len;
                        r_row_word_count <= 16'd0;
                        r_row_base       <= ext_addr;
                        xfer_count       <= 16'd0;
                        if (!dir)
                            state <= S_LOAD;
                        else
                            state <= S_STORE_RD;
                    end
                end

                // ─── LOAD path: MEM -> SRAM Port A ───
                S_LOAD: begin
                    if (abort) begin
                        wb_cyc_o <= 1'b0;
                        wb_stb_o <= 1'b0;
                        state    <= S_DONE;
                    end else begin
                        wb_cyc_o <= 1'b1;
                        wb_stb_o <= 1'b1;
                        wb_we_o  <= 1'b0;
                        wb_sel_o <= {SEL_W{1'b1}};
                        wb_adr_o <= align_ext_addr(r_ext_addr);
                        if (wb_cyc_o && wb_stb_o && wb_ack_i) begin
                            sram_en     <= 1'b1;
                            sram_we     <= ld_sram_we_comb;
                            sram_addr_o <= r_sram_addr;
                            sram_wdata  <= ld_wdata_comb;
                            r_sram_addr <= r_sram_addr + {{(SRAM_ADDR_W-3){1'b0}}, ld_nwords};
                            xfer_count  <= xfer_count + {13'd0, ld_nwords};
                            r_row_word_count <= r_row_word_count + {13'd0, ld_nwords};

                            if (ld_row_end) begin
                                if (effective_src_len > r_row_len)
                                    r_sram_addr <= r_sram_addr
                                                 + {{(SRAM_ADDR_W-3){1'b0}}, ld_nwords}
                                                 + (effective_src_len - r_row_len);
                                r_ext_addr       <= r_row_base + r_stride;
                                r_row_base       <= r_row_base + r_stride;
                                r_row_count      <= r_row_count - 16'd1;
                                r_row_word_count <= 16'd0;
                                wb_adr_o         <= align_ext_addr(r_row_base + r_stride);
                            end else if (non_contig_1d) begin
                                r_ext_addr <= r_ext_addr + r_stride;
                                wb_adr_o   <= align_ext_addr(r_ext_addr + r_stride);
                            end else begin
                                r_ext_addr <= r_ext_addr + {27'd0, ld_nwords, 2'b00};
                                wb_adr_o   <= align_ext_addr(r_ext_addr + {27'd0, ld_nwords, 2'b00});
                            end

                            if (ld_xfer_end) begin
                                wb_cyc_o <= 1'b0;
                                wb_stb_o <= 1'b0;
                                state    <= S_DONE;
                            end else begin
                                wb_cyc_o <= 1'b1;
                                wb_stb_o <= 1'b1;
                                state    <= S_LOAD;
                            end
                        end
                    end
                end

                // ─── STORE path: OFM Wide Port B -> MEM ───
                S_STORE_RD: begin
                    store_pf_valid <= 1'b0;
                    pf_state       <= 2'd0;
                    if (abort) begin
                        state <= S_DONE;
                    end else begin
                        sram_wide_en   <= 1'b1;
                        sram_wide_addr <= r_sram_addr;
                        state          <= S_STORE_LATCH;
                    end
                end

                S_STORE_LATCH: begin
                    if (abort) begin
                        wb_cyc_o <= 1'b0;
                        wb_stb_o <= 1'b0;
                        state    <= S_DONE;
                    end else if (sram_wide_wait) begin
                        sram_wide_en <= 1'b1;
                    end else begin
                        state <= S_STORE_REQ;
                    end
                end

                S_STORE_REQ: begin
                    if (abort) begin
                        wb_cyc_o <= 1'b0;
                        wb_stb_o <= 1'b0;
                        state    <= S_DONE;
                    end else begin
                        store_wide_buf <= sram_rdata_wide;
                        store_wide_idx <= 5'd0;
                        r_cur_nwords   <= req_nwords;
                        wb_cyc_o       <= 1'b1;
                        wb_stb_o       <= 1'b1;
                        wb_we_o        <= 1'b1;
                        wb_adr_o       <= align_ext_addr(r_ext_addr);
                        wb_dat_o       <= req_dat_comb;
                        wb_sel_o       <= req_sel_comb;
                        state          <= S_STORE;
                    end
                end

                S_STORE: begin
                    if (abort) begin
                        wb_cyc_o       <= 1'b0;
                        wb_stb_o       <= 1'b0;
                        store_pf_valid <= 1'b0;
                        pf_state       <= 2'd0;
                        state          <= S_DONE;
                    end else begin
                        // Early 1-beat-ahead 512-bit prefetch (completes in 3 cycles,
                        // sustaining 0-bubble streaming even on 128-bit 4-word/cyc buses).
                        if (pf_state == 2'd0) begin
                            if (!store_pf_valid
                                && (store_wide_idx <= 5'd12)
                                && ({1'b0, xfer_count} + {11'd0, rem_in_beat} < {1'b0, r_xfer_len})
                                && (!mode_2d || ({1'b0, r_row_word_count} + {11'd0, rem_in_beat} < {1'b0, r_row_len}))) begin
                                sram_wide_en   <= 1'b1;
                                sram_wide_addr <= r_sram_addr + {{(SRAM_ADDR_W-6){1'b0}}, rem_in_beat};
                                pf_state       <= 2'd1;
                            end
                        end else if (pf_state == 2'd1) begin
                            if (sram_wide_wait)
                                sram_wide_en <= 1'b1;
                            else
                                pf_state <= 2'd2;
                        end else begin
                            pf_state       <= 2'd0;
                            store_pf_buf   <= sram_rdata_wide;
                            store_pf_valid <= 1'b1;
                        end

                        wb_cyc_o <= 1'b1;
                        wb_stb_o <= 1'b1;
                        wb_we_o  <= 1'b1;
                        wb_adr_o <= align_ext_addr(r_ext_addr);

                        if (wb_cyc_o && wb_stb_o && wb_ack_i) begin
                            r_sram_addr      <= r_sram_addr + {{(SRAM_ADDR_W-3){1'b0}}, r_cur_nwords};
                            xfer_count       <= xfer_count + {13'd0, r_cur_nwords};
                            r_row_word_count <= r_row_word_count + {13'd0, r_cur_nwords};

                            if (st_row_end) begin
                                if (effective_src_len > r_row_len)
                                    r_sram_addr <= r_sram_addr
                                                 + {{(SRAM_ADDR_W-3){1'b0}}, r_cur_nwords}
                                                 + (effective_src_len - r_row_len);
                                r_ext_addr       <= r_row_base + r_stride;
                                r_row_base       <= r_row_base + r_stride;
                                r_row_count      <= r_row_count - 16'd1;
                                r_row_word_count <= 16'd0;
                            end else begin
                                r_ext_addr <= st_next_ext_contig;
                            end

                            if (st_xfer_end) begin
                                wb_cyc_o       <= 1'b0;
                                wb_stb_o       <= 1'b0;
                                store_pf_valid <= 1'b0;
                                pf_state       <= 2'd0;
                                state          <= S_DONE;
                            end else if (st_burst_end) begin
                                wb_cyc_o       <= 1'b0;
                                wb_stb_o       <= 1'b0;
                                store_pf_valid <= 1'b0;
                                pf_state       <= 2'd0;
                                state          <= S_STORE_RD;
                            end else begin
                                if (st_wide_wrap) begin
                                    store_wide_buf <= store_pf_buf;
                                    store_pf_valid <= 1'b0;
                                end
                                store_wide_idx <= nxt_wide_idx;
                                r_cur_nwords   <= nxt_nwords;
                                wb_cyc_o       <= 1'b1;
                                wb_stb_o       <= 1'b1;
                                wb_adr_o       <= align_ext_addr(st_next_ext_contig);
                                wb_dat_o       <= nxt_dat_comb;
                                wb_sel_o       <= nxt_sel_comb;
                                state          <= S_STORE;
                            end
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
