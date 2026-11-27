// Open-NPU RTL — Wishbone-B4 Burst to AXI4 INCR Burst Master Bridge
// SPDX-License-Identifier: Apache-2.0
//
// Lightweight, fully registered AXI4 Master adapter for NPU DMA:
//   - Supports parameterized DATA_W in {32, 64, 128} bits
//   - Multi-beat AXI4 INCR reads (up to 16 beats, 4KB-boundary safe) using
//     wb_burst_beats_i exported by npu_dma
//   - Multi-beat AXI4 INCR writes collected into a 16-beat synchronous buffer
//     so AWLEN is always exact and WDATA streams without bubbles
//   - Strictly registered AXI4 VALID outputs (zero combinational ready->valid paths)

`include "npu_defines.vh"

module npu_wb_to_axi4 #(
    parameter ADDR_W = 32,
    parameter DATA_W = `DMA_BUS_WIDTH
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // ─── Internal Wishbone-B4 Burst Slave (from NPU DMA arbiter) ───
    input  wire                  wb_cyc_i,
    input  wire                  wb_stb_i,
    input  wire                  wb_we_i,
    input  wire [ADDR_W-1:0]     wb_adr_i,
    input  wire [DATA_W-1:0]     wb_dat_i,
    input  wire [(DATA_W/8)-1:0] wb_sel_i,
    input  wire [2:0]            wb_cti_i,
    input  wire [4:0]            wb_burst_beats_i,
    output wire                  wb_ack_o,
    output wire [DATA_W-1:0]     wb_dat_o,

    // ─── External AXI4 Master Interface ───
    output reg  [ADDR_W-1:0]     m_axi_awaddr,
    output reg  [7:0]            m_axi_awlen,
    output wire [2:0]            m_axi_awsize,
    output wire [1:0]            m_axi_awburst,
    output reg                   m_axi_awvalid,
    input  wire                  m_axi_awready,

    output reg  [DATA_W-1:0]     m_axi_wdata,
    output reg  [(DATA_W/8)-1:0] m_axi_wstrb,
    output reg                   m_axi_wlast,
    output reg                   m_axi_wvalid,
    input  wire                  m_axi_wready,

    input  wire [1:0]            m_axi_bresp,
    input  wire                  m_axi_bvalid,
    output reg                   m_axi_bready,

    output reg  [ADDR_W-1:0]     m_axi_araddr,
    output reg  [7:0]            m_axi_arlen,
    output wire [2:0]            m_axi_arsize,
    output wire [1:0]            m_axi_arburst,
    output reg                   m_axi_arvalid,
    input  wire                  m_axi_arready,

    input  wire [DATA_W-1:0]     m_axi_rdata,
    input  wire [1:0]            m_axi_rresp,
    input  wire                  m_axi_rlast,
    input  wire                  m_axi_rvalid,
    output wire                  m_axi_rready
);

    localparam SEL_W     = DATA_W / 8;
    localparam BUS_BYTES = DATA_W / 8;
    localparam [2:0] AXSIZE_VAL = (DATA_W == 128) ? 3'd4 :
                                  (DATA_W == 64)  ? 3'd3 : 3'd2;

    wire _unused_ok = &{1'b0, m_axi_bresp, m_axi_rresp};

    assign m_axi_awsize  = AXSIZE_VAL;
    assign m_axi_awburst = 2'b01; // INCR
    assign m_axi_arsize  = AXSIZE_VAL;
    assign m_axi_arburst = 2'b01; // INCR

    // ─── Read FSM ───
    localparam RD_IDLE = 2'd0;
    localparam RD_AR   = 2'd1;
    localparam RD_DATA = 2'd2;

    reg [1:0] rd_state;
    reg [4:0] rd_rem_beats;

    assign m_axi_rready = (rd_state == RD_DATA);
    wire   wb_ack_rd    = (rd_state == RD_DATA) && m_axi_rvalid
                        && wb_cyc_i && wb_stb_i && !wb_we_i;
    assign wb_dat_o     = m_axi_rdata;

    wire [4:0] req_rd_beats = (wb_burst_beats_i == 5'd0) ? 5'd1 : wb_burst_beats_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state      <= RD_IDLE;
            rd_rem_beats  <= 5'd0;
            m_axi_araddr  <= {ADDR_W{1'b0}};
            m_axi_arlen   <= 8'd0;
            m_axi_arvalid <= 1'b0;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    if (wb_cyc_i && wb_stb_i && !wb_we_i) begin
                        m_axi_araddr  <= wb_adr_i;
                        m_axi_arlen   <= {3'd0, req_rd_beats - 5'd1};
                        m_axi_arvalid <= 1'b1;
                        rd_rem_beats  <= req_rd_beats;
                        rd_state      <= RD_AR;
                    end
                end

                RD_AR: begin
                    if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        rd_state      <= RD_DATA;
                    end
                end

                RD_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rlast || (rd_rem_beats <= 5'd1))
                            rd_state <= RD_IDLE;
                        else
                            rd_rem_beats <= rd_rem_beats - 5'd1;
                    end
                end

                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    // ─── Write FSM + 16-beat synchronous burst buffer ───
    // Non-terminal beats of a burst are accepted into wbuf at 1 beat/cycle.
    // The terminal beat (CTI=111, 16th beat, or 4KB boundary) withholds wb_ack_o
    // until AXI4 BVALID/BREADY completes, guaranteeing strict write-completion
    // ordering before NPU DMA asserts done_pulse / IRQ!
    localparam WR_COLLECT = 2'd0;
    localparam WR_PREP    = 2'd1;
    localparam WR_SEND    = 2'd2;
    localparam WR_RESP    = 2'd3;

    reg [1:0]        wr_state;
    reg [3:0]        wr_cnt;
    reg [3:0]        wr_total;
    reg [3:0]        wr_send_idx;
    reg [ADDR_W-1:0] wr_base_addr;
    reg [DATA_W-1:0] wbuf_dat  [0:15];
    reg [SEL_W-1:0]  wbuf_strb [0:15];

    wire wr_req_valid   = wb_cyc_i && wb_stb_i && wb_we_i;
    wire wr_4k_end      = ((wb_adr_i[11:0] + BUS_BYTES[11:0]) == 12'h000);
    wire wr_end_collect = (wb_cti_i == 3'b111) || (wr_cnt == 4'd15) || wr_4k_end;

    wire wb_ack_wr = ((wr_state == WR_COLLECT) && wr_req_valid && !wr_end_collect)
                  || ((wr_state == WR_RESP) && m_axi_bvalid && m_axi_bready);

    wire aw_done_now = (!m_axi_awvalid) || m_axi_awready;
    wire w_done_now  = (!m_axi_wvalid)  || (m_axi_wready && (wr_send_idx == wr_total));

    assign wb_ack_o = wb_we_i ? wb_ack_wr : wb_ack_rd;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state      <= WR_COLLECT;
            wr_cnt        <= 4'd0;
            wr_total      <= 4'd0;
            wr_send_idx   <= 4'd0;
            wr_base_addr  <= {ADDR_W{1'b0}};
            m_axi_awaddr  <= {ADDR_W{1'b0}};
            m_axi_awlen   <= 8'd0;
            m_axi_awvalid <= 1'b0;
            m_axi_wdata   <= {DATA_W{1'b0}};
            m_axi_wstrb   <= {SEL_W{1'b0}};
            m_axi_wlast   <= 1'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_bready  <= 1'b0;
        end else begin
            case (wr_state)
                WR_COLLECT: begin
                    if (wr_req_valid) begin
                        wbuf_dat[wr_cnt]  <= wb_dat_i;
                        wbuf_strb[wr_cnt] <= wb_sel_i;
                        if (wr_cnt == 4'd0)
                            wr_base_addr <= wb_adr_i;
                        if (wr_end_collect) begin
                            m_axi_awaddr  <= (wr_cnt == 4'd0) ? wb_adr_i : wr_base_addr;
                            m_axi_awlen   <= {4'd0, wr_cnt};
                            m_axi_awvalid <= 1'b1;
                            wr_total      <= wr_cnt;
                            wr_send_idx   <= 4'd0;
                            wr_state      <= WR_PREP;
                        end else begin
                            wr_cnt <= wr_cnt + 4'd1;
                        end
                    end
                end

                WR_PREP: begin
                    if (m_axi_awvalid && m_axi_awready)
                        m_axi_awvalid <= 1'b0;
                    m_axi_wvalid <= 1'b1;
                    m_axi_wdata  <= wbuf_dat[0];
                    m_axi_wstrb  <= wbuf_strb[0];
                    m_axi_wlast  <= (wr_total == 4'd0);
                    wr_send_idx  <= 4'd0;
                    wr_state     <= WR_SEND;
                end

                WR_SEND: begin
                    if (m_axi_awvalid && m_axi_awready)
                        m_axi_awvalid <= 1'b0;
                    if (m_axi_wvalid && m_axi_wready) begin
                        if (wr_send_idx == wr_total) begin
                            m_axi_wvalid <= 1'b0;
                            m_axi_wlast  <= 1'b0;
                        end else begin
                            wr_send_idx <= wr_send_idx + 4'd1;
                            m_axi_wdata <= wbuf_dat[wr_send_idx + 4'd1];
                            m_axi_wstrb <= wbuf_strb[wr_send_idx + 4'd1];
                            m_axi_wlast <= ((wr_send_idx + 4'd1) == wr_total);
                        end
                    end
                    if (aw_done_now && w_done_now) begin
                        m_axi_bready <= 1'b1;
                        wr_state     <= WR_RESP;
                    end
                end

                WR_RESP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 1'b0;
                        wr_cnt       <= 4'd0;
                        wr_state     <= WR_COLLECT;
                    end
                end

                default: wr_state <= WR_COLLECT;
            endcase
        end
    end

endmodule
