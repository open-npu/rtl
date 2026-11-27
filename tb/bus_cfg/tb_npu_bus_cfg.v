// ============================================================================
// Self-checking testbench for parameterized DMA bus configurations:
//   DMA_BUS_WIDTH = 32 / 64 / 128
//   DMA_BUS_PROTO = 0 (Wishbone-B4 Master) / 1 (AXI4 Burst Master via DMA_AXI4)
//
// Verifies:
//   1. REG_HW_CONFIG [29:27] reflects DMA_BUS_WIDTH & DMA_BUS_PROTO accurately
//   2. End-to-end npu_top Conv2D + PPU inference over 32/64/128-bit WB or AXI4
//   3. Non-16B-aligned external addresses (offset +4B, +8B, +12B) for both
//      LOAD and STORE without clobbering adjacent sentinel words in DDR
//   4. 2D strided NHWC LOAD and STORE with odd row word counts & unaligned
//      row strides under random external bus wait states / backpressure
//
// SPDX-License-Identifier: Apache-2.0
// ============================================================================

`timescale 1ns / 1ps
`include "npu_defines.vh"

module tb_npu_bus_cfg;

    localparam integer BUS_W     = `DMA_BUS_WIDTH;
    localparam integer SEL_W     = BUS_W / 8;
    localparam integer BUS_WORDS = BUS_W / 32;
    localparam integer MEM_WORDS = 65536; // 256 KB external RAM in 32-bit words

    reg clk = 1'b0;
    always #2.5 clk = ~clk; // 200 MHz

    reg rst_n = 1'b0;

    // CSR Wishbone Slave (32-bit)
    reg         slv_cyc = 1'b0;
    reg         slv_stb = 1'b0;
    reg         slv_we  = 1'b0;
    reg  [11:0] slv_adr = 12'd0;
    reg  [31:0] slv_dat_w = 32'd0;
    reg  [3:0]  slv_sel = 4'hF;
    wire [31:0] slv_dat_r;
    wire        slv_ack;
    wire        irq;

    // Shared 32-bit-word external RAM model (little-endian)
    reg [31:0] ext_mem [0:MEM_WORDS-1];

    // Pseudo-random stall generator for realistic backpressure
    reg [15:0] lfsr = 16'hACE1;
    always @(posedge clk) begin
        if (!rst_n)
            lfsr <= 16'hACE1;
        else
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    end

`ifdef DMA_AXI4
    wire [31:0]      m_axi_awaddr;
    wire [7:0]       m_axi_awlen;
    wire [2:0]       m_axi_awsize;
    wire [1:0]       m_axi_awburst;
    wire             m_axi_awvalid;
    wire             m_axi_awready;
    wire [BUS_W-1:0] m_axi_wdata;
    wire [SEL_W-1:0] m_axi_wstrb;
    wire             m_axi_wlast;
    wire             m_axi_wvalid;
    wire             m_axi_wready;
    wire [1:0]       m_axi_bresp = 2'b00;
    reg              m_axi_bvalid;
    wire             m_axi_bready;
    wire [31:0]      m_axi_araddr;
    wire [7:0]       m_axi_arlen;
    wire [2:0]       m_axi_arsize;
    wire [1:0]       m_axi_arburst;
    wire             m_axi_arvalid;
    wire             m_axi_arready;
    reg  [BUS_W-1:0] m_axi_rdata;
    wire [1:0]       m_axi_rresp = 2'b00;
    reg              m_axi_rlast;
    reg              m_axi_rvalid;
    wire             m_axi_rready;

    npu_top #(
        .DMA_DATA_W(BUS_W)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .wb_slv_cyc_i  (slv_cyc),
        .wb_slv_stb_i  (slv_stb),
        .wb_slv_we_i   (slv_we),
        .wb_slv_adr_i  (slv_adr),
        .wb_slv_dat_i  (slv_dat_w),
        .wb_slv_sel_i  (slv_sel),
        .wb_slv_dat_o  (slv_dat_r),
        .wb_slv_ack_o  (slv_ack),
        .m_axi_awaddr  (m_axi_awaddr),
        .m_axi_awlen   (m_axi_awlen),
        .m_axi_awsize  (m_axi_awsize),
        .m_axi_awburst (m_axi_awburst),
        .m_axi_awvalid (m_axi_awvalid),
        .m_axi_awready (m_axi_awready),
        .m_axi_wdata   (m_axi_wdata),
        .m_axi_wstrb   (m_axi_wstrb),
        .m_axi_wlast   (m_axi_wlast),
        .m_axi_wvalid  (m_axi_wvalid),
        .m_axi_wready  (m_axi_wready),
        .m_axi_bresp   (m_axi_bresp),
        .m_axi_bvalid  (m_axi_bvalid),
        .m_axi_bready  (m_axi_bready),
        .m_axi_araddr  (m_axi_araddr),
        .m_axi_arlen   (m_axi_arlen),
        .m_axi_arsize  (m_axi_arsize),
        .m_axi_arburst (m_axi_arburst),
        .m_axi_arvalid (m_axi_arvalid),
        .m_axi_arready (m_axi_arready),
        .m_axi_rdata   (m_axi_rdata),
        .m_axi_rresp   (m_axi_rresp),
        .m_axi_rlast   (m_axi_rlast),
        .m_axi_rvalid  (m_axi_rvalid),
        .m_axi_rready  (m_axi_rready),
        .irq_o         (irq)
    );

    // AXI4 Slave memory model with randomized ready stalls & burst protocol checks
    reg        rd_active;
    reg [31:0] rd_addr;
    reg [7:0]  rd_rem;

    reg        wr_active;
    reg [31:0] wr_addr;
    reg [7:0]  wr_rem;

    assign m_axi_arready = !rd_active && lfsr[0];
    assign m_axi_awready = !wr_active && !m_axi_bvalid && lfsr[1];
    assign m_axi_wready  = wr_active && lfsr[2];

    integer ln, by;
    reg [31:0] w_idx_r, w_idx_w, old_w, new_w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_active    <= 1'b0;
            rd_addr      <= 32'd0;
            rd_rem       <= 8'd0;
            m_axi_rvalid <= 1'b0;
            m_axi_rlast  <= 1'b0;
            m_axi_rdata  <= {BUS_W{1'b0}};

            wr_active    <= 1'b0;
            wr_addr      <= 32'd0;
            wr_rem       <= 8'd0;
            m_axi_bvalid <= 1'b0;
        end else begin
            // Read channel
            if (m_axi_arvalid && m_axi_arready) begin
                if (m_axi_arburst !== 2'b01)
                    $fatal(1, "FAIL: unexpected ARBURST=%b", m_axi_arburst);
                if ((1 << m_axi_arsize) != SEL_W)
                    $fatal(1, "FAIL: unexpected ARSIZE=%0d for SEL_W=%0d", m_axi_arsize, SEL_W);
                if ((m_axi_araddr & (SEL_W - 1)) != 0)
                    $fatal(1, "FAIL: unaligned ARADDR=0x%08x for SEL_W=%0d", m_axi_araddr, SEL_W);
                rd_active <= 1'b1;
                rd_addr   <= m_axi_araddr;
                rd_rem    <= m_axi_arlen;
            end else if (rd_active) begin
                if (!m_axi_rvalid || m_axi_rready) begin
                    if (m_axi_rvalid && m_axi_rlast) begin
                        m_axi_rvalid <= 1'b0;
                        m_axi_rlast  <= 1'b0;
                        rd_active    <= 1'b0;
                    end else if (lfsr[3]) begin
                        w_idx_r = (rd_addr >> 2) & (MEM_WORDS - 1);
                        for (ln = 0; ln < BUS_WORDS; ln = ln + 1)
                            m_axi_rdata[ln*32 +: 32] <= ext_mem[(w_idx_r + ln) & (MEM_WORDS - 1)];
                        m_axi_rvalid <= 1'b1;
                        m_axi_rlast  <= (rd_rem == 8'd0);
                        rd_addr      <= rd_addr + SEL_W;
                        if (rd_rem != 8'd0)
                            rd_rem <= rd_rem - 8'd1;
                    end else begin
                        m_axi_rvalid <= 1'b0;
                        m_axi_rlast  <= 1'b0;
                    end
                end
            end

            // Write channel
            if (m_axi_bvalid && m_axi_bready)
                m_axi_bvalid <= 1'b0;

            if (m_axi_awvalid && m_axi_awready) begin
                if (m_axi_awburst !== 2'b01)
                    $fatal(1, "FAIL: unexpected AWBURST=%b", m_axi_awburst);
                if ((1 << m_axi_awsize) != SEL_W)
                    $fatal(1, "FAIL: unexpected AWSIZE=%0d for SEL_W=%0d", m_axi_awsize, SEL_W);
                if ((m_axi_awaddr & (SEL_W - 1)) != 0)
                    $fatal(1, "FAIL: unaligned AWADDR=0x%08x for SEL_W=%0d", m_axi_awaddr, SEL_W);
                wr_active <= 1'b1;
                wr_addr   <= m_axi_awaddr;
                wr_rem    <= m_axi_awlen;
            end else if (wr_active && m_axi_wvalid && m_axi_wready) begin
                if (m_axi_wlast !== (wr_rem == 8'd0))
                    $fatal(1, "FAIL: WLAST=%b mismatch at wr_rem=%0d", m_axi_wlast, wr_rem);
                w_idx_w = (wr_addr >> 2) & (MEM_WORDS - 1);
                for (ln = 0; ln < BUS_WORDS; ln = ln + 1) begin
                    old_w = ext_mem[(w_idx_w + ln) & (MEM_WORDS - 1)];
                    new_w = old_w;
                    for (by = 0; by < 4; by = by + 1) begin
                        if (m_axi_wstrb[ln*4 + by])
                            new_w[by*8 +: 8] = m_axi_wdata[(ln*4 + by)*8 +: 8];
                    end
                    ext_mem[(w_idx_w + ln) & (MEM_WORDS - 1)] <= new_w;
                end
                wr_addr <= wr_addr + SEL_W;
                if (wr_rem == 8'd0) begin
                    wr_active    <= 1'b0;
                    m_axi_bvalid <= 1'b1;
                end else begin
                    wr_rem <= wr_rem - 8'd1;
                end
            end
        end
    end

`else
    wire             wb_mst_cyc;
    wire             wb_mst_stb;
    wire             wb_mst_we;
    wire [31:0]      wb_mst_adr;
    wire [BUS_W-1:0] wb_mst_dat_w;
    wire [SEL_W-1:0] wb_mst_sel;
    wire [2:0]       wb_mst_cti;
    reg              wb_mst_ack;
    reg  [BUS_W-1:0] wb_mst_dat_r;

    npu_top #(
        .DMA_DATA_W(BUS_W)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .wb_slv_cyc_i  (slv_cyc),
        .wb_slv_stb_i  (slv_stb),
        .wb_slv_we_i   (slv_we),
        .wb_slv_adr_i  (slv_adr),
        .wb_slv_dat_i  (slv_dat_w),
        .wb_slv_sel_i  (slv_sel),
        .wb_slv_dat_o  (slv_dat_r),
        .wb_slv_ack_o  (slv_ack),
        .wb_mst_cyc_o  (wb_mst_cyc),
        .wb_mst_stb_o  (wb_mst_stb),
        .wb_mst_we_o   (wb_mst_we),
        .wb_mst_adr_o  (wb_mst_adr),
        .wb_mst_dat_o  (wb_mst_dat_w),
        .wb_mst_sel_o  (wb_mst_sel),
        .wb_mst_cti_o  (wb_mst_cti),
        .wb_mst_ack_i  (wb_mst_ack),
        .wb_mst_dat_i  (wb_mst_dat_r),
        .irq_o         (irq)
    );

    integer ln, by;
    reg [31:0] w_idx, old_w, new_w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_mst_ack   <= 1'b0;
            wb_mst_dat_r <= {BUS_W{1'b0}};
        end else begin
            wb_mst_ack <= 1'b0;
            if (wb_mst_cyc && wb_mst_stb && !wb_mst_ack && lfsr[0]) begin
                if ((wb_mst_adr & (SEL_W - 1)) != 0)
                    $fatal(1, "FAIL: unaligned wb_mst_adr=0x%08x for SEL_W=%0d", wb_mst_adr, SEL_W);
                wb_mst_ack <= 1'b1;
                w_idx = (wb_mst_adr >> 2) & (MEM_WORDS - 1);
                if (wb_mst_we) begin
                    for (ln = 0; ln < BUS_WORDS; ln = ln + 1) begin
                        old_w = ext_mem[(w_idx + ln) & (MEM_WORDS - 1)];
                        new_w = old_w;
                        for (by = 0; by < 4; by = by + 1) begin
                            if (wb_mst_sel[ln*4 + by])
                                new_w[by*8 +: 8] = wb_mst_dat_w[(ln*4 + by)*8 +: 8];
                        end
                        ext_mem[(w_idx + ln) & (MEM_WORDS - 1)] <= new_w;
                    end
                end else begin
                    for (ln = 0; ln < BUS_WORDS; ln = ln + 1)
                        wb_mst_dat_r[ln*32 +: 32] <= ext_mem[(w_idx + ln) & (MEM_WORDS - 1)];
                end
            end
        end
    end
`endif

    // CSR helper tasks (clean single-beat Wishbone handshake)
    task csr_write(input [11:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            slv_cyc   <= 1'b1;
            slv_stb   <= 1'b1;
            slv_we    <= 1'b1;
            slv_adr   <= addr;
            slv_dat_w <= data;
            @(posedge clk);
            while (!slv_ack) @(posedge clk);
            slv_cyc   <= 1'b0;
            slv_stb   <= 1'b0;
            slv_we    <= 1'b0;
            @(posedge clk);
        end
    endtask

    task csr_read(input [11:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            slv_cyc <= 1'b1;
            slv_stb <= 1'b1;
            slv_we  <= 1'b0;
            slv_adr <= addr;
            @(posedge clk);
            while (!slv_ack) @(posedge clk);
            data    = slv_dat_r;
            slv_cyc <= 1'b0;
            slv_stb <= 1'b0;
            @(posedge clk);
        end
    endtask

    task wait_npu_done;
        integer timeout;
        reg [31:0] irq_st, st;
        begin
            timeout = 0;
            irq_st = 32'd0;
            while ((irq_st & 32'h1) == 0) begin // REG_IRQ_STATUS[0] = sticky DONE
                csr_read(12'h00C, irq_st);
                csr_read(12'h004, st);
                if ((st & 32'h4) != 0)
                    $fatal(1, "FAIL: NPU reported STAT_ERROR (status=0x%08x)", st);
                timeout = timeout + 1;
                if (timeout > 20000)
                    $fatal(1, "FAIL: timeout waiting for NPU DONE (status=0x%08x irq=0x%08x)", st, irq_st);
            end
            // Clear sticky IRQ_STATUS and issue CTRL_SOFT_RST before next op
            csr_write(12'h00C, 32'h0000_000F);
            csr_write(12'h000, 32'h0000_0004);
        end
    endtask

    // Run 1x1 Conv2D (H=2, W=3, Cin=8, Cout=8) -> 6 pixels * 8B = 12 words (48B)
    // K-major tight [K=8, Cout=8] weights: w[k==oc] = 2; PPU bias = 10, mult M = 1, shift = 0
    // -> out = 2 * in + 10 for every byte!
    // Exercises S_LOAD_WGT, S_LOAD_ACT, S_LOAD_PARAM, compute + PPU, and S_STORE
    // across arbitrary 4B-aligned external addresses and optional 2D NHWC strides.
    task run_conv_test(
        input [31:0] in_byte_addr,
        input [31:0] wgt_byte_addr,
        input [31:0] prm_byte_addr,
        input [31:0] out_byte_addr,
        input [31:0] in_row_stride_bytes,   // 0 = contiguous (6 words/row = 24B/row)
        input [31:0] out_row_stride_bytes   // 0 = contiguous
    );
        integer r, c, ch, idx;
        integer in_stride_w, out_stride_w;
        reg [31:0] in_base_w, wgt_base_w, prm_base_w, out_base_w;
        reg [7:0]  v0, v1, v2, v3, exp_b;
        reg [31:0] exp_w, got_w;
        begin
            in_base_w    = in_byte_addr >> 2;
            wgt_base_w   = wgt_byte_addr >> 2;
            prm_base_w   = prm_byte_addr >> 2;
            out_base_w   = out_byte_addr >> 2;
            in_stride_w  = (in_row_stride_bytes == 0)  ? 6 : (in_row_stride_bytes >> 2);
            out_stride_w = (out_row_stride_bytes == 0) ? 6 : (out_row_stride_bytes >> 2);

            // Fill guard sentinels around input and output regions
            for (idx = -8; idx < 64; idx = idx + 1) begin
                ext_mem[in_base_w + idx]  = 32'hCAFE_0000 | (idx & 16'hFFFF);
                ext_mem[out_base_w + idx] = 32'hDEAD_0000 | (idx & 16'hFFFF);
            end

            // Populate input: 2 rows x 3 cols x 8 channels (6 words per row)
            for (r = 0; r < 2; r = r + 1) begin
                for (c = 0; c < 6; c = c + 1) begin
                    v0 = (r * 24 + c * 4 + 0 + 1);
                    v1 = (r * 24 + c * 4 + 1 + 1);
                    v2 = (r * 24 + c * 4 + 2 + 1);
                    v3 = (r * 24 + c * 4 + 3 + 1);
                    ext_mem[in_base_w + r * in_stride_w + c] = {v3, v2, v1, v0};
                end
            end

            // Populate weights: tight K-major [k=0..7, oc=0..7] (64 bytes = 16 words)
            for (idx = 0; idx < 16; idx = idx + 1)
                ext_mem[wgt_base_w + idx] = 32'd0;
            for (ch = 0; ch < 8; ch = ch + 1) begin
                idx = (ch * 8 + ch) >> 2;
                ext_mem[wgt_base_w + idx] = ext_mem[wgt_base_w + idx]
                                          | (32'd2 << (((ch * 8 + ch) & 3) * 8));
            end

            // Populate per-channel PPU params (4 words/ch * 8 ch = 32 words):
            //   w0 = {shift=0 [21:16], mult_m=1 [14:0]}
            //   w1 = {bias[15:0]=10 [31:16], zp=0 [15:0]}
            //   w2 = bias[47:16] = 0
            //   w3 = bias[63:48] = 0
            for (ch = 0; ch < 8; ch = ch + 1) begin
                ext_mem[prm_base_w + ch*4 + 0] = 32'h0000_0001;
                ext_mem[prm_base_w + ch*4 + 1] = 32'h000A_0000;
                ext_mem[prm_base_w + ch*4 + 2] = 32'd0;
                ext_mem[prm_base_w + ch*4 + 3] = 32'd0;
            end

            // Program NPU CSRs (exact register offsets from npu_defines.vh / main.c)
            csr_write(12'h040, 32'h0000_0000);       // REG_LAYER_MODE: Conv2D, INT8
            csr_write(12'h14C, 32'h0000_0001);       // REG_WGT_LAYOUT: K-major row engine
            csr_write(12'h044, {16'd2, 16'd3});      // REG_IN_DIM_HW: H=2, W=3
            csr_write(12'h048, 32'd8);               // REG_IN_DIM_C: 8
            csr_write(12'h04C, {16'd2, 16'd3});      // REG_OUT_DIM_HW: H=2, W=3
            csr_write(12'h050, 32'd8);               // REG_OUT_DIM_C: 8
            csr_write(12'h054, 32'h0000_0101);       // REG_KERNEL_SIZE: 1x1
            csr_write(12'h058, 32'h0000_0101);       // REG_STRIDE: 1x1
            csr_write(12'h05C, 32'd0);               // REG_PADDING: 0
            csr_write(12'h070, {16'd3, 16'd2});      // REG_TILE_CFG: {tile_w=3, tile_h=2}
            csr_write(12'h074, {16'd1, 16'd1});      // REG_TILE_COUNT: 1x1
            csr_write(12'h078, 32'd0);               // REG_SRAM_BASE: 0

            csr_write(12'h100, in_byte_addr);        // REG_DMA_IN_ADDR
            csr_write(12'h104, out_byte_addr);       // REG_DMA_OUT_ADDR
            csr_write(12'h108, wgt_byte_addr);       // REG_DMA_WGT_ADDR
            csr_write(12'h10C, prm_byte_addr);       // REG_DMA_PARAM_ADDR
            csr_write(12'h110, in_row_stride_bytes); // REG_DMA_IN_STRIDE
            csr_write(12'h114, out_row_stride_bytes);// REG_DMA_OUT_STRIDE
            csr_write(12'h118, 32'd0);               // REG_DMA_CTRL: load all + store out
            csr_write(12'h128, 32'd48);              // REG_DMA_IN_SIZE (48 bytes = 12 words)
            csr_write(12'h12C, 32'd64);              // REG_DMA_WGT_SIZE (64 bytes = 16 words)
            csr_write(12'h130, 32'd48);              // REG_DMA_OUT_SIZE (48 bytes = 12 words)

            csr_write(12'h180, 32'h0000_0060);       // REG_POST_CTRL: CONV_REQ (bias + zp)
            csr_write(12'h188, 32'd8);               // REG_POST_PARAM_CNT: 8 channels
            csr_write(12'h18C, 32'h0000_7F80);       // REG_POST_CLAMP: [-128, 127]

            // Launch (CTRL_START = bit 0)
            csr_write(12'h000, 32'h0000_0001);
            wait_npu_done();

            // Verify guard words BEFORE output start were NOT clobbered
            for (idx = -4; idx < 0; idx = idx + 1) begin
                got_w = ext_mem[out_base_w + idx];
                exp_w = 32'hDEAD_0000 | (idx & 16'hFFFF);
                if (got_w !== exp_w)
                    $fatal(1, "FAIL: pre-guard clobbered at out_base%0d: got=0x%08x exp=0x%08x",
                           idx, got_w, exp_w);
            end

            // Verify 2 rows x 6 words output AND inter-row / post-row sentinel words
            for (r = 0; r < 2; r = r + 1) begin
                for (c = 0; c < 6; c = c + 1) begin
                    exp_w = 32'd0;
                    for (ch = 0; ch < 4; ch = ch + 1) begin
                        exp_b = 2 * (r * 24 + c * 4 + ch + 1) + 10;
                        exp_w[ch*8 +: 8] = exp_b;
                    end
                    got_w = ext_mem[out_base_w + r * out_stride_w + c];
                    if (got_w !== exp_w)
                        $fatal(1, "FAIL: mismatch row=%0d word=%0d got=0x%08x exp=0x%08x",
                               r, c, got_w, exp_w);
                end
                // Check inter-row gap guard words when 2D strided
                for (c = 6; c < out_stride_w; c = c + 1) begin
                    idx   = r * out_stride_w + c;
                    got_w = ext_mem[out_base_w + idx];
                    exp_w = 32'hDEAD_0000 | (idx & 16'hFFFF);
                    if (got_w !== exp_w)
                        $fatal(1, "FAIL: stride gap clobbered at +%0d: got=0x%08x exp=0x%08x",
                               idx, got_w, exp_w);
                end
            end

            // Check post-output guard words
            for (idx = 2 * out_stride_w; idx < 2 * out_stride_w + 4; idx = idx + 1) begin
                got_w = ext_mem[out_base_w + idx];
                exp_w = 32'hDEAD_0000 | (idx & 16'hFFFF);
                if (got_w !== exp_w)
                    $fatal(1, "FAIL: post-guard clobbered at +%0d: got=0x%08x exp=0x%08x",
                           idx, got_w, exp_w);
            end
        end
    endtask

    // Run 2x2 spatially tiled 1x1 Conv2D over full 4x5x8 NHWC tensor (40B = 10 words/row,
    // NOT 16B-aligned!) with tile_h=2, tile_w=3 (tiles: 2x3, 2x2, 2x3, 2x2) and unaligned
    // external base addresses! Exercises true 2D NHWC strided LOAD & STORE + border clipping.
    task run_tiled_2d_conv_test(
        input [31:0] in_byte_addr,
        input [31:0] wgt_byte_addr,
        input [31:0] prm_byte_addr,
        input [31:0] out_byte_addr
    );
        integer r, c, ch, idx;
        reg [31:0] in_base_w, wgt_base_w, prm_base_w, out_base_w;
        reg [7:0]  v0, v1, v2, v3, exp_b;
        reg [31:0] exp_w, got_w;
        begin
            in_base_w  = in_byte_addr >> 2;
            wgt_base_w = wgt_byte_addr >> 2;
            prm_base_w = prm_byte_addr >> 2;
            out_base_w = out_byte_addr >> 2;

            for (idx = -8; idx < 64; idx = idx + 1) begin
                ext_mem[in_base_w + idx]  = 32'hCAFE_0000 | (idx & 16'hFFFF);
                ext_mem[out_base_w + idx] = 32'hDEAD_0000 | (idx & 16'hFFFF);
            end

            // Full 4x5x8 tensor = 4 rows x 10 words/row = 40 words (160B)
            for (r = 0; r < 4; r = r + 1) begin
                for (c = 0; c < 10; c = c + 1) begin
                    v0 = ((r * 40 + c * 4 + 0) % 50) + 1;
                    v1 = ((r * 40 + c * 4 + 1) % 50) + 1;
                    v2 = ((r * 40 + c * 4 + 2) % 50) + 1;
                    v3 = ((r * 40 + c * 4 + 3) % 50) + 1;
                    ext_mem[in_base_w + r * 10 + c] = {v3, v2, v1, v0};
                end
            end

            for (idx = 0; idx < 16; idx = idx + 1)
                ext_mem[wgt_base_w + idx] = 32'd0;
            for (ch = 0; ch < 8; ch = ch + 1) begin
                idx = (ch * 8 + ch) >> 2;
                ext_mem[wgt_base_w + idx] = ext_mem[wgt_base_w + idx]
                                          | (32'd2 << (((ch * 8 + ch) & 3) * 8));
            end

            for (ch = 0; ch < 8; ch = ch + 1) begin
                ext_mem[prm_base_w + ch*4 + 0] = 32'h0000_0001;
                ext_mem[prm_base_w + ch*4 + 1] = 32'h000A_0000;
                ext_mem[prm_base_w + ch*4 + 2] = 32'd0;
                ext_mem[prm_base_w + ch*4 + 3] = 32'd0;
            end

            csr_write(12'h040, 32'h0000_0000);       // REG_LAYER_MODE: Conv2D, INT8
            csr_write(12'h14C, 32'h0000_0001);       // REG_WGT_LAYOUT: K-major
            csr_write(12'h044, {16'd4, 16'd5});      // REG_IN_DIM_HW: H=4, W=5
            csr_write(12'h048, 32'd8);               // REG_IN_DIM_C: 8
            csr_write(12'h04C, {16'd4, 16'd5});      // REG_OUT_DIM_HW: H=4, W=5
            csr_write(12'h050, 32'd8);               // REG_OUT_DIM_C: 8
            csr_write(12'h054, 32'h0000_0101);       // REG_KERNEL_SIZE: 1x1
            csr_write(12'h058, 32'h0000_0101);       // REG_STRIDE: 1x1
            csr_write(12'h05C, 32'd0);               // REG_PADDING: 0
            csr_write(12'h070, {16'd3, 16'd2});      // REG_TILE_CFG: {tile_w=3, tile_h=2}
            csr_write(12'h074, {16'd2, 16'd2});      // REG_TILE_COUNT: 2x2 tiles
            csr_write(12'h078, 32'd0);               // REG_SRAM_BASE: 0

            csr_write(12'h100, in_byte_addr);        // REG_DMA_IN_ADDR
            csr_write(12'h104, out_byte_addr);       // REG_DMA_OUT_ADDR
            csr_write(12'h108, wgt_byte_addr);       // REG_DMA_WGT_ADDR
            csr_write(12'h10C, prm_byte_addr);       // REG_DMA_PARAM_ADDR
            csr_write(12'h110, 32'd40);              // REG_DMA_IN_STRIDE: 40B/row (2D load)
            csr_write(12'h114, 32'd40);              // REG_DMA_OUT_STRIDE: 40B/row
            csr_write(12'h118, 32'd0);               // REG_DMA_CTRL
            csr_write(12'h128, 32'd160);             // REG_DMA_IN_SIZE (160B)
            csr_write(12'h12C, 32'd64);              // REG_DMA_WGT_SIZE (64B)
            csr_write(12'h130, 32'd160);             // REG_DMA_OUT_SIZE (160B)
            csr_write(12'h134, 32'd48);              // REG_DMA_TILE_IN_SIZE (2x3x8 = 48B)
            csr_write(12'h138, 32'd48);              // REG_DMA_TILE_OUT_SIZE (2x3x8 = 48B)
            csr_write(12'h140, 32'd1);               // REG_DMA_STORE_MODE: per-tile 2D NHWC store

            csr_write(12'h180, 32'h0000_0060);       // REG_POST_CTRL: CONV_REQ
            csr_write(12'h188, 32'd8);               // REG_POST_PARAM_CNT: 8
            csr_write(12'h18C, 32'h0000_7F80);       // REG_POST_CLAMP: [-128, 127]

            csr_write(12'h000, 32'h0000_0001);
            wait_npu_done();

            // Reset tiling CSRs back to untiled defaults
            csr_write(12'h134, 32'd0);
            csr_write(12'h138, 32'd0);
            csr_write(12'h140, 32'd0);

            for (idx = -4; idx < 0; idx = idx + 1) begin
                got_w = ext_mem[out_base_w + idx];
                exp_w = 32'hDEAD_0000 | (idx & 16'hFFFF);
                if (got_w !== exp_w)
                    $fatal(1, "FAIL: tiled pre-guard clobbered at %0d: got=0x%08x exp=0x%08x",
                           idx, got_w, exp_w);
            end

            for (r = 0; r < 4; r = r + 1) begin
                for (c = 0; c < 10; c = c + 1) begin
                    exp_w = 32'd0;
                    for (ch = 0; ch < 4; ch = ch + 1) begin
                        exp_b = 2 * (((r * 40 + c * 4 + ch) % 50) + 1) + 10;
                        exp_w[ch*8 +: 8] = exp_b;
                    end
                    got_w = ext_mem[out_base_w + r * 10 + c];
                    if (got_w !== exp_w)
                        $fatal(1, "FAIL: tiled mismatch r=%0d c=%0d got=0x%08x exp=0x%08x",
                               r, c, got_w, exp_w);
                end
            end

            for (idx = 40; idx < 44; idx = idx + 1) begin
                got_w = ext_mem[out_base_w + idx];
                exp_w = 32'hDEAD_0000 | (idx & 16'hFFFF);
                if (got_w !== exp_w)
                    $fatal(1, "FAIL: tiled post-guard clobbered at +%0d: got=0x%08x exp=0x%08x",
                           idx, got_w, exp_w);
            end
        end
    endtask

    reg [31:0] hw_cfg;
    reg [1:0]  exp_dw_enc;
    reg        exp_proto;

    initial begin
        integer i;
        for (i = 0; i < MEM_WORDS; i = i + 1)
            ext_mem[i] = 32'd0;

        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        // 1. Check REG_HW_CONFIG [29:27]
        csr_read(12'h018, hw_cfg);
        exp_dw_enc = (BUS_W == 128) ? 2'd2 : (BUS_W == 64) ? 2'd1 : 2'd0;
        exp_proto  = (`DMA_BUS_PROTO != 0) ? 1'b1 : 1'b0;
        if (hw_cfg[28:27] !== exp_dw_enc || hw_cfg[29] !== exp_proto)
            $fatal(1, "FAIL: HW_CONFIG=0x%08x [29:27]=%b want {%b,%b}",
                   hw_cfg, hw_cfg[29:27], exp_proto, exp_dw_enc);

        // 2. Aligned contiguous transfer & compute
        run_conv_test(32'h0000_1000, 32'h0000_2000, 32'h0000_3000, 32'h0000_4000, 32'd0, 32'd0);

        // 3. Unaligned +4B, +8B, +12B external start addresses (tests partial first/last beat strobes)
        run_conv_test(32'h0000_5004, 32'h0000_6008, 32'h0000_700C, 32'h0000_8004, 32'd0, 32'd0);
        run_conv_test(32'h0000_900C, 32'h0000_A004, 32'h0000_B008, 32'h0000_C008, 32'd0, 32'd0);
        run_conv_test(32'h0000_D008, 32'h0000_E00C, 32'h0000_F004, 32'h0001_000C, 32'd0, 32'd0);

        // 4. 2x2 spatially tiled 2D NHWC LOAD & STORE with 40B/row (unaligned row pitch + border clip)
        run_tiled_2d_conv_test(32'h0001_2004, 32'h0001_3000, 32'h0001_4000, 32'h0001_500C);

        $display("PASS: tb_npu_bus_cfg (BUS_W=%0d, PROTO=%s, HW_CONFIG=0x%08x)",
                 BUS_W, exp_proto ? "AXI4" : "WB", hw_cfg);
        $finish(0);
    end

endmodule
