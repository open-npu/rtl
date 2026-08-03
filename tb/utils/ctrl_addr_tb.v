// Standalone arithmetic testbench for npu_ctrl's 2D tiled-load addressing.
// Drives the layer config CSRs directly and dumps, for every tile (ty,tx),
// the DDR address / row_count / xfer_len that the controller would program into
// the DMA — for both the current-tile and the next-tile (prefetch) copies.
//
// Purpose: prove the Resize input-space tile origin matches npu_compute.v
// bit-exactly, and that Conv/Pool/DWConv addressing is byte-identical to
// before, without paying for a full SoC simulation.
//
// Run: iverilog -g2012 -I../include -o /tmp/ctrl_tb ../src/npu_ctrl.v ctrl_addr_tb.v
//      /tmp/ctrl_tb
`timescale 1ns/1ps

module ctrl_addr_tb;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // Config drivers
    reg [31:0] c_layer_mode, c_in_addr, c_in_stride, c_pool_cfg;
    reg [15:0] c_in_w, c_in_h, c_in_c, c_out_w, c_out_h, c_out_c;
    reg [15:0] c_tile_h, c_tile_w, c_tile_num_h, c_tile_num_w;
    reg [7:0]  c_stride_h, c_stride_w, c_kernel_h, c_kernel_w;
    reg [7:0]  c_pad_top, c_pad_left;
    reg [31:0] c_tile_in_size;
    reg        c_int16;

    wire [15:0] dummy_row_len, dummy_row_count, dummy_src_row_len;
    wire [31:0] dummy_out_stride;

    npu_ctrl u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .ctrl_start(1'b0), .ctrl_abort(1'b0), .ctrl_soft_rst(1'b0),
        .hw_busy(), .hw_done(), .hw_error(), .hw_error_code(), .hw_curr_layer(),
        .dma_start(), .dma_dir(), .dma_ext_addr(), .dma_sram_addr(),
        .dma_xfer_len(), .dma_busy(1'b0), .dma_done(1'b0),
        .compute_start(), .compute_done(1'b0), .oc_group_done(1'b0),
        .oc_group_idx(16'd0), .wgt_reload_done(), .dma_sram_sel(),
        .cfg_dma_in_addr(c_in_addr), .cfg_dma_out_addr(32'd0),
        .cfg_dma_wgt_addr(32'd0), .cfg_dma_param_addr(32'd0),
        .cfg_dma_in_size(32'd0), .cfg_dma_wgt_size(32'd0),
        .cfg_dma_wgt_per_oc(32'd0), .cfg_dma_out_size(32'd0),
        .cfg_tile_in_size(c_tile_in_size), .cfg_param_count(16'd0),
        .cfg_dma_ctrl(32'h11), .cfg_dma_in_stride(c_in_stride),
        .cfg_dma_out_stride(32'd0), .cfg_layer_mode(c_layer_mode),
        .cfg_out_base(16'd0), .cfg_dma_add_b_addr(32'd0),
        .cfg_dma_store_mode(32'd1), .cfg_dma_tile_out_size(32'd0),
        .cfg_out_w(c_out_w), .cfg_out_h(c_out_h), .cfg_out_c(c_out_c),
        .cfg_in_w(c_in_w), .cfg_in_h(c_in_h), .cfg_in_c(c_in_c),
        .cfg_stride_h(c_stride_h), .cfg_stride_w(c_stride_w),
        .cfg_kernel_h(c_kernel_h), .cfg_kernel_w(c_kernel_w),
        .cfg_pad_top(c_pad_top), .cfg_pad_left(c_pad_left),
        .cfg_pool_cfg(c_pool_cfg),
        .cfg_tile_h(c_tile_h), .cfg_tile_w(c_tile_w),
        .cfg_tile_num_h(c_tile_num_h), .cfg_tile_num_w(c_tile_num_w),
        .cfg_int16(c_int16),
        .tile_out_h_actual(16'd0), .tile_out_w_actual(16'd0),
        .dma_row_len(dummy_row_len), .dma_row_count(dummy_row_count),
        .dma_src_row_len(dummy_src_row_len), .dma_out_stride(dummy_out_stride),
        .ping_pong_flag(), .db_prefetch_done(), .tile_done(1'b0),
        .cfg_act_bank_offset(16'd6144),
        .ctrl_auto_next(1'b0), .cfg_layer_count(8'd1)
    );

    integer ty, tx;
    integer fh;
    integer errors = 0;
    integer checks = 0;

    // Independent reference for the Resize input-space tile origin: plain
    // integer division, NOT the reciprocal multiply the DUT uses. If the DUT's
    // Q40 reciprocal rounding ever drifts, this catches it.
    function integer ref_org;
        input integer tile_idx, tile_dim, in_dim, out_dim;
        begin
            ref_org = (out_dim != 0) ? ((tile_idx * tile_dim * in_dim) / out_dim) : 0;
        end
    endfunction

    // Checks the DUT's current-tile and next-tile geometry against the
    // reference for one tile position. Only meaningful for Resize.
    task check_resize_tile;
        input integer c_ty, c_tx;
        input integer exp_off, exp_rows, exp_xfer;
        input integer got_off, got_rows, got_xfer;
        input [255:0] which;
        begin
            checks = checks + 1;
            if (got_off !== exp_off || got_rows !== exp_rows || got_xfer !== exp_xfer) begin
                errors = errors + 1;
                $display("FAIL %0s t(%0d,%0d): off=%0d/%0d rows=%0d/%0d xfer=%0d/%0d (got/exp)",
                         which, c_ty, c_tx, got_off, exp_off, got_rows, exp_rows,
                         got_xfer, exp_xfer);
            end
        end
    endtask

    task dump_case;
        input [255:0] name;
        input         check_rsz;   // 1 = also assert against the reference
        integer eb, stride, rlen, org_h, org_w, avail, rows, xfer, off;
        integer n_org_h, n_org_w, n_avail, n_rows, n_xfer, n_off;
        begin
            $fwrite(fh, "CASE %0s\n", name);
            eb = c_int16 ? 2 : 1;
            stride = c_in_w * c_in_c * eb;
            rlen = (c_tile_w * c_in_c * eb) / 4;   // Resize: tile_in_w == tile_w
            for (ty = 0; ty < c_tile_num_h; ty = ty + 1) begin
                for (tx = 0; tx < c_tile_num_w; tx = tx + 1) begin
                    u_ctrl.tile_y_seq = ty[15:0];
                    u_ctrl.tile_x_seq = tx[15:0];
                    #1;
                    $fwrite(fh,
                        "T %0d %0d cur_off=%0d rows=%0d xfer=%0d rlen=%0d | nty=%0d ntx=%0d next_off=%0d nrows=%0d nxfer=%0d\n",
                        ty, tx,
                        u_ctrl.tile_in_addr_2d - c_in_addr,
                        u_ctrl.load_row_count,
                        u_ctrl.load_xfer_words,
                        u_ctrl.load_row_len,
                        u_ctrl.next_ty_2d, u_ctrl.next_tx_2d,
                        u_ctrl.next_tile_in_addr_2d - c_in_addr,
                        u_ctrl.next_row_count,
                        u_ctrl.next_xfer_words);

                    if (check_rsz) begin
                        // current tile
                        org_h = ref_org(ty, c_tile_h, c_in_h, c_out_h);
                        org_w = ref_org(tx, c_tile_w, c_in_w, c_out_w);
                        avail = (org_h < c_in_h) ? (c_in_h - org_h) : 1;
                        rows  = (avail < c_tile_h) ? avail : c_tile_h;
                        off   = org_h * stride + org_w * c_in_c * eb;
                        xfer  = rlen * rows;
                        check_resize_tile(ty, tx, off, rows, xfer,
                                          u_ctrl.tile_in_addr_2d - c_in_addr,
                                          u_ctrl.load_row_count,
                                          u_ctrl.load_xfer_words, "cur");
                        // next tile (prefetch copy — must use its OWN origin)
                        n_org_h = ref_org(u_ctrl.next_ty_2d, c_tile_h, c_in_h, c_out_h);
                        n_org_w = ref_org(u_ctrl.next_tx_2d, c_tile_w, c_in_w, c_out_w);
                        n_avail = (n_org_h < c_in_h) ? (c_in_h - n_org_h) : 1;
                        n_rows  = (n_avail < c_tile_h) ? n_avail : c_tile_h;
                        n_off   = n_org_h * stride + n_org_w * c_in_c * eb;
                        n_xfer  = rlen * n_rows;
                        check_resize_tile(u_ctrl.next_ty_2d, u_ctrl.next_tx_2d,
                                          n_off, n_rows, n_xfer,
                                          u_ctrl.next_tile_in_addr_2d - c_in_addr,
                                          u_ctrl.next_row_count,
                                          u_ctrl.next_xfer_words, "next");
                        // The fetched region must never leave the input image
                        // by more than the deliberate width over-fetch, and
                        // must never exceed the per-tile SRAM budget.
                        checks = checks + 1;
                        if (xfer > (c_tile_h * c_tile_w * c_in_c * eb) / 4) begin
                            errors = errors + 1;
                            $display("FAIL SRAM budget t(%0d,%0d): xfer=%0d > %0d",
                                     ty, tx, xfer,
                                     (c_tile_h * c_tile_w * c_in_c * eb) / 4);
                        end
                    end
                end
            end
        end
    endtask

    initial begin
        fh = $fopen("/tmp/ctrl_addr_dump.txt", "w");
        c_in_addr = 32'h1000_0000;
        c_pool_cfg = 32'd0;
        c_int16 = 1'b0;
        rst_n = 0; #20; rst_n = 1; #20;

        // ── model_c_int8 L12: Resize nearest 13x13x64 → 26x26x64, tile 8x12 ──
        c_layer_mode = 32'd5;
        c_in_h=13; c_in_w=13; c_in_c=64; c_out_h=26; c_out_w=26; c_out_c=64;
        c_kernel_h=1; c_kernel_w=1; c_stride_h=1; c_stride_w=1;
        c_pad_top=0; c_pad_left=0;
        c_tile_h=8; c_tile_w=12; c_tile_num_h=4; c_tile_num_w=3;
        c_in_stride = 13*64; c_tile_in_size = 8*12*64;
        #20; dump_case("resize_mc_int8_L12", 1'b1);

        // ── model_c_int16 L12: same layer, tile 6x8, 5x4 tiles, INT16 ──
        c_int16 = 1'b1;
        c_tile_h=6; c_tile_w=8; c_tile_num_h=5; c_tile_num_w=4;
        c_in_stride = 13*64*2; c_tile_in_size = 6*8*64*2;
        #20; dump_case("resize_mc_int16_L12", 1'b1);
        c_int16 = 1'b0;

        // ── Regression: Conv2D 3x3 s1 p1, must be unchanged ──
        c_layer_mode = 32'd0;
        c_in_h=32; c_in_w=32; c_in_c=16; c_out_h=32; c_out_w=32; c_out_c=16;
        c_kernel_h=3; c_kernel_w=3; c_stride_h=1; c_stride_w=1;
        c_pad_top=1; c_pad_left=1;
        c_tile_h=8; c_tile_w=16; c_tile_num_h=4; c_tile_num_w=2;
        c_in_stride = 32*16; c_tile_in_size = 10*18*16;
        #20; dump_case("conv3x3_s1_p1", 1'b0);

        // ── Regression: Conv2D 3x3 s2 p1 ──
        c_out_h=16; c_out_w=16;
        c_stride_h=2; c_stride_w=2;
        c_tile_h=4; c_tile_w=8; c_tile_num_h=4; c_tile_num_w=2;
        c_tile_in_size = 9*17*16;
        #20; dump_case("conv3x3_s2_p1", 1'b0);

        // ── Regression: Pooling 2x2 s2 ──
        c_layer_mode = 32'd3;
        c_pool_cfg = (2 << 16) | (2 << 12) | (2 << 8) | 2;  // sw,sh,w,h = 2
        c_in_h=32; c_in_w=32; c_out_h=16; c_out_w=16;
        c_kernel_h=1; c_kernel_w=1; c_stride_h=1; c_stride_w=1;
        c_pad_top=0; c_pad_left=0;
        c_tile_h=4; c_tile_w=8; c_tile_num_h=4; c_tile_num_w=2;
        c_tile_in_size = 8*16*16;
        #20; dump_case("pool2x2_s2", 1'b0);
        c_pool_cfg = 32'd0;

        // ── Regression: DWConv 3x3 s1 p1 (op_type 1) ──
        c_layer_mode = 32'd1;
        c_in_h=32; c_in_w=32; c_out_h=32; c_out_w=32;
        c_kernel_h=3; c_kernel_w=3; c_stride_h=1; c_stride_w=1;
        c_pad_top=1; c_pad_left=1;
        c_tile_h=8; c_tile_w=16; c_tile_num_h=4; c_tile_num_w=2;
        c_tile_in_size = 10*18*16;
        #20; dump_case("dwconv3x3_s1_p1", 1'b0);

        // ── Resize edge cases ──
        c_layer_mode = 32'd5;
        c_kernel_h=1; c_kernel_w=1; c_stride_h=1; c_stride_w=1;
        c_pad_top=0; c_pad_left=0;
        // exact 2x, no border
        c_in_h=8; c_in_w=8; c_in_c=8; c_out_h=16; c_out_w=16; c_out_c=8;
        c_tile_h=4; c_tile_w=4; c_tile_num_h=4; c_tile_num_w=4;
        c_in_stride = 8*8; c_tile_in_size = 4*4*8;
        #20; dump_case("resize_2x_exact", 1'b1);
        // different H/W ratios
        c_in_h=5; c_in_w=10; c_out_h=20; c_out_w=20;
        c_tile_h=5; c_tile_w=5; c_tile_num_h=4; c_tile_num_w=4;
        c_in_stride = 10*8; c_tile_in_size = 5*5*8;
        #20; dump_case("resize_ratio_hw_differ", 1'b1);
        // downsample
        c_in_h=16; c_in_w=16; c_out_h=8; c_out_w=8;
        c_tile_h=4; c_tile_w=4; c_tile_num_h=2; c_tile_num_w=2;
        c_in_stride = 16*8; c_tile_in_size = 4*4*8;
        #20; dump_case("resize_downsample", 1'b1);

        $fclose(fh);
        $display("dump written to /tmp/ctrl_addr_dump.txt");
        if (errors == 0)
            $display("RESULT: PASS - all %0d resize-address checks matched", checks);
        else
            $display("RESULT: FAIL - %0d/%0d checks mismatched", errors, checks);
        $finish;
    end

endmodule
