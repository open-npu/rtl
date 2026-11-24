`timescale 1ns/1ps
`include "npu_defines.vh"

module npu_compute_single_tile_tb;
    localparam WGT_AW = $clog2(`WGT_DEPTH_W);
    localparam PARAM_AW = $clog2(`PARAM_DEPTH_W);
    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    reg [7:0] op_type = 8'd6;
    reg wgt_layout = 0;
    reg single_tile_en = 1;
    reg [31:0] pool_cfg = 32'h0001_1110;
    wire done, tile_done;
    wire [15:0] actual_h, actual_w;
    wire wgt_rd_en, param_rd_en;
    wire [WGT_AW-1:0] wgt_rd_addr;
    wire [PARAM_AW-1:0] param_rd_addr;
    wire ppu_in_valid;
    wire [`MAC_LANES-1:0] ppu_valid_w;
    wire act_wr_en;
    wire [`SRAM_B_WIDTH-1:0] act_wr_wide;
    reg saw_tile_done = 0;
    reg saw_wgt_read = 0;
    reg saw_param_read = 0;

    always #5 clk = ~clk;
    always @(posedge clk)
        if (tile_done) saw_tile_done <= 1;
    always @(posedge clk) begin
        if (wgt_rd_en) begin
            saw_wgt_read <= 1;
            if (wgt_rd_addr < WGT_AW'(37))
                $fatal(1, "weight read escaped descriptor base");
        end
        if (param_rd_en) begin
            saw_param_read <= 1;
            if (param_rd_addr < PARAM_AW'(91))
                $fatal(1, "parameter read escaped descriptor base");
        end
    end

    npu_compute #(
        .WGT_ADDR_W(WGT_AW),
        .PARAM_ADDR_W(PARAM_AW)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .tile_done(tile_done), .oc_group_done(), .oc_group_out(),
        .wgt_k_start(), .wgt_reload_done(1'b1), .cfg_wgt_per_oc(32'd0),
        .cfg_wgt_layout(wgt_layout), .cfg_group_major(1'b0),
        .db_prefetch_done(1'b1),
        .cfg_single_tile_en(single_tile_en),
        .cfg_external_tile_y(16'd1), .cfg_external_tile_x(16'd2),
        .cfg_external_tile_h_actual(16'd3),
        .cfg_external_tile_w_actual(16'd2),
        .cfg_external_origin_en(1'b0),
        .cfg_external_origin_h(17'sd0), .cfg_external_origin_w(17'sd0),
        .cfg_external_tile_in_w(16'd0), .cfg_halo_expand_en(1'b0),
        .tile_out_h_actual(actual_h), .tile_out_w_actual(actual_w),
        // Unsupported deconv exits after setup, which isolates this test to
        // START latching, geometry, base ownership and completion semantics.
        .cfg_op_type(op_type), .cfg_int16(1'b0), .cfg_in_c(16'd1),
        .cfg_out_h(16'd8), .cfg_out_w(16'd12), .cfg_out_c(16'd1),
        .cfg_kernel_h(8'd1), .cfg_kernel_w(8'd1),
        .cfg_dilation_h(8'd1), .cfg_dilation_w(8'd1),
        .cfg_stride_h(8'd1), .cfg_stride_w(8'd1),
        .cfg_pad_top(8'd0), .cfg_pad_left(8'd0),
        .cfg_tile_h(16'd4), .cfg_tile_w(16'd4),
        .cfg_tile_num_h(16'd2), .cfg_tile_num_w(16'd3),
        .cfg_in_w(16'd12), .cfg_in_h(16'd8), .cfg_in_zp(16'sd0),
        .cfg_act_base(13'd0), .cfg_out_base(13'd0), .cfg_ofm_base(13'd0),
        .cfg_wgt_base(WGT_AW'(37)), .cfg_param_base(PARAM_AW'(91)),
        .cfg_pool_cfg(pool_cfg), .cfg_resize_cfg(32'd0),
        .cfg_deconv_cfg(32'd0), .cfg_concat_cfg(32'd0), .cfg_2d_load(1'b0),
        .wgt_rd_en(wgt_rd_en), .wgt_rd_addr(wgt_rd_addr),
        .wgt_rd_data({`SRAM_B_WIDTH{1'b0}}),
        .act_rd_en(), .act_rd_addr(), .act_rd_data({`SRAM_B_WIDTH{1'b0}}),
        .act_rd_ofm(), .act_wr_en(act_wr_en), .act_wr_addr(), .act_wr_data(),
        .act_wr_mask(), .act_wr_wide(act_wr_wide),
        .param_rd_en(param_rd_en), .param_rd_addr(param_rd_addr),
        .param_rd_data(32'd0),
        .mac_fire(), .mac_lanes(),
        .dw_wgt_load(), .dw_wgt_valid(), .dw_wgt_data(),
        .dw_in_valid(), .dw_in_data(), .dw_acc_clear(),
        .dw_acc_out({`ACC_WIDTH{1'b0}}), .dw_out_valid(1'b0),
        .dw_wgt_valid_w(), .dw_wgt_data_w(), .dw_in_valid_w(),
        .dw_in_data_w(), .dw_acc_w({(`ACC_WIDTH*`ARRAY_SIZE){1'b0}}),
        .dw_out_valid_w({`ARRAY_SIZE{1'b0}}),
        .ppu_acc_in(), .ppu_in_valid(ppu_in_valid), .ppu_bias(), .ppu_mult_m(),
        .ppu_shift_s(), .ppu_zero_point(), .ppu_out_data({`DATA_WIDTH{1'b0}}),
        .ppu_out_valid(ppu_in_valid), .ppu_acc_w(), .ppu_valid_w(ppu_valid_w),
        .ppu_bias_w(),
        .ppu_mult_w(), .ppu_shift_w(), .ppu_zp_w(),
        .ppu_out_w({(`DATA_WIDTH*`MAC_LANES){1'b0}}),
        .ppu_vout_w(ppu_valid_w), .dbg_state()
    );

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait (done);

        if (actual_h !== 16'd3 || actual_w !== 16'd2)
            $fatal(1, "external actual geometry");
        if (dut.tile_y !== 16'd1 || dut.tile_x !== 16'd2
            || dut.tile_oh_origin !== 16'd4 || dut.tile_ow_origin !== 16'd8)
            $fatal(1, "external tile coordinates/origin");
        if (dut.wgt_base !== WGT_AW'(37)
            || dut.param_base !== PARAM_AW'(91))
            $fatal(1, "compute SRAM bases");
        if (saw_tile_done)
            $fatal(1, "single-tile mode emitted tile_done");

        // Run a supported 1x1 row-convolution far enough to observe both
        // physical SRAM read ports using the non-zero descriptor bases.
        @(negedge clk); rst_n = 0;
        @(negedge clk);
        rst_n = 1;
        op_type = 8'd0;
        wgt_layout = 1;
        single_tile_en = 1;
        saw_wgt_read = 0;
        saw_param_read = 0;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        fork
            begin
                wait (saw_wgt_read && saw_param_read);
            end
            begin
                repeat (200) @(negedge clk);
                $fatal(1, "timed out waiting for based SRAM reads");
            end
        join_any
        disable fork;

        // A real pooling datapath reaches S_TILE_NEXT.  The external mode
        // must turn that boundary into DONE without traversing tile (1,0).
        @(negedge clk); rst_n = 0;
        @(negedge clk);
        rst_n = 1;
        op_type = 8'd3;
        wgt_layout = 0;
        saw_tile_done = 0;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        fork
            begin
                wait (done);
            end
            begin
                repeat (2000) @(negedge clk);
                $fatal(1, "single-tile pooling did not complete");
            end
        join_any
        disable fork;
        if (saw_tile_done || dut.tile_y !== 16'd1 || dut.tile_x !== 16'd2)
            $fatal(1, "compute traversed beyond the external tile");

        // Isolate the row-output capture to prove halo fill comes from each
        // output channel's quantized zero-point, rather than arithmetic zero.
        @(negedge clk); rst_n = 0;
        @(negedge clk); rst_n = 1;
        dut.param_cache[1]  = 32'h0000_0011;
        dut.param_cache[5]  = 32'h0000_0022;
        dut.param_cache[9]  = 32'h0000_0033;
        dut.param_cache[13] = 32'h0000_0044;
        force dut.row_ppu = 1'b1;
        force dut.row_drain = 1'b0;
        force dut.halo_expand_active = 1'b1;
        force dut.external_origin_h = -17'sd1;
        force dut.external_origin_w = 17'sd0;
        force dut.ppu_oh = 16'd0;
        force dut.ppu_ow = 16'd0;
        force dut.out_tile_w = 16'd2;
        force dut.oc_group = 16'd0;
        force dut.row_group_ch = 7'd64;
        force dut.ppu_nlane = 9'd4;
        force dut.ppu_npix = 5'd1;
        force dut.ppu_vout_w = {{(`MAC_LANES-4){1'b0}}, 4'hf};
        @(posedge clk);
        #1;
        if (!act_wr_en || act_wr_wide[31:0] !== 32'h4433_2211)
            $fatal(1, "halo fill did not use per-channel output zero-points");
        release dut.row_ppu;
        release dut.row_drain;
        release dut.halo_expand_active;
        release dut.external_origin_h;
        release dut.external_origin_w;
        release dut.ppu_oh;
        release dut.ppu_ow;
        release dut.out_tile_w;
        release dut.oc_group;
        release dut.row_group_ch;
        release dut.ppu_nlane;
        release dut.ppu_npix;
        release dut.ppu_vout_w;

        $display("PASS: compute single external tile and all local bases");
        $finish;
    end
endmodule
