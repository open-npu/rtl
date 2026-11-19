`timescale 1ns/1ps
`include "npu_defines.vh"

module npu_ctrl_super_tb;
    reg clk = 0, rst_n = 0, ctrl_start = 0;
    reg ctrl_abort = 0, auto_dma_done = 1, auto_compute_done = 1;
    reg dma_done = 0, compute_done = 0;
    reg bad_pattern = 0, bad_overlap = 0, wide_case = 0;
    wire dma_start, dma_dir, compute_start, hw_busy, hw_done, hw_error;
    wire [1:0] dma_sram_sel, super_slot;
    wire [3:0] hw_error_code;
    wire [15:0] dma_sram_addr, ext_h, ext_w, ext_in_w;
    wire signed [16:0] ext_oh, ext_ow;
    wire single_tile, origin_en, halo_en;
    integer wgt_loads = 0, param_loads = 0, act_loads = 0;
    integer stores = 0, computes = 0, halo_computes = 0, dones = 0;

    wire [31:0] wgt_size = wide_case
        ? ((super_slot == 1) ? 32'd1008 : 32'd6272)
        : ((super_slot == 1) ? 32'd1728 : 32'd18432);
    wire [15:0] param_count = wide_case
        ? ((super_slot == 2) ? 16'd56 : 16'd112)
        : ((super_slot == 2) ? 16'd96 : 16'd192);
    wire [31:0] wgt_base = (super_slot == 0) ? 0 :
                           wide_case ? ((super_slot == 1) ? 1568 : 1820)
                                     : ((super_slot == 1) ? 4608 : 5040);
    wire [31:0] param_base = (super_slot == 0) ? 0 :
                             (super_slot == 1) ? (wide_case ? 448 : 768) :
                             bad_overlap ? 0 : (wide_case ? 896 : 1536);
    wire [15:0] in_c = (super_slot == 0) ? (wide_case ? 56 : 96)
                                         : (wide_case ? 112 : 192);
    wire [15:0] out_c = (super_slot == 2) ? (wide_case ? 56 : 96)
                                          : (wide_case ? 112 : 192);
    wire [7:0] kernel = (super_slot == 1) ? 3 : 1;

    always #5 clk = ~clk;
    always @(posedge clk) begin
        dma_done <= auto_dma_done && dma_start;
        compute_done <= auto_compute_done && compute_start;
        if (!rst_n) begin
            wgt_loads <= 0; param_loads <= 0; act_loads <= 0;
            stores <= 0; computes <= 0; halo_computes <= 0; dones <= 0;
        end else if (dma_start) begin
            if (!dma_dir && dma_sram_sel == 0) wgt_loads <= wgt_loads + 1;
            if (!dma_dir && dma_sram_sel == 2) param_loads <= param_loads + 1;
            if (!dma_dir && dma_sram_sel == 1) act_loads <= act_loads + 1;
            if ( dma_dir && dma_sram_sel == 3) stores <= stores + 1;
        end
        if (rst_n && compute_start) begin
            computes <= computes + 1;
            if (!single_tile || !origin_en)
                $fatal(1, "compute escaped external single-tile ownership");
            if (halo_en) halo_computes <= halo_computes + 1;
        end
        if (rst_n && hw_done) dones <= dones + 1;
    end

    npu_ctrl dut (
        .clk(clk), .rst_n(rst_n),
        .ctrl_start(ctrl_start), .ctrl_abort(ctrl_abort), .ctrl_soft_rst(1'b0),
        .ctrl_super_req(1'b1), .ctrl_super_slot_valid(1'b1),
        .ctrl_super_slot(2'd0), .super_slot_select(super_slot),
        .hw_busy(hw_busy), .hw_done(hw_done), .hw_error(hw_error),
        .hw_error_code(hw_error_code),
        .dma_start(dma_start), .dma_dir(dma_dir), .dma_ext_addr(),
        .dma_sram_sel(dma_sram_sel), .dma_sram_addr(dma_sram_addr),
        .dma_xfer_len(), .dma_busy(1'b0), .dma_done(dma_done),
        .compute_start(compute_start), .compute_done(compute_done),
        .compute_single_tile_en(single_tile),
        .compute_external_origin_en(origin_en),
        .compute_external_tile_y(), .compute_external_tile_x(),
        .compute_external_tile_h(ext_h), .compute_external_tile_w(ext_w),
        .compute_external_origin_h(ext_oh), .compute_external_origin_w(ext_ow),
        .compute_external_tile_in_w(ext_in_w), .compute_halo_expand_en(halo_en),
        .oc_group_done(1'b0), .oc_group_idx(16'd0), .wgt_k_start(16'd0),
        .wgt_reload_done(), .group_major_en(), .act_role_swap(),
        .cfg_dma_in_addr(32'h1000_0000), .cfg_dma_out_addr(32'h2000_0000),
        .cfg_dma_wgt_addr(32'h3000_0000 + {super_slot, 20'd0}),
        .cfg_dma_param_addr(32'h4000_0000 + {super_slot, 20'd0}),
        .cfg_dma_in_size(wide_case ? 32'd175616 : 32'd18816),
        .cfg_dma_wgt_size(wgt_size),
        .cfg_wgt_sram_base(wgt_base), .cfg_param_sram_base(param_base),
        .cfg_dma_wgt_per_oc(32'd0), .cfg_wgt_layout(1'b1),
        .cfg_dma_out_size(32'd18816), .cfg_tile_in_size(32'd0),
        .cfg_param_count(param_count), .cfg_dma_ctrl(32'd0),
        .cfg_dma_in_stride(wide_case ? 32'd3136 : 32'd1344),
        .cfg_dma_out_stride(32'd0),
        .cfg_layer_mode((super_slot == 1) ? 32'd1 : 32'd0),
        .cfg_out_base(16'd0), .cfg_dma_add_b_addr(32'd0),
        .cfg_param_preloaded(1'b0), .cfg_param_bank(1'b0),
        .cfg_next_param_en(1'b0), .cfg_next_param_addr(32'd0),
        .cfg_next_param_words(16'd0), .cfg_next_param_bank(1'b0),
        .next_param_ready(), .compute_param_base(),
        .cfg_dma_store_mode(32'd0), .cfg_dma_tile_out_size(32'd0),
        .cfg_out_w(wide_case ? 16'd56 : 16'd14), .cfg_out_c(out_c),
        .cfg_out_h(wide_case ? 16'd56 : 16'd14),
        .cfg_in_w(wide_case ? 16'd56 : 16'd14),
        .cfg_in_h(wide_case ? 16'd56 : 16'd14), .cfg_in_c(in_c),
        .cfg_stride_h(8'd1), .cfg_stride_w(8'd1),
        .cfg_kernel_h(kernel), .cfg_kernel_w(kernel),
        .cfg_pad_top((super_slot == 1) ? 8'd1 : 8'd0),
        .cfg_pad_left((super_slot == 1) ? 8'd1 : 8'd0),
        .cfg_pool_cfg(32'd0), .cfg_tile_h(wide_case ? 16'd28 : 16'd14),
        .cfg_tile_w(bad_pattern ? 16'd11 : (wide_case ? 16'd10 : 16'd12)),
        .cfg_tile_num_h(wide_case ? 16'd2 : 16'd1),
        .cfg_tile_num_w(wide_case ? 16'd6 : 16'd2),
        .cfg_int16(1'b0), .tile_out_h_actual(16'd14),
        .tile_out_w_actual(16'd12), .dma_row_len(), .dma_row_count(),
        .dma_src_row_len(), .dma_out_stride(), .ping_pong_flag(),
        .db_prefetch_done(), .ctrl_store_phase(), .tile_done(1'b0),
        .cfg_act_bank_offset(16'd4096), .cfg_ofm_bank_offset(16'd6144),
        .ctrl_auto_next(1'b0), .cfg_layer_count(8'd0)
    );

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;
        fork
            begin wait (hw_done); @(posedge clk); #1; end
            begin repeat (20000) @(negedge clk); $fatal(1, "super timeout"); end
        join_any
        disable fork;
        if (hw_error) $fatal(1, "unexpected E%0d", hw_error_code);
        if (wgt_loads != 3 || param_loads != 3)
            $fatal(1, "resident preload count w=%0d p=%0d", wgt_loads, param_loads);
        if (act_loads != 2 || computes != 6 || halo_computes != 2
                || stores != 2 || dones != 1)
            $fatal(1, "tile schedule a=%0d c=%0d h=%0d s=%0d d=%0d",
                   act_loads, computes, halo_computes, stores, dones);

        // 56x56 exercises both generic x rollover and the second tile row.
        @(negedge clk); rst_n = 0;
        @(negedge clk); rst_n = 1; wide_case = 1;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;
        fork
            begin wait (hw_done); @(posedge clk); #1; end
            begin repeat (80000) @(negedge clk); $fatal(1, "56x56 timeout"); end
        join_any
        disable fork;
        if (hw_error || act_loads != 12 || computes != 36
                || halo_computes != 12 || stores != 12 || dones != 1)
            $fatal(1, "56x56 schedule a=%0d c=%0d h=%0d s=%0d d=%0d E%0d",
                   act_loads, computes, halo_computes, stores, dones,
                   hw_error_code);

        @(negedge clk); rst_n = 0;
        @(negedge clk); rst_n = 1; wide_case = 0; bad_pattern = 1;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;
        wait (hw_error);
        if (hw_error_code != 4'd13)
            $fatal(1, "strict pattern guard E%0d", hw_error_code);

        @(negedge clk); rst_n = 0;
        @(negedge clk); rst_n = 1; bad_pattern = 0; bad_overlap = 1;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;
        wait (hw_error);
        if (hw_error_code != 4'd14)
            $fatal(1, "resident overlap guard E%0d", hw_error_code);

        // ABORT while a preload DMA completion is deliberately withheld.
        @(negedge clk); rst_n = 0;
        @(negedge clk);
        rst_n = 1; bad_overlap = 0; auto_dma_done = 0;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;
        wait (dut.state == 6'd34); // S_SUPER_WAIT_WGT
        @(negedge clk); ctrl_abort = 1;
        @(negedge clk); ctrl_abort = 0;
        fork
            begin wait (hw_done); wait (!hw_busy && dut.state == 6'd0); end
            begin repeat (20) @(negedge clk); $fatal(1, "DMA-wait ABORT hung"); end
        join_any
        disable fork;
        if (single_tile || origin_en || halo_en)
            $fatal(1, "DMA-wait ABORT left external compute mode active");

        // ABORT while compute_done is deliberately withheld.
        @(negedge clk); rst_n = 0;
        @(negedge clk);
        rst_n = 1; auto_dma_done = 1; auto_compute_done = 0;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;
        wait (dut.state == 6'd42); // S_SUPER_WAIT_COMP
        @(negedge clk); ctrl_abort = 1;
        @(negedge clk); ctrl_abort = 0;
        fork
            begin wait (hw_done); wait (!hw_busy && dut.state == 6'd0); end
            begin repeat (20) @(negedge clk); $fatal(1, "compute-wait ABORT hung"); end
        join_any
        disable fork;
        if (single_tile || origin_en || halo_en)
            $fatal(1, "compute-wait ABORT left external compute mode active");

        $display("PASS: strict super preload/tile schedule and one DONE");
        $finish;
    end
endmodule
