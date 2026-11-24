`timescale 1ns/1ps
`include "npu_defines.vh"

module npu_ctrl_base_tb;
    reg clk = 0;
    reg rst_n = 0;
    reg ctrl_start = 0;
    reg dma_done = 0;
    reg [31:0] wgt_base = 32'd37;
    reg [31:0] param_base = 32'd91;
    wire dma_start;
    wire [1:0] dma_sram_sel;
    wire [15:0] dma_sram_addr;
    wire hw_error;
    wire [3:0] hw_error_code;

    always #5 clk = ~clk;

    npu_ctrl dut (
        .clk(clk), .rst_n(rst_n),
        .ctrl_start(ctrl_start), .ctrl_abort(1'b0), .ctrl_soft_rst(1'b0),
        .ctrl_super_req(1'b0), .ctrl_super_slot_valid(1'b1),
        .hw_error(hw_error), .hw_error_code(hw_error_code),
        .dma_start(dma_start), .dma_dir(), .dma_ext_addr(),
        .dma_sram_addr(dma_sram_addr), .dma_xfer_len(),
        .dma_busy(1'b0), .dma_done(dma_done),
        .compute_start(), .compute_done(1'b0), .oc_group_done(1'b0),
        .oc_group_idx(16'd0), .wgt_k_start(16'd0), .wgt_reload_done(),
        .group_major_en(), .dma_sram_sel(dma_sram_sel), .act_role_swap(),
        .cfg_dma_in_addr(32'd0), .cfg_dma_out_addr(32'd0),
        .cfg_dma_wgt_addr(32'h1000), .cfg_dma_param_addr(32'h2000),
        .cfg_dma_in_size(32'd0), .cfg_dma_wgt_size(32'd4),
        .cfg_wgt_sram_base(wgt_base), .cfg_param_sram_base(param_base),
        .cfg_dma_wgt_per_oc(32'd0), .cfg_wgt_layout(1'b1),
        .cfg_dma_out_size(32'd4), .cfg_tile_in_size(32'd0),
        .cfg_param_count(16'd1), .cfg_dma_ctrl(32'd0),
        .cfg_dma_in_stride(32'd0), .cfg_dma_out_stride(32'd0),
        .cfg_layer_mode(32'd0), .cfg_out_base(16'd0),
        .cfg_dma_add_b_addr(32'd0), .cfg_param_preloaded(1'b0),
        .cfg_param_bank(1'b0), .cfg_next_param_en(1'b0),
        .cfg_next_param_addr(32'd0), .cfg_next_param_words(16'd0),
        .cfg_next_param_bank(1'b0), .next_param_ready(),
        .compute_param_base(),
        .cfg_dma_store_mode(32'd0), .cfg_dma_tile_out_size(32'd0),
        .cfg_out_w(16'd1), .cfg_out_c(16'd1), .cfg_out_h(16'd1),
        .cfg_in_w(16'd1), .cfg_in_h(16'd1), .cfg_in_c(16'd1),
        .cfg_stride_h(8'd1), .cfg_stride_w(8'd1),
        .cfg_kernel_h(8'd1), .cfg_kernel_w(8'd1),
        .cfg_dilation_h(8'd1), .cfg_dilation_w(8'd1),
        .cfg_pad_top(8'd0), .cfg_pad_left(8'd0), .cfg_pool_cfg(32'd0),
        .cfg_tile_h(16'd0), .cfg_tile_w(16'd0),
        .cfg_tile_num_h(16'd0), .cfg_tile_num_w(16'd0),
        .cfg_int16(1'b0), .tile_out_h_actual(16'd1),
        .tile_out_w_actual(16'd1), .dma_row_len(), .dma_row_count(),
        .dma_src_row_len(), .dma_out_stride(), .ping_pong_flag(),
        .db_prefetch_done(), .ctrl_store_phase(), .tile_done(1'b0),
        .cfg_act_bank_offset(16'd4096), .cfg_ofm_bank_offset(16'd6144),
        .ctrl_auto_next(1'b0), .cfg_layer_count(8'd0)
    );

    task pulse_dma_done;
        begin
            @(negedge clk); dma_done = 1;
            @(negedge clk); dma_done = 0;
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;

        wait (dma_start && dma_sram_sel == 2'd0);
        if (dma_sram_addr !== 16'd37)
            $fatal(1, "weight DMA base: %0d", dma_sram_addr);
        pulse_dma_done();

        wait (dma_start && dma_sram_sel == 2'd2);
        if (dma_sram_addr !== 16'd91)
            $fatal(1, "parameter DMA base: %0d", dma_sram_addr);

        // An out-of-range base must fail before any truncated DMA address is
        // issued.
        @(negedge clk); rst_n = 0;
        @(negedge clk); rst_n = 1; wgt_base = `WGT_DEPTH_W;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;
        wait (hw_error);
        if (hw_error_code !== 4'd8)
            $fatal(1, "weight base capacity guard: E%0d", hw_error_code);

        @(negedge clk); rst_n = 0;
        @(negedge clk);
        rst_n = 1;
        wgt_base = 0;
        param_base = `PARAM_DEPTH_W;
        @(negedge clk); ctrl_start = 1;
        @(negedge clk); ctrl_start = 0;
        wait (hw_error);
        if (hw_error_code !== 4'd9)
            $fatal(1, "parameter base capacity guard: E%0d", hw_error_code);

        $display("PASS: controller weight/parameter DMA bases");
        $finish;
    end
endmodule
