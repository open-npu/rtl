`timescale 1ns/1ps
`include "npu_defines.vh"

module npu_csr_super_tb;
    reg clk = 0;
    reg rst_n = 0;
    reg cyc = 0, stb = 0, we = 0;
    reg [11:0] adr = 0;
    reg [31:0] wdata = 0;
    wire [31:0] rdata;
    wire ack;
    reg hw_done = 0, hw_error = 0;
    wire ctrl_start, super_req, super_valid;
    wire [1:0] super_slot;
    wire [31:0] layer_mode, wgt_base, param_base;

    always #5 clk = ~clk;

    npu_csr dut (
        .clk(clk), .rst_n(rst_n),
        .wb_cyc_i(cyc), .wb_stb_i(stb), .wb_we_i(we),
        .wb_adr_i(adr), .wb_dat_i(wdata), .wb_sel_i(4'hf),
        .wb_dat_o(rdata), .wb_ack_o(ack),
        .hw_busy(1'b0), .hw_dma_busy(1'b0), .hw_done(hw_done),
        .hw_error(hw_error), .hw_dma_done(1'b0), .hw_error_code(4'd0),
        .hw_curr_layer(8'd0), .hw_perf_cnt(32'd0), .hw_mac_cnt(32'd0),
        .hw_perf_dma_cnt(32'd0), .hw_perf_compute_cnt(32'd0),
        .hw_perf_overlap_cnt(32'd0), .hw_perf_st_mac_cnt(32'd0),
        .hw_perf_st_wait_cnt(32'd0), .hw_perf_st_wb_cnt(32'd0),
        .hw_perf_st_pool_cnt(32'd0), .hw_perf_st_add_cnt(32'd0),
        .hw_perf_st_rsz_cnt(32'd0), .hw_perf_st_cfg_cnt(32'd0),
        .hw_perf_st_oth_cnt(32'd0), .hw_perf_wait_act_cnt(32'd0),
        .hw_perf_wait_wgt_cnt(32'd0), .hw_perf_wait_db_cnt(32'd0),
        .hw_perf_mac_idle_cnt(32'd0), .hw_perf_wait_store_cnt(32'd0),
        .hw_next_param_ready(1'b0),
        .hw_super_slot_en(1'b0), .hw_super_slot(2'd0),
        .ctrl_start(ctrl_start), .ctrl_super_req(super_req),
        .ctrl_super_slot(super_slot), .ctrl_super_slot_valid(super_valid),
        .reg_layer_mode(layer_mode), .reg_wgt_sram_base(wgt_base),
        .reg_param_sram_base(param_base)
    );

    task wb_write(input [11:0] a, input [31:0] d);
        begin
            @(negedge clk); adr = a; wdata = d; we = 1; cyc = 1; stb = 1;
            @(negedge clk); cyc = 0; stb = 0; we = 0;
            @(negedge clk);
        end
    endtask

    // START, one mandatory Wishbone ACK turnaround cycle, then the earliest
    // legal write an adversarial master can issue.
    task start_then_attack(input [11:0] a, input [31:0] d);
        begin
            @(negedge clk); adr = 12'h000; wdata = 32'h1;
            we = 1; cyc = 1; stb = 1;
            @(negedge clk); cyc = 0; stb = 0; we = 0;
            @(negedge clk); adr = a; wdata = d; we = 1; cyc = 1; stb = 1;
            @(negedge clk); cyc = 0; stb = 0; we = 0;
            @(negedge clk);
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1;

        // Legacy path is unchanged while SUPER_CTRL.request is clear.
        wb_write(12'h040, 32'h1111_0000);
        if (layer_mode !== 32'h1111_0000) $fatal(1, "legacy mux changed");

        // Program slot 1 through the descriptor window, including SRAM bases.
        wb_write(12'h160, 32'h0000_0100); // window slot=1, request=0
        wb_write(12'h400, 32'h2222_0001); // descriptor word 0
        wb_write(12'h4A8, 32'h0000_0123); // word 42: WGT base
        wb_write(12'h4AC, 32'h0000_0456); // word 43: PARAM base
        wb_write(12'h160, 32'h0000_0103); // request=1, START slot=1
        if (!super_req || !super_valid || super_slot != 1) $fatal(1, "slot select");
        if (layer_mode !== 32'h2222_0001) $fatal(1, "slot mux");
        if (wgt_base !== 32'h123 || param_base !== 32'h456) $fatal(1, "base mux");

        // Freeze is established on the START acceptance edge. Attack the
        // first legal following bus slot: SUPER_CTRL must not disable the
        // request or switch back to the legacy descriptor.
        start_then_attack(12'h160, 32'h0000_0001);
        if (super_slot != 1 || layer_mode !== 32'h2222_0001)
            $fatal(1, "START-adjacent SUPER_CTRL polluted active selection");

        // Release and repeat with the descriptor window as the first attack.
        @(negedge clk); hw_error = 1;
        @(negedge clk); hw_error = 0;
        start_then_attack(12'h400, 32'hDEAD_BEEF);
        if (super_slot != 1 || layer_mode !== 32'h2222_0001)
            $fatal(1, "START-adjacent descriptor write polluted active slot");

        // ERROR releases the freeze; software can explicitly disable feature.
        @(negedge clk); hw_error = 1;
        @(negedge clk); hw_error = 0;
        wb_write(12'h160, 32'h0);
        if (super_req || layer_mode !== 32'h1111_0000)
            $fatal(1, "legacy restore");

        $display("PASS: super descriptor slots/freeze/default isolation");
        $finish;
    end
endmodule
