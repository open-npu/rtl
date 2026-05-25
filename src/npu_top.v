// Open-NPU RTL — Top-Level Integration
// SPDX-License-Identifier: Apache-2.0
//
// Integrates all NPU sub-modules:
//   - CSR (Wishbone slave for CPU configuration)
//   - Controller (layer sequencer FSM)
//   - DMA (Wishbone master for external memory)
//   - SRAMs: activation, weight, parameter (dual-port)
//   - Systolic Array (weight-stationary N×N)
//   - DW Convolution Engine
//   - PPU (post-processing unit)
//
// External interfaces:
//   - Wishbone slave (CPU ↔ CSR registers)
//   - Wishbone master (DMA ↔ external memory)
//   - IRQ output
//
// Simplified V1: single DMA channel, no ping-pong, no tiling loop.

`include "npu_defines.vh"

module npu_top #(
    parameter ARRAY_SIZE = `ARRAY_SIZE,
    parameter SPAD_KB    = `SPAD_KB,
    // SRAM depths (words)
    parameter ACT_DEPTH   = 8192,   // 32KB / 4B = 8K words
    parameter WGT_DEPTH   = 16384,  // 64KB / 4B = 16K words
    parameter PARAM_DEPTH = 2048    // 8KB / 4B = 2K words
)(
    input  wire         clk,
    input  wire         rst_n,

    // ─── Wishbone Slave Interface (CPU → CSR) ───
    input  wire         wb_slv_cyc_i,
    input  wire         wb_slv_stb_i,
    input  wire         wb_slv_we_i,
    input  wire [11:0]  wb_slv_adr_i,
    input  wire [31:0]  wb_slv_dat_i,
    input  wire [3:0]   wb_slv_sel_i,
    output wire [31:0]  wb_slv_dat_o,
    output wire         wb_slv_ack_o,

    // ─── Wishbone Master Interface (DMA → External Memory) ───
    output wire         wb_mst_cyc_o,
    output wire         wb_mst_stb_o,
    output wire         wb_mst_we_o,
    output wire [31:0]  wb_mst_adr_o,
    output wire [31:0]  wb_mst_dat_o,
    output wire [3:0]   wb_mst_sel_o,
    input  wire         wb_mst_ack_i,
    input  wire [31:0]  wb_mst_dat_i,

    // ─── Interrupt Output ───
    output wire         irq_o
);

    // ════════════════════════════════════════════════════════════════════
    // Internal wires
    // ════════════════════════════════════════════════════════════════════

    // --- CSR ↔ Controller ---
    wire        ctrl_start, ctrl_abort, ctrl_soft_rst, ctrl_auto_next;
    wire        hw_busy, hw_done, hw_error;
    wire [3:0]  hw_error_code;
    wire [7:0]  hw_curr_layer;

    // --- CSR Layer Config outputs ---
    wire [31:0] reg_layer_mode, reg_in_dim_hw, reg_in_dim_c;
    wire [31:0] reg_out_dim_hw, reg_out_dim_c, reg_kernel_size;
    wire [31:0] reg_stride, reg_padding, reg_pool_cfg;
    wire [31:0] reg_resize_cfg, reg_deconv_cfg, reg_concat_cfg;
    wire [31:0] reg_tile_cfg, reg_tile_count;

    // --- CSR DMA Config outputs ---
    wire [31:0] reg_dma_in_addr, reg_dma_out_addr;
    wire [31:0] reg_dma_wgt_addr, reg_dma_param_addr;
    wire [31:0] reg_dma_in_stride, reg_dma_out_stride;
    wire [31:0] reg_dma_ctrl, reg_dma_add_b_addr;
    wire [31:0] reg_dma_add_param_addr;
    wire [31:0] reg_dma_in_size, reg_dma_wgt_size;

    // --- CSR Post-Processing Config outputs ---
    wire [31:0] reg_post_ctrl, reg_post_param_addr;
    wire [31:0] reg_post_param_count, reg_post_clamp;
    wire [31:0] reg_post_act_cfg, reg_post_add_param_addr;
    wire [31:0] reg_post_add_input_addr, reg_post_add_stride;

    // --- Controller ↔ DMA ---
    wire        dma_start, dma_dir;
    wire [31:0] dma_ext_addr;
    wire [15:0] dma_sram_addr, dma_xfer_len;
    wire        dma_busy, dma_done;

    // --- Controller ↔ Compute ---
    wire        compute_start, compute_done;

    // --- Controller ↔ PPU ---
    wire        ppu_start, ppu_done;

    // --- DMA ↔ SRAM MUX ---
    wire        dma_sram_en, dma_sram_we;
    wire [15:0] dma_sram_addr_o;
    wire [31:0] dma_sram_wdata, dma_sram_rdata;

    // --- Performance counters (stub) ---
    reg  [31:0] perf_cycle_cnt;
    wire [31:0] mac_cnt = 32'd0;  // TODO: real MAC counter

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            perf_cycle_cnt <= 32'd0;
        else if (hw_busy)
            perf_cycle_cnt <= perf_cycle_cnt + 1;
    end

    // ════════════════════════════════════════════════════════════════════
    // CSR — Wishbone Slave Register File
    // ════════════════════════════════════════════════════════════════════

    npu_csr #(
        .ARRAY_SZ   (ARRAY_SIZE),
        .SPAD_KB_P  (SPAD_KB)
    ) u_csr (
        .clk            (clk),
        .rst_n          (rst_n),
        // Wishbone slave
        .wb_cyc_i       (wb_slv_cyc_i),
        .wb_stb_i       (wb_slv_stb_i),
        .wb_we_i        (wb_slv_we_i),
        .wb_adr_i       (wb_slv_adr_i),
        .wb_dat_i       (wb_slv_dat_i),
        .wb_sel_i       (wb_slv_sel_i),
        .wb_dat_o       (wb_slv_dat_o),
        .wb_ack_o       (wb_slv_ack_o),
        // Hardware status inputs
        .hw_busy        (hw_busy),
        .hw_dma_busy    (dma_busy),
        .hw_done        (hw_done),
        .hw_error       (hw_error),
        .hw_dma_done    (dma_done),
        .hw_error_code  (hw_error_code),
        .hw_curr_layer  (hw_curr_layer),
        .hw_perf_cnt    (perf_cycle_cnt),
        .hw_mac_cnt     (mac_cnt),
        // Control outputs
        .ctrl_start     (ctrl_start),
        .ctrl_abort     (ctrl_abort),
        .ctrl_soft_rst  (ctrl_soft_rst),
        .ctrl_auto_next (ctrl_auto_next),
        .irq_o          (irq_o),
        // Layer parameters
        .reg_layer_mode (reg_layer_mode),
        .reg_in_dim_hw  (reg_in_dim_hw),
        .reg_in_dim_c   (reg_in_dim_c),
        .reg_out_dim_hw (reg_out_dim_hw),
        .reg_out_dim_c  (reg_out_dim_c),
        .reg_kernel_size(reg_kernel_size),
        .reg_stride     (reg_stride),
        .reg_padding    (reg_padding),
        .reg_pool_cfg   (reg_pool_cfg),
        .reg_resize_cfg (reg_resize_cfg),
        .reg_deconv_cfg (reg_deconv_cfg),
        .reg_concat_cfg (reg_concat_cfg),
        .reg_tile_cfg   (reg_tile_cfg),
        .reg_tile_count (reg_tile_count),
        // DMA config
        .reg_dma_in_addr    (reg_dma_in_addr),
        .reg_dma_out_addr   (reg_dma_out_addr),
        .reg_dma_wgt_addr   (reg_dma_wgt_addr),
        .reg_dma_param_addr (reg_dma_param_addr),
        .reg_dma_in_stride  (reg_dma_in_stride),
        .reg_dma_out_stride (reg_dma_out_stride),
        .reg_dma_ctrl       (reg_dma_ctrl),
        .reg_dma_add_b_addr (reg_dma_add_b_addr),
        .reg_dma_add_param_addr(reg_dma_add_param_addr),
        .reg_dma_in_size    (reg_dma_in_size),
        .reg_dma_wgt_size   (reg_dma_wgt_size),
        // Post-processing config
        .reg_post_ctrl          (reg_post_ctrl),
        .reg_post_param_addr    (reg_post_param_addr),
        .reg_post_param_count   (reg_post_param_count),
        .reg_post_clamp         (reg_post_clamp),
        .reg_post_act_cfg       (reg_post_act_cfg),
        .reg_post_add_param_addr(reg_post_add_param_addr),
        .reg_post_add_input_addr(reg_post_add_input_addr),
        .reg_post_add_stride    (reg_post_add_stride)
    );

    // ════════════════════════════════════════════════════════════════════
    // Controller — Layer Sequencer FSM
    // ════════════════════════════════════════════════════════════════════

    npu_ctrl u_ctrl (
        .clk            (clk),
        .rst_n          (rst_n),
        // CSR interface
        .ctrl_start     (ctrl_start),
        .ctrl_abort     (ctrl_abort),
        .ctrl_soft_rst  (ctrl_soft_rst),
        .hw_busy        (hw_busy),
        .hw_done        (hw_done),
        .hw_error       (hw_error),
        .hw_error_code  (hw_error_code),
        .hw_curr_layer  (hw_curr_layer),
        // DMA control
        .dma_start      (dma_start),
        .dma_dir        (dma_dir),
        .dma_ext_addr   (dma_ext_addr),
        .dma_sram_addr  (dma_sram_addr),
        .dma_xfer_len   (dma_xfer_len),
        .dma_busy       (dma_busy),
        .dma_done       (dma_done),
        // Compute control
        .compute_start  (compute_start),
        .compute_done   (compute_done),
        // PPU control
        .ppu_start      (ppu_start),
        .ppu_done       (ppu_done),
        // Layer configuration from CSR
        .cfg_dma_in_addr    (reg_dma_in_addr),
        .cfg_dma_out_addr   (reg_dma_out_addr),
        .cfg_dma_wgt_addr   (reg_dma_wgt_addr),
        .cfg_dma_param_addr (reg_dma_param_addr),
        .cfg_dma_in_size    (reg_dma_in_size),
        .cfg_dma_wgt_size   (reg_dma_wgt_size),
        .cfg_param_count    (reg_post_param_count[15:0]),
        .cfg_layer_mode     (reg_layer_mode)
    );

    // ════════════════════════════════════════════════════════════════════
    // DMA Engine — Wishbone Master + SRAM Interface
    // ════════════════════════════════════════════════════════════════════

    npu_dma #(
        .ADDR_W      (32),
        .DATA_W      (32),
        .SRAM_ADDR_W (16)
    ) u_dma (
        .clk            (clk),
        .rst_n          (rst_n),
        // Control
        .start          (dma_start),
        .abort          (ctrl_abort),
        .dir            (dma_dir),
        .ext_addr       (dma_ext_addr),
        .sram_addr      (dma_sram_addr),
        .xfer_len       (dma_xfer_len),
        .burst_cfg      (reg_dma_ctrl[5:4]),
        // Status
        .busy           (dma_busy),
        .done_pulse     (dma_done),
        .xfer_count     (),  // unused at top level
        // Wishbone master
        .wb_cyc_o       (wb_mst_cyc_o),
        .wb_stb_o       (wb_mst_stb_o),
        .wb_we_o        (wb_mst_we_o),
        .wb_adr_o       (wb_mst_adr_o),
        .wb_dat_o       (wb_mst_dat_o),
        .wb_sel_o       (wb_mst_sel_o),
        .wb_ack_i       (wb_mst_ack_i),
        .wb_dat_i       (wb_mst_dat_i),
        // SRAM interface (muxed by controller phase)
        .sram_en        (dma_sram_en),
        .sram_we        (dma_sram_we),
        .sram_addr_o    (dma_sram_addr_o),
        .sram_wdata     (dma_sram_wdata),
        .sram_rdata     (dma_sram_rdata)
    );

    // ════════════════════════════════════════════════════════════════════
    // SRAM Banks
    // ════════════════════════════════════════════════════════════════════

    // Address width calculations
    localparam ACT_ADDR_W   = $clog2(ACT_DEPTH);
    localparam WGT_ADDR_W   = $clog2(WGT_DEPTH);
    localparam PARAM_ADDR_W = $clog2(PARAM_DEPTH);

    // --- Activation SRAM ---
    // Port A: DMA access (load input / store output)
    // Port B: Compute engine read + write (output writeback)
    wire                    act_a_en, act_a_we;
    wire [ACT_ADDR_W-1:0]  act_a_addr;
    wire [31:0]             act_a_wdata, act_a_rdata;
    wire                    act_b_en;
    wire [ACT_ADDR_W-1:0]  act_b_addr;
    wire [31:0]             act_b_rdata;

    npu_sram #(
        .DATA_W (32),
        .DEPTH  (ACT_DEPTH)
    ) u_sram_act (
        .clk    (clk),
        .a_en   (act_a_en),
        .a_we   (act_a_we),
        .a_addr (act_a_addr),
        .a_wdata(act_a_wdata),
        .a_rdata(act_a_rdata),
        .b_en   (act_b_en),
        .b_we   (act_b_wr_en),
        .b_addr (act_b_addr),
        .b_wdata(act_b_wr_data),
        .b_rdata(act_b_rdata)
    );

    // --- Weight SRAM ---
    // Port A: DMA access (load weights)
    // Port B: Compute engine read
    wire                    wgt_a_en, wgt_a_we;
    wire [WGT_ADDR_W-1:0]  wgt_a_addr;
    wire [31:0]             wgt_a_wdata, wgt_a_rdata;
    wire                    wgt_b_en;
    wire [WGT_ADDR_W-1:0]  wgt_b_addr;
    wire [31:0]             wgt_b_rdata;

    npu_sram #(
        .DATA_W (32),
        .DEPTH  (WGT_DEPTH)
    ) u_sram_wgt (
        .clk    (clk),
        .a_en   (wgt_a_en),
        .a_we   (wgt_a_we),
        .a_addr (wgt_a_addr),
        .a_wdata(wgt_a_wdata),
        .a_rdata(wgt_a_rdata),
        .b_en   (wgt_b_en),
        .b_we   (1'b0),
        .b_addr (wgt_b_addr),
        .b_wdata(32'd0),
        .b_rdata(wgt_b_rdata)
    );

    // --- Parameter SRAM ---
    // Port A: DMA access (load per-channel params)
    // Port B: PPU reads during post-processing
    wire                      param_a_en, param_a_we;
    wire [PARAM_ADDR_W-1:0]   param_a_addr;
    wire [31:0]               param_a_wdata, param_a_rdata;
    wire                      param_b_en;
    wire [PARAM_ADDR_W-1:0]   param_b_addr;
    wire [31:0]               param_b_rdata;

    npu_sram #(
        .DATA_W (32),
        .DEPTH  (PARAM_DEPTH)
    ) u_sram_param (
        .clk    (clk),
        .a_en   (param_a_en),
        .a_we   (param_a_we),
        .a_addr (param_a_addr),
        .a_wdata(param_a_wdata),
        .a_rdata(param_a_rdata),
        .b_en   (param_b_en),
        .b_we   (1'b0),
        .b_addr (param_b_addr),
        .b_wdata(32'd0),
        .b_rdata(param_b_rdata)
    );

    // ════════════════════════════════════════════════════════════════════
    // DMA ↔ SRAM MUX
    // ════════════════════════════════════════════════════════════════════
    //
    // The controller drives DMA with a direction and phase.
    // During weight load: DMA writes to weight SRAM
    // During act load: DMA writes to activation SRAM
    // During param load: DMA writes to parameter SRAM
    // During output store: DMA reads from activation SRAM
    //
    // We use the controller's FSM state (exposed via dma_dir and the
    // address range) to select which SRAM the DMA connects to.
    // V1 simplified: controller uses a dedicated SRAM select based on phase.
    //
    // For V1 we route based on the DMA ext_addr range from the controller.
    // Alternative: controller could expose a sram_select signal.
    // Here we use a simple approach: during controller phases, the DMA
    // sram_addr matches the target bank. The controller ensures no overlap.
    //
    // Since the controller drives DMA sequentially (wgt → act → param → store),
    // and we know which phase from dma_ext_addr vs cfg registers, we use the
    // controller's internal state. However, npu_ctrl doesn't export state.
    //
    // Best V1 approach: Add a 2-bit sram_bank_sel output to npu_ctrl.
    // For now, we decode from the ext_addr matching cfg addresses.

    // SRAM bank select: registered, captured on dma_start pulse
    // 0 = weight, 1 = activation, 2 = parameter
    reg [1:0] dma_bank_sel;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_bank_sel <= 2'd1;
        end else if (dma_start) begin
            // Latch bank select at transfer start based on ext_addr
            if (dma_ext_addr == reg_dma_wgt_addr)
                dma_bank_sel <= 2'd0;  // weight
            else if (dma_ext_addr == reg_dma_param_addr)
                dma_bank_sel <= 2'd2;  // param
            else
                dma_bank_sel <= 2'd1;  // activation (load or store)
        end
    end

    // Route DMA SRAM signals to correct bank
    assign act_a_en    = dma_sram_en && (dma_bank_sel == 2'd1);
    assign act_a_we    = dma_sram_we;
    assign act_a_addr  = dma_sram_addr_o[ACT_ADDR_W-1:0];
    assign act_a_wdata = dma_sram_wdata;

    assign wgt_a_en    = dma_sram_en && (dma_bank_sel == 2'd0);
    assign wgt_a_we    = dma_sram_we;
    assign wgt_a_addr  = dma_sram_addr_o[WGT_ADDR_W-1:0];
    assign wgt_a_wdata = dma_sram_wdata;

    assign param_a_en    = dma_sram_en && (dma_bank_sel == 2'd2);
    assign param_a_we    = dma_sram_we;
    assign param_a_addr  = dma_sram_addr_o[PARAM_ADDR_W-1:0];
    assign param_a_wdata = dma_sram_wdata;

    // DMA read data mux (for store direction)
    assign dma_sram_rdata = (dma_bank_sel == 2'd0) ? wgt_a_rdata :
                            (dma_bank_sel == 2'd2) ? param_a_rdata :
                                                     act_a_rdata;

    // ════════════════════════════════════════════════════════════════════
    // Systolic Array
    // ════════════════════════════════════════════════════════════════════

    // Systolic array signals (driven by npu_compute)
    wire [1:0]                    sa_cmd;
    wire                          sa_cmd_valid;
    wire signed [`DATA_WIDTH-1:0] sa_wgt_data  [0:ARRAY_SIZE-1];
    wire                          sa_wgt_valid;
    wire signed [`DATA_WIDTH-1:0] sa_act_data  [0:ARRAY_SIZE-1];
    wire                          sa_act_valid;
    wire [$clog2(ARRAY_SIZE)-1:0] sa_drain_col_sel;
    wire signed [`ACC_WIDTH-1:0]  sa_acc_out   [0:ARRAY_SIZE-1];
    wire                          sa_acc_out_valid;
    wire                          sa_busy, sa_ready;

    npu_systolic #(
        .ROWS   (ARRAY_SIZE),
        .COLS   (ARRAY_SIZE),
        .DATA_W (`DATA_WIDTH),
        .ACC_W  (`ACC_WIDTH)
    ) u_systolic (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd            (sa_cmd),
        .cmd_valid      (sa_cmd_valid),
        .wgt_data       (sa_wgt_data),
        .wgt_valid      (sa_wgt_valid),
        .act_data       (sa_act_data),
        .act_valid      (sa_act_valid),
        .drain_col_sel  (sa_drain_col_sel),
        .acc_out        (sa_acc_out),
        .acc_out_valid  (sa_acc_out_valid),
        .busy           (sa_busy),
        .ready          (sa_ready)
    );

    // ════════════════════════════════════════════════════════════════════
    // DW Convolution Engine (single lane)
    // ════════════════════════════════════════════════════════════════════

    wire signed [`ACC_WIDTH-1:0]  dw_acc_out;
    wire                          dw_out_valid;
    wire                          dw_wgt_load;
    wire                          dw_wgt_valid;
    wire signed [`DATA_WIDTH-1:0] dw_wgt_data;
    wire                          dw_in_valid;
    wire signed [`DATA_WIDTH-1:0] dw_in_data;
    wire                          dw_acc_clear;

    npu_dw_conv #(
        .DATA_W  (`DATA_WIDTH),
        .ACC_W   (`ACC_WIDTH),
        .MAX_KSZ (7)
    ) u_dw_conv (
        .clk        (clk),
        .rst_n      (rst_n),
        .kernel_h   (reg_kernel_size[3:0]),
        .kernel_w   (reg_kernel_size[7:4]),
        .wgt_load   (dw_wgt_load),
        .wgt_valid  (dw_wgt_valid),
        .wgt_data   (dw_wgt_data),
        .in_valid   (dw_in_valid),
        .in_data    (dw_in_data),
        .acc_clear  (dw_acc_clear),
        .acc_out    (dw_acc_out),
        .out_valid  (dw_out_valid)
    );

    // ════════════════════════════════════════════════════════════════════
    // PPU — Post-Processing Unit (single lane)
    // ════════════════════════════════════════════════════════════════════

    wire signed [`DATA_WIDTH-1:0] ppu_out_data;
    wire                          ppu_out_valid;
    wire signed [`ACC_WIDTH-1:0]  ppu_acc_in;
    wire                          ppu_in_valid;
    wire signed [`ACC_WIDTH-1:0]  ppu_bias;
    wire [`PARAM_M_BITS-1:0]      ppu_mult_m;
    wire [`PARAM_S_BITS-1:0]      ppu_shift_s;
    wire signed [`PARAM_ZP_BITS-1:0] ppu_zero_point;

    npu_ppu #(
        .ACC_W   (`ACC_WIDTH),
        .DATA_W  (`DATA_WIDTH),
        .BIAS_W  (`BIAS_WIDTH),
        .MULT_W  (`PARAM_M_BITS),
        .SHIFT_W (`PARAM_S_BITS),
        .ZP_W    (`PARAM_ZP_BITS)
    ) u_ppu (
        .clk        (clk),
        .rst_n      (rst_n),
        .mode       (reg_post_ctrl[1:0]),
        .relu_en    (reg_post_ctrl[2]),
        .bias_en    (reg_post_ctrl[6]),
        .zp_en      (reg_post_ctrl[5]),
        .acc_in     (ppu_acc_in),
        .in_valid   (ppu_in_valid),
        .bias       (ppu_bias),
        .mult_m     (ppu_mult_m),
        .shift_s    (ppu_shift_s),
        .zero_point (ppu_zero_point),
        .out_data   (ppu_out_data),
        .out_valid  (ppu_out_valid)
    );

    // ════════════════════════════════════════════════════════════════════
    // Compute Micro-Sequencer (V2)
    // ════════════════════════════════════════════════════════════════════

    wire        compute_done_w;
    wire        act_b_wr_en;
    wire [ACT_ADDR_W-1:0] act_b_wr_addr;
    wire [31:0] act_b_wr_data;
    wire        act_b_rd_en;
    wire [ACT_ADDR_W-1:0] act_b_rd_addr;

    // Both compute_done and ppu_done come from the compute engine's done signal
    // (the V2 compute engine handles the full compute+PPU+writeback pipeline internally)
    assign compute_done = compute_done_w;
    assign ppu_done     = compute_done_w;

    npu_compute #(
        .ARRAY_SIZE   (ARRAY_SIZE),
        .ACT_ADDR_W   (ACT_ADDR_W),
        .WGT_ADDR_W   (WGT_ADDR_W),
        .PARAM_ADDR_W (PARAM_ADDR_W),
        .DATA_W       (`DATA_WIDTH),
        .ACC_W        (`ACC_WIDTH)
    ) u_compute (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (compute_start),
        .done           (compute_done_w),
        // Layer config from CSR
        .cfg_op_type    (reg_layer_mode[7:0]),
        .cfg_in_c       (reg_in_dim_c[15:0]),
        .cfg_out_h      (reg_out_dim_hw[31:16]),
        .cfg_out_w      (reg_out_dim_hw[15:0]),
        .cfg_out_c      (reg_out_dim_c[15:0]),
        .cfg_kernel_h   (reg_kernel_size[7:0]),
        .cfg_kernel_w   (reg_kernel_size[15:8]),
        .cfg_stride_h   (reg_stride[7:0]),
        .cfg_stride_w   (reg_stride[15:8]),
        .cfg_pad_top    (reg_padding[7:0]),
        .cfg_pad_left   (reg_padding[15:8]),
        .cfg_tile_h     (reg_tile_cfg[15:0]),
        .cfg_tile_w     (reg_tile_cfg[31:16]),
        .cfg_tile_num_h (reg_tile_count[15:0]),
        .cfg_tile_num_w (reg_tile_count[31:16]),
        .cfg_in_w       (reg_in_dim_hw[15:0]),
        .cfg_in_h       (reg_in_dim_hw[31:16]),
        // Weight SRAM Port B
        .wgt_rd_en      (wgt_b_en),
        .wgt_rd_addr    (wgt_b_addr),
        .wgt_rd_data    (wgt_b_rdata),
        // Activation SRAM Port B
        .act_rd_en      (act_b_rd_en),
        .act_rd_addr    (act_b_rd_addr),
        .act_rd_data    (act_b_rdata),
        .act_wr_en      (act_b_wr_en),
        .act_wr_addr    (act_b_wr_addr),
        .act_wr_data    (act_b_wr_data),
        // Param SRAM Port B
        .param_rd_en    (param_b_en),
        .param_rd_addr  (param_b_addr),
        .param_rd_data  (param_b_rdata),
        // Systolic
        .sa_cmd         (sa_cmd),
        .sa_cmd_valid   (sa_cmd_valid),
        .sa_wgt_data    (sa_wgt_data),
        .sa_wgt_valid   (sa_wgt_valid),
        .sa_act_data    (sa_act_data),
        .sa_act_valid   (sa_act_valid),
        .sa_drain_col_sel(sa_drain_col_sel),
        .sa_acc_out     (sa_acc_out),
        .sa_acc_out_valid(sa_acc_out_valid),
        .sa_busy        (sa_busy),
        .sa_ready       (sa_ready),
        // DW Conv
        .dw_wgt_load    (dw_wgt_load),
        .dw_wgt_valid   (dw_wgt_valid),
        .dw_wgt_data    (dw_wgt_data),
        .dw_in_valid    (dw_in_valid),
        .dw_in_data     (dw_in_data),
        .dw_acc_clear   (dw_acc_clear),
        .dw_acc_out     (dw_acc_out),
        .dw_out_valid   (dw_out_valid),
        // PPU
        .ppu_acc_in     (ppu_acc_in),
        .ppu_in_valid   (ppu_in_valid),
        .ppu_bias       (ppu_bias),
        .ppu_mult_m     (ppu_mult_m),
        .ppu_shift_s    (ppu_shift_s),
        .ppu_zero_point (ppu_zero_point),
        .ppu_out_data   (ppu_out_data),
        .ppu_out_valid  (ppu_out_valid)
    );

    // ─── SRAM Port B connections (compute engine) ───
    // Activation SRAM Port B: needs both read and write for compute
    assign act_b_en   = act_b_rd_en | act_b_wr_en;
    assign act_b_addr = act_b_wr_en ? act_b_wr_addr : act_b_rd_addr;

endmodule

