//      // verilator_coverage annotation
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
 004959     input  wire         clk,
 000011     input  wire         rst_n,
        
            // ─── Wishbone Slave Interface (CPU → CSR) ───
 000134     input  wire         wb_slv_cyc_i,
 000132     input  wire         wb_slv_stb_i,
 000017     input  wire         wb_slv_we_i,
~000052     input  wire [11:0]  wb_slv_adr_i,
~000035     input  wire [31:0]  wb_slv_dat_i,
 000015     input  wire [3:0]   wb_slv_sel_i,
~000020     output wire [31:0]  wb_slv_dat_o,
 000160     output wire         wb_slv_ack_o,
        
            // ─── Wishbone Master Interface (DMA → External Memory) ───
 000228     output wire         wb_mst_cyc_o,
 000228     output wire         wb_mst_stb_o,
%000000     output wire         wb_mst_we_o,
~000223     output wire [31:0]  wb_mst_adr_o,
%000000     output wire [31:0]  wb_mst_dat_o,
%000001     output wire [3:0]   wb_mst_sel_o,
 000450     input  wire         wb_mst_ack_i,
~000024     input  wire [31:0]  wb_mst_dat_i,
        
            // ─── Interrupt Output ───
%000003     output wire         irq_o
        );
        
            // ════════════════════════════════════════════════════════════════════
            // Internal wires
            // ════════════════════════════════════════════════════════════════════
        
            // --- CSR ↔ Controller ---
%000004     wire        ctrl_start, ctrl_abort, ctrl_soft_rst, ctrl_auto_next;
%000000     wire [7:0]  reg_layer_count;
%000004     wire        hw_busy, hw_done, hw_error;
%000000     wire [3:0]  hw_error_code;
%000003     wire [7:0]  hw_curr_layer;
        
            // --- CSR Layer Config outputs ---
%000006     wire [31:0] reg_layer_mode, reg_in_dim_hw, reg_in_dim_c;
%000005     wire [31:0] reg_out_dim_hw, reg_out_dim_c, reg_kernel_size;
%000005     wire [31:0] reg_stride, reg_padding, reg_pool_cfg;
%000000     wire [31:0] reg_resize_cfg, reg_deconv_cfg, reg_concat_cfg;
%000000     wire [31:0] reg_tile_cfg, reg_tile_count, reg_sram_base;
        
            // --- CSR DMA Config outputs ---
%000004     wire [31:0] reg_dma_in_addr, reg_dma_out_addr;
%000003     wire [31:0] reg_dma_wgt_addr, reg_dma_param_addr;
%000000     wire [31:0] reg_dma_in_stride, reg_dma_out_stride;
%000003     wire [31:0] reg_dma_ctrl, reg_dma_add_b_addr;
%000000     wire [31:0] reg_dma_add_param_addr;
%000002     wire [31:0] reg_dma_in_size, reg_dma_wgt_size;
%000000     wire [31:0] reg_dma_out_size;
        
            // --- CSR Post-Processing Config outputs ---
%000003     wire [31:0] reg_post_ctrl, reg_post_param_addr;
%000003     wire [31:0] reg_post_param_count, reg_post_clamp;
%000000     wire [31:0] reg_post_act_cfg, reg_post_add_param_addr;
%000000     wire [31:0] reg_post_add_input_addr, reg_post_add_stride;
        
            // --- Controller ↔ DMA ---
~000012     wire        dma_start, dma_dir;
%000007     wire [31:0] dma_ext_addr;
%000003     wire [15:0] dma_sram_addr, dma_xfer_len;
 000012     wire        dma_busy, dma_done;
        
            // --- Controller ↔ Compute ---
%000004     wire        compute_start, compute_done;
        
            // --- DMA ↔ SRAM MUX ---
 000448     wire        dma_sram_en, dma_sram_we;
~000223     wire [15:0] dma_sram_addr_o;
~000024     wire [31:0] dma_sram_wdata, dma_sram_rdata;
        
            // --- Performance counters ---
~002206     reg  [31:0] perf_cycle_cnt;
~000032     reg  [31:0] mac_cnt;
        
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n) begin
 000035             perf_cycle_cnt <= 32'd0;
 000035             mac_cnt        <= 32'd0;
%000000         end else if (ctrl_soft_rst) begin
%000000             perf_cycle_cnt <= 32'd0;
%000000             mac_cnt        <= 32'd0;
 002205         end else if (hw_busy) begin
 002205             perf_cycle_cnt <= perf_cycle_cnt + 1;
                    // Conv2D: ARRAY_SIZE MACs per sa_act_valid pulse
                    // DW Conv: 1 MAC per dw_in_valid pulse
%000000             if (sa_act_valid)
%000000                 mac_cnt <= mac_cnt + ARRAY_SIZE;
 002173             else if (dw_in_valid)
 000032                 mac_cnt <= mac_cnt + 1;
                end
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
                .reg_layer_count(reg_layer_count),
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
                .reg_sram_base  (reg_sram_base),
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
                .reg_dma_out_size   (reg_dma_out_size),
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
                // Layer configuration from CSR
                .cfg_dma_in_addr    (reg_dma_in_addr),
                .cfg_dma_out_addr   (reg_dma_out_addr),
                .cfg_dma_wgt_addr   (reg_dma_wgt_addr),
                .cfg_dma_param_addr (reg_dma_param_addr),
                .cfg_dma_in_size    (reg_dma_in_size),
                .cfg_dma_wgt_size   (reg_dma_wgt_size),
                .cfg_dma_out_size   (reg_dma_out_size),
                .cfg_param_count    (reg_post_param_count[15:0]),
                .cfg_dma_ctrl       (reg_dma_ctrl),
                .cfg_layer_mode     (reg_layer_mode),
                .cfg_out_base       ({3'd0, reg_sram_base[ACT_ADDR_W+16-1:16]}),
                .cfg_dma_add_b_addr (reg_dma_add_b_addr),
                // Auto-restart
                .ctrl_auto_next  (ctrl_auto_next),
                .cfg_layer_count (reg_layer_count)
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
 000448     wire                    act_a_en, act_a_we;
~000223     wire [ACT_ADDR_W-1:0]  act_a_addr;
~000024     wire [31:0]             act_a_wdata, act_a_rdata;
 000192     wire                    act_b_en;
%000007     wire [ACT_ADDR_W-1:0]  act_b_addr;
%000008     wire [31:0]             act_b_rdata;
        
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
 000448     wire                    wgt_a_en, wgt_a_we;
~000223     wire [WGT_ADDR_W-1:0]  wgt_a_addr;
~000024     wire [31:0]             wgt_a_wdata, wgt_a_rdata;
 000064     wire                    wgt_b_en;
%000007     wire [WGT_ADDR_W-1:0]  wgt_b_addr;
%000000     wire [31:0]             wgt_b_rdata;
        
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
 000448     wire                      param_a_en, param_a_we;
~000223     wire [PARAM_ADDR_W-1:0]   param_a_addr;
~000024     wire [31:0]               param_a_wdata, param_a_rdata;
 000256     wire                      param_b_en;
~000127     wire [PARAM_ADDR_W-1:0]   param_b_addr;
%000000     wire [31:0]               param_b_rdata;
        
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
%000008     reg [1:0] dma_bank_sel;
        
 002485     always @(posedge clk or negedge rst_n) begin
 002450         if (!rst_n) begin
 000035             dma_bank_sel <= 2'd1;
~002444         end else if (dma_start) begin
                    // Latch bank select at transfer start based on ext_addr
%000002             if (dma_ext_addr == reg_dma_wgt_addr)
%000002                 dma_bank_sel <= 2'd0;  // weight
%000002             else if (dma_ext_addr == reg_dma_param_addr ||
                             dma_ext_addr == reg_dma_add_param_addr)
%000002                 dma_bank_sel <= 2'd2;  // param
                    else
%000002                 dma_bank_sel <= 2'd1;  // activation (load A, load Add B, or store)
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
%000000     wire [1:0]                    sa_cmd;
%000000     wire                          sa_cmd_valid;
%000000     wire [`DATA_WIDTH*ARRAY_SIZE-1:0] sa_wgt_data_flat;
%000000     wire                          sa_wgt_valid;
%000000     wire [`DATA_WIDTH*ARRAY_SIZE-1:0] sa_act_data_flat;
%000000     wire                          sa_act_valid;
%000000     wire [$clog2(ARRAY_SIZE)-1:0] sa_drain_col_sel;
            wire [`ACC_WIDTH*ARRAY_SIZE-1:0]  sa_acc_out_flat;
%000000     wire                          sa_acc_out_valid;
%000000     wire                          sa_busy, sa_ready;
        
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
                .wgt_data_flat  (sa_wgt_data_flat),
                .wgt_valid      (sa_wgt_valid),
                .act_data_flat  (sa_act_data_flat),
                .act_valid      (sa_act_valid),
                .drain_col_sel  (sa_drain_col_sel),
                .acc_out_flat   (sa_acc_out_flat),
                .acc_out_valid  (sa_acc_out_valid),
                .busy           (sa_busy),
                .ready          (sa_ready)
            );
        
            // ════════════════════════════════════════════════════════════════════
            // DW Convolution Engine (single lane)
            // ════════════════════════════════════════════════════════════════════
        
%000000     wire signed [`ACC_WIDTH-1:0]  dw_acc_out;
 000064     wire                          dw_out_valid;
 000064     wire                          dw_wgt_load;
 000064     wire                          dw_wgt_valid;
%000000     wire signed [`DATA_WIDTH-1:0] dw_wgt_data;
 000064     wire                          dw_in_valid;
~000015     wire signed [`DATA_WIDTH-1:0] dw_in_data;
 000128     wire                          dw_acc_clear;
        
            npu_dw_conv #(
                .DATA_W  (`DATA_WIDTH),
                .ACC_W   (`ACC_WIDTH),
                .MAX_KSZ (7)
            ) u_dw_conv (
                .clk        (clk),
                .rst_n      (rst_n),
                .kernel_h   (reg_kernel_size[3:0]),
                .kernel_w   (reg_kernel_size[11:8]),
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
        
%000000     wire signed [`DATA_WIDTH-1:0] ppu_out_data;
 000064     wire                          ppu_out_valid;
%000000     wire signed [`ACC_WIDTH-1:0]  ppu_acc_in;
 000064     wire                          ppu_in_valid;
%000000     wire signed [`ACC_WIDTH-1:0]  ppu_bias;
%000000     wire [`PARAM_M_BITS-1:0]      ppu_mult_m;
%000000     wire [`PARAM_S_BITS-1:0]      ppu_shift_s;
%000000     wire signed [`PARAM_ZP_BITS-1:0] ppu_zero_point;
        
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
                .int16_mode (reg_post_ctrl[7]),
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
        
%000004     wire        compute_done_w;
 000064     wire        act_b_wr_en;
%000007     wire [ACT_ADDR_W-1:0] act_b_wr_addr;
~000016     wire [31:0] act_b_wr_data;
 000128     wire        act_b_rd_en;
%000007     wire [ACT_ADDR_W-1:0] act_b_rd_addr;
        
            // Compute engine handles full pipeline (compute + PPU + writeback) internally
            assign compute_done = compute_done_w;
        
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
                .cfg_op_type    ({4'd0, reg_layer_mode[3:0]}),
                .cfg_int16      (reg_post_ctrl[7]),
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
                .cfg_act_base   (reg_sram_base[ACT_ADDR_W-1:0]),
                .cfg_out_base   (reg_sram_base[ACT_ADDR_W+16-1:16]),
                .cfg_pool_cfg   (reg_pool_cfg),
                .cfg_resize_cfg (reg_resize_cfg),
                .cfg_deconv_cfg (reg_deconv_cfg),
                .cfg_concat_cfg (reg_concat_cfg),
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
                .sa_wgt_data_flat(sa_wgt_data_flat),
                .sa_wgt_valid   (sa_wgt_valid),
                .sa_act_data_flat(sa_act_data_flat),
                .sa_act_valid   (sa_act_valid),
                .sa_drain_col_sel(sa_drain_col_sel),
                .sa_acc_out_flat (sa_acc_out_flat),
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
        
