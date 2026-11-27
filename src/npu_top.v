// Open-NPU RTL — Top-Level Integration
// SPDX-License-Identifier: Apache-2.0
//
// Integrates all NPU sub-modules:
//   - CSR (Wishbone slave for CPU configuration)
//   - Controller (layer sequencer FSM)
//   - DMA (Wishbone master for external memory)
//   - SRAMs: IFM, OFM, weight, parameter (dual-port)
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
    parameter SPAD_KB    = `SPAD_KB,          // total KB reported in HW_CONFIG
    // IFM 32KB, OFM 48KB, weight 80KB, param 12KB. See npu_defines.vh.
    parameter IFM_DEPTH   = `IFM_DEPTH,
    parameter OFM_DEPTH   = `OFM_DEPTH,
    parameter WGT_DEPTH   = `WGT_DEPTH_W,
    parameter PARAM_DEPTH = `PARAM_DEPTH_W
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
    output wire [2:0]   wb_mst_cti_o,
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
    wire        ctrl_super_req, ctrl_super_has_add, ctrl_super_stream_param;
    wire        ctrl_super_slot_valid;
    wire [1:0]  ctrl_super_slot;
    wire        super_slot_en;
    wire [1:0]  super_slot_select;
    // Cancel all in-flight datapath activity on SOFT_RST or ABORT. SRAM
    // contents and programmed CSRs intentionally remain intact.
    wire        engine_rst_n = rst_n & ~(ctrl_soft_rst | ctrl_abort);
    wire [7:0]  reg_layer_count;
    wire        hw_busy, hw_done, hw_error;
    wire [3:0]  hw_error_code;
    wire [7:0]  hw_curr_layer;

    // --- DB_EN (Double-Buffer) wires ---
    wire        ping_pong_flag;
    wire        tile_done;
    wire        db_prefetch_done;
    wire [15:0] tile_out_h_actual, tile_out_w_actual;  // Border-clipped tile dims
    wire [15:0] dma_row_len;
    wire [15:0] dma_row_count;
    wire [15:0] dma_src_row_len;
    wire [31:0] dma_out_stride;
    wire        db_en_is_active = reg_dma_ctrl[0] && hw_busy;

    wire [7:0]  eff_dilation_h = (reg_kernel_size[23:16] == 8'd0)
                               ? 8'd1 : reg_kernel_size[23:16];
    wire [7:0]  eff_dilation_w = (reg_kernel_size[31:24] == 8'd0)
                               ? 8'd1 : reg_kernel_size[31:24];

    // --- CSR Layer Config outputs ---
    wire [31:0] reg_layer_mode, reg_in_dim_hw, reg_in_dim_c;
    wire [31:0] reg_out_dim_hw, reg_out_dim_c, reg_kernel_size;
    wire [31:0] reg_stride, reg_padding, reg_pool_cfg;
    wire [31:0] reg_resize_cfg, reg_concat_cfg;
    wire [31:0] reg_tile_cfg, reg_tile_count, reg_sram_base;

    // --- CSR DMA Config outputs ---
    wire [31:0] reg_dma_in_addr, reg_dma_out_addr;
    wire [31:0] reg_dma_wgt_addr, reg_dma_param_addr;
    wire [31:0] reg_dma_in_stride, reg_dma_out_stride;
    wire        npu_2d_load = (reg_dma_in_stride != 32'd0);
    wire [31:0] reg_dma_ctrl, reg_dma_add_b_addr;
    wire [31:0] reg_dma_add_param_addr;
    wire [31:0] reg_dma_in_size, reg_dma_wgt_size;
    wire [31:0] reg_dma_out_size;
    wire [31:0] reg_dma_tile_in_size;
    wire [31:0] reg_dma_tile_out_size;
    wire [31:0] reg_dma_store_mode;
    wire [31:0] reg_dma_row_cfg;
    wire [31:0] reg_dma_wgt_per_oc;
    wire [31:0] reg_wgt_layout;
    wire [31:0] reg_tile_in_hw;
    wire [31:0] reg_next_param_addr, reg_next_param_cfg;
    wire [31:0] reg_param_bank_ctrl;
    wire [31:0] reg_wgt_sram_base, reg_param_sram_base;

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
    wire        st_dma_start, st_dma_active;
    wire [31:0] st_dma_ext_addr, st_dma_out_stride;
    wire [15:0] st_dma_sram_addr, st_dma_xfer_len;
    wire [15:0] st_dma_row_len, st_dma_row_count, st_dma_src_row_len;
    wire        rd_dma_busy, rd_dma_done;
    wire        wr_dma_busy, wr_dma_done;

    // --- Controller ↔ Compute ---
    wire        compute_start, compute_done;
    wire        compute_single_tile_en;
    wire [15:0] compute_external_tile_y, compute_external_tile_x;
    wire [15:0] compute_external_tile_h, compute_external_tile_w;
    wire        compute_external_origin_en, compute_halo_expand_en;
    wire signed [16:0] compute_external_origin_h, compute_external_origin_w;
    wire [15:0] compute_external_tile_in_w;
    wire        oc_group_done, wgt_reload_done;
    wire        group_major_en;
    wire [15:0] oc_group_idx;
    wire [15:0] wgt_k_start;
    wire [1:0]  dma_sram_sel;
    wire        act_role_swap;
    wire        next_param_ready;
    wire [15:0] compute_param_base;
    wire [15:0] compute_wgt_base;

    // --- DMA ↔ SRAM MUX ---
    wire        dma_sram_en, dma_sram_we;
    wire [15:0] dma_sram_addr_o;
    wire [31:0] dma_sram_wdata, dma_sram_rdata;
    wire        dma_sram_wide_en;
    wire [15:0] dma_sram_wide_addr;
    wire [`SRAM_B_WIDTH-1:0] dma_sram_rdata_wide;
    wire        compute_uses_ofm_b;

    // --- Performance counters ---
    reg  [31:0] perf_cycle_cnt;
    reg  [31:0] mac_cnt;
    reg  [31:0] perf_dma_cnt;
    reg  [31:0] perf_compute_cnt;
    reg  [31:0] perf_overlap_cnt;
    reg         compute_active;
    wire        mac_fire;
    wire [6:0]  mac_lanes;
    wire [6:0]  compute_state;

    // Compute FSM cycle histogram. The five original counters say how much
    // of a layer is DMA and how much overlaps; these say what the engine is
    // doing in the cycles it is not issuing MACs, which is what decides
    // whether a layer is limited by data supply, writeback or setup.
    reg  [31:0] perf_st_mac_cnt;   // row / depthwise MAC issue
    reg  [31:0] perf_st_wait_cnt;  // gather, weight stream, DB wait
    reg  [31:0] perf_st_wb_cnt;    // PPU and SRAM stores
    reg  [31:0] perf_st_pool_cnt;  // pool read / accumulate / divide
    reg  [31:0] perf_st_add_cnt;   // elementwise add
    reg  [31:0] perf_st_rsz_cnt;   // resize
    reg  [31:0] perf_st_cfg_cnt;   // layer, tile and oc-group setup
    reg  [31:0] perf_st_oth_cnt;   // anything not classified above

    // Second-level split of the wait bucket. The three sources need
    // completely different fixes: activation supply wants deeper
    // pipelining, weight streaming wants residency, tile switching is a
    // scheduling problem.
    reg  [31:0] perf_wait_act_cnt; // row gather filling, pool/add act reads
    reg  [31:0] perf_wait_wgt_cnt; // weight stream and oc-group reload
    reg  [31:0] perf_wait_db_cnt;  // tile ping-pong handshake
    reg  [31:0] perf_wait_store_cnt; // ... of which waiting for the DDR writeback
    // Cycles inside the MAC state that did not issue a MAC. Non-zero here
    // means the gather is not running ahead of the array.
    reg  [31:0] perf_mac_idle_cnt;
    wire        ctrl_store_phase;

    // State groups mirror the npu_compute localparams. Keeping the numbers
    // here avoids threading the FSM enum through the hierarchy.
    wire st_is_mac  = (compute_state == 7'd70)   // S_ROW_MAC
                   || (compute_state == 7'd20);  // S_DW_COMPUTE
    wire st_is_wait = (compute_state == 7'd74)   // S_ROW_FILL
                   || (compute_state == 7'd68)   // S_ROW_WGT
                   || (compute_state == 7'd62)   // S_TILE_WAIT_DB
                   || (compute_state == 7'd63);  // S_WAIT_WGT_RELOAD
    wire st_is_wb   = (compute_state == 7'd71)   // S_ROW_PUSH
                   || (compute_state == 7'd72)   // S_ROW_TAIL
                   || (compute_state == 7'd21)   // S_DW_DRAIN
                   || (compute_state == 7'd25)   // S_DW_PPU_WAIT
                   || (compute_state == 7'd35)   // S_POOL_PPU
                   || (compute_state == 7'd36)   // S_POOL_PPU_WAIT
                   || (compute_state == 7'd37)   // S_POOL_WB
                   || (compute_state == 7'd45)   // S_ADD_PPU
                   || (compute_state == 7'd46)   // S_ADD_PPU_WAIT
                   || (compute_state == 7'd47)   // S_ADD_WB
                   || (compute_state == 7'd57)   // S_RESIZE_PPU
                   || (compute_state == 7'd58)   // S_RESIZE_PPU_WAIT
                   || (compute_state == 7'd59)   // S_RESIZE_WB
                   || (compute_state == 7'd78)   // S_ADD_VEC_WAIT
                   || (compute_state == 7'd79)   // S_ADD_VEC_WB
                   || (compute_state == 7'd84)   // S_POOL_VEC_WAIT
                   || (compute_state == 7'd85);  // S_POOL_VEC_WB
    wire st_is_pool = (compute_state == 7'd81)   // S_POOL_VEC_READ
                   || (compute_state == 7'd82)   // S_POOL_VEC_ACC
                   || (compute_state == 7'd87)   // S_POOL_VEC_DIV
                   || (compute_state == 7'd32)   // S_POOL_READ
                   || (compute_state == 7'd33)   // S_POOL_ACC
                   || (compute_state == 7'd34);  // S_POOL_DIV
    wire st_is_add  = (compute_state == 7'd75)   // S_ADD_VEC_READ_A
                   || (compute_state == 7'd76)   // S_ADD_VEC_READ_B
                   || (compute_state == 7'd77)   // S_ADD_VEC_COMPUTE
                   || (compute_state == 7'd42)   // S_ADD_READ_A
                   || (compute_state == 7'd43)   // S_ADD_READ_B
                   || (compute_state == 7'd44);  // S_ADD_COMPUTE
    wire st_is_rsz  = (compute_state >= 7'd49 && compute_state <= 7'd61)
                   || (compute_state == 7'd64)
                   || (compute_state == 7'd65);
    wire st_is_cfg  = (compute_state == 7'd1)    // S_TILE_SETUP
                   || (compute_state == 7'd2)    // S_OC_SETUP
                   || (compute_state == 7'd16)   // S_OC_NEXT
                   || (compute_state == 7'd17)   // S_TILE_NEXT
                   || (compute_state == 7'd18)   // S_DONE
                   || (compute_state == 7'd19)   // S_DW_WGT_LOAD
                   || (compute_state == 7'd22)   // S_DW_PARAM
                   || (compute_state == 7'd30)   // S_POOL_SETUP
                   || (compute_state == 7'd31)   // S_POOL_CH_SETUP
                   || (compute_state == 7'd39)   // S_POOL_CH_NEXT
                   || (compute_state == 7'd40)   // S_ADD_SETUP
                   || (compute_state == 7'd41)   // S_ADD_PARAM
                   || (compute_state == 7'd48)   // S_ADD_NEXT
                   || (compute_state == 7'd73)   // S_ROW_GRP
                   || (compute_state == 7'd80)   // S_POOL_VEC_PARAM
                   || (compute_state == 7'd86);  // S_POOL_VEC_NEXT
    wire st_active  = compute_active;
    wire st_cls     = st_is_mac | st_is_wait | st_is_wb | st_is_pool
                    | st_is_add | st_is_rsz | st_is_cfg;

    // Split of the wait bucket by what the engine is blocked on.
    wire wt_wgt = (compute_state == 7'd68)   // S_ROW_WGT
               || (compute_state == 7'd63);  // S_WAIT_WGT_RELOAD
    wire wt_db  = (compute_state == 7'd62);  // S_TILE_WAIT_DB
    wire wt_act = st_is_wait & ~wt_wgt & ~wt_db;
    wire wt_store = wt_db & ctrl_store_phase;
    // A MAC-state cycle with no MAC issue: the array is starved.
    wire mac_idle = st_is_mac & ~mac_fire;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_cycle_cnt <= 32'd0;
            mac_cnt        <= 32'd0;
            perf_dma_cnt   <= 32'd0;
            perf_compute_cnt <= 32'd0;
            perf_overlap_cnt <= 32'd0;
            compute_active <= 1'b0;
            perf_st_mac_cnt  <= 32'd0;
            perf_st_wait_cnt <= 32'd0;
            perf_st_wb_cnt   <= 32'd0;
            perf_st_pool_cnt <= 32'd0;
            perf_st_add_cnt  <= 32'd0;
            perf_st_rsz_cnt  <= 32'd0;
            perf_st_cfg_cnt  <= 32'd0;
            perf_st_oth_cnt  <= 32'd0;
            perf_wait_act_cnt <= 32'd0;
            perf_wait_wgt_cnt <= 32'd0;
            perf_wait_db_cnt  <= 32'd0;
            perf_mac_idle_cnt <= 32'd0;
            perf_wait_store_cnt <= 32'd0;
        end else if (ctrl_soft_rst) begin
            perf_cycle_cnt <= 32'd0;
            mac_cnt        <= 32'd0;
            perf_dma_cnt   <= 32'd0;
            perf_compute_cnt <= 32'd0;
            perf_overlap_cnt <= 32'd0;
            compute_active <= 1'b0;
            perf_st_mac_cnt  <= 32'd0;
            perf_st_wait_cnt <= 32'd0;
            perf_st_wb_cnt   <= 32'd0;
            perf_st_pool_cnt <= 32'd0;
            perf_st_add_cnt  <= 32'd0;
            perf_st_rsz_cnt  <= 32'd0;
            perf_st_cfg_cnt  <= 32'd0;
            perf_st_oth_cnt  <= 32'd0;
            perf_wait_act_cnt <= 32'd0;
            perf_wait_wgt_cnt <= 32'd0;
            perf_wait_db_cnt  <= 32'd0;
            perf_mac_idle_cnt <= 32'd0;
            perf_wait_store_cnt <= 32'd0;
        end else begin
            if (compute_start)
                compute_active <= 1'b1;
            if (compute_done)
                compute_active <= 1'b0;
            if (hw_busy) begin
                perf_cycle_cnt <= perf_cycle_cnt + 1;
                if (dma_busy)
                    perf_dma_cnt <= perf_dma_cnt + 1;
                if (compute_active || compute_start)
                    perf_compute_cnt <= perf_compute_cnt + 1;
                if (dma_busy && (compute_active || compute_start))
                    perf_overlap_cnt <= perf_overlap_cnt + 1;
                // Row & DW MACs: mac_lanes products on a mac_fire cycle.
                if (mac_fire)
                    mac_cnt <= mac_cnt + {25'd0, mac_lanes};
            end
            // Histogram runs while the engine is active, so the buckets sum
            // to the compute-active span even when the DMA is the limiter.
            if (st_active) begin
                if (st_is_mac)
                    perf_st_mac_cnt  <= perf_st_mac_cnt + 1;
                else if (st_is_wait)
                    perf_st_wait_cnt <= perf_st_wait_cnt + 1;
                else if (st_is_wb)
                    perf_st_wb_cnt   <= perf_st_wb_cnt + 1;
                else if (st_is_pool)
                    perf_st_pool_cnt <= perf_st_pool_cnt + 1;
                else if (st_is_add)
                    perf_st_add_cnt  <= perf_st_add_cnt + 1;
                else if (st_is_rsz)
                    perf_st_rsz_cnt  <= perf_st_rsz_cnt + 1;
                else if (st_is_cfg)
                    perf_st_cfg_cnt  <= perf_st_cfg_cnt + 1;
                else
                    perf_st_oth_cnt  <= perf_st_oth_cnt + 1;
            end
            if (wt_act)
                perf_wait_act_cnt <= perf_wait_act_cnt + 1;
            if (wt_wgt)
                perf_wait_wgt_cnt <= perf_wait_wgt_cnt + 1;
            if (wt_db)
                perf_wait_db_cnt  <= perf_wait_db_cnt + 1;
            if (wt_store)
                perf_wait_store_cnt <= perf_wait_store_cnt + 1;
            if (mac_idle)
                perf_mac_idle_cnt <= perf_mac_idle_cnt + 1;
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // CSR — Wishbone Slave Register File
    // ════════════════════════════════════════════════════════════════════

    npu_csr #(
        .ARRAY_SZ   (ARRAY_SIZE),
        .SPAD_KB_P  (SPAD_KB),
        .PARAM_DEPTH_P(PARAM_DEPTH)
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
        .hw_perf_dma_cnt(perf_dma_cnt),
        .hw_perf_compute_cnt(perf_compute_cnt),
        .hw_perf_overlap_cnt(perf_overlap_cnt),
        .hw_perf_st_mac_cnt (perf_st_mac_cnt),
        .hw_perf_st_wait_cnt(perf_st_wait_cnt),
        .hw_perf_st_wb_cnt  (perf_st_wb_cnt),
        .hw_perf_st_pool_cnt(perf_st_pool_cnt),
        .hw_perf_st_add_cnt (perf_st_add_cnt),
        .hw_perf_st_rsz_cnt (perf_st_rsz_cnt),
        .hw_perf_st_cfg_cnt (perf_st_cfg_cnt),
        .hw_perf_st_oth_cnt (perf_st_oth_cnt),
        .hw_perf_wait_act_cnt(perf_wait_act_cnt),
        .hw_perf_wait_wgt_cnt(perf_wait_wgt_cnt),
        .hw_perf_wait_db_cnt (perf_wait_db_cnt),
        .hw_perf_mac_idle_cnt(perf_mac_idle_cnt),
        .hw_perf_wait_store_cnt(perf_wait_store_cnt),
        .hw_next_param_ready(next_param_ready),
        .hw_super_slot_en(super_slot_en),
        .hw_super_slot   (super_slot_select),
        // Control outputs
        .ctrl_start     (ctrl_start),
        .ctrl_abort     (ctrl_abort),
        .ctrl_soft_rst  (ctrl_soft_rst),
        .ctrl_auto_next (ctrl_auto_next),
        .ctrl_super_req (ctrl_super_req),
        .ctrl_super_has_add(ctrl_super_has_add),
        .ctrl_super_stream_param(ctrl_super_stream_param),
        .ctrl_super_slot(ctrl_super_slot),
        .ctrl_super_slot_valid(ctrl_super_slot_valid),
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
        .reg_dma_tile_in_size(reg_dma_tile_in_size),
        .reg_dma_tile_out_size(reg_dma_tile_out_size),
        .reg_dma_store_mode (reg_dma_store_mode),
        .reg_dma_row_cfg    (reg_dma_row_cfg),
        .reg_dma_wgt_per_oc (reg_dma_wgt_per_oc),
        .reg_wgt_layout     (reg_wgt_layout),
        .reg_tile_in_hw      (reg_tile_in_hw),
        .reg_next_param_addr (reg_next_param_addr),
        .reg_next_param_cfg  (reg_next_param_cfg),
        .reg_param_bank_ctrl (reg_param_bank_ctrl),
        .reg_wgt_sram_base   (reg_wgt_sram_base),
        .reg_param_sram_base (reg_param_sram_base),
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

    npu_ctrl #(
        .IFM_DEPTH   (IFM_DEPTH),
        .OFM_DEPTH   (OFM_DEPTH),
        .WGT_DEPTH   (WGT_DEPTH),
        .PARAM_DEPTH (PARAM_DEPTH)
    ) u_ctrl (
        .clk            (clk),
        .rst_n          (rst_n),
        // CSR interface
        .ctrl_start     (ctrl_start),
        .ctrl_abort     (ctrl_abort),
        .ctrl_soft_rst  (ctrl_soft_rst),
        .ctrl_super_req (ctrl_super_req),
        .ctrl_super_has_add(ctrl_super_has_add),
        .ctrl_super_stream_param(ctrl_super_stream_param),
        .ctrl_super_slot(ctrl_super_slot),
        .ctrl_super_slot_valid(ctrl_super_slot_valid),
        .super_slot_en   (super_slot_en),
        .super_slot_select(super_slot_select),
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
        .dma_busy       (rd_dma_busy),
        .dma_done       (dma_done),
        // Dedicated Store DMA control
        .st_dma_en          (1'b1),
        .st_dma_start       (st_dma_start),
        .st_dma_ext_addr    (st_dma_ext_addr),
        .st_dma_sram_addr   (st_dma_sram_addr),
        .st_dma_xfer_len    (st_dma_xfer_len),
        .st_dma_row_len     (st_dma_row_len),
        .st_dma_row_count   (st_dma_row_count),
        .st_dma_src_row_len (st_dma_src_row_len),
        .st_dma_out_stride  (st_dma_out_stride),
        .st_dma_busy        (wr_dma_busy),
        .st_dma_done        (wr_dma_done),
        .st_dma_active      (st_dma_active),
        // Compute control
        .compute_start  (compute_start),
        .compute_single_tile_en(compute_single_tile_en),
        .compute_external_tile_y(compute_external_tile_y),
        .compute_external_tile_x(compute_external_tile_x),
        .compute_external_tile_h(compute_external_tile_h),
        .compute_external_tile_w(compute_external_tile_w),
        .compute_external_origin_en(compute_external_origin_en),
        .compute_external_origin_h(compute_external_origin_h),
        .compute_external_origin_w(compute_external_origin_w),
        .compute_external_tile_in_w(compute_external_tile_in_w),
        .compute_halo_expand_en(compute_halo_expand_en),
        .compute_done   (compute_done),
        .oc_group_done  (oc_group_done),
        .oc_group_idx   (oc_group_idx),
        .wgt_k_start    (wgt_k_start),
        .wgt_reload_done(wgt_reload_done),
        .group_major_en (group_major_en),
        .dma_sram_sel   (dma_sram_sel),
        .act_role_swap  (act_role_swap),
        // Layer configuration from CSR
        .cfg_dma_in_addr    (reg_dma_in_addr),
        .cfg_dma_out_addr   (reg_dma_out_addr),
        .cfg_dma_wgt_addr   (reg_dma_wgt_addr),
        .cfg_dma_param_addr (reg_dma_param_addr),
        .cfg_dma_in_size    (reg_dma_in_size),
        .cfg_dma_wgt_size   (reg_dma_wgt_size),
        .cfg_wgt_sram_base  (reg_wgt_sram_base),
        .cfg_param_sram_base(reg_param_sram_base),
        .cfg_dma_wgt_per_oc (reg_dma_wgt_per_oc),
        .cfg_wgt_layout     (reg_wgt_layout[0]),
        .cfg_dma_out_size   (reg_dma_out_size),
        .cfg_tile_in_size   (reg_dma_tile_in_size),
        .cfg_param_count    (reg_post_param_count[15:0]),
        .cfg_dma_ctrl       (reg_dma_ctrl),
        .cfg_dma_in_stride  (reg_dma_in_stride),
        .cfg_dma_out_stride (reg_dma_out_stride),
        .cfg_layer_mode     (reg_layer_mode),
        .cfg_out_base       ({{(16-ACT_ADDR_W){1'b0}},
                              reg_sram_base[ACT_ADDR_W+16-1:16]}),
        .cfg_dma_add_b_addr (reg_dma_add_b_addr),
        .cfg_param_preloaded(reg_param_bank_ctrl[1]),
        .cfg_param_bank     (reg_param_bank_ctrl[0]),
        .cfg_next_param_en  (reg_next_param_cfg[31]),
        .cfg_next_param_addr(reg_next_param_addr),
        .cfg_next_param_words(reg_next_param_cfg[15:0]),
        .cfg_next_param_bank(reg_param_bank_ctrl[2]),
        .next_param_ready   (next_param_ready),
        .compute_param_base (compute_param_base),
        .compute_wgt_base   (compute_wgt_base),
        // Per-tile store (2D DMA for NHWC)
        .cfg_dma_store_mode    (reg_dma_store_mode),
        .cfg_dma_tile_out_size (reg_dma_tile_out_size),
        .cfg_out_w             (reg_out_dim_hw[15:0]),
        .cfg_out_h             (reg_out_dim_hw[31:16]),
        .cfg_out_c             (reg_out_dim_c[15:0]),
        .cfg_in_w              (reg_in_dim_hw[15:0]),
        .cfg_in_h              (reg_in_dim_hw[31:16]),
        .cfg_in_c              (reg_in_dim_c[15:0]),
        .cfg_stride_h          (reg_stride[7:0]),
        .cfg_stride_w          (reg_stride[15:8]),
        .cfg_kernel_h          (reg_kernel_size[7:0]),
        .cfg_kernel_w          (reg_kernel_size[15:8]),
        .cfg_dilation_h        (eff_dilation_h),
        .cfg_dilation_w        (eff_dilation_w),
        .cfg_pad_top           (reg_padding[7:0]),
        .cfg_pad_left          (reg_padding[15:8]),
        .cfg_pool_cfg          (reg_pool_cfg),
        .cfg_tile_h            (reg_tile_cfg[15:0]),
        .cfg_tile_w            (reg_tile_cfg[31:16]),
        .cfg_tile_num_h        (reg_tile_count[15:0]),
        .cfg_tile_num_w        (reg_tile_count[31:16]),
        .cfg_int16             (reg_post_ctrl[7]),
        .tile_out_h_actual     (tile_out_h_actual),
        .tile_out_w_actual     (tile_out_w_actual),
        .dma_row_len           (dma_row_len),
        .dma_row_count         (dma_row_count),
        .dma_src_row_len       (dma_src_row_len),
        .dma_out_stride        (dma_out_stride),
        // Double-Buffer
        .ping_pong_flag     (ping_pong_flag),
        .db_prefetch_done   (db_prefetch_done),
        .ctrl_store_phase   (ctrl_store_phase),
        .tile_done          (tile_done),
        .cfg_act_bank_offset({1'b0, IFM_DEPTH[15:1]}),
        .cfg_ofm_bank_offset({1'b0, OFM_DEPTH[15:1]}),
        // Auto-restart
        .ctrl_auto_next  (ctrl_auto_next),
        .cfg_layer_count (reg_layer_count)
    );

    // ════════════════════════════════════════════════════════════════════
    // Decoupled Read/Write DMA Engines + Burst-Atomic Wishbone Arbiter
    // ════════════════════════════════════════════════════════════════════

    wire        rd_wb_cyc, rd_wb_stb, rd_wb_we;
    wire [31:0] rd_wb_adr, rd_wb_dat;
    wire [3:0]  rd_wb_sel;
    wire [2:0]  rd_wb_cti;
    wire        rd_wb_ack;

    wire        wr_wb_cyc, wr_wb_stb, wr_wb_we;
    wire [31:0] wr_wb_adr, wr_wb_dat;
    wire [3:0]  wr_wb_sel;
    wire [2:0]  wr_wb_cti;
    wire        wr_wb_ack;

    wire        wr_use_bg = st_dma_active || st_dma_start;
    assign      dma_busy  = rd_dma_busy || wr_dma_busy;
    assign      dma_done  = rd_dma_done || (!st_dma_active && wr_dma_done);

    npu_dma #(
        .ADDR_W      (32),
        .DATA_W      (32),
        .SRAM_ADDR_W (16)
    ) u_dma_rd (
        .clk            (clk),
        .rst_n          (engine_rst_n),
        // Load channel (DDR → SRAM Port A)
        .start          (dma_start && !dma_dir),
        .abort          (ctrl_abort),
        .dir            (1'b0),
        .ext_addr       (dma_ext_addr),
        .sram_addr      (dma_sram_addr),
        .xfer_len       (dma_xfer_len),
        .burst_cfg      (reg_dma_ctrl[5:4]),
        .cfg_in_stride  (reg_dma_in_stride),
        .cfg_out_stride (dma_out_stride),
        .cfg_row_len    (dma_row_len),
        .cfg_row_count  (dma_row_count),
        .cfg_src_row_len(dma_src_row_len),
        .busy           (rd_dma_busy),
        .done_pulse     (rd_dma_done),
        .xfer_count     (),
        // Wishbone master (read channel)
        .wb_cyc_o       (rd_wb_cyc),
        .wb_stb_o       (rd_wb_stb),
        .wb_we_o        (rd_wb_we),
        .wb_adr_o       (rd_wb_adr),
        .wb_dat_o       (rd_wb_dat),
        .wb_sel_o       (rd_wb_sel),
        .wb_cti_o       (rd_wb_cti),
        .wb_ack_i       (rd_wb_ack),
        .wb_dat_i       (wb_mst_dat_i),
        // SRAM Port A write interface
        .sram_en        (dma_sram_en),
        .sram_we        (dma_sram_we),
        .sram_addr_o    (dma_sram_addr_o),
        .sram_wdata     (dma_sram_wdata),
        .sram_rdata     (dma_sram_rdata),
        .sram_wide_en   (),
        .sram_wide_addr (),
        .sram_rdata_wide({`SRAM_B_WIDTH{1'b0}}),
        .sram_wide_wait (1'b0)
    );

    npu_dma #(
        .ADDR_W      (32),
        .DATA_W      (32),
        .SRAM_ADDR_W (16)
    ) u_dma_wr (
        .clk            (clk),
        .rst_n          (engine_rst_n),
        // Store channel (OFM Wide Port B → DDR)
        .start          (st_dma_start || (!st_dma_active && dma_start && dma_dir)),
        .abort          (ctrl_abort),
        .dir            (1'b1),
        .ext_addr       (wr_use_bg ? st_dma_ext_addr    : dma_ext_addr),
        .sram_addr      (wr_use_bg ? st_dma_sram_addr   : dma_sram_addr),
        .xfer_len       (wr_use_bg ? st_dma_xfer_len    : dma_xfer_len),
        .burst_cfg      (reg_dma_ctrl[5:4]),
        .cfg_in_stride  (32'd0),
        .cfg_out_stride (wr_use_bg ? st_dma_out_stride  : dma_out_stride),
        .cfg_row_len    (wr_use_bg ? st_dma_row_len     : dma_row_len),
        .cfg_row_count  (wr_use_bg ? st_dma_row_count   : dma_row_count),
        .cfg_src_row_len(wr_use_bg ? st_dma_src_row_len : dma_src_row_len),
        .busy           (wr_dma_busy),
        .done_pulse     (wr_dma_done),
        .xfer_count     (),
        // Wishbone master (write channel)
        .wb_cyc_o       (wr_wb_cyc),
        .wb_stb_o       (wr_wb_stb),
        .wb_we_o        (wr_wb_we),
        .wb_adr_o       (wr_wb_adr),
        .wb_dat_o       (wr_wb_dat),
        .wb_sel_o       (wr_wb_sel),
        .wb_cti_o       (wr_wb_cti),
        .wb_ack_i       (wr_wb_ack),
        .wb_dat_i       (wb_mst_dat_i),
        // SRAM Wide Port B read interface
        .sram_en        (),
        .sram_we        (),
        .sram_addr_o    (),
        .sram_wdata     (),
        .sram_rdata     (dma_sram_rdata),
        .sram_wide_en   (dma_sram_wide_en),
        .sram_wide_addr (dma_sram_wide_addr),
        .sram_rdata_wide(dma_sram_rdata_wide),
        .sram_wide_wait (compute_uses_ofm_b)
    );

    // Zero-latency, burst-atomic round-robin Wishbone B4 arbiter
    reg  wb_locked;
    reg  wb_owner_wr;
    reg  wb_rr_last_wr;
    wire cur_owner_active = wb_locked && (wb_owner_wr ? wr_wb_cyc : rd_wb_cyc);
    wire grant_wr = cur_owner_active ? wb_owner_wr
                  : (rd_wb_cyc && wr_wb_cyc) ? ~wb_rr_last_wr
                  : wr_wb_cyc;
    wire any_wb_req = rd_wb_cyc || wr_wb_cyc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_locked     <= 1'b0;
            wb_owner_wr   <= 1'b0;
            wb_rr_last_wr <= 1'b0;
        end else if (ctrl_soft_rst || ctrl_abort) begin
            wb_locked     <= 1'b0;
            wb_owner_wr   <= 1'b0;
            wb_rr_last_wr <= 1'b0;
        end else begin
            wb_locked   <= any_wb_req;
            wb_owner_wr <= grant_wr;
            if (any_wb_req)
                wb_rr_last_wr <= grant_wr;
        end
    end

    assign wb_mst_cyc_o = grant_wr ? wr_wb_cyc : rd_wb_cyc;
    assign wb_mst_stb_o = grant_wr ? wr_wb_stb : rd_wb_stb;
    assign wb_mst_we_o  = grant_wr ? wr_wb_we  : rd_wb_we;
    assign wb_mst_adr_o = grant_wr ? wr_wb_adr : rd_wb_adr;
    assign wb_mst_dat_o = grant_wr ? wr_wb_dat : rd_wb_dat;
    assign wb_mst_sel_o = grant_wr ? wr_wb_sel : rd_wb_sel;
    assign wb_mst_cti_o = grant_wr ? wr_wb_cti : rd_wb_cti;
    assign rd_wb_ack    = !grant_wr && wb_mst_ack_i;
    assign wr_wb_ack    =  grant_wr && wb_mst_ack_i;

    // ════════════════════════════════════════════════════════════════════
    // SRAM Banks
    // ════════════════════════════════════════════════════════════════════

    // Address width calculations
    localparam ACT_ADDR_W   = $clog2(
        (IFM_DEPTH > OFM_DEPTH) ? IFM_DEPTH : OFM_DEPTH);
    localparam IFM_ADDR_W   = $clog2(IFM_DEPTH);
    localparam WGT_ADDR_W   = $clog2(WGT_DEPTH);
    localparam PARAM_ADDR_W = $clog2(PARAM_DEPTH);

    // --- IFM / OFM SRAMs (phys0 = u_sram_act, phys1 = u_sram_ofm) ---
    // Port A: 32-bit DMA (load → IFM, store → OFM; concat preload → OFM)
    // Port B: IFM 256-bit read; OFM 256-bit RMW read + 32-bit write
    // act_role_swap swaps the two physical banks so a fused next layer
    // reads the previous OFM without a memcpy.
    wire                    act_a_en, act_a_we;
    wire [ACT_ADDR_W-1:0]  act_a_addr;
    wire [31:0]             act_a_wdata, act_a_rdata;
    wire                    ofm_a_en, ofm_a_we;
    wire [ACT_ADDR_W-1:0]  ofm_a_addr;
    wire [31:0]             ofm_a_wdata, ofm_a_rdata;
    wire                    act_b_en;
    wire [ACT_ADDR_W-1:0]  act_b_addr;
    wire [`SRAM_B_WIDTH-1:0] act_b_rdata;
    wire                    ofm_b_en;
    wire [ACT_ADDR_W-1:0]  ofm_b_addr;
    wire [`SRAM_B_WIDTH-1:0] ofm_b_rdata;
    wire                    act_phys0_b_we, ofm_phys1_b_we;

    npu_sram_wide #(
        .DEPTH  (IFM_DEPTH)
    ) u_sram_act (
        .clk    (clk),
        .a_en   (act_a_en),
        .a_we   (act_a_we),
        .a_addr (act_a_addr[IFM_ADDR_W-1:0]),
        .a_wdata(act_a_wdata),
        .a_rdata(act_a_rdata),
        .b_en   (act_b_en),
        .b_we   (act_phys0_b_we),
        .b_addr (act_b_addr[IFM_ADDR_W-1:0]),
        .b_wdata(act_b_wr_data),
        .b_wmask(act_phys0_b_we ? act_b_wr_mask : 64'd0),
        .b_wdata_wide(act_b_wr_wide),
        .b_rdata(act_b_rdata)
    );

    npu_sram_wide #(
        .DEPTH  (OFM_DEPTH)
    ) u_sram_ofm (
        .clk    (clk),
        .a_en   (ofm_a_en),
        .a_we   (ofm_a_we),
        .a_addr (ofm_a_addr),
        .a_wdata(ofm_a_wdata),
        .a_rdata(ofm_a_rdata),
        .b_en   (ofm_b_en),
        .b_we   (ofm_phys1_b_we),
        .b_addr (ofm_b_addr),
        .b_wdata(act_b_wr_data),
        .b_wmask(ofm_phys1_b_we ? act_b_wr_mask : 64'd0),
        .b_wdata_wide(act_b_wr_wide),
        .b_rdata(ofm_b_rdata)
    );

    // --- Weight SRAM ---
    // Port A: 32-bit DMA access (load weights)
    // Port B: 256-bit compute read
    wire                    wgt_a_en, wgt_a_we;
    wire [WGT_ADDR_W-1:0]  wgt_a_addr;
    wire [31:0]             wgt_a_wdata, wgt_a_rdata;
    wire                    wgt_b_en;
    wire [WGT_ADDR_W-1:0]  wgt_b_addr;
    wire [`SRAM_B_WIDTH-1:0] wgt_b_rdata;

    npu_sram_wide #(
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
        .b_wmask(64'd0),
        .b_wdata_wide({`SRAM_B_WIDTH{1'b0}}),
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
    // During act load: DMA writes to IFM
    // During param load: DMA writes to parameter SRAM
    // During output store: DMA reads from OFM
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

    // SRAM bank select: latch on dma_start (same timing as old address-compare)
    // Controller sets dma_sram_sel in same cycle as dma_start (both non-blocking)
    reg [1:0] dma_bank_sel;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dma_bank_sel <= 2'd1;  // default: activation
        else if (ctrl_soft_rst || ctrl_abort)
            dma_bank_sel <= 2'd1;
        else if (dma_start && !dma_dir)
            dma_bank_sel <= dma_sram_sel;  // latch at load start, takes effect next cycle
    end

    // Logical IFM/OFM Port A (DMA). act_role_swap maps them onto phys0/phys1.
    wire ifm_dma_en = dma_sram_en && (dma_bank_sel == 2'd1);
    wire ofm_dma_en = dma_sram_en && (dma_bank_sel == 2'd3);

    assign act_a_en    = act_role_swap ? ofm_dma_en : ifm_dma_en;
    assign act_a_we    = dma_sram_we;
    assign act_a_addr  = dma_sram_addr_o[ACT_ADDR_W-1:0];
    assign act_a_wdata = dma_sram_wdata;

    assign ofm_a_en    = act_role_swap ? ifm_dma_en : ofm_dma_en;
    assign ofm_a_we    = dma_sram_we;
    assign ofm_a_addr  = dma_sram_addr_o[ACT_ADDR_W-1:0];
    assign ofm_a_wdata = dma_sram_wdata;

    assign wgt_a_en    = dma_sram_en && (dma_bank_sel == 2'd0);
    assign wgt_a_we    = dma_sram_we;
    assign wgt_a_addr  = dma_sram_addr_o[WGT_ADDR_W-1:0];
    assign wgt_a_wdata = dma_sram_wdata;

    assign param_a_en    = dma_sram_en && (dma_bank_sel == 2'd2);
    assign param_a_we    = dma_sram_we;
    assign param_a_addr  = dma_sram_addr_o[PARAM_ADDR_W-1:0];
    assign param_a_wdata = dma_sram_wdata;

    // DMA read data mux (store reads OFM = sel 3)
    wire [31:0] ofm_dma_rdata = act_role_swap ? act_a_rdata : ofm_a_rdata;
    assign dma_sram_rdata = (dma_bank_sel == 2'd0) ? wgt_a_rdata :
                            (dma_bank_sel == 2'd2) ? param_a_rdata :
                                                     ofm_dma_rdata;

    // ════════════════════════════════════════════════════════════════════
    // PPU — Post-Processing Unit (8-lane time-multiplexed bank)
    // ════════════════════════════════════════════════════════════════════

    wire signed [`DATA_WIDTH-1:0] ppu_out_data;
    wire                          ppu_out_valid;
    wire signed [`ACC_WIDTH-1:0]  ppu_acc_in;
    wire                          ppu_in_valid;
    wire signed [`ACC_WIDTH-1:0]  ppu_bias;
    wire [`PARAM_M_BITS-1:0]      ppu_mult_m;
    wire [`PARAM_S_BITS-1:0]      ppu_shift_s;
    wire signed [`PARAM_ZP_BITS-1:0] ppu_zero_point;

    wire [`ACC_WIDTH*`PPU_LANES-1:0]  ppu_acc_w, ppu_bias_w;
    wire [`PPU_LANES-1:0]              ppu_valid_w, ppu_vout_w;
    wire [15*`PPU_LANES-1:0]           ppu_mult_w;
    wire [6*`PPU_LANES-1:0]            ppu_shift_w;
    wire [16*`PPU_LANES-1:0]           ppu_zp_w;
    wire [`DATA_WIDTH*`PPU_LANES-1:0] ppu_out_w;

    npu_ppu_bank #(
        .N       (`PPU_LANES),
        .ACC_W   (`ACC_WIDTH),
        .DATA_W  (`DATA_WIDTH),
        .BIAS_W  (`BIAS_WIDTH),
        .MULT_W  (`PARAM_M_BITS),
        .SHIFT_W (`PARAM_S_BITS),
        .ZP_W    (`PARAM_ZP_BITS)
    ) u_ppu_bank (
        .clk        (clk),
        .rst_n      (engine_rst_n),
        .mode       (reg_post_ctrl[1:0]),
        .relu_en    (reg_post_ctrl[2]),
        .relu6_en   (reg_post_ctrl[3]),
        .bias_en    (reg_post_ctrl[6]),
        .zp_en      (reg_post_ctrl[5]),
        .int16_mode (reg_post_ctrl[7]),
        .clamp_max  (reg_post_clamp[`PARAM_ZP_BITS-1:0]),
        .acc_wide   (ppu_acc_w),
        .valid_wide (ppu_valid_w),
        .bias_wide  (ppu_bias_w),
        .mult_wide  (ppu_mult_w),
        .shift_wide (ppu_shift_w),
        .zp_wide    (ppu_zp_w),
        .acc_s      (ppu_acc_in),
        .valid_s    (ppu_in_valid),
        .bias_s     (ppu_bias),
        .mult_s     (ppu_mult_m),
        .shift_s    (ppu_shift_s),
        .zp_s       (ppu_zero_point),
        .out_wide   (ppu_out_w),
        .vout_wide  (ppu_vout_w),
        .out_s      (ppu_out_data),
        .vout_s     (ppu_out_valid)
    );

    // ════════════════════════════════════════════════════════════════════
    // Compute Micro-Sequencer (V2)
    // ════════════════════════════════════════════════════════════════════

    wire        compute_done_w;
    wire        act_b_wr_en;
    wire [ACT_ADDR_W-1:0] act_b_wr_addr;
    wire [31:0] act_b_wr_data;
    wire [63:0] act_b_wr_mask;
    wire [`SRAM_B_WIDTH-1:0] act_b_wr_wide;
    wire        act_b_rd_en;
    wire [ACT_ADDR_W-1:0] act_b_rd_addr;
    wire        act_rd_ofm;
    wire [`SRAM_B_WIDTH-1:0] act_rd_mux;

    // Compute engine handles full pipeline (compute + PPU + writeback) internally
    assign compute_done = compute_done_w;

    // DB_EN places the current tile in one half of the bank compute is
    // reading. After a fusion swap that bank is the physical OFM, whose
    // half is larger than the IFM half.
    wire [ACT_ADDR_W-1:0] ifm_db_half = act_role_swap
        ? OFM_DEPTH[ACT_ADDR_W-1:0] >> 1
        : IFM_DEPTH[ACT_ADDR_W-1:0] >> 1;
    wire [ACT_ADDR_W-1:0] ofm_db_half = act_role_swap
        ? IFM_DEPTH[ACT_ADDR_W-1:0] >> 1
        : OFM_DEPTH[ACT_ADDR_W-1:0] >> 1;
    wire [ACT_ADDR_W-1:0] effective_act_base = reg_sram_base[ACT_ADDR_W-1:0]
        + (db_en_is_active && ping_pong_flag ? ifm_db_half : {ACT_ADDR_W{1'b0}});
    // Store DMA adds this same OFM half. The IFM half is a different size.
    wire [ACT_ADDR_W-1:0] effective_ofm_base = reg_sram_base[ACT_ADDR_W+16-1:16]
        + (db_en_is_active && ping_pong_flag ? ofm_db_half : {ACT_ADDR_W{1'b0}});

    npu_compute #(
        .ARRAY_SIZE   (ARRAY_SIZE),
        .ACT_ADDR_W   (ACT_ADDR_W),
        .WGT_ADDR_W   (WGT_ADDR_W),
        .PARAM_ADDR_W (PARAM_ADDR_W),
        .DATA_W       (`DATA_WIDTH),
        .ACC_W        (`ACC_WIDTH)
    ) u_compute (
        .clk            (clk),
        .rst_n          (engine_rst_n),
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
        .cfg_dilation_h (eff_dilation_h),
        .cfg_dilation_w (eff_dilation_w),
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
        .cfg_in_zp      (reg_layer_mode[23:8]),
        .cfg_act_base   (effective_act_base),
        .cfg_out_base   (reg_sram_base[ACT_ADDR_W+16-1:16]),
        .cfg_ofm_base   (effective_ofm_base),
        .cfg_wgt_base   (compute_wgt_base[WGT_ADDR_W-1:0]),
        .cfg_pool_cfg   (reg_pool_cfg),
        .cfg_resize_cfg (reg_resize_cfg),
        .cfg_concat_cfg (reg_concat_cfg),
        .cfg_param_base (compute_param_base[PARAM_ADDR_W-1:0]),
        .cfg_2d_load    (npu_2d_load),  // 2D load when in_stride set
        // Weight SRAM Port B
        .wgt_rd_en      (wgt_b_en),
        .wgt_rd_addr    (wgt_b_addr),
        .wgt_rd_data    (wgt_b_rdata),
        // Activation SRAM Port B
        .act_rd_en      (act_b_rd_en),
        .act_rd_addr    (act_b_rd_addr),
        .act_rd_data    (act_rd_mux),
        .act_rd_ofm     (act_rd_ofm),
        .act_wr_en      (act_b_wr_en),
        .act_wr_addr    (act_b_wr_addr),
        .act_wr_data    (act_b_wr_data),
        .act_wr_mask    (act_b_wr_mask),
        .act_wr_wide    (act_b_wr_wide),
        // Param SRAM Port B
        .param_rd_en    (param_b_en),
        .param_rd_addr  (param_b_addr),
        .param_rd_data  (param_b_rdata),
        .mac_fire       (mac_fire),
        .mac_lanes      (mac_lanes),
        // PPU
        .ppu_acc_in     (ppu_acc_in),
        .ppu_in_valid   (ppu_in_valid),
        .ppu_bias       (ppu_bias),
        .ppu_mult_m     (ppu_mult_m),
        .ppu_shift_s    (ppu_shift_s),
        .ppu_zero_point (ppu_zero_point),
        .ppu_out_data   (ppu_out_data),
        .ppu_out_valid  (ppu_out_valid),
        .ppu_acc_w      (ppu_acc_w),
        .ppu_valid_w    (ppu_valid_w),
        .ppu_bias_w     (ppu_bias_w),
        .ppu_mult_w     (ppu_mult_w),
        .ppu_shift_w    (ppu_shift_w),
        .ppu_zp_w       (ppu_zp_w),
        .ppu_out_w      (ppu_out_w),
        .ppu_vout_w     (ppu_vout_w),
        // DB_EN
        .tile_done          (tile_done),
        .oc_group_done      (oc_group_done),
        .oc_group_out       (oc_group_idx),
        .wgt_k_start        (wgt_k_start),
        .wgt_reload_done    (wgt_reload_done),
        .cfg_wgt_per_oc     (reg_dma_wgt_per_oc),
        .cfg_wgt_layout     (reg_wgt_layout[0]),
        .cfg_group_major    (group_major_en),
        .db_prefetch_done   (db_prefetch_done),
        .cfg_single_tile_en (compute_single_tile_en),
        .cfg_external_tile_y(compute_external_tile_y),
        .cfg_external_tile_x(compute_external_tile_x),
        .cfg_external_tile_h_actual(compute_external_tile_h),
        .cfg_external_tile_w_actual(compute_external_tile_w),
        .cfg_external_origin_en(compute_external_origin_en),
        .cfg_external_origin_h(compute_external_origin_h),
        .cfg_external_origin_w(compute_external_origin_w),
        .cfg_external_tile_in_w(compute_external_tile_in_w),
        .cfg_halo_expand_en(compute_halo_expand_en),
        .tile_out_h_actual  (tile_out_h_actual),
        .tile_out_w_actual  (tile_out_w_actual),
        .dbg_state          (compute_state)
    );

    // ─── SRAM Port B: IFM read ∥ OFM write (and OFM RMW read) ───
    // Logical IFM = phys0 when !swap, phys1 when swap.
    // Compute always owns OFM Port B when it writes or RMW-reads; a concurrent
    // background store wide read yields for one cycle via sram_wide_wait.
    wire ifm_b_en = act_b_rd_en && !act_rd_ofm;
    assign compute_uses_ofm_b = (act_b_rd_en && act_rd_ofm) || act_b_wr_en;
    wire ofm_b_en_log = compute_uses_ofm_b || dma_sram_wide_en;
    wire [ACT_ADDR_W-1:0] ofm_b_addr_log
        = act_b_wr_en ? act_b_wr_addr
        : (act_b_rd_en && act_rd_ofm) ? act_b_rd_addr
        : dma_sram_wide_addr[ACT_ADDR_W-1:0];

    assign act_b_en        = act_role_swap ? ofm_b_en_log : ifm_b_en;
    assign act_phys0_b_we  = act_role_swap ? act_b_wr_en : 1'b0;
    assign act_b_addr      = act_role_swap ? ofm_b_addr_log : act_b_rd_addr;

    assign ofm_b_en        = act_role_swap ? ifm_b_en : ofm_b_en_log;
    assign ofm_phys1_b_we  = act_role_swap ? 1'b0 : act_b_wr_en;
    assign ofm_b_addr      = act_role_swap ? act_b_rd_addr : ofm_b_addr_log;

    wire [`SRAM_B_WIDTH-1:0] ifm_b_rdata = act_role_swap ? ofm_b_rdata : act_b_rdata;
    wire [`SRAM_B_WIDTH-1:0] ofm_b_rdata_log = act_role_swap ? act_b_rdata : ofm_b_rdata;
    assign act_rd_mux = act_rd_ofm ? ofm_b_rdata_log : ifm_b_rdata;
    assign dma_sram_rdata_wide = ofm_b_rdata_log;

`ifdef VCD_TRACE
// Minimal VCD: selected row-engine signals only
initial begin
    $dumpfile("/tmp/npu_nano.vcd");
    $dumpvars(0, npu_top.u_compute.act_wr_data);
    $dumpvars(0, npu_top.u_compute.ppu_acc_in);
    $dumpvars(0, npu_top.u_compute.ppu_out_data);
    $dumpvars(0, npu_top.u_compute.mac1d_fire);
    $dumpvars(0, npu_top.u_compute.sp_oh);
    $dumpvars(0, npu_top.u_compute.tile_x);
end
`endif

endmodule
