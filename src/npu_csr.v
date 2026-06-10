// Open-NPU RTL — CSR Register File (Wishbone Slave)
// SPDX-License-Identifier: Apache-2.0
//
// Wishbone B4 single-cycle slave implementing the full register map:
//   Group 0 (0x000-0x03F): Control & Status
//   Group 1 (0x040-0x0FF): Layer parameters
//   Group 2 (0x100-0x17F): DMA configuration
//   Group 3 (0x180-0x1FF): Post-processing configuration
//   Group 4 (0x200-0x3FF): LUT data (256 entries)
//
// Access types: RW, RO, W1C, self-clearing

`include "npu_defines.vh"

module npu_csr #(
    parameter ADDR_W    = 12,   // Internal address width (4KB space)
    parameter DATA_W    = 32,   // Data bus width
    parameter ARRAY_SZ  = `ARRAY_SIZE,
    parameter SPAD_KB_P = `SPAD_KB,
    // Version info (V1.0.0)
    parameter VERSION_MAJOR = 8'd1,
    parameter VERSION_MINOR = 8'd0,
    parameter VERSION_PATCH = 8'd0,
    // Hardware config
    parameter HAS_INT16 = 1,
    parameter HAS_LUT   = 1,
    parameter HAS_IPU   = 0
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ─── Wishbone Slave Interface ───
    input  wire                 wb_cyc_i,
    input  wire                 wb_stb_i,
    input  wire                 wb_we_i,
    input  wire [ADDR_W-1:0]   wb_adr_i,
    input  wire [DATA_W-1:0]   wb_dat_i,
    input  wire [3:0]          wb_sel_i,
    output reg  [DATA_W-1:0]   wb_dat_o,
    output reg                  wb_ack_o,

    // ─── Hardware Status Inputs (from other modules) ───
    input  wire                 hw_busy,        // NPU compute busy
    input  wire                 hw_dma_busy,    // DMA busy
    input  wire                 hw_done,        // Layer done pulse (1 clk)
    input  wire                 hw_error,       // Error pulse (1 clk)
    input  wire                 hw_dma_done,    // DMA done pulse (1 clk)
    input  wire [3:0]          hw_error_code,  // Error code from hardware
    input  wire [7:0]          hw_curr_layer,  // Current layer number
    input  wire [31:0]         hw_perf_cnt,    // Cycle counter
    input  wire [31:0]         hw_mac_cnt,     // MAC counter

    // ─── Control Outputs (to other modules) ───
    output wire                 ctrl_start,     // START pulse
    output wire                 ctrl_abort,     // ABORT pulse
    output wire                 ctrl_soft_rst,  // Soft reset pulse
    output wire                 ctrl_auto_next, // AUTO_NEXT mode

    // ─── Layer Count Output ───
    output wire [7:0]           reg_layer_count, // Total layer count for auto-next

    // ─── Interrupt Output ───
    output wire                 irq_o,

    // ─── Layer Parameter Outputs ───
    output wire [DATA_W-1:0]   reg_layer_mode,
    output wire [DATA_W-1:0]   reg_in_dim_hw,
    output wire [DATA_W-1:0]   reg_in_dim_c,
    output wire [DATA_W-1:0]   reg_out_dim_hw,
    output wire [DATA_W-1:0]   reg_out_dim_c,
    output wire [DATA_W-1:0]   reg_kernel_size,
    output wire [DATA_W-1:0]   reg_stride,
    output wire [DATA_W-1:0]   reg_padding,
    output wire [DATA_W-1:0]   reg_pool_cfg,
    output wire [DATA_W-1:0]   reg_resize_cfg,
    output wire [DATA_W-1:0]   reg_deconv_cfg,
    output wire [DATA_W-1:0]   reg_concat_cfg,
    output wire [DATA_W-1:0]   reg_tile_cfg,
    output wire [DATA_W-1:0]   reg_tile_count,
    output wire [DATA_W-1:0]   reg_sram_base,

    // ─── DMA Config Outputs ───
    output wire [DATA_W-1:0]   reg_dma_in_addr,
    output wire [DATA_W-1:0]   reg_dma_out_addr,
    output wire [DATA_W-1:0]   reg_dma_wgt_addr,
    output wire [DATA_W-1:0]   reg_dma_param_addr,
    output wire [DATA_W-1:0]   reg_dma_in_stride,
    output wire [DATA_W-1:0]   reg_dma_out_stride,
    output wire [DATA_W-1:0]   reg_dma_ctrl,
    output wire [DATA_W-1:0]   reg_dma_add_b_addr,
    output wire [DATA_W-1:0]   reg_dma_add_param_addr,
    output wire [DATA_W-1:0]   reg_dma_in_size,
    output wire [DATA_W-1:0]   reg_dma_wgt_size,
    output wire [DATA_W-1:0]   reg_dma_out_size,
    output wire [DATA_W-1:0]   reg_dma_tile_in_size,

    // ─── Post-Processing Config Outputs ───
    output wire [DATA_W-1:0]   reg_post_ctrl,
    output wire [DATA_W-1:0]   reg_post_param_addr,
    output wire [DATA_W-1:0]   reg_post_param_count,
    output wire [DATA_W-1:0]   reg_post_clamp,
    output wire [DATA_W-1:0]   reg_post_act_cfg,
    output wire [DATA_W-1:0]   reg_post_add_param_addr,
    output wire [DATA_W-1:0]   reg_post_add_input_addr,
    output wire [DATA_W-1:0]   reg_post_add_stride
);

    // ─── Internal address decoding ───
    wire valid_access = wb_cyc_i & wb_stb_i;
    wire [ADDR_W-1:0] addr = wb_adr_i;

    // ─── Group 0: Control & Status Registers ───
    reg [DATA_W-1:0] r_ctrl;           // 0x000 — self-clearing bits
    reg [DATA_W-1:0] r_layer_count;   // 0x030 — total layer count for auto-next
    reg [DATA_W-1:0] r_irq_en;        // 0x008
    reg [DATA_W-1:0] r_irq_status;    // 0x00C — W1C

    // ─── Group 1: Layer Parameters ───
    reg [DATA_W-1:0] r_layer_mode;    // 0x040
    reg [DATA_W-1:0] r_in_dim_hw;     // 0x044
    reg [DATA_W-1:0] r_in_dim_c;      // 0x048
    reg [DATA_W-1:0] r_out_dim_hw;    // 0x04C
    reg [DATA_W-1:0] r_out_dim_c;     // 0x050
    reg [DATA_W-1:0] r_kernel_size;   // 0x054
    reg [DATA_W-1:0] r_stride;        // 0x058
    reg [DATA_W-1:0] r_padding;       // 0x05C
    reg [DATA_W-1:0] r_pool_cfg;      // 0x060
    reg [DATA_W-1:0] r_resize_cfg;    // 0x064
    reg [DATA_W-1:0] r_deconv_cfg;    // 0x068
    reg [DATA_W-1:0] r_concat_cfg;    // 0x06C
    reg [DATA_W-1:0] r_tile_cfg;      // 0x070
    reg [DATA_W-1:0] r_tile_count;    // 0x074
    reg [DATA_W-1:0] r_sram_base;    // 0x078 — act_base[12:0], out_base[28:16]

    // ─── Group 2: DMA Configuration ───
    reg [DATA_W-1:0] r_dma_in_addr;   // 0x100
    reg [DATA_W-1:0] r_dma_out_addr;  // 0x104
    reg [DATA_W-1:0] r_dma_wgt_addr;  // 0x108
    reg [DATA_W-1:0] r_dma_param_addr;// 0x10C (dual-mapped w/ 0x184)
    reg [DATA_W-1:0] r_dma_in_stride; // 0x110
    reg [DATA_W-1:0] r_dma_out_stride;// 0x114
    reg [DATA_W-1:0] r_dma_ctrl;      // 0x118
    reg [DATA_W-1:0] r_dma_add_b_addr;// 0x120 (dual-mapped w/ 0x198)
    reg [DATA_W-1:0] r_dma_add_param; // 0x124 (dual-mapped w/ 0x194)
    reg [DATA_W-1:0] r_dma_in_size;   // 0x128
    reg [DATA_W-1:0] r_dma_wgt_size;  // 0x12C
    reg [DATA_W-1:0] r_dma_out_size;  // 0x130
    reg [DATA_W-1:0] r_dma_tile_in_size; // 0x134 — per-tile input size (bytes), for DB_EN tiled layers

    // ─── Group 3: Post-Processing Configuration ───
    reg [DATA_W-1:0] r_post_ctrl;     // 0x180
    // r_dma_param_addr serves as POST_PARAM_ADDR (0x184) — dual-mapped
    reg [DATA_W-1:0] r_post_param_count; // 0x188
    reg [DATA_W-1:0] r_post_clamp;    // 0x18C
    reg [DATA_W-1:0] r_post_act_cfg;  // 0x190
    // r_dma_add_param serves as POST_ADD_PARAM_ADDR (0x194) — dual-mapped
    // r_dma_add_b_addr serves as POST_ADD_INPUT_ADDR (0x198) — dual-mapped
    reg [DATA_W-1:0] r_post_add_stride; // 0x19C

    // ─── Group 4: LUT Data (256 entries packed 4-per-word for INT8) ───
    reg [DATA_W-1:0] r_lut [0:127];   // 128 words covers both INT8(64) and INT16(128)

    // ─── Read-Only Hardware Config ───
    wire [DATA_W-1:0] hw_version = {8'd0, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH};

    // Compute log2 of DW_CHANNELS
    // DW_CHANNELS = ARRAY_SZ, so log2 depends on ARRAY_SZ
    function [3:0] clog2_val;
        input [7:0] v;
        integer i;
        begin
            clog2_val = 0;
            for (i = 0; i < 8; i = i + 1)
                if (v > (1 << i))
                    clog2_val = i[3:0] + 1;
        end
    endfunction

    wire [3:0] dw_ch_log2 = clog2_val(ARRAY_SZ[7:0]);
    wire [DATA_W-1:0] hw_config = {
        5'd0,                            // [31:27] reserved
        HAS_IPU[0],                      // [26]
        HAS_LUT[0],                      // [25]
        HAS_INT16[0],                    // [24]
        SPAD_KB_P[7:0],                  // [23:16] SPAD_SIZE_4KB (units of 4KB)
        dw_ch_log2,                      // [15:12]
        4'd1,                            // [11:8] NUM_ARRAYS = 1
        ARRAY_SZ[7:0]                    // [7:0]
    };

    // Serial and vendor (hardcoded for open-source)
    wire [DATA_W-1:0] hw_serial_lo = 32'h0000_0001;
    wire [DATA_W-1:0] hw_serial_hi = 32'h0000_0000;
    wire [DATA_W-1:0] hw_vendor_id = {16'h0001, 16'h4F4E}; // Product=0x0001, Vendor="ON"

    // ─── DMA Status (directly from hw inputs) ───
    wire [DATA_W-1:0] dma_status = {16'd0, 8'd0, 5'd0, hw_dma_busy, hw_dma_busy, hw_dma_busy};

    // ─── IRQ generation ───
    // Set on pulse, clear on W1C write
    wire irq_done_set  = hw_done;
    wire irq_error_set = hw_error;
    wire irq_dma_set   = hw_dma_done;

    // IRQ output = any enabled & pending
    assign irq_o = |(r_irq_status & r_irq_en);

    // ─── Control outputs ───
    // Self-clearing: read as 0, pulse for 1 cycle on write
    assign ctrl_start     = r_ctrl[0];
    assign ctrl_abort     = r_ctrl[1];
    assign ctrl_soft_rst  = r_ctrl[2];
    assign ctrl_auto_next = r_ctrl[3];
    assign reg_layer_count = r_layer_count[7:0];

    // ─── Register outputs ───
    assign reg_layer_mode       = r_layer_mode;
    assign reg_in_dim_hw        = r_in_dim_hw;
    assign reg_in_dim_c         = r_in_dim_c;
    assign reg_out_dim_hw       = r_out_dim_hw;
    assign reg_out_dim_c        = r_out_dim_c;
    assign reg_kernel_size      = r_kernel_size;
    assign reg_stride           = r_stride;
    assign reg_padding          = r_padding;
    assign reg_pool_cfg         = r_pool_cfg;
    assign reg_resize_cfg       = r_resize_cfg;
    assign reg_deconv_cfg       = r_deconv_cfg;
    assign reg_concat_cfg       = r_concat_cfg;
    assign reg_tile_cfg         = r_tile_cfg;
    assign reg_tile_count       = r_tile_count;
    assign reg_sram_base        = r_sram_base;
    assign reg_dma_in_addr      = r_dma_in_addr;
    assign reg_dma_out_addr     = r_dma_out_addr;
    assign reg_dma_wgt_addr     = r_dma_wgt_addr;
    assign reg_dma_param_addr   = r_dma_param_addr;
    assign reg_dma_in_stride    = r_dma_in_stride;
    assign reg_dma_out_stride   = r_dma_out_stride;
    assign reg_dma_ctrl         = r_dma_ctrl;
    assign reg_dma_add_b_addr   = r_dma_add_b_addr;
    assign reg_dma_add_param_addr = r_dma_add_param;
    assign reg_dma_in_size      = r_dma_in_size;
    assign reg_dma_wgt_size     = r_dma_wgt_size;
    assign reg_dma_out_size     = r_dma_out_size;
    assign reg_dma_tile_in_size = r_dma_tile_in_size;
    assign reg_post_ctrl        = r_post_ctrl;
    assign reg_post_param_addr  = r_dma_param_addr; // dual-mapped
    assign reg_post_param_count = r_post_param_count;
    assign reg_post_clamp       = r_post_clamp;
    assign reg_post_act_cfg     = r_post_act_cfg;
    assign reg_post_add_param_addr = r_dma_add_param; // dual-mapped
    assign reg_post_add_input_addr = r_dma_add_b_addr; // dual-mapped
    assign reg_post_add_stride  = r_post_add_stride;

    // ─── Status register (assembled from hw inputs) ───
    wire [DATA_W-1:0] status_reg = {16'd0, hw_curr_layer, 4'd0,
                                     hw_done, hw_error, hw_dma_busy, hw_busy};

    // ─── LUT initialization ───
    integer lut_i;

    // ─── Main register write/read logic ───
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_ack_o     <= 1'b0;
            wb_dat_o     <= 32'd0;
            r_ctrl       <= 32'd0;
            r_layer_count<= 32'd0;
            r_irq_en     <= 32'd0;
            r_irq_status <= 32'd0;
            r_layer_mode <= 32'd0;
            r_in_dim_hw  <= 32'd0;
            r_in_dim_c   <= 32'd0;
            r_out_dim_hw <= 32'd0;
            r_out_dim_c  <= 32'd0;
            r_kernel_size<= 32'd0;
            r_stride     <= 32'd0;
            r_padding    <= 32'd0;
            r_pool_cfg   <= 32'd0;
            r_resize_cfg <= 32'd0;
            r_deconv_cfg <= 32'd0;
            r_concat_cfg <= 32'd0;
            r_tile_cfg   <= 32'd0;
            r_tile_count <= 32'd0;
            r_sram_base  <= 32'd0;
            r_dma_in_addr   <= 32'd0;
            r_dma_out_addr  <= 32'd0;
            r_dma_wgt_addr  <= 32'd0;
            r_dma_param_addr<= 32'd0;
            r_dma_in_stride <= 32'd0;
            r_dma_out_stride<= 32'd0;
            r_dma_ctrl      <= 32'd0;
            r_dma_add_b_addr<= 32'd0;
            r_dma_add_param <= 32'd0;
            r_dma_in_size   <= 32'd0;
            r_dma_wgt_size  <= 32'd0;
            r_dma_out_size  <= 32'd0;
            r_dma_tile_in_size <= 32'd0;
            r_post_ctrl     <= 32'd0;
            r_post_param_count <= 32'd0;
            r_post_clamp    <= 32'd0;
            r_post_act_cfg  <= 32'd0;
            r_post_add_stride <= 32'd0;
            for (lut_i = 0; lut_i < 128; lut_i = lut_i + 1)
                r_lut[lut_i] <= 32'd0;
        end else begin
            // ─── Self-clearing control bits (pulse for 1 cycle) ───
            r_ctrl[0] <= 1'b0;  // START auto-clear
            r_ctrl[1] <= 1'b0;  // ABORT auto-clear
            r_ctrl[2] <= 1'b0;  // SOFT_RST auto-clear

            // ─── IRQ status set from hardware pulses ───
            if (irq_done_set)  r_irq_status[0] <= 1'b1;
            if (irq_error_set) r_irq_status[1] <= 1'b1;
            if (irq_dma_set)   r_irq_status[2] <= 1'b1;

            // ─── Wishbone bus handling ───
            if (valid_access && !wb_ack_o) begin
                wb_ack_o <= 1'b1;

                if (wb_we_i) begin
                    // ════════════ WRITE ════════════
                    casez (addr)
                        // Group 0: Control & Status
                        12'h000: r_ctrl    <= wb_dat_i;  // bits [0:2] self-clear next cycle
                        12'h008: r_irq_en  <= wb_dat_i;
                        12'h030: r_layer_count <= wb_dat_i;
                        12'h00C: r_irq_status <= r_irq_status & ~wb_dat_i; // W1C

                        // Group 1: Layer Parameters
                        12'h040: r_layer_mode <= wb_dat_i;
                        12'h044: r_in_dim_hw  <= wb_dat_i;
                        12'h048: r_in_dim_c   <= wb_dat_i;
                        12'h04C: r_out_dim_hw <= wb_dat_i;
                        12'h050: r_out_dim_c  <= wb_dat_i;
                        12'h054: r_kernel_size<= wb_dat_i;
                        12'h058: r_stride     <= wb_dat_i;
                        12'h05C: r_padding    <= wb_dat_i;
                        12'h060: r_pool_cfg   <= wb_dat_i;
                        12'h064: r_resize_cfg <= wb_dat_i;
                        12'h068: r_deconv_cfg <= wb_dat_i;
                        12'h06C: r_concat_cfg <= wb_dat_i;
                        12'h070: r_tile_cfg   <= wb_dat_i;
                        12'h074: r_tile_count <= wb_dat_i;
                        12'h078: r_sram_base  <= wb_dat_i;

                        // Group 2: DMA Configuration
                        12'h100: r_dma_in_addr   <= wb_dat_i;
                        12'h104: r_dma_out_addr  <= wb_dat_i;
                        12'h108: r_dma_wgt_addr  <= wb_dat_i;
                        12'h10C: r_dma_param_addr<= wb_dat_i; // dual-map w/ 0x184
                        12'h110: r_dma_in_stride <= wb_dat_i;
                        12'h114: r_dma_out_stride<= wb_dat_i;
                        12'h118: r_dma_ctrl      <= wb_dat_i;
                        12'h120: r_dma_add_b_addr<= wb_dat_i; // dual-map w/ 0x198
                        12'h124: r_dma_add_param <= wb_dat_i; // dual-map w/ 0x194
                        12'h128: r_dma_in_size   <= wb_dat_i;
                        12'h12C: r_dma_wgt_size  <= wb_dat_i;
                        12'h130: r_dma_out_size  <= wb_dat_i;
                        12'h134: r_dma_tile_in_size <= wb_dat_i;

                        // Group 3: Post-Processing Configuration
                        12'h180: r_post_ctrl        <= wb_dat_i;
                        12'h184: r_dma_param_addr   <= wb_dat_i; // dual-map
                        12'h188: r_post_param_count <= wb_dat_i;
                        12'h18C: r_post_clamp       <= wb_dat_i;
                        12'h190: r_post_act_cfg     <= wb_dat_i;
                        12'h194: r_dma_add_param    <= wb_dat_i; // dual-map
                        12'h198: r_dma_add_b_addr   <= wb_dat_i; // dual-map
                        12'h19C: r_post_add_stride  <= wb_dat_i;

                        // Group 4: LUT Data (0x200-0x3FF)
                        12'b0010_????_????: r_lut[addr[8:2]] <= wb_dat_i;
                        12'b0011_????_????: r_lut[addr[8:2]] <= wb_dat_i;

                        default: ; // ignore writes to RO/reserved
                    endcase
                end else begin
                    // ════════════ READ ════════════
                    casez (addr)
                        // Group 0: Control & Status
                        12'h000: wb_dat_o <= {28'd0, r_ctrl[3], 3'd0}; // START/ABORT/RST read as 0
                        12'h004: wb_dat_o <= status_reg;
                        12'h008: wb_dat_o <= r_irq_en;
                        12'h00C: wb_dat_o <= r_irq_status;
                        12'h010: wb_dat_o <= {28'd0, hw_error_code};
                        12'h014: wb_dat_o <= hw_version;
                        12'h018: wb_dat_o <= hw_config;
                        12'h01C: wb_dat_o <= hw_perf_cnt;
                        12'h020: wb_dat_o <= hw_mac_cnt;
                        12'h024: wb_dat_o <= hw_serial_lo;
                        12'h028: wb_dat_o <= hw_serial_hi;
                        12'h02C: wb_dat_o <= hw_vendor_id;
                        12'h030: wb_dat_o <= r_layer_count;

                        // Group 1: Layer Parameters
                        12'h040: wb_dat_o <= r_layer_mode;
                        12'h044: wb_dat_o <= r_in_dim_hw;
                        12'h048: wb_dat_o <= r_in_dim_c;
                        12'h04C: wb_dat_o <= r_out_dim_hw;
                        12'h050: wb_dat_o <= r_out_dim_c;
                        12'h054: wb_dat_o <= r_kernel_size;
                        12'h058: wb_dat_o <= r_stride;
                        12'h05C: wb_dat_o <= r_padding;
                        12'h060: wb_dat_o <= r_pool_cfg;
                        12'h064: wb_dat_o <= r_resize_cfg;
                        12'h068: wb_dat_o <= r_deconv_cfg;
                        12'h06C: wb_dat_o <= r_concat_cfg;
                        12'h070: wb_dat_o <= r_tile_cfg;
                        12'h074: wb_dat_o <= r_tile_count;
                        12'h078: wb_dat_o <= r_sram_base;

                        // Group 2: DMA Configuration
                        12'h100: wb_dat_o <= r_dma_in_addr;
                        12'h104: wb_dat_o <= r_dma_out_addr;
                        12'h108: wb_dat_o <= r_dma_wgt_addr;
                        12'h10C: wb_dat_o <= r_dma_param_addr;
                        12'h110: wb_dat_o <= r_dma_in_stride;
                        12'h114: wb_dat_o <= r_dma_out_stride;
                        12'h118: wb_dat_o <= r_dma_ctrl;
                        12'h11C: wb_dat_o <= dma_status;
                        12'h120: wb_dat_o <= r_dma_add_b_addr;
                        12'h124: wb_dat_o <= r_dma_add_param;
                        12'h128: wb_dat_o <= r_dma_in_size;
                        12'h12C: wb_dat_o <= r_dma_wgt_size;
                        12'h130: wb_dat_o <= r_dma_out_size;
                        12'h134: wb_dat_o <= r_dma_tile_in_size;

                        // Group 3: Post-Processing Configuration
                        12'h180: wb_dat_o <= r_post_ctrl;
                        12'h184: wb_dat_o <= r_dma_param_addr; // dual-map
                        12'h188: wb_dat_o <= r_post_param_count;
                        12'h18C: wb_dat_o <= r_post_clamp;
                        12'h190: wb_dat_o <= r_post_act_cfg;
                        12'h194: wb_dat_o <= r_dma_add_param;  // dual-map
                        12'h198: wb_dat_o <= r_dma_add_b_addr; // dual-map
                        12'h19C: wb_dat_o <= r_post_add_stride;

                        // Group 4: LUT Data (0x200-0x3FF)
                        12'b0010_????_????: wb_dat_o <= r_lut[addr[8:2]];
                        12'b0011_????_????: wb_dat_o <= r_lut[addr[8:2]];

                        default: wb_dat_o <= 32'd0;
                    endcase
                end
            end else begin
                wb_ack_o <= 1'b0;
            end
        end
    end

endmodule
