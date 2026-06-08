// Open-NPU RTL — Global Defines
// SPDX-License-Identifier: Apache-2.0
//
// Hardware parameters (configurable via -D flags or override here)

`ifndef NPU_DEFINES_VH
`define NPU_DEFINES_VH

// ─── Array configuration ───
`ifndef ARRAY_SIZE
`define ARRAY_SIZE     16     // Systolic array dimension (N×N)
`endif

`ifndef SPAD_KB
`define SPAD_KB        128    // Total scratchpad size in KB
`endif

// ─── Derived parameters ───
`define NUM_PES        (`ARRAY_SIZE * `ARRAY_SIZE)
`define DW_CHANNELS    `ARRAY_SIZE

// SRAM sizes (bytes) — for reference; actual depths derived in npu_top.v from SPAD_KB
`define ACT_BANK_SIZE  (`SPAD_KB * 1024 / 4)   // Act SRAM = SPAD/4
`define WEIGHT_BUF_SIZE (`SPAD_KB * 1024 / 2)  // Weight SRAM = SPAD/2
`define PARAM_BUF_SIZE (`SPAD_KB * 1024 / 16)  // Param SRAM = SPAD/16

// ─── Data widths ───
`define DATA_WIDTH     16     // Data port width (always 16-bit; INT8 sign-extends)
`define ACC_WIDTH      40     // Accumulator width (supports C_in<=512)
`define MULT_WIDTH     16     // Multiplier operand width (for M parameter)
`define BIAS_WIDTH     64     // Per-channel bias width
`define SHIFT_WIDTH    6      // Shift parameter width

// ─── Element packing ───
`define ELEM_BYTES_INT8   1   // Bytes per element in INT8 mode
`define ELEM_BYTES_INT16  2   // Bytes per element in INT16 mode

// ─── Bus configuration ───
`define WB_DATA_WIDTH  32     // Wishbone data bus width
`define WB_ADDR_WIDTH  32     // Wishbone address bus width
`define WB_SEL_WIDTH   4      // Wishbone byte select width

// ─── Register map base addresses ───
`define CSR_BASE       12'h000  // Control & Status (Group 0)
`define LAYER_BASE     12'h040  // Layer parameters (Group 1)
`define DMA_BASE       12'h100  // DMA config (Group 2)
`define PPU_BASE       12'h180  // Post-processing (Group 3)
`define LUT_BASE       12'h200  // LUT data (Group 4)

// ─── Key CSR offsets ───
`define CSR_CTRL       12'h000
`define CSR_STATUS     12'h004
`define CSR_IRQ_EN     12'h008
`define CSR_IRQ_STATUS 12'h00C
`define CSR_ERROR      12'h010
`define CSR_VERSION    12'h014
`define CSR_HW_CONFIG  12'h018
`define CSR_PERF_CNT   12'h01C

// ─── Control bits ───
`define CTRL_START     0
`define CTRL_ABORT     1
`define CTRL_SOFT_RST  2
`define CTRL_AUTO_NEXT 3

// ─── Status bits ───
`define STAT_BUSY      0
`define STAT_DMA_BUSY  1
`define STAT_ERROR     2
`define STAT_DONE      3

// ─── Per-channel parameter layout (14 bytes/channel) ───
`define PARAM_BYTES_PER_CH  14
`define PARAM_M_BITS        15
`define PARAM_S_BITS        6
`define PARAM_ZP_BITS       16
`define PARAM_BIAS_BITS     64

`endif // NPU_DEFINES_VH
