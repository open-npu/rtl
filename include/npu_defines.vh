// Open-NPU RTL — Global Defines
// SPDX-License-Identifier: Apache-2.0
//
// Hardware parameters (configurable via -D flags or override here)

`ifndef NPU_DEFINES_VH
`define NPU_DEFINES_VH

// ─── Compute-lane configuration ───
// ARRAY_SIZE is retained as a build-option compatibility alias. It now sets
// only the depthwise bank width; dense operators always use MAC_LANES.
`ifndef ARRAY_SIZE
`define ARRAY_SIZE     8
`endif

// Physical scratchpad is IFM+OFM+WGT+PARAM = 164KB, not one scaled SPAD_KB.
// IFM is 24KB: the largest double-buffered input tile in the checked models
// is ~11.5KB, so a 12KB half still holds it. OFM stays 48KB because a
// double-buffered output tile is 22528 bytes and the half must hold it.
// Weight is 80KB: the largest resident block is 73728 bytes (8ch 3x3x512
// int16). Param stays 12KB so a 576-channel table that is not reloaded
// still fits.
`ifndef IFM_KB
`define IFM_KB         24
`endif
`ifndef OFM_KB
`define OFM_KB         48
`endif
`ifndef WGT_KB
`define WGT_KB         80
`endif
`ifndef PARAM_KB
`define PARAM_KB       12
`endif
`ifndef SPAD_KB
`define SPAD_KB        (`IFM_KB + `OFM_KB + `WGT_KB + `PARAM_KB)
`endif

// ─── Derived parameters ───
`define DW_CHANNELS    `ARRAY_SIZE

// Depths are 32-bit words. 1KB = 256 words.
`define IFM_DEPTH      (`IFM_KB * 256)
`define OFM_DEPTH      (`OFM_KB * 256)
`define WGT_DEPTH_W    (`WGT_KB * 256)
`define PARAM_DEPTH_W  (`PARAM_KB * 256)

`define ACT_BANK_SIZE   (`IFM_KB * 1024)       // IFM bytes (OFM is OFM_KB)
`define WEIGHT_BUF_SIZE (`WGT_KB * 1024)
`define PARAM_BUF_SIZE  (`PARAM_KB * 1024)

// ─── Data widths ───
`define DATA_WIDTH     16     // Data port width (always 16-bit; INT8 sign-extends)
`ifndef ACC_WIDTH
`define ACC_WIDTH      44     // Phase-1 baseline; supports INT16 model accumulation
`endif
`define MULT_WIDTH     16     // Multiplier operand width (for M parameter)
`define BIAS_WIDTH     64     // Per-channel bias width
`define SHIFT_WIDTH    6      // Shift parameter width

// ─── Element packing ───
`define ELEM_BYTES_INT8   1   // Bytes per element in INT8 mode
`define ELEM_BYTES_INT16  2   // Bytes per element in INT16 mode

// ─── DW global-pool streaming ───
// Non-tiled DW Conv whose kernel covers the whole input (global-pool style,
// e.g. model_a L61: 14x14x512 -> 1x1x512) is streamed in 16-channel groups:
// per group the controller DMA-loads a [pos][ch_local] act slice plus a
// 16-channel weight block; the 1x1xC output accumulates in a fixed high
// region of act SRAM (top 1024 words), decoupled from the slice at SRAM[0].
`define DW_STREAM_OUT_BASE (`OFM_DEPTH - 1024)

// ─── Bus configuration ───
`define WB_DATA_WIDTH  32     // Wishbone data bus width
`define WB_ADDR_WIDTH  32     // Wishbone address bus width
`define WB_SEL_WIDTH   4      // Wishbone byte select width
`ifndef SRAM_B_WIDTH
`define SRAM_B_WIDTH   512    // Compute port: 64 bytes, one INT8 lane-group
`endif
`define SRAM_B_WORDS   (`SRAM_B_WIDTH / 32)
`define MAC_LANES      64

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
