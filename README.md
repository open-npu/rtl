# Open-NPU RTL

[中文版](README_CN.md)

Hardware implementation of the Open-NPU neural network accelerator in synthesizable Verilog.

## Architecture

- **64-lane INT8/INT16 MAC Row** — K-major dense convolution, fully connected, and unified depthwise execution (sharing `wrow` weight banks and `racc` accumulators)
- **8-lane Time-Multiplexed Post-Processing Bank** — per-channel requantization (bias → multiply → shift → zp → clamp → ReLU/ReLU6)
- **Compute Micro-Sequencer** — handles tiling, OC grouping, inverted-residual super-blocks, and weight/activation streaming
- **Wishbone CSR Register File** — configuration, status, and 4-slot Super-Block descriptor interface
- **Parameterized Dual-Channel DMA (`32 / 64 / 128-bit`, `Wishbone-B4` or `AXI4`)** — decoupled Load/Store engines with arbitrary 4B-aligned byte masking, 1D/2D strided transfers, and optional synthesizable AXI4 `INCR` burst bridge (`npu_wb_to_axi4.v`)

## Directory Structure

```
src/            Verilog source modules
  npu_top.v         Top-level integration (IFM/OFM/WGT/PARAM 4-bank SRAM + dual DMA)
  npu_compute.v     Compute micro-sequencer (Row/DW/Pool/Add/Resize)
  npu_ppu.v         Post-processing lane pipeline
  npu_ppu_bank.v    8-lane post-processing bank
  npu_sram.v        Dual-port synchronous SRAM (parameterized 32/64/128-bit Port A)
  npu_sram_wide.v   512-bit compute-port SRAM (IFM / OFM / WGT)
  npu_csr.v         CSR register file (Wishbone)
  npu_dma.v         Parameterized 32/64/128-bit burst DMA controller
  npu_wb_to_axi4.v  Synthesizable Wishbone-to-AXI4 INCR burst master bridge
  npu_ctrl.v        Top-level controller FSM
include/        Shared defines (npu_defines.vh: SRAM sizes, DMA_BUS_WIDTH, DMA_BUS_PROTO)
tb/             Cocotb & Verilog self-checking testbenches
```

## Running Tests

Prerequisites: [Icarus Verilog](http://iverilog.icarus.com/) and [cocotb](https://www.cocotb.org/)

```bash
# Run all 6 external DMA bus configurations (32/64/128-bit x Wishbone/AXI4)
make test-bus-cfg

cd tb

# Run the compute integration suite
make DUT=npu_compute_tb MODULE=integration.test_npu_compute_tb SIM=icarus

# Run individual module tests
make DUT=npu_ppu SIM=icarus
```

The compute integration suite currently contains 29 tests.

## Related Repositories

- [open-npu/csim](https://github.com/open-npu/csim) — C cycle-approximate simulator
- [open-npu/tools](https://github.com/open-npu/tools) — ONNX converter & quantization toolchain
- [open-npu/design](https://github.com/open-npu/design) — Architecture specifications

## License

Apache-2.0
