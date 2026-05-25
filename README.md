# Open-NPU RTL

[中文版](README_CN.md)

Hardware implementation of the Open-NPU neural network accelerator in synthesizable Verilog.

## Architecture

- **16×16 INT8 Systolic Array** — configurable via `ARRAY_SIZE` parameter
- **Depthwise Convolution Engine** — up to 7×7 kernel with hardware padding
- **Post-Processing Unit (PPU)** — per-channel requantization (bias → multiply → shift → clamp)
- **Compute Micro-Sequencer** — handles tiling, OC grouping, weight/activation streaming
- **Wishbone CSR Register File** — configuration and status interface
- **DMA Controller** — burst transfers between external bus and internal SRAMs

## Directory Structure

```
src/            Verilog source modules
  npu_top.v         Top-level integration
  npu_compute.v     Compute micro-sequencer (Conv2D + DW Conv)
  npu_systolic.v    Systolic array
  npu_pe.v          Processing element
  npu_ppu.v         Post-processing unit
  npu_dw_conv.v     Depthwise convolution engine
  npu_sram.v        Dual-port synchronous SRAM
  npu_csr.v         CSR register file (Wishbone)
  npu_dma.v         DMA controller
  npu_ctrl.v        Top-level controller FSM
include/        Shared defines (npu_defines.vh)
tb/             Cocotb testbenches
```

## Running Tests

Prerequisites: [Icarus Verilog](http://iverilog.icarus.com/) and [cocotb](https://www.cocotb.org/)

```bash
cd tb

# Run all tests for a specific module
make DUT=npu_compute_tb ARRAY_SIZE=4 SIM=icarus

# Run individual module tests
make DUT=npu_pe SIM=icarus
make DUT=npu_systolic SIM=icarus
make DUT=npu_ppu SIM=icarus
```

Current status: **81/81 tests passing** (ARRAY_SIZE=4).

## Related Repositories

- [open-npu/csim](https://github.com/open-npu/csim) — C cycle-approximate simulator
- [open-npu/tools](https://github.com/open-npu/tools) — ONNX converter & quantization toolchain
- [open-npu/design](https://github.com/open-npu/design) — Architecture specifications

## License

Apache-2.0
