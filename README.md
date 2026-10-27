# Open-NPU RTL

[中文版](README_CN.md)

Hardware implementation of the Open-NPU neural network accelerator in synthesizable Verilog.

## Architecture

- **64-lane INT8/INT16 MAC Row** — dense convolution and fully connected layers
- **8-lane Depthwise Engine** — up to 7×7 kernels, hardware padding, and streamed global kernels
- **64-lane Post-Processing Bank** — per-channel requantization (bias → multiply → shift → clamp)
- **Compute Micro-Sequencer** — handles tiling, OC grouping, weight/activation streaming
- **Wishbone CSR Register File** — configuration and status interface
- **DMA Controller** — burst transfers between external bus and internal SRAMs

## Directory Structure

```
src/            Verilog source modules
  npu_top.v         Top-level integration
  npu_compute.v     Compute micro-sequencer
  npu_ppu.v         Post-processing unit
  npu_ppu_bank.v    64-lane post-processing bank
  npu_dw_conv.v     Depthwise convolution engine
  npu_dw_bank.v     8-lane depthwise bank
  npu_sram.v        Dual-port synchronous SRAM
  npu_sram_wide.v   512-bit compute-port SRAM
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
