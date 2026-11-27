# Open-NPU RTL

[English](README.md)

Open-NPU 神经网络加速器的可综合 Verilog 硬件实现。

## 架构

- **64 路 INT8/INT16 MAC 行引擎** — 统一承载 K-major 普通卷积、全连接与深度卷积（复用 `wrow` 权重库与 `racc` 累加器）
- **8 路分时复用后处理单元组** — per-channel 重量化（bias → 乘 → 移位 → zp → clamp → ReLU/ReLU6）
- **计算微序列器** — 自动 tiling、OC 分组、跨层 Super-Block 融合与权重/激活流式传输
- **Wishbone CSR 寄存器组** — 配置、状态与 4 槽 Super-Block 描述符接口
- **参数化双通道 DMA（`32 / 64 / 128-bit`，`Wishbone-B4` 或 `AXI4`）** — 读写解耦突发引擎，原生支持任意 4B 对齐首尾掩码、1D/2D strided 传输及可选可综合 AXI4 `INCR` 突发桥（`npu_wb_to_axi4.v`）

## 目录结构

```
src/            Verilog 源码
  npu_top.v         顶层集成（IFM/OFM/WGT/PARAM 四库 SRAM + 读写双通道 DMA）
  npu_compute.v     计算微序列器（Row/DW/Pool/Add/Resize）
  npu_ppu.v         单路后处理流水线
  npu_ppu_bank.v    8 路分时复用后处理单元组
  npu_sram.v        双端口同步 SRAM（参数化 32/64/128-bit Port A）
  npu_sram_wide.v   512 位计算端口 SRAM（IFM / OFM / WGT）
  npu_csr.v         CSR 寄存器组（Wishbone）
  npu_dma.v         参数化 32/64/128-bit 突发 DMA 控制器
  npu_wb_to_axi4.v  可综合 Wishbone 转 AXI4 INCR 突发主接口桥
  npu_ctrl.v        顶层控制器 FSM
include/        共享宏定义 (npu_defines.vh：SRAM 容量、DMA_BUS_WIDTH、DMA_BUS_PROTO)
tb/             cocotb 与 Verilog 自检测试平台
```

## 运行测试

依赖：[Icarus Verilog](http://iverilog.icarus.com/) 和 [cocotb](https://www.cocotb.org/)

```bash
# 运行全部 6 种外部 DMA 总线配置自检（32/64/128-bit x Wishbone/AXI4）
make test-bus-cfg

cd tb

# 运行计算集成测试
make DUT=npu_compute_tb MODULE=integration.test_npu_compute_tb SIM=icarus

# 运行单个模块测试
make DUT=npu_ppu SIM=icarus
```

计算集成测试当前包含 29 个用例。

## 相关仓库

- [open-npu/csim](https://github.com/open-npu/csim) — C 周期近似模拟器
- [open-npu/tools](https://github.com/open-npu/tools) — ONNX 转换器与量化工具链
- [open-npu/design](https://github.com/open-npu/design) — 架构设计文档

## 许可证

Apache-2.0
