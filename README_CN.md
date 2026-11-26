# Open-NPU RTL

[English](README.md)

Open-NPU 神经网络加速器的可综合 Verilog 硬件实现。

## 架构

- **64 路 INT8/INT16 MAC 行引擎** — 统一承载 K-major 普通卷积、全连接与深度卷积（复用 `wrow` 权重库与 `racc` 累加器）
- **8 路分时复用后处理单元组** — per-channel 重量化（bias → 乘 → 移位 → zp → clamp → ReLU/ReLU6）
- **计算微序列器** — 自动 tiling、OC 分组、跨层 Super-Block 融合与权重/激活流式传输
- **Wishbone CSR 寄存器组** — 配置与状态接口
- **DMA 控制器** — 外部总线与片上 172KB 四库 512-bit SRAM 之间的 1D/2D strided 传输

## 目录结构

```
src/            Verilog 源码
  npu_top.v         顶层集成（IFM/OFM/WGT/PARAM 四库 SRAM）
  npu_compute.v     计算微序列器（Row/DW/Pool/Add/Resize）
  npu_ppu.v         单路后处理流水线
  npu_ppu_bank.v    8 路分时复用后处理单元组
  npu_sram.v        双端口同步 32 位 SRAM（PARAM）
  npu_sram_wide.v   512 位计算端口 SRAM（IFM / OFM / WGT）
  npu_csr.v         CSR 寄存器组（Wishbone）
  npu_dma.v         DMA 控制器
  npu_ctrl.v        顶层控制器 FSM
include/        共享宏定义 (npu_defines.vh)
tb/             cocotb 测试平台
```

## 运行测试

依赖：[Icarus Verilog](http://iverilog.icarus.com/) 和 [cocotb](https://www.cocotb.org/)

```bash
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
