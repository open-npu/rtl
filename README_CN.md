# Open-NPU RTL

[English](README.md)

Open-NPU 神经网络加速器的可综合 Verilog 硬件实现。

## 架构

- **16×16 INT8 脉动阵列** — 可通过 `ARRAY_SIZE` 参数配置
- **深度可分离卷积引擎** — 支持最大 7×7 卷积核，硬件 padding
- **后处理单元 (PPU)** — per-channel 重量化（bias → 乘 → 移位 → clamp）
- **计算微序列器** — 自动 tiling、OC 分组、权重/激活流式传输
- **Wishbone CSR 寄存器组** — 配置与状态接口
- **DMA 控制器** — 外部总线与内部 SRAM 之间的突发传输

## 目录结构

```
src/            Verilog 源码
  npu_top.v         顶层集成
  npu_compute.v     计算微序列器（Conv2D + DW Conv）
  npu_systolic.v    脉动阵列
  npu_pe.v          处理单元
  npu_ppu.v         后处理单元
  npu_dw_conv.v     深度卷积引擎
  npu_sram.v        双端口同步 SRAM
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

# 运行指定模块的全部测试
make DUT=npu_compute_tb ARRAY_SIZE=4 SIM=icarus

# 运行单个模块测试
make DUT=npu_pe SIM=icarus
make DUT=npu_systolic SIM=icarus
make DUT=npu_ppu SIM=icarus
```

当前状态：**81/81 测试通过**（ARRAY_SIZE=4）。

## 相关仓库

- [open-npu/csim](https://github.com/open-npu/csim) — C 周期近似模拟器
- [open-npu/tools](https://github.com/open-npu/tools) — ONNX 转换器与量化工具链
- [open-npu/design](https://github.com/open-npu/design) — 架构设计文档

## 许可证

Apache-2.0
