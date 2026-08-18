# debug_archive — 调试脚本存档（不再维护）

这里归档 2026 年 6-8 月开发期的逐层/逐场景调试脚本（~212 个）：
每层一个的 `test_ma_l*_only`、`test_r18c_l*`、`test_md_*`、`test_mc_*`、
`test_me_*` 系列，以及 chain2d/fused/scan 类一次性脚本。

## 地位

- **唯一维护的回归套件是上级目录的 `test_npu_dma_e2e.py`**（50 测试全量绿，
  2026-08-17 基线）+ `unit/` + `integration/` + `robustness/`。
- 本目录脚本当时都为特定 bug 猎杀/层验证写过并跑通过，但**不随 RTL 演进维护**；
  若失效，参考其价值在于复现方法而非当前可运行性。
- 新 bug 复现请优先在维护套件中加用例；这里的脚本适合作为单层复现的起点模板。

## 运行方式（import 路径已适配）

```bash
make -C /data/sam/open-npu/rtl/tb DUT=npu_top \
    MODULE=e2e.debug_archive.test_ma_l1_only SIM=verilator ACC_WIDTH=44
```

（MODULE 路径加 `debug_archive.` 前缀；脚本的 sys.path 已指向 e2e/ 以复用
`test_npu_dma_e2e.py` 的 WbSlave/WbMasterMem/load_golden 等工具函数。）
