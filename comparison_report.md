# Apple M-Chip Benchmark: ONNX vs CoreML

## 1. 性能对比 (Performance)

| 模式 | 分辨率 | 后端设备 | 总耗时 | 吞吐量 |
| :--- | :--- | :--- | :--- | :--- |
| **ONNX** | 1280px | M4 CPU | 72.5 s | 13.8 img/s |
| **CoreML** | 1280px | **Apple Neural Engine (ANE)** | **53.8 s** | **18.6 img/s** |

> **结论**: CoreML 版本的整体筛选流程比 ONNX 版本快了 **35%**。

## 2. 筛选结果对比 (Label Consistency)

| 模式 | Keep (保留) | Reject (拒绝) | 1星 | 2星 | 3星 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ONNX** | 170 | 830 | 112 | 56 | 2 |
| **CoreML** | 171 | 829 | 108 | 61 | 2 |

> **一致性分析**: 两者结果几乎完全一致（差异 < 0.1%）。
