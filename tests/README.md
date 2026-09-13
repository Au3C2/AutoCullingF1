# Auto-Culling 测试体系设计规范

本项目采用分层测试架构（Layered Testing Architecture），按照 **核心逻辑（Engine）→ 交互呈现（GUI）→ 交付安装（Package）** 的包含关系逐层递进，既杜绝重复测试引入的开销与噪声，又针对性防御打包环境下的底层“静默性能暴跌”。

---

## 架构层级划分

```text
[Layer 1: test_engine/] ─── 核心引擎（纯 CLI / 源码态）
   │   • 协议与边界测试 (Stdio JSON Lines)
   │   • 金样绝对精度测试 (严格位零漂移)
   │   • 细颗粒度稳态性能基准 (run_benchmarks.py)
   ▼
[Layer 2: test_gui/]    ─── 交互呈现（开发态 GUI / 前端，待接入）
   │   • 界面端到端状态机与交互模拟 (选择目录/启停/筛选)
   │   • 异常保护 (引擎崩溃时不假死，取消后状态重置)
   │   • 多语言界面文案切换与渲染校验
   ▼
[Layer 3: test_package/]─── 交付安装（成品打包安装包）
       • 双端安装与打包流程验证 (DMG 挂载, NSIS 静默安装/卸载, Portable 解包)
       • 平铺布局完整性与引擎自拉起通信 (auto_culling.exe + auto_culling_cli + lib/)
       • 打包防劣化金样比对 (打包 CLI vs 源码 CLI 精度 100% 一致)
       • 防静默性能暴跌断言 (断言无纯软解回退，断言 GPU/ANE 推理后端激活)
```

---

## 核心设计原则

### 1. 严格禁止在 GUI 层测精度与性能
GUI 仅是无状态的交互呈现壳。如果在 GUI 中测量吞吐量，测出的是 DOM 刷新、IPC 序列化和 Webview 性能，而非算法本身，且极易受 UI 动效干扰。
**GUI 层的唯一使命是检验状态流与交互体验**（几张轻量图片即可在几秒内跑完）。

### 2. 打包测试（Package）专打“静默性能暴跌”
Python 打包本质是重构底层的 C/C++ 动态链接环境。跨平台底层库常自带降级机制（如缺少驱动静默降级为 CPU 计算，缺少 pyav 软解回退导致性能暴跌 20 倍）。
打包层测试必须包含：
- **执行器探针断言（Execution Provider Assertion）**：检查运行时日志，严禁出现软解 fallback（如 `pillow_heif software fallback`），断言平台硬件加速（CoreML / DirectML / CUDA）激活。
- **阶跃式性能下限守卫（Floor Assertion）**：通过小样本（50~100 张图）设定低而刚性的吞吐下限，在秒级时间内阻断数量级级的性能回退，同时免疫云主机的轻微负载抖动。

---

## 目录结构映射

```text
tests/
├── README.md                      # 本文档：测试体系规范
├── conftest.py                    # pytest 共享 fixture（环境与配置）
├── score_gate.py                  # 金样精度断言公共辅助逻辑
├── baselines/                     # 提交的金样真值（deterministic.json 等）
├── test_img/                      # 轻量测试图片集合（6 张 JPG 金样）
│
├── test_engine/                   # 【Layer 1】引擎源码态测试
│   ├── test_protocol.py           # 引擎 Stdio JSON Lines 通信协议规范
│   ├── test_protocol_stress.py    # 引擎协议异常与高压边界容错（坏图、并发预览、瞬时取消等）
│   ├── test_deterministic_baseline.py # 跨平台 CPU 确定性真值与 GPU 对齐比对
│   ├── test_cull.py               # 6 张 JPG 金样基础端到端评分测试
│   ├── test_precision_heif.py     # 24 张 HEIF 真实相机文件精度守护
│   └── test_precision_raw.py      # 20 张 ARW + 20 张 NEF 原始 RAW 精度守护
│
├── test_gui/                      # 【Layer 2】开发态 GUI 自动化交互测试（预留待接入）
│
├── test_package/                  # 【Layer 3】打包与交付产物测试
│   └── test_package.py            # 验证打包后的 CLI (dist/engine/auto_culling_cli) 精度与源码 100% 一致
│
└── ci/                            # CI 运行专用的种子扩展工具与配置
    ├── ci_config.json             # 云端 Runner 校准基线
    ├── sample/                    # 格式种子图片
    └── seed_precision.py          # 种子复制精度比对工具

packaging/
├── test.py                        # 本地全流程回归集成入口
└── test_gui_package.py            # 双端安装包全套安装测试与防静默降级探针
```
