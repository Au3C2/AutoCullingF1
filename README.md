# Auto-Culling 🏎️📸

F1 赛车摄影自动筛选工具 —— 从连拍数百张的 HIF 原片中，基于规则与深度学习自动挑选出最值得保留的照片，并生成 XMP 星级标注文件供 Lightroom 直接导入。

## 功能概览

- **连拍分组**：基于 EXIF 时间戳自动将连拍序列编组
- **多阶段评分流水线**：综合锐度、构图、车辆朝向和完整性进行智能打分
- **Top-N 自动优选**：在每组连拍中自动挑选最优的 N 张照片
- **XMP 输出**：生成 Lightroom 可直接识别的 `.xmp` 星级评分文件

## 流水线架构

```
HIF 原片
  │
  ├─ P0: 锐度评估 (HF Ratio)
  │     └─ 计算高频细节占比，过滤严重失焦/模糊
  │
  ├─ P1: 构图评分 (YOLO 检测 + 规则)
  │     ├─ F1 赛车专用 YOLO 目标检测
  │     ├─ 面积占比 + 居中度 + 完整性罚分
  │     └─ 检测置信度加权
  │
  ├─ P3: 铁丝网检测 (MobileNetV2 二分类) [默认关闭]
  │     └─ 识别铁丝网遮挡（精度高但对最终指标贡献有限）
  │
  ├─ P4: 车辆朝向与完整性 (MobileNetV3 多任务)
  │     ├─ 5 类朝向分类: front / front_angle / side / rear_angle / rear
  │     ├─ 2 类完整性分类: full / cut
  │     ├─ 正尾部(rear)一票否决
  │     └─ 截断车辆(cut)软性扣分 (penalty=0.6)
  │
  └─ 综合打分 → Top-N 筛选 → XMP 输出
```

## 评分公式

```
raw_score = 1.5 × s_sharp + 2.5 × s_comp - (0.6 if cut else 0)
```

- `s_sharp`：锐度得分 ∈ [0, 1]（基于高频比率 HF Ratio）
- `s_comp`：构图得分 ∈ [0, 1]（面积占比 + 居中度 + 完整性）
- 否决条件：无检测目标 / 锐度 < 0.05 / raw < 3.1 / 正尾部朝向

## 快速开始

### 环境准备

```bash
# 使用 uv 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### 运行自动筛选

```bash
python cull_photos.py /path/to/hif/folder --top-n 11 --output /path/to/output
```

## macOS 性能优化与 CoreML 支持

在 Apple M 系列芯片（M1/M2/M3/M4）上，本项目支持使用 **Apple Neural Engine (ANE)** 进行硬件加速推理。

### 1. 性能基准 (1280px / 1000张 HEIF)

| 模式 | 后端设备 | 吞吐量 | 耗时 |
| :--- | :--- | :--- | :--- |
| **ONNX (原生)** | M-Chip CPU | 13.8 img/s | 72.5 s |
| **CoreML (加速)** | **Neural Engine (ANE)** | **18.6 img/s** | **53.8 s** |

> **提示**: CoreML 版本比 ONNX 版本整体快了约 **35%**，且大幅降低了 CPU 负载。

### 2. 精确度分析 (ONNX vs CoreML)
经过 1000 张照片的对比测试，CoreML 与 ONNX 的筛选结果**一致性高达 97.6%**。仅 2.4% 的照片因推理引擎精度差异在分值临界点产生星级波动（±1星），不影响最终筛选逻辑。

### 3. 推荐使用方式 (Recommended Usage)

系统会自动检测 macOS 环境并优先加载 `models/*.mlpackage` 模型。

#### 方案 A: 使用 uv 直接运行 (推荐)
无需手动激活环境，速度最快：
```bash
~/.local/bin/uv run cull_photos.py --input-dir /你的/照片/路径 --scale-width 1280 --workers 12
```

#### 方案 B: 标准虚拟环境运行
```bash
# 激活环境
source .venv/bin/activate

# 执行筛选
python3 cull_photos.py --input-dir /你的/照片/路径 --scale-width 1280 --workers 12
```

#### 进阶控制 (环境变量)
```bash
# 强制使用 CoreML (ANE 加速)
export CULL_BACKEND=coreml
python cull_photos.py ...

# 强制使用 ONNX (CPU 运行)
export CULL_BACKEND=onnx
python cull_photos.py ...
```

### 4. 硬件解码要求
为了在 Mac 上获得最佳速度，请确保已安装 `ffmpeg` (支持 `videotoolbox` 硬件加速):
```bash
brew install ffmpeg
```

### 评估流水线性能

```bash
# 在采样测试集上运行评估
python eval_multi_session.py --test-set test_set.csv --output scores.csv
```

## 项目结构

```
auto_culling/
├── cull/                          # 核心评分模块
│   ├── composition.py             # P1: 构图评分（面积/居中/完整性）
│   ├── detector.py                # YOLO 检测器封装
│   ├── exif_reader.py             # EXIF 读取与连拍分组
│   ├── fence_classifier.py        # P3: 铁丝网分类器
│   ├── p4_classifier.py           # P4: 朝向+完整性分类器
│   ├── scorer.py                  # 综合评分与 Top-N 选择
│   └── sharpness.py               # P0: 锐度评估
│
├── cull_photos.py                 # 主程序入口
├── eval_multi_session.py          # 多 session 评估脚本
├── sample_test_set.py             # 测试集采样工具
├── tune_params.py                 # 离线参数调优
│
├── train_fence_classifier.py      # P3 铁丝网分类器训练
├── train_fence_classifier_v2.py   # P3 v2 训练脚本
├── train_p4_multitask.py          # P4 多任务模型训练
├── train_f1_yolo.py               # F1 YOLO 检测器训练
├── extract_p4_rois.py             # P4 训练数据 ROI 提取
│
├── models/                        # ONNX 模型文件 (gitignored)
├── fence_classifier_checkpoints/  # P3 模型权重
├── p4_model_checkpoints/          # P4 模型权重
│
├── AGENTS.md                      # AI 辅助开发规范
├── P4_ANNOTATION_GUIDE.md         # P4 数据标注指南
└── pyproject.toml                 # 项目配置
```

## 性能基准

在 7 个拍摄场次、2018 张测试图片上的评估结果：

| 阶段配置 | F1 Score | Precision | Recall |
|:---|:---|:---|:---|
| P0 旧参数 (旧标签) | 0.5579 | — | — |
| P0 旧参数 (修正标签) | 0.4338 | — | — |
| **P0+P1 优化参数** | **0.5827** | 0.525 | 0.656 |
| P0+P1+P4 (最终) | **0.5952** | 0.558 | 0.637 |

> P0 参数优化贡献了约 **+16%** 的绝对 F1 提升（在修正标签后）。
> P4 额外贡献了 **+1.2%** 的 F1 提升，同时显著提升了 Precision (+3.3%)。

## 技术要点

- **Python 3.10** + PyTorch 2.6 + ONNX Runtime
- **YOLO v8** 目标检测（F1 赛车专用微调模型 + COCO 通用模型双路检测）
- **MobileNetV3** 多任务分类（朝向 + 完整性双 Head）
- **Step-based 训练** + AMP 混合精度 + Early Stopping
- **动态数据增强**：从 Full 样本裁剪生成无限 Cut 训练数据

## 许可

Private project. All rights reserved.
