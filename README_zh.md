# Auto-Culling (F1 专用) 🏎️📸 —— 赛车摄影自动筛选工具

**中文版** | [English](README.md)

本工具专为 F1 及各类赛车摄影设计。利用深度学习与启发式规则，从连拍产生的数千张原片（HIF/RAW）中高效筛选出最值得保留的高质量照片，并同步生成可供 Lightroom 直接导入的 XMP 星级标注与自动裁剪信息。

---

## 🌟 核心功能

- **连拍分组**：基于 EXIF 时间戳自动将瞬时爆发的连拍序列智能编组。
- **多阶段评分流水线**：
  - **P0 锐度评估**：基于高频细节比率（HF Ratio）精准过滤失焦或模糊的照片。
  - **P1 构图评分**：使用赛车专用 YOLO 目标检测模型，评估主体面积、位置居中度。
  - **P4 朝向与完整性**：MobileNetV3 多任务模型自动识别车辆朝向（一票否决正后方视角）并检测主体是否被铁丝网等遮挡。
- **Top-N 筛选策略**：在每一组连拍中自动挑选得分最高的 $N$ 张。
- **自动裁剪**：基于车辆主体位置与目标纵横比（3:2/2:3）自动生成最佳裁剪方案。
- **Lightroom 深度集成**：生成对应的 `.xmp` 附属文件，Lightroom Classic 导入时将自动识别星级评分 (1-5★) 与裁剪框。

---

## 🚀 端到端性能基准 (End-to-End Performance)

测试基于 1000 张 HEIF 连拍原片（1280px 解码缩放）。**「端到端」** 吞吐量涵盖了完整的工作流：文件搜集、EXIF 读取、图像解码、多阶段 AI 推理以及 XMP 生成。

### macOS (Apple Silicon M4 Pro)
针对 Apple Neural Engine (ANE) 进行 CoreML 深度优化。

| 推理后端 | 硬件设备 | 端到端吞吐量 | 
| :--- | :--- | :--- |
| **ONNX Runtime** | M-Chip CPU | ~13.8 张/秒 |
| **CoreML** | **ANE 神经网络引擎** | **~18.6 张/秒 (+35%)** |

### Windows (Intel i9 + RTX 4070 Ti)
得益于强大的 CUDA 核心加速与多线程预取技术。

| 推理后端 | 硬件设备 | 端到端吞吐量 | 
| :--- | :--- | :--- |
| **CUDA** | **NVIDIA RTX 4070 Ti** | **~35.0 张/秒** |
| **CUDA** | **NVIDIA RTX 4090** | **~52.0+ 张/秒** |

---

## 🛠️ 快速上手

### 1. 依赖环境

- **Python 3.10+**
- **FFmpeg**: 用于高性能 HIF 视频预览流解码。
  - **macOS**: `brew install ffmpeg`
  - **Windows**: [官网下载](https://ffmpeg.org/download.html) 并添加到环境变量 `PATH`。

### 2. 安装与配置

推荐使用 [uv](https://github.com/astral-sh/uv) 进行高效依赖管理及虚拟环境创建：

**macOS / Linux:**
```bash
uv sync
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
uv sync
.venv\Scripts\activate.ps1
```

### 3. 使用示例

扫描指定文件夹并生成 XMP 评分文件：

**macOS:**
```bash
python cull_photos.py --input-dir /你的/照片/路径 --workers 8 --scale-width 1280
```

**Windows:**
```powershell
python cull_photos.py --input-dir C:\Photos\F1 --workers 12 --scale-width 1280
```

**常用参数说明:**
- `--workers N`: 设置并行解码与预加载的 Worker 线程数。
- `--scale-width 1280`: 解码时进行缩放处理，极大提升推理速度。
- `--top-n 11`: 每组连拍保留的最大数量。
- `--force`: 忽略已有 XMP 评分，强制全量重新检测。

---

## 📂 项目结构

```text
auto_culling/
├── cull/                  # 核心计算包（锐度、构图、检测器、综合打分）
├── eval/                  # 评估与基准测试脚本
├── train/                 # 模型训练模块（YOLO、各类分类器）
├── utils/                 # 工具脚本（自动裁剪补全、EXIF 整理、模型下载）
├── models/                # 模型权重文件（本地 ONNX/CoreML 模型）
├── results/               # 评测结果与基准报告
├── tests/                 # 自动化测试套件
└── cull_photos.py         # 主程序入口
```

---

## 📊 评分逻辑说明

最终 `raw_score` 计算公式：
$$score = 1.5 \times S_{锐度} + 2.5 \times S_{构图} - 惩罚_{截断/遮挡}$$

**否决项 (一票否决):**
- 未检测到任何目标。
- 锐度得分低于 0.05（严重模糊）。
- 车辆朝向为 "Rear"（正尾部视角）。
- 综合总分过低（低于 3.1）。

---

## 🧪 自动化测试

运行集成测试套件以验证后端推理准确性与 XMP 字段正确性：

```bash
pytest tests/test_cull.py
```

---

## 📜 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。
