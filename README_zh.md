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
- **极简便携 (Lite)**：彻底移除了 Torch、OpenCV 等沉重依赖。全流程基于 **ONNX Runtime** 与 **Pillow**，压缩包大小仅约 50MB。

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

## 📦 极简发行版 (LITE)

**LITE 版本** 是一个独立的可执行文件，无需安装 Python 或任何复杂的 AI 框架即可在你的系统上直接运行。它针对便携性与极速分发进行了极致优化。

### 1. 下载与使用 (预编译)
1. 从 Release 页面下载最新的 `cull_photos_lite.zip`。
2. 解压到任意文件夹。
3. 直接运行命令（以管理员权限运行性能更好）：
   - **macOS:** `./auto_cull_v0.1_macos_arm64 --input-dir /你的照片路径`
   - **Windows:** `.\auto_cull_v0.1_win_x64.exe --input-dir C:\Photos`

### 2. 自行打包 (Packaging)

打包产物为单文件可执行，无需 Python、无需安装模型与 exiftool（全部内置）。
支持 macOS（Apple Silicon）与 Windows 双平台（spec 按平台分支）。

**第一步：安装 PyInstaller**
```bash
uv pip install pyinstaller
```

**第二步：执行打包脚本**
```bash
python packaging/build.py                 # 单文件产物 → 项目根目录
python packaging/build.py --onedir        # 目录形态（性能验证用，见下）
```
产物为 `auto_cull_v0.1_macos_arm64` / `auto_cull_v0.1_win_x64.exe`。

**在打包产物上跑现有测试脚本**（`CULL_EXE` 环境变量使精度/性能脚本指向二进制）：

```bash
# 精度门（JPG + 24 HEIF + 20 ARW + 20 NEF 逐文件比对基线）
CULL_EXE=./auto_cull_v0.1_macos_arm64 pytest tests/test_package.py \
    tests/test_precision_heif.py tests/test_precision_raw.py

# 性能门（4 数据集吞吐，JPG 14.0 / HEIF 6.0 / ARW 5.0 / NEF 5.5 img/s）
CULL_EXE=dist/auto_cull_v0.1_macos_arm64/auto_cull_v0.1_macos_arm64 \
    python benchmarks/run_benchmarks.py
```

**打包策略**：性能与精度验证一律使用 **onedir（目录形态）**——单文件形态在 macOS 上每次运行都要重新支付代码签名验证税（约 15–25 秒，且稳态吞吐因冷页缓存下降 10–50%）。onedir 由 `build.py --onedir` 产出，zip 后可分发；`build.py`（默认）仍产出单文件形态用于简单分发。

**统一回归看护**（覆盖源码精度/性能、打包流程、打包产物精度/性能）：
```bash
python packaging/guards.py                # 5 项全部，约 12-15 分钟
python packaging/guards.py --perf-only    # 仅性能门（源码 + 打包，各约 3 分钟）
python packaging/guards.py --skip-build   # 复用已有 dist 产物（精度第 2 次起）
```

## 🤖 GitHub Actions CI（精度 / 打包 / 性能三层）

仓库内 `ci_sample/` 每种格式只存 1 张种子（约 70MB），CI 运行时复制成
约 500 张数据集，无需把 1.3GB 相机数据集入库。

- **精度门**（无需校准）：`benchmarks/ci_seed_precision.py --compare` 用
  源码与打包产物分别评分同一批种子副本，逐文件断言 raw_score（±0.002 容差，
  吸收 ANE 每次运行的 ±0.0004 抖动）与 rating 多重集一致。同源复制会进入
  同一 burst 并被 Top-N 降级，因此只看"打包 vs 源码"的一致性而非绝对评级。
- **打包流程门**：`build.py --onedir` + 产物存在性检查。
- **性能门**（需校准）：`run_benchmarks.py --seed-dir ci_sample
  --baseline-file ci_config.json --tolerance 0.85`。CI 托管的 macOS runner
  无 ANE、硅片与本地 M4 不同，**基线必须在目标 runner 上实测**：手动触发
  `perf-calibrate` workflow → 下载产物 → 把 baselines 提交进 `ci_config.json`。
  校准前性能门自动跳过（精度与打包门始终运行）。

workflow 见 `.github/workflows/guards.yml`；CI 统一入口 `packaging/ci_guard.py`。
本地 seed 协议参考值（Apple M4，源码，workers=4）：JPG 84.5 / HEIF 42.6 /
ARW 48.7 / NEF 69.9 img/s——HEIF 单张代表性差（DSC00827 走 COCO 回退），
CI 基线必须以实测为准。

**macOS 平台说明（冷启动签名验证税）**：macOS 内核对包内每个 Mach-O 的 adhoc
签名做首次加载验证（按 inode 缓存）。单文件形态每次运行都解包到新临时目录，
因此每次启动多付约 15–25 s（文件越多越慢）；目录形态（`--onedir`）只在每次
开机后的首个进程付一次，之后吞吐与源码一致。

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
