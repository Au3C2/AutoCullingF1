# Auto-Culling (F1 专用) 🏎️📸 —— 赛车摄影自动筛选工具

**中文版** | [English](README.md)

本工具专为 F1 及各类赛车摄影设计。面向相机直出的整个文件夹（HEIF/RAW/JPEG），自动完成连拍分组、逐帧多阶段 AI 评分、每组 Top-N 精选，并直接写入 Lightroom 兼容的星级/拒绝标记与自动裁剪参数——无需人工初筛。

- **输入**：相机直出文件夹（索尼 ARW、尼康 NEF、佳能 CR2/CR3、富士 RAF、奥之心 ORF、松下 RW2、佳能/索尼 HEIF `.hif/.heif/.heic`、JPEG、PNG、TIFF）
- **输出**：Lightroom `.xmp` 附属文件（RAW/HEIF）或文件内 XMP 元数据（JPEG/HIF），包含星级、拒绝标记与裁剪参数
- **运行时**：纯 ONNX Runtime 推理——运行时不需要 PyTorch

---

## 核心功能

- **连拍分组**：基于 EXIF 时间戳（辅以时间间隔回退策略）自动编组。
- **多阶段评分流水线**：
  - **P0 锐度**：基于 FFT 的高频能量比，叠加主体 ROI 加权；失焦帧一票否决。
  - **P1 构图**：YOLO 检测（F1 专用 14 类模型，640px；F1 未命中时级联 COCO `yolov8n` 兜底），评估主体面积、位置居中与画面留白。
  - **P4 朝向与完整性**：MobileNetV3 多任务分类器（224px），否决正后方视角、惩罚切割/遮挡主体。
  - **P3 铁丝网否决**：可选围栏分类器（默认关闭）。
- **Top-N 筛选**：每组连拍按原始分保留最优 *N* 张（默认 11）。
- **自动裁剪**：围绕检测主体生成 Lightroom 裁剪框（3:2 / 2:3）并写入 `crs:` 参数。
- **Lightroom 深度集成**：导入即显示星级；已评分文件自动跳过（`--force` 强制重跑）。
- **确定性模式**：`--deterministic`（或 `CULL_DETERMINISTIC=1`）强制纯 CPU ONNX + 软件解码，在 macOS / Windows / Linux 上输出位一致——这是所有精度门禁锁定的跨平台真值。

## 端到端性能

门协议：每格式约 500 张真实相机文件，`--dry-run`，测稳态吞吐（解码 + 连拍分组 + AI 推理 + 元数据写入）。基线与方法论见 [`results/performance_baseline.md`](results/performance_baseline.md)。

### macOS — Apple M4（10 核），workers = 4

| 格式 | 吞吐量 |
| :--- | ---: |
| JPEG | 83.5 img/s |
| HEIF | 65.5 img/s |
| 索尼 ARW | 49.9 img/s |
| 尼康 NEF | 70.0 img/s |

硬件加速：VideoToolbox HEIF 硬解、ImageIO JPEG 硬解、CoreML(+ANE) YOLO 静态图、exiftool 分片 EXIF 扫描。评分链串行 ≈ 37.5 fps。

### Windows — Ryzen 7 5700X + RTX 4070 Ti，workers = 4（默认 8）

| 格式 | workers = 4 | workers = 8（默认） |
| :--- | ---: | ---: |
| JPEG | 28.1 img/s | 40–42 img/s |
| HEIF | 38.0 img/s | 38.5 img/s |
| 索尼 ARW | 31.9 img/s | 35 img/s |
| 尼康 NEF | 45.0 img/s | 46 img/s |

推理跑在 **DirectML**（onnxruntime-directml；评分链 6.8 ms/帧，CUDA EP 为 12.6 ms）。消费级 NVIDIA GPU 没有 HEVC 4:2:2 与 JPEG 硬件解码器，解码走 libjpeg-turbo / libav 软解（实测与论证见 [`results/performance_baseline.md`](results/performance_baseline.md)）。

> 单帧成本中，JPEG/RAW 以解码为主（24MP 约 115 ms——Huffman 熵解码与分辨率无关），HEIF/NEF 以推理链为主。`--workers` 控制解码并行度；星级与 worker 数无关。

## 快速开始

### 方式一：运行可执行文件（无需 Python）

从 GitHub Releases 获取预编译二进制（或自行构建，见[打包](#打包独立可执行文件)）：

- Windows：`auto_cull_v0.1_win_x64.exe`
- macOS（Apple Silicon）：`auto_cull_v0.1_macos_arm64`

```powershell
# Windows
.\auto_cull_v0.1_win_x64.exe --input-dir C:\Photos\F1 --recursive --force
```

```bash
# macOS
./auto_cull_v0.1_macos_arm64 --input-dir /path/to/photos --recursive --force
```

省略 `--input-dir` 会弹出文件夹选择器。[常用参数](#常用参数)与源码 CLI 完全一致。二进制已内置 ONNX 模型、exiftool 与 ffmpeg 运行时组件，无需额外安装。

### 方式二：从源码运行

前置条件：

- **Python 3.10+**，推荐用 [uv](https://github.com/astral-sh/uv) 管理依赖
- **ffmpeg** 在 PATH 中（macOS：`brew install ffmpeg`；Windows：`external/ffmpeg/` 已内置）
- **exiftool**：已内置于 `external/exiftool/`（macOS 使用系统自带 perl）；Windows CI 通过 `choco install exiftool` 安装
- GPU 可选：NVIDIA（DirectML/CUDA EP）或 Apple Silicon（CoreML）加速推理；全部自动回退 CPU。

```bash
uv sync
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python cull_photos.py --input-dir /path/to/photos --recursive --force
```

#### 常用参数

| 参数 | 说明 |
| :--- | :--- |
| `--workers N` | 解码进程池大小（默认 8；星级与 worker 数无关） |
| `--top-n 11` | 每组连拍最多保留张数 |
| `--scale-width 1280` | 评分链解码分辨率 |
| `--p4-policy` | `always`（默认）/ `never` / `auto`（仅 F1/GP 目录启用 P4） |
| `--crop-off` | 关闭自动裁剪写入 |
| `--dry-run` | 只评分与报告，不写任何元数据 |
| `--dump-scores FILE` | 导出逐图 CSV（锐度/构图/原始分/星级） |
| `--force` | 强制重分析已评分文件 |
| `--deterministic` | 跨平台位一致的 CPU 路径（较慢） |

省略 `--input-dir` 时还有基于 **customtkinter** 的小型 GUI（`cull/gui`）。

## 打包（独立可执行文件）

```bash
uv pip install pyinstaller

python packaging/build.py            # onefile -> dist/ 并复制到仓库根目录
python packaging/build.py --onedir   # 目录形态（推荐：无每次启动的解压/签名税）
```

产物：`auto_cull_v0.1_win_x64.exe`（Windows，onedir 约 478 MiB）、
`auto_cull_v0.1_macos_arm64`（macOS，onefile 161 MiB）。spec 打包了冻结的 ONNX 图、
exiftool（perl 形态）与 ffmpeg 运行时组件，目标机器无需安装 Python。
打包回归统一入口：`python packaging/guards.py`（精度 → 性能 → 构建 → 打包精度 → 打包性能）。

## 项目结构

```text
auto_culling/
├── cull/                  # 核心引擎：loader（解码）、detector（YOLO）、sharpness、
│                          # composition、scorer（P0-P4 + 星级）、p4_classifier、
│                          # fence_classifier、engine（解码池 + 单消费者）、
│                          # exif_reader、xmp_writer、gui
├── models/                # ONNX 权重（f1_yolov8n、yolov8n、p4_car_model 及静态图版本）
├── train/                 # 训练流水线（YOLO 微调、P4 多任务、围栏分类器、调参）
├── packaging/             # build.py + guards.py（PyInstaller，统一回归套件）
├── benchmarks/            # run_benchmarks.py —— 分格式稳态性能门
├── tests/                 # 精度门禁 + 确定性真值 + CI harness
├── scripts/               # 工具脚本（基线生成、性能剖面、精度报告）
├── eval/                  # 离线评估工具
├── docs/                  # 优化计划、标注指南
├── results/               # performance_baseline.md（权威数据与实验史）
├── external/              # 内置 exiftool（Windows 另含 ffmpeg）
└── cull_photos.py         # CLI 入口
```

## 评分逻辑

```
raw_score = 1.5 × S_sharp + 2.5 × S_comp − 0.6 × [P4 判定切割/遮挡]

星级：raw < 3.11 → 1★，< 3.40 → 2★，< 3.80 → 3★，< 4.20 → 4★，否则 5★
```

一票否决（rating = −1）：

- 画面中未检测到主体
- 锐度低于阈值（0.05）
- 车辆朝向为**正后方**
- 原始分低于下限（3.1）

逐帧评分后由 `select_best_n` 在每组连拍内保留 Top-N 并微调组内星级。

## 测试与门禁

```bash
# 精度：70 张门文件的 rating + raw 对比已提交的确定性真值
pytest tests/ -m deterministic                       # 真值平台检查（严格）
pytest tests/test_cull.py tests/test_precision_heif.py tests/test_precision_raw.py

# 性能：分格式稳态门（需约 1.3GB 相机数据集）
python benchmarks/run_benchmarks.py

# 全套（精度 → 性能 → 构建 → 打包门）：约 12-15 分钟
python packaging/guards.py

# 有意调整评分逻辑后重新生成确定性真值
CULL_DETERMINISTIC=1 python scripts/generate_deterministic_baseline.py
```

CI（`.github/workflows/`）在 GitHub 托管的 macOS/Windows runner 上用已提交的种子样本（`tests/ci/`）跑同一套门禁。精度采用一致性判定：打包二进制必须与源码流水线逐文件一致（raw ±0.002），rating 必须完全相同。

延伸阅读：[`results/performance_baseline.md`](results/performance_baseline.md)（实测数据、优化史、各平台基线）、[`docs/P4_LABELING.md`](docs/P4_LABELING.md)（P4 标注指南）。

## 许可证

[Apache License 2.0](LICENSE)。
