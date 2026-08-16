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
   - **macOS:** `./dist/cull_photos/cull_photos --input-dir /你的照片路径`
   - **Windows:** `.\dist\cull_photos\cull_photos.exe --input-dir C:\Photos`

### 2. 自行打包 (Packaging)
如果你希望基于当前代码自行编译二进制文件：

**第一步：安装 PyInstaller**
```bash
uv pip install pyinstaller
```

**第二步：执行打包脚本**
```bash
# 使用已优化的 spec 文件（已排除 Torch/CV2 等冗余库）
pyinstaller cull_photos.spec --noconfirm
```
打包产物将生成在 `dist/cull_photos/` 目录下。

---

## 🖥️ 桌面图形界面

主桌面应用为 **Tauri 2** 壳（`ui/` 静态前端 + `src-tauri/` Rust 应用）：深色现代 UI 由系统 WebView 渲染（Windows 用 WebView2，macOS 用 WKWebView），外壳本体约 10MB。筛选引擎以打包的 Python sidecar（`cull_photos.py --json-lines`）形式运行，通过 stdio 流式推送事件——GUI 每次任务只启动一次引擎，任务结束后进程驻留用于解码预览，点击预览无需重新启动进程。

- **流式结果**：每帧打分完成后立即插入结果列表，无需等待全部筛选结束即可开始点选预览。
- **实时进度**：权重重映射的阶段进度条（打分阶段占据大部分进度）+ 逐帧计数（"已打分 X/Y，保留 A / 丢弃 B"）。
- **功能齐全**：参数面板覆盖全部 CLI 选项（基本 + 高级）；设置保存在浏览器 localStorage 中，跨会话持久。
- **结果查看**：可排序/筛选的评分表格（星级、评分、否决原因）；点击行即可在**固定尺寸预览窗格**中查看照片，窗格大小永不随内容变化。
- **随时取消**：运行中可停止任务——已打分的结果保留展示，且不会写入任何文件。
- **日志与导出**：实时日志面板、汇总统计（吞吐、保留/丢弃、星级分布）、原生保存对话框一键导出 CSV。

**硬件后端（按平台自动选择）**：Windows 默认优先 CUDA（`onnxruntime-gpu` + `nvidia-cudnn-cu12`），macOS 优先 MLX 再 CoreML，其余平台使用 CPU；不可用的后端自动降级到 CPU。Windows 上源码方式运行会自动启用 CUDA；打包后的可执行文件为纯 CPU 版（CUDA 运行时无法随包分发，否则体积增加约 1GB）。

源码方式运行：
```bash
python cull_gui.py                    # 轻量 CustomTkinter 备用 GUI
cd src-tauri && cargo tauri dev       # Tauri UI（需要 Rust 工具链）
```

以不抢焦点方式启动打包后的 Tauri 应用（例如不打断全屏游戏）：
```powershell
Start-Process -FilePath "...\auto-culling-gui.exe" -WindowStyle Minimized
```
Rust 侧诊断日志写入可执行文件旁的 `gui.log`。

构建 Tauri 应用：
```bash
pyinstaller cull_sidecar.spec --noconfirm          # 1. 构建 windowed sidecar
cp dist/cull_sidecar.exe src-tauri/binaries/cull-sidecar-x86_64-pc-windows-gnu.exe
tauri build --no-bundle                            # 2. 构建外壳 + sidecar
# Windows: src-tauri/target/release/auto-culling.exe（旁边带 cull-sidecar.exe）
# macOS:   src-tauri/target/release/auto-culling
```

旧版 CustomTkinter GUI 仍可用 `pyinstaller cull_gui.spec --noconfirm` 构建（产物 `dist/auto_cull_gui_v0.1_win_x64.exe`）。打包后的程序以 `CREATE_NO_WINDOW` 方式启动子进程，任务开始阶段不会再闪现命令行窗口。

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

运行完整测试套件（CLI 流水线、GUI 逻辑与视图层、取消语义）：

```bash
pytest tests/
```

说明：
- `tests/test_package.py` 额外验证打包后的二进制，需要构建产物位于项目根目录（否则自动跳过）。
- GUI 视图层测试会实例化真实窗口并通过泵事件循环驱动。它们需要显示环境，无显示时自动跳过；无头 Linux CI 可用 `xvfb-run pytest tests/`。
- `pyproject.toml` 中配置了 `--capture=sys`：fd 级捕获在 Windows 上会导致 Tcl/Tk 初始化偶发失败。

---

## 📜 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。
