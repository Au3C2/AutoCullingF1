# Auto-Culling: macOS 迁移与验证任务（交接文档）

> 由 Windows 开发机（2026-08-24）转交。目标：在 Mac（192.168.10.143）上完成环境搭建、
> 精度门重锁、性能实测，并回答"20fps 在苹果芯片上是否成立"。

## 0. 项目简介

F1 连拍照片自动筛选工具。流水线：解码（JPG/HEIF/ARW/NEF）→ YOLO 检测（ONNX/ORT）
→ 锐度（cv2.dft）/构图/P4 完整性-朝向（MobileNetV3 多任务）→ 评分 → XMP/元数据写回。
Windows 侧已完成大量优化并全部门绿；Mac 侧从零验证。

**Windows 对照性能（2026-08-24，门协议 workers=4）**：
JPG 10.8 / HEIF 5.9 / ARW 4.5 / NEF 4.7 img/s；打分链干净 46-63fps（单线程），
引擎内受内存带宽争抢膨胀 2.4×（稳态 ~12-13fps）。这是需要对比的基线。

## 1. Mac 上已就位的数据（~/code/auto_culling/）

| 路径 | 内容 |
|---|---|
| test_import/ | 24 张索尼 HEIF（精度门数据集，*完整目录非此子集，勿全量） |
| test_arw/ | 20 张 ARW |
| test_nef/ | 20 张 NEF |
| p4_data/labeled/ | 5451 张 P4 标注+合成 ROI（966 full + 4485 合成 cut，含 RAW 域） |
| p4_data/unlabeled_raw/ | 178 张 RAW 域未标注 ROI |
| p4_model_checkpoints/ | p4_best.pt(v1)、p4_car_model_v1_legacy.onnx、**p4_car_model_v21_large.onnx（生产 Large 备份）** |
| models/p4_best.pt | ⚠️ 是 Small 架构的 state_dict（最后一次 --arch small 训练产物），仅参考 |
| runs/f1_detect/train/weights/ | F1 YOLO 的 ultralytics torch 权重（best.pt + last.pt，训练的源头） |
| fence_classifier_checkpoints/ | mobilenetv2/resnet18/resnet50 三个 fence 分类器权重（生产未启用） |
| yolov8n.pt | 原版 YOLOv8n 权重（参考/导出用） |

**代码尚未就位**：`git clone git@github.com:Au3C2/auto_culling.git`（develop 分支，最新
提交 4779f13）。模型 models/*.onnx（f1/coco/p4 v2.1 Large）随 git 到达。
tests/test_img/（6 张 JPG 门数据）也在 git 中。若 GitHub 不可达，向用户索取代码包。

## 2. 环境搭建

```bash
cd ~/code/auto_culling
python3 -m venv .venv && source .venv/bin/activate   # Python 3.10+（M 系建议 3.11/3.12）
pip install -U pip
pip install numpy opencv-python Pillow psutil pytest onnxruntime av
# 可选（重训 P4 才需要）：
pip install torch torchvision       # arm64 wheel，自动装
brew install ffmpeg                 # HEIF 回退路径
brew install exiftool               # RAW 提取(持久会话)与元数据; 若不在 PATH, loader 自动回退
```

关键点：
- onnxruntime 在 macOS ARM 上原生带 **CoreMLExecutionProvider**，代码里 provider 优先级
  已是 CoreML → CUDA → CPU（detector/p4_classifier），无需改动。
- JPG 解码用 Pillow + draft，HEIF 优先 pyav（进程内 libav），RAW 用 exiftool 持久会话
  （`-stay_open`，`-b -w` 文件帧界协议，见 cull/loader.py）。
- ffprobe/ffmpeg 探测与解码都由 PATH 解析；brew 默认路径即生效。

## 3. 验证任务（按顺序执行）

### 任务 1：精度门 + 基线重锁
```bash
python -m pytest tests/test_cull.py tests/test_precision_heif.py tests/test_precision_raw.py -q
```
- **预期 HEIF 门会挂**：Mac 的 pyav/libav 版本与 Windows 不同，HEVC 解码/转 RGB 的
  像素可能有 ±1 LSB 差异。ARW/NEF 也可能有（exiftool Perl 版本差异导致提取字节不同的话）。
- 按仓库惯例处理：
  1. `tests/score_gate.py` 的 `run_cull_on_copies` 逐文件对比（注意它只报第一个失败文件，
     要写循环取全量清单）；
  2. 统计：评级翻转数、raw 漂移量、最大幅度；
  3. **只有评级保持 1-5/-1 语义稳定（相对 Windows 锁 ≤小比例翻转，优先零翻转）才重锁**；
    记录每个变化的文件到 results/performance_baseline.md 新建小节"macOS 平台基线 2026-08-24"，
    并更新三个测试文件的 BASELINE 注释头（注明平台/日期/原因）。
- 若 HEIF 大范围损坏（>20% 翻转）：不要重锁，改为定位：对比 pyav 解码与 ffmpeg 解码
  像素差；考虑 `-hwaccel videotoolbox` 路径（见任务 3）。

### 任务 2：端到端性能基线
```bash
python benchmarks/run_benchmarks.py --verbose    # workers=4 默认
```
- 对照：JPG 10.8 / HEIF 5.9 / ARW 4.5 / NEF 4.7。
- 机器漂移纪律：任何 A/B 必须交错跑（同一代码连续两轮取范围），禁止单次下结论。

### 任务 3：解码专项（Mac 的核心收益点）
```bash
# 单线程解码计时（四格式）
python -c "import time;from pathlib import Path;from cull.loader import load_image_rgb
for g in ['tests/test_img/*.jpg','test_import/*.heif','test_arw/*.ARW','test_nef/*.nef']:
  ts=[]
  for p in sorted(Path('.').glob(g))[:6]:
    t0=time.perf_counter();load_image_rgb(p,1280);ts.append((time.perf_counter()-t0)*1000)
  ts.sort();print(g,ts[len(ts)//2])"
```
- **重点实验**：给 HEIF 路径加回硬件解码。Windows 上 `-hwaccel cuda` 失败是因为
  NVDEC 不支持 4:2:2 10-bit；**Mac 的 VideoToolbox 支持 HEVC Rext**，loader.py 的
  ffmpeg 回退路径可加 `-hwaccel videotoolbox`（或 pyav 的 hw_device_ctx）。
  若硬解生效，HEIF 解码应从 ~50ms 降到 ~5-15ms。像素与软解可能 ±1 LSB → 必须过门。
- pyav `frame.reformat(width=1280)` 提前缩放在 Windows 测为中性，Mac 上可复测。
- RAW：exiftool 提取字节跨平台一致性要验证（对比 Windows 侧提取结果哈希，若有记录）；
  若 Perl 版本差异导致字节不同 → ARW/NEF 门会大范围挂 → 门重锁前先统一 exiftool 版本
  （brew 或官网版）。

### 任务 4：打分链与 GPU 利用
- `benchmarks/bench_consumer_scaling.py`：单链 fps + GPU util（M4 用 `powermetrics` 或
  Activity Monitor；脚本里的 nvidia-smi 采样在 Mac 上无效，跳过 GPU 行）。
- 重点：CoreML EP 是否真的激活（模型加载日志会打印 providers）；打分链 M4 上预期
  干净 40-60fps；因带宽宽松，`--consumer-threads 2` 在 Mac 上可能真的有收益——值得
  交错 A/B（Windows 上无收益是因为带宽平台，Mac 结构不同，不能照搬结论）。

### 任务 5：报告交付
- 更新 results/performance_baseline.md + AGENTS.md（新节/追加行，注明 macOS 日期）；
- 提交（提交信息英文）：平台基线重锁 + 性能数据；
- 向用户输出中文总结：四格式门数字 vs Windows、解码单线程 vs Windows、
  打分链 fps、竞品结论"20fps 是否达成"，以及达成/未达成的差距与下一步。

## 4. 雷区与纪律（必读，Windows 侧血的教训）

1. **P4 决策面对像素极敏感**（v1 时代 5% 翻转、v2.1 后 RAW 域 ~1%）：任何解码/缩放/
   预处理核的改动都必须过精度门才能留；Mac 平台基线首次重锁视为"平台差异"可接受，
   但改动后（如加 videotoolbox）仍需过 Mac 门。
2. **机器状态漂移 ±40%**：所有对比必须交错 A/B。
3. 不要提交：.gitignore 的本地修改、p4_data/、test_import/ 等已忽略数据、
   .codegraph/ .zcode/ 工具目录、*.log、训练日志。
4. 代码/注释/提交信息用英文；与用户沟通用中文。不引入 emoji。
5. 训练脚本（若重训 P4）：`python train/train_p4_multitask.py --arch large --num-workers 8`
   （数据本地、缓存 p4_data/cache_raw 首轮自动重建 ~1 分钟）；改架构/增广需过门。
6. exiftool/ffmpeg 找不到时 loader 有回退链，但性能会退化——先用 brew 装齐再测性能。

## 5. 交付物清单（任务完成标准）

- [ ] venv 依赖装齐，pytest 可跑
- [ ] 精度门重锁完成（全绿 ×2 轮）+ 基线 diff 已记录到 results/performance_baseline.md
- [ ] run_benchmarks 四格式数字 + 解码单线程 + 打分链数字，与 Windows 对照表
- [ ] HEIF 硬解实验结论（videotoolbox 是否生效/收益/是否过门）
- [ ] "--consumer-threads 2 在 Mac 上"的交错 A/B 结论
- [ ] 提交（含文档）+ 中文总结给用户

## 6. 登录备忘

- Mac 登录方式（主机/账号/密码）由交接人在仓库外另行提供，此处不落盘。
- 项目路径：`~/code/auto_culling/`；Windows 侧 git log 与 results/performance_baseline.md
  存有全部决策历史，可随时核对。