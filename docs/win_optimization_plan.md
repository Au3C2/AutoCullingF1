# Windows 优化计划（借鉴 macOS 已验证优化）

> 目标：在 Windows 上复现 macOS 优化路径的收益，预期端到端 1.5–2× 提升
> （起点基线：JPG 10.8 / HEIF 5.9 / ARW 4.4 / NEF 3.3 img/s @ workers=4，2026-08-22 门协议）。
>
> 执行纪律（macOS 侧教训，2026-08-28）：
> 1. 每项优化用**交错 A/B + 冷却**验证——连续满载会热节流，绝不信单次串行长批；
> 2. 每项改动后跑 **rating 门**：`pytest tests/ -m deterministic`（70 张，CULL_DETERMINISTIC=1）
>    ——任何星级翻转立刻暴露，行为回归不允许伪装成性能提升合入；
> 3. 性能验收用本地 500 张协议（`benchmarks/run_benchmarks.py`），先 `--no-guard` 校准
>    Windows 基线，再 ±10% 窗口看护；
> 4. `opencv-python==5.0.0.93` 钉死，**不要混装 headless**——混装状态曾让 knife-edge
>    文件间歇翻转（macOS 教训，Windows 同理）。

## Phase 0 — 现状校准（先做，半天）

- [ ] 跑 `packaging/guards.py`（或 Windows workflow）确认当前门禁全绿，记录起点数字；
- [ ] `run_benchmarks.py --no-guard` 校准 Windows 本地 500 张基线，写入
      `ci_config.json` / `results/performance_baseline.md`；
- [ ] 确认 `--workers 2/4/6/8` 在 Win 上的最优值（核心数与 M4 不同，不要沿用 4）。

## Phase 1 — 直接移植（macOS 已验证，纯跨平台代码）

- [ ] **RAW 持久 exiftool stay_open**（`-b -w` 文件框架）
      mac 实测：33 vs 460 ms/file，ARW +14% / NEF +21%。
      验收：`exiftool.exe` stay_open 行为与 perl 版一致（提取字节逐位比对），
      ARW/NEF 稳态提升 ≥10%。
- [ ] **Range I/O RAW 预览提取**（只读文件前 ~6%，IFD+JPEG 定位）
      mac 实测：ARW -7.7 ms、NEF -2.4 ms/帧。
      验收：40 张 gate 文件提取字节逐位一致 + ARW/NEF 解码耗时下降。
- [ ] **向量化 YOLO 后处理**（若 Win 侧代码未同源）
      mac 实测：detect 阶段 -39%。
      验收：检测结果逐位一致 + detect 耗时对比。
- [ ] **批量元数据写入**（exiftool stay_open 单进程）
      Win 历史实测：60 vs 458 ms/file（7.6×）。确认未回退。
- [ ] **BGR-first resize**（先缩后转色）
      mac 实测：-0.68 ms/帧 RAW，0 像素漂移。
      验收：位一致 + RAW 解码耗时下降。
- [ ] **架构确认**：解码进程池 + 单消费者推理、P4 warm-up 前置（#21）、
      psutil worker 降优先级（BELOW_NORMAL_PRIORITY_CLASS）——确认 develop
      同源且在 Win 生效。

## Phase 2 — Windows 专属硬解码（等价实现，收益预计最大）

- [ ] **HEIF 硬解**：pyav `HWAccel("cuda"/"dxva2")` 探测
      （`cull/loader.py` 291–333 行脚手架已就位，从未实测）。
      mac 参照：VideoToolbox 12.4 vs 21.8 ms 软解（1.76×）。
      验收：与软解位一致性用 rating 门验证（色彩范围对齐逻辑必须保留），
      HEIF 稳态提升 ≥30%。
- [ ] **JPEG 硬解**：`ffmpeg -hwaccel`（mjpeg / cuvid）
      （`_hw_decode_jpeg_ffmpeg` 脚手架已就位）。
      mac 参照：ImageIO 硬解是软解的 ~2-3×。
      验收：像素容差用 rating 门 + raw 平台窗验证，JPG 稳态提升 ≥30%。
      注意：硬解像素若与软解有 ±1 LSB 差异，属可接受范围（rating 不变即可）。
- [ ] **DirectML EP 评估**（`DmlExecutionProvider`，Windows ORT 原生）：
      与 CUDA EP A/B 对比 P4 / YOLO 单帧推理耗时，择优锁定。
      mac 参照：锁定 EP 消除了静默切换导致的 rating 翻转（DSC00849 教训），
      **Win 上同样要锁定单一 EP**，不允许运行时静默回退。

## Phase 3 — 推理与打包

- [ ] **静态图 YOLO + CUDA 调优**：onnxsim 常量折叠静态导出
      （`models/f1_yolov8n_static.onnx` 已有），A/B CUDA EP 吞吐；
      mac 参照：全链 40.2 → 26.4 ms/帧。
      验收：detection 输出位一致（或 rating 门通过）+ YOLO 阶段耗时下降。
- [ ] **打包**：`cull_photos.spec` Win 分支沿用（动态模型 + exiftool.exe），
      验证 onefile/onedir 在 Win 的启动税差异（mac 的签名税问题 Win 无，
      但 onedir 仍是推荐形态）。
- [ ] **cv2 版本钉死**：`opencv-python==5.0.0.93`（不混装 headless），
      knife-edge 文件 IMG_20260314_160318_240.jpg / DSC00849.heif 每次换依赖后复测。

## 已知边界（不作为失败）

- `IMG_20260314_160318_240.jpg`（P4 朝向 knife-edge，-1↔3）与
  `DSC00849.heif`（raw 压 1★/2★ 边界 3.11，1↔2）：加速后端上 rating 允许
  与确定性真值不同，已列入 `tests/conftest.py::KNOWN_RATING_DIVERGENCE`；
- `DSC00942.heif` / `IMG_20260315_164133_810.nef`：P4 cut 判定边界
  （raw 差 ≈ P4_CUT_PENALTY 0.6，rating 不变），列入 `KNOWN_CUT_BOUNDARY`；
- 任何优化若翻转**其他**文件的 rating = 立即回退。

## 预期收益排序（先吃大的）

1. HEIF 硬解（Phase 2）——HEIF 是 Win 第二慢格式，硬解预计 1.5-2× 单级提升；
2. RAW exiftool stay_open + Range I/O（Phase 1）——ARW/NEF 合计 +15-25%；
3. YOLO 静态图 + EP 调优（Phase 3）——推理占比高，预计全格式 +10-20%；
4. JPEG 硬解（Phase 2）——JPG 已较快，收益相对小。

## 已验证不可行 / 需回避（macOS 教训，Windows 大概率同样）

- ffmpeg `-vf scale` 解码内缩放：像素漂移（mac REJECTED #1）；
- RAW 批量提取：无像素安全路径（DROPPED #4）；
- sharpness 移入 worker / 独立 sharpness pool：零收益或 ROI 耦合（#5/#7）；
- EXIF stay_open 未验证版本：回退过一次（#8），本次务必带字节比对验证。
