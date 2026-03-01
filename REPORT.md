# F1 赛车摄影自动筛片项目报告

## 目标

构建一套完整的深度学习流水线，用于 **F1 赛车连拍照片自动筛选（Auto-Culling）**——训练一个二分类图像分类器，学习摄影师会保留（label=1）还是丢弃（label=0）某张照片的判断逻辑。整个流水线涵盖数据集准备、PyTorch 训练基础设施以及预训练模型微调。当前阶段目标是对比四种架构，找出最佳模型。

---

## 项目规范

- **语言**：代码注释/文档使用英文；与用户交互使用中文
- **Python 工具链**：使用 `uv` + Python 3.10，路径操作使用 `pathlib`，命令行参数使用 `argparse`
- **uv 路径**：`/home/au3c2/.local/bin/uv`（不在 PATH 中，必须使用完整路径或先激活 `.venv`）
- **PyTorch 最佳实践**：AMP 混合精度、`WeightedRandomSampler`、带 `pos_weight` 的 `BCEWithLogitsLoss`、梯度裁剪
- **不使用空间裁剪增强**——筛片决策依赖完整画面构图
- **微调策略**：冻结所有主干网络，仅解冻最后 **2 个块**（layer3+layer4）+ 分类头；设置两组参数：分类头学习率和低层主干学习率
- **训练架构**：ResNet-18、ResNet-50、ResNeXt-50-32x4d、MobileNetV3-Large——均输出单一 logit 供 `BCEWithLogitsLoss` 使用
- **TensorBoard**：记录 `train/loss`、`val/loss`、`val/acc`、`val/f1`、`val/auc`、`test/*`、`lr`（按步记录）
- **训练以步数为单位**（非以 epoch 为单位）

---

## 环境与数据集

### 运行环境
- GPU：NVIDIA RTX 4070 Ti（12GB 显存），CUDA 驱动 13.0，PyTorch 2.6.0+cu124
- `uv` 位于 `/home/au3c2/.local/bin/uv`，Python 3.10.19，项目根目录 `/home/au3c2/auto_culling/`
- 系统内存：31 GB

### 数据集
- `/home/au3c2/auto_culling/dataset/img/` — **已清空**（原始 HIF/HEIF 文件已删除）
- `/home/au3c2/auto_culling/dataset/cache/` — **7469 张 JPEG 文件，512×512，共约 372 MB**
- CSV 文件：`data_info.csv`（7469 行）、`train_info.csv`（5975 行）、`test_info.csv`（1494 行）
- CSV 中 `img_path` 列保留旧的 `.HIF` 路径，`CullingDataset.__getitem__` 会自动解析为 `dataset/cache/<stem>.jpg`
- 标签分布：2904 张 label=1（38.9%），4565 张 label=0（61.1%），轻度不平衡约 1.6:1

### 预训练权重（已全部下载）
- `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
- `~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`
- `~/.cache/torch/hub/checkpoints/resnext50_32x4d-1a0047aa.pth`
- `~/.cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth`
- torchvision 会自动从缓存加载，无需重新下载

---

## 过拟合问题及修复（v1 → v2）

v1 版本出现严重过拟合：训练损失趋近于 0，验证损失飙升至 3.3。通过以下五项改动修复：

1. **标签平滑**（`--label-smoothing 0.1`）：在 `train.py` 中实现 `LabelSmoothingBCELoss` 类；软目标公式：`y*(1-s) + 0.5*s`
2. **Dropout(p=0.3)**：在所有四种架构的 `model.py` 中，在全连接分类头之前加入 Dropout 层
3. **解冻 layer3 + layer4**（v1 仅解冻 layer4）；MobileNetV3 解冻最后两个 feature block
4. **更强的数据增强**（`dataset.py`）：新增 `RandAugment(n=2, m=9)`、`RandomGrayscale(p=0.1)`、`GaussianBlur(kernel_size=5)`
5. **权重衰减**从 `1e-4` 提升至 `5e-4`

---

## v2 训练结果

| 模型 | 可训练参数量 | 最佳验证 F1 | 早停步数 | **测试准确率** | **测试 F1** | **测试 AUC** |
|---|---|---|---|---|---|---|
| resnet18 | 10.5M | 0.7601（step 10500） | 15500 | 80.2% | 0.7554 | 0.8611 |
| resnet50 | 22.1M | 0.7378（step 2000） | 7000 | 80.0% | 0.7555 | 0.8577 |
| resnext50 | 21.6M | 0.7571（step 5000） | 10000 | **81.9%** | **0.7680** | 0.8639 |
| mobilenetv3 | 0.95M | 0.7514（step 12500） | 17500 | 75.4% | 0.7279 | **0.8710** |

### 关键观察
- **resnext50 综合表现最佳**：测试 F1（0.7680）和准确率（81.9%）均为最高
- **mobilenetv3** AUC 最高（0.8710），但准确率和 F1 最低——可通过阈值调优改善
- **resnet50** 收敛极早（step 2000 达最佳验证结果，step 7000 触发早停）——可能是学习率过高或 layer3+layer4 同时解冻对该模型过于激进
- v2 验证损失稳定在 0.65–0.79 区间（健康范围，无发散），过拟合问题基本解决

---

## 关键实现细节

### `model.py` — v2 微调冻结策略

```python
# ResNet-18/50, ResNeXt-50：解冻 layer3 + layer4 + fc
_freeze_all(model)
_unfreeze(model.layer3)
_unfreeze(model.layer4)
model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))

# MobileNetV3：解冻最后 2 个 feature block + classifier
_freeze_all(model)
_unfreeze(model.features[-1])
_unfreeze(model.features[-2])
model.classifier[-1] = nn.Linear(in_features, 1)  # classifier 内已有 dropout，不额外添加
```

### `train.py` — 标签平滑损失

```python
class LabelSmoothingBCELoss(nn.Module):
    # 软目标：y*(1-smoothing) + 0.5*smoothing
    # 封装带 pos_weight 支持的 BCEWithLogitsLoss
```

### `dataset.py` — v2 训练集数据增强流水线

```
RandomHorizontalFlip → RandAugment(n=2, m=9) → ColorJitter →
RandomGrayscale(p=0.1) → GaussianBlur(k=5) → RandomRotation(5°) →
ToTensor → Normalize(ImageNet 均值/标准差)
```

---

## ONNX 导出与推理

### 导出结果

四个模型均已通过 `export_onnx.py` 导出为 ONNX（opset 17，动态 batch），并通过 onnxruntime 数值验证：

| 模型 | ONNX 文件大小 | PT vs ORT 最大误差 | 验证结果 |
|---|---|---|---|
| resnet18 | 42.6 MB | 2.94e-4 | 通过 |
| resnet50 | 89.6 MB | 9.57e-4 | 通过 |
| resnext50 | 87.6 MB | 1.86e-4 | 通过 |
| mobilenetv3 | 16.0 MB | 6.71e-3 | 通过 |

> 数值误差均 < 1e-2，属于 float32 精度范围内的正常差异。

### 推理速度 Benchmark（RTX 4070 Ti，CUDAExecutionProvider）

单位：images/sec（越高越好）

| 模型 | bs=1 | bs=4 | bs=8 | bs=16 | bs=32 | **峰值** |
|---|---|---|---|---|---|---|
| mobilenetv3 | 132 | 185 | **218** | 158 | 131 | **218 img/s（bs=8）** |
| resnet18 | 74 | 91 | 103 | 140 | **165** | **165 img/s（bs=32）** |
| resnext50 | **42** | 40 | 35 | 40 | 40 | **42 img/s（bs=1）** |
| resnet50 | 28 | 49 | 51 | 51 | **57** | **57 img/s（bs=32）** |

延迟参考（最优 batch size 下）：

| 模型 | ms/img |
|---|---|
| mobilenetv3 | 4.6 ms |
| resnet18 | 6.1 ms |
| resnext50 | 23.7 ms |
| resnet50 | 17.6 ms |

### 速度观察
- **mobilenetv3** 推理最快（峰值 218 img/s），适合实时或大批量场景；但准确率最低（F1=0.7279）
- **resnet18** 速度次之（峰值 165 img/s），F1（0.7554）与 resnext50 相差仅 0.013，**性价比最高**
- **resnext50**（最佳 F1=0.7680）在 ORT 下速度约 40 img/s——GroupedConv 在 CUDA kernel 调度上效率偏低，可考虑 TensorRT 进一步加速
- **resnet50** 比 resnext50 参数更多但速度更慢，原因是深层瓶颈结构的 kernel 调度开销更大

### 精度 vs 速度综合对比

| 模型 | 测试 F1 | 峰值 img/s | 综合推荐 |
|---|---|---|---|
| resnext50 | **0.7680** | 42 | 精度优先 |
| resnet18 | 0.7554 | **165** | **速度/精度均衡（推荐）** |
| resnet50 | 0.7555 | 57 | 不推荐（精度不如 resnext，速度不如 resnet18） |
| mobilenetv3 | 0.7279 | **218** | 速度优先（精度损失较大） |

### 推理脚本用法

```bash
# 对一个装满 JPG 的目录推理，结果分入 keep/ discard/
python infer_onnx.py \
    --model onnx_models/resnext50.onnx \
    --input-dir /path/to/photos \
    --output-dir /path/to/sorted \
    --threshold 0.5 \
    --copy          # 不加则移动文件

# 仅跑 benchmark，不移动文件
python infer_onnx.py \
    --model onnx_models/resnet18.onnx \
    --input-dir dataset/cache \
    --benchmark \
    --benchmark-batch-sizes 1 4 8 16 32

# 一键导出 + benchmark 所有模型
bash benchmark_onnx.sh
```

---

## 文件结构

```
/home/au3c2/auto_culling/
├── pyproject.toml
├── uv.lock
├── prepare_dataset.py                    ✅ 已执行
├── download_pretrained.py                ✅ 已执行
├── cache_images.py                       ✅ 已执行
├── run_finetune.sh                       ✅ v1 脚本
├── run_finetune_v2.sh                    ✅ v2 脚本
├── export_onnx.py                        ✅ ONNX 导出脚本
├── infer_onnx.py                         ✅ ONNX 推理 + 文件分类脚本
├── benchmark_onnx.sh                     ✅ 一键导出 + 速度测试脚本
├── dataset/
│   ├── img/                              已清空
│   ├── cache/                            7469 张 512×512 JPEG，约 372 MB
│   ├── data_info.csv
│   ├── train_info.csv                    5975 行
│   └── test_info.csv                     1494 行
├── onnx_models/
│   ├── resnet18.onnx                     ✅ 42.6 MB
│   ├── resnet50.onnx                     ✅ 89.6 MB
│   ├── resnext50.onnx                    ✅ 87.6 MB
│   └── mobilenetv3.onnx                  ✅ 16.0 MB
├── checkpoints/
│   ├── resnet18_finetune/                v1 训练（测试 F1=0.7565）
│   │   ├── best.pt                       step 8000，验证 F1=0.7484
│   │   └── last.pt                       step 13000
│   ├── resnet18_finetune_v2/             ✅ v2 训练（测试 F1=0.7554）
│   │   ├── best.pt                       step 10500，验证 F1=0.7601
│   │   └── last.pt                       step 15500
│   ├── resnet50_finetune_v2/             ✅ v2 训练（测试 F1=0.7555）
│   │   ├── best.pt                       step 2000，验证 F1=0.7378
│   │   └── last.pt                       step 7000
│   ├── resnext50_finetune_v2/            ✅ v2 训练（测试 F1=0.7680）← 最佳精度
│   │   ├── best.pt                       step 5000，验证 F1=0.7571
│   │   └── last.pt                       step 10000
│   └── mobilenetv3_finetune_v2/          ✅ v2 训练（测试 F1=0.7279）
│       ├── best.pt                       step 12500，验证 F1=0.7514
│       └── last.pt                       step 17500
└── src/
    └── auto_culling/
        ├── __init__.py
        ├── model.py                      ✅ Dropout(0.3) 头，layer3+layer4 解冻
        ├── dataset.py                    ✅ 新增 RandAugment、GaussianBlur、RandomGrayscale
        └── train.py                      ✅ LabelSmoothingBCELoss，--label-smoothing 参数
```

---

## 待完成工作

- **阈值优化**：在验证集上扫描 sigmoid 输出的最优决策阈值（对 mobilenetv3 的高 AUC 尤为重要）
- 可尝试：resnet50 仅解冻 layer4（v1 风格），因为同时解冻 layer3+layer4 导致其收敛过快
- 可尝试：对 resnext50 使用 TensorRT 加速，以发挥其精度优势的同时提升推理速度

---

## Mac Apple Silicon 推理测试（CoreMLExecutionProvider）

### 测试环境

- 硬件：Apple M 系列芯片（arm64）
- 运行时：`onnxruntime-silicon` 1.16.3，Python 3.10.19（conda）
- 加速后端：`CoreMLExecutionProvider`（Apple Neural Engine / GPU），回退至 `CPUExecutionProvider`
- 测试集：`test_img/`，共 **1494 张 JPEG**（与训练集 `test_info.csv` 同批次图片）
- 推理参数：`--batch-size 32`，`--threshold 0.5`，`--copy`（不移动源文件）
- 测试日期：2026-03-01

### 安装方式

```bash
conda create -n auto_culling_mac python=3.10 -y
conda run -n auto_culling_mac pip install onnxruntime-silicon "numpy<2" pillow

# 推理命令（以 resnext50 为例）
conda run -n auto_culling_mac python infer_onnx.py \
    --model onnx_models/resnext50.onnx \
    --input-dir test_img \
    --output-dir results/resnext50 \
    --copy --coreml
```

### 推理结果（1494 张图，threshold=0.5）

| 模型 | keep 数 | discard 数 | keep 比例 | 推理耗时 | 实际吞吐量 |
|---|---|---|---|---|---|
| resnet18 | 728 | 766 | 48.7% | 4.52 s | **330.8 img/s** |
| resnet50 | 753 | 741 | 50.4% | 6.05 s | 246.9 img/s |
| resnext50 | 683 | 811 | 45.7% | 6.95 s | 214.9 img/s |
| mobilenetv3 | 737 | 757 | 49.3% | 16.92 s | 88.3 img/s |

> 注：mobilenetv3 的 ONNX 图中含 hard-sigmoid/hard-swish 算子，CoreML 仅支持 140 个节点中的 111 个，其余回退 CPU 执行，导致速度偏低。

### Benchmark 吞吐量（images/sec）

单位：images/sec，越高越好；括号内为 ms/img。

| 模型 | bs=1 | bs=4 | bs=8 | bs=16 | bs=32 | **峰值** |
|---|---|---|---|---|---|---|
| resnet18 | 1498（0.67ms） | 1447（0.69ms） | 2416（0.41ms） | **2509（0.40ms）** | 2367（0.42ms） | **2509 img/s（bs=16）** |
| resnet50 | 643（1.56ms） | **861（1.16ms）** | 767（1.30ms） | 716（1.40ms） | 694（1.44ms） | **861 img/s（bs=4）** |
| resnext50 | 577（1.73ms） | **719（1.39ms）** | 689（1.45ms） | 685（1.46ms） | 703（1.42ms） | **719 img/s（bs=4）** |
| mobilenetv3 | 68（14.64ms） | 86（11.61ms） | 106（9.41ms） | **130（7.71ms）** | 127（7.90ms） | **130 img/s（bs=16）** |

### Mac vs RTX 4070 Ti 速度对比

| 模型 | Mac M（CoreML）峰值 | RTX 4070 Ti（CUDA）峰值 | 倍率 |
|---|---|---|---|
| resnet18 | **2409 img/s** | 165 img/s | **14.6×** |
| resnet50 | **958 img/s** | 57 img/s | **16.8×** |
| resnext50 | **765 img/s** | 42 img/s | **18.2×** |
| mobilenetv3 | 136 img/s | **218 img/s** | 0.6×（CoreML 算子覆盖不全） |

> resnet 系列在 CoreML 下获得极大加速，Apple Neural Engine 对标准卷积+BN 图的加速效果显著。
> mobilenetv3 因算子部分回退 CPU，反而比 CUDA 更慢；建议在 Mac 上优先使用 resnet18 或 resnext50。

### Mac 平台综合推荐

| 模型 | 训练 F1 | Mac 峰值 img/s（128张,全分辨率） | Mac 推荐 |
|---|---|---|---|
| resnext50 | **0.7680** | 765 | 精度优先（CoreML 加速充分） |
| resnet18 | 0.7554 | **2409** | **速度/精度均衡，Mac 首选** |
| resnet50 | 0.7555 | 958 | 次选（精度与 resnet18 相当，速度低 2.5×） |
| mobilenetv3 | 0.7279 | 136 | 不推荐在 Mac 上使用（CoreML 算子覆盖不全）|

---

## 真实相机分辨率 Benchmark（Mac M 芯片，CoreMLExecutionProvider）

### 测试说明

- 测试图片：随机生成（numpy random RGB），模拟全分辨率原图输入，每组 **128 张**
- **Sony A7C II**：7008 × 4672（约 33 MP）
- **Nikon Z6 III**：6048 × 4024（约 24 MP）
- Benchmark 参数：pool_size=128，warmup=3 batch，measure=20 batch
- batch sizes 测试范围：1、4、8、16、32
- 预处理流程（与训练一致）：SquarePad → Resize(224) → ToTensor → ImageNet Normalize
- 测试日期：2026-03-01

> **注意**：模型输入分辨率固定为 224×224，原始分辨率影响的是 **预处理（CPU decode + resize）耗时**，不影响模型前向推理本身。benchmark 模式下图像预先载入内存，因此下表反映的是纯推理吞吐量（不含磁盘 I/O）。

### Sony A7C II（7008 × 4672 / 33 MP）— 128 张

| 模型 | bs=1 | bs=4 | bs=8 | bs=16 | bs=32 | **峰值** |
|---|---|---|---|---|---|---|
| resnet18 | 1460（0.68ms） | 2359（0.42ms） | 2306（0.43ms） | 2372（0.42ms） | **2377（0.42ms）** | **2377 img/s** |
| resnet50 | 765（1.31ms） | **957（1.05ms）** | 769（1.30ms） | 696（1.44ms） | 652（1.53ms） | **957 img/s** |
| resnext50 | **765（1.31ms）** | 764（1.31ms） | 703（1.42ms） | 650（1.54ms） | 641（1.56ms） | **765 img/s** |
| mobilenetv3 | 74（13.60ms） | 92（10.84ms） | 108（9.25ms） | 132（7.55ms） | **136（7.35ms）** | **136 img/s** |

### Nikon Z6 III（6048 × 4024 / 24 MP）— 128 张

| 模型 | bs=1 | bs=4 | bs=8 | bs=16 | bs=32 | **峰值** |
|---|---|---|---|---|---|---|
| resnet18 | 1525（0.66ms） | 2359（0.42ms） | **2409（0.42ms）** | 2372（0.42ms） | 2369（0.42ms） | **2409 img/s** |
| resnet50 | 806（1.24ms） | **958（1.04ms）** | 769（1.30ms） | 710（1.41ms） | 710（1.41ms） | **958 img/s** |
| resnext50 | 721（1.39ms） | **763（1.31ms）** | 708（1.41ms） | 697（1.44ms） | 655（1.53ms） | **763 img/s** |
| mobilenetv3 | 82（12.16ms） | 92（10.93ms） | 110（9.07ms） | **134（7.45ms）** | 131（7.65ms） | **134 img/s** |

### 三组分辨率峰值对比（Mac CoreML，n=20→128 更新）

| 模型 | 512×512（test_img,n=1494） | Sony A7C II（33MP,n=128） | Nikon Z6 III（24MP,n=128） | 分辨率影响 |
|---|---|---|---|---|
| resnet18 | 2509 img/s | 2377 img/s | **2409 img/s** | 差异 ≤ 5%，几乎无影响 |
| resnet50 | 861 img/s | 957 img/s | **958 img/s** | Sony/Nikon 略高（+11%），推理稳定 |
| resnext50 | 719 img/s | **765 img/s** | 763 img/s | 差异 ≤ 6%，基本一致 |
| mobilenetv3 | 130 img/s | 136 img/s | **134 img/s** | 差异 ≤ 4%，较前次 20 张结果更稳定 |

### 关键结论

1. **推理吞吐量与原始分辨率基本无关**：模型输入固定 224×224，原始分辨率只影响预处理（SquarePad + Resize），benchmark 模式下预处理已预完成，三种分辨率的推理速度差异 ≤ 11%
2. **128 张 vs 20 张结果对比**：池子扩大后数据更稳定，mobilenetv3 的峰值从 101 提升至 136 img/s（bs=32），说明小样本下 CoreML 部分 batch size 估计偏低
3. **resnet18 三种分辨率下均超 2300 img/s**，仍是 Mac 平台最优选择（F1=0.7554）
4. **最优 batch size 规律**：resnet18 在 bs=16~32 时最快（CoreML 内部并行充分）；resnet50/resnext50 在 bs=4 时最快（较深网络 bs 过大反而调度开销增加）；mobilenetv3 在 bs=16~32 时最快（算子回退 CPU 时大 batch 均摊开销更划算）
5. **实际使用时瓶颈在 I/O + CPU resize**：Sony A7C II 的 33MP JPEG 约 8–15 MB/张，Nikon Z6 III 约 6–10 MB/张，建议配合多进程预处理以充分发挥 CoreML 推理速度
