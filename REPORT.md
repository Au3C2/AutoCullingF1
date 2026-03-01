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
