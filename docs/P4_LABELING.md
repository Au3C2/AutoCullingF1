# P4 标注指南（多相机鲁棒性数据扩充）

## 背景

P4 完整性分类器（`models/p4_car_model.onnx`）v1 对缩放算法和相机类型不够鲁棒：
换插值核时约 5% 的真实照片会翻转 keep/reject 判定。v2 重训已在训练端加了
缩放核随机化 + 相机管线抖动，但要真正覆盖"不同相机"，需要补充多相机标注数据。

## 当前数据分布（p4_data/labeled/，共 1143 张有效）

| 类别 | full | cut |
|---|---|---|
| front | 59 | 1 |
| front_angle | 267 | 34 |
| side | 247 | 75 |
| rear_angle | 363 | 65 |
| rear | 30 | 2 |

短板：**cut 样本太少（177 张）**、rear/front 几乎没有、相机来源单一（以 Sony A7C2 为主）。

## 提取待标注 ROI

```bash
# 从任意照片目录批量提取主车 ROI（含 15% 上下文边距，与训练集同协议）
python utils/extract_p4_rois_dir.py --dir D:/photos/nikon_gp --out-dir p4_data/unlabeled_nikon
python utils/extract_p4_rois_dir.py --dir D:/photos/sony_hif --out-dir p4_data/unlabeled_sony
python utils/extract_p4_rois_dir.py --dir D:/photos/phone_jpg --out-dir p4_data/unlabeled_phone
```

## 标注方法

把提取出的 ROI 图片移动（剪切）到 `p4_data/labeled/<朝向>_<完整性>/` 文件夹：

- 朝向：`front` / `front_angle` / `side` / `rear_angle` / `rear`
- 完整性：`full`（车身完整）或 `cut`（车尾/车头被画面边缘裁掉）

文件夹名示例：`side_cut`、`front_angle_full`。看不清或不是单辆 F1 车的丢进 `无效数据/`。

### full vs cut 的判定标准（与训练增广一致）

- **full**：整车（含车轮）都在画面内，车身与边缘有明显间隙。
- **cut**：车头或车尾被画面边缘明显切掉（约 1/3 以上的车长出画）。轻微贴边的算 full。
- 拿不准的放 `无效数据/`，不要硬标——边界模糊的标签会直接教坏决策面。

## 优先级

1. **各相机的 cut 样本**（现在最缺，尤其 rear_cut / front_cut）
2. 尼康 Z6III（NEF）和手机 JPG 的 full 样本
3. 车身贴边但没裁掉的"临界 full"样本（帮模型把决策面推离刀刃）

每补一批后重跑：

```bash
python train/train_p4_multitask.py            # 重训（自动包含新标注）
python eval/eval_p4_robustness.py             # 验证翻转率是否下降
```
