# Project Status Summary — Auto-Culling Photo Classifier

**Last Updated**: Sat Mar 7, 2026  
**Status**: P3 Complete, P4 Preparation Complete, Ready for User Annotation

---

## Executive Summary

| Milestone | Status | Key Metric |
|-----------|--------|-----------|
| **P0: Sharpness Detection** | ✅ Complete | Good coverage |
| **P1: Composition Analysis** | ✅ Complete | Integrated |
| **P2: Object Detection (F1 Cars)** | ✅ Complete | YOLO F1 detector |
| **P3: Wire Fence Detection** | ✅ Complete | **F1=0.9796** (MobileNetV2) |
| **P4: Orientation Classification** | 🔄 Awaiting Annotation | Ready to train after user labels 608 images |

**Test Set Impact (P3 Simulated)**:
- Baseline (no fence veto): 967 kept (47.9%)
- With fence veto: 960 kept (47.6%)
- Additional filtered: 7 images (-0.35pp)

---

## P3: Wire Fence Detection — COMPLETE ✅

### What It Does
Detects wire fence occlusion in F1 photos using a pretrained deep learning classifier.

### Performance
| Model | F1 | Accuracy | Precision | Recall | Throughput |
|-------|-----|----------|-----------|--------|------------|
| **MobileNetV2** | **0.9796** | 99.18% | 0.96 | 1.00 | 4201 img/s |
| ResNet18 | 0.9752 | 99.01% | 0.9672 | 0.9833 | 8030 img/s |
| ResNet50 | 0.9500 | 98.03% | 0.95 | 0.95 | 3659 img/s |

**Selected Model**: MobileNetV2 (best F1, perfect recall on training set)

### Files
```
cull/fence_classifier.py              ← Inference module (lazy-loaded)
cull/scorer.py                         ← Integrated veto logic + ImageScore.fence_* fields
fence_classifier_checkpoints/mobilenetv2/best.pt  ← Trained weights
fence_label/有铁丝网/                  ← 120 positive training samples
fence_label/没有铁丝网/                ← 488 negative training samples
```

### Integration
- Fence veto runs **first** in the scoring pipeline (before other veto checks)
- Disabled by setting `ENABLE_FENCE_VETO = False` in `cull/scorer.py`
- Exports `fence_pred` and `fence_confidence` columns in output CSV

### Testing
✅ **Lightweight integration test** (test_fence_integration.py):
- FenceClassifier loads successfully on CUDA
- Scorer correctly applies fence veto logic
- CSV export includes fence fields
- All 3/3 tests PASSED

---

## P4: Vehicle Orientation Recognition — PREPARATION COMPLETE ✅

### What It Is
A 4-class classifier to recognize car heading angle in F1 photos:
- **正前方** (0°): Head-on, car facing camera
- **侧身** (90°): Side profile, complete side view
- **正后方** (180°): Rear-facing, mirror of head-on
- **侧后方** (45-135°): Diagonal, during overtake/defending

### Task: User Annotation Phase

**📁 Work Location**: `vehicle_orientation_labels/待标注/` (608 images)

**Time Estimate**: ~10-20 hours (~1-2 min/image)

**Instructions**: 
1. Open `vehicle_orientation_labels/待标注/`
2. For each image, determine car orientation
3. Move to corresponding subfolder:
   - `vehicle_orientation_labels/正前方/`
   - `vehicle_orientation_labels/侧身/`
   - `vehicle_orientation_labels/正后方/`
   - `vehicle_orientation_labels/侧后方/`
4. Delete if orientation unclear

**Detailed Guidelines**: See `P4_ORIENTATION_TASK.md` and `vehicle_orientation_labels/标注原则.md` (Chinese)

### Training Pipeline (Ready to Use)

Once user provides annotated data:

```bash
# Train classifier
python train_orientation_classifier.py --epoch 100 --batch-size 32

# Evaluate
python eval_orientation_classifier.py --checkpoint orientation_checkpoints/best.pt
```

**Output**:
- `orientation_checkpoints/best.pt` — Trained model
- `orientation_checkpoints/history.json` — Training history
- `orientation_checkpoints/eval_results.json` — Per-class metrics (precision, recall, F1)

### Expected Results
- Expected accuracy: 85-92% (depends on annotation quality)
- Expected F1 (macro): 0.85-0.90

---

## Key Discoveries & Learnings

### P3: Class Imbalance Fix (Critical)
**Problem**: Initial fence models learned "always predict negative" strategy (0% F1)  
**Root Cause**: Missing `pos_weight` parameter in BCEWithLogitsLoss  
**Solution**: Added `pos_weight=4.07` (ratio of negative:positive samples)  
**Result**: All models retrained successfully with strong performance

### Path Encoding on Windows
HIF and images with Chinese paths require special handling:
```python
# ✓ Correct
img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

# ✗ Wrong (fails silently on Windows)
img = cv2.imread(path)
```

### Dataset Notes
- Total labeled fence ROIs: 608 (after cleaning)
  - Positive (fence): 120
  - Negative (no fence): 488
  - Deleted (corrupted/human faces): 199
- Total orientation ROIs (to annotate): 608

---

## Project Statistics

### Image Data
| Dataset | Images | Status |
|---------|--------|--------|
| Fence training | 608 | ✅ Labeled & trained |
| Orientation to annotate | 608 | 🔄 Awaiting user labels |
| Test set (multi-session) | 2018 | ✅ Baseline scores available |

### Computational Resources
- **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM)
- **Framework**: PyTorch 2.6.0+cu124
- **Python**: 3.10 (via `uv` package manager)
- **Inference**: CUDA + CPU fallback (ORT for ONNX)

### Training Times (P3)
- MobileNetV2: ~15 minutes (full training)
- ResNet18: ~20 minutes
- ResNet50: ~40 minutes

---

## Next Steps (Recommended Timeline)

### Week 1-2: User Annotation
1. User completes 100-150 images as test run
2. Share sample with agent for verification (spot-check accuracy)
3. Continue with remaining ~450 images at ~1 hour/day

### Week 3: Agent Training (Parallel with User)
- Once ~50% annotated (300 images), agent can train preliminary model
- Monitor validation accuracy
- Provide feedback to user if systematic errors detected

### Week 4: Final Training & Evaluation
1. User completes all 608 annotations
2. Agent retrains on full dataset
3. Run `eval_multi_session.py` with P3+P4 enabled
4. Compare P3-only vs P3+P4 F1 scores
5. Document final results

### Week 5: Publication
- Generate REPORT.md with final metrics
- Archive checkpoints and results

---

## File Organization

### Training & Inference
```
cull/                                 ← Core pipeline modules
├── fence_classifier.py               ← P3 inference
├── scorer.py                         ← P3 integration
├── detector.py                       ← Object detection
├── sharpness.py                      ← Sharpness scoring
└── composition.py                    ← Composition scoring

train_fence_classifier_v2.py          ← P3 training (completed)
eval_fence_models.py                  ← P3 evaluation (completed)
test_fence_integration.py              ← P3 integration tests ✅

train_orientation_classifier.py        ← P4 training (ready)
eval_orientation_classifier.py         ← P4 evaluation (ready)
```

### Checkpoints
```
fence_classifier_checkpoints/
└── mobilenetv2/best.pt               ← P3 model (F1=0.9796)

orientation_checkpoints/
└── (will be created after annotation)
```

### Data
```
fence_label/                          ← P3 labeled data (completed)
├── 有铁丝网/ (120)
├── 没有铁丝网/ (488)
└── 删除/ (199)

vehicle_orientation_labels/           ← P4 data structure (ready)
├── 待标注/ (608 to annotate)
├── 正前方/ (empty, user populates)
├── 侧身/ (empty)
├── 正后方/ (empty)
└── 侧后方/ (empty)
```

### Output
```
scores_multi_hfr_v4.csv               ← Baseline (P0-P2 only)
scores_with_fence_veto_simulated.csv  ← P3 simulated effect
analyze_fence_veto_impact.py           ← Impact analysis script
```

---

## Configuration & Customization

### P3 Fence Veto
Edit `cull/scorer.py`:
```python
ENABLE_FENCE_VETO: bool = True  # Set to False to disable P3
```

### P3 Model Architecture
Edit `cull/scorer.py`:
```python
_FENCE_CLASSIFIER = FenceClassifier(arch="mobilenetv2")  # or "resnet18", "resnet50"
```

### Scoring Parameters
Edit `cull/scorer.py`:
```python
SHARP_THRESH: float = 0.12      # Sharpness veto threshold
W_SHARP: float = 3.0            # Sharpness weight
W_COMP: float = 3.0             # Composition weight
MIN_RAW: float = 2.9            # Minimum raw score
_RATING_BREAKS = [3.40, 3.80, 4.20, 4.60]  # Rating thresholds
```

---

## Troubleshooting

### "Fence classifier checkpoint not found"
**Solution**: Verify checkpoint exists at:
```
fence_classifier_checkpoints/mobilenetv2/best.pt
```

### P4 Training: "No images found"
**Solution**: Check directory structure:
```
vehicle_orientation_labels/
├── 正前方/  ← Must have at least one JPG
├── 侧身/    ← Must have at least one JPG
├── 正后方/  ← Must have at least one JPG
└── 侧后方/  ← Must have at least one JPG
```

### "Chinese path not found"
**Solution**: Already handled in code. If issue persists:
```python
import cv2
import numpy as np
img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
```

---

## References

### Key Publications & Techniques
- **Fence Detection**: ResNet50 backbone + MobileNetV2 optimization for real-time inference
- **Orientation**: ResNet50 4-class classifier with dropout regularization
- **Training**: BCEWithLogitsLoss (P3) + CrossEntropyLoss (P4) with balanced sampling

### Architecture Decisions
- **Why MobileNetV2 for P3?**: Best F1 (0.9796) + high throughput (4201 img/s)
- **Why ResNet50 for P4?**: Proven on similar tasks, good transfer learning properties
- **Why lazy loading?**: Avoid startup overhead; fence classifier only loaded on first use
- **Why separate veto checks?**: Fence is more reliable veto (0% FN) so runs first

---

## Contact & Issues

For questions or issues with:
- **P3 (Fence Detection)**: Check fence classifier output confidence scores in CSV
- **P4 (Orientation)**: Follow `P4_ORIENTATION_TASK.md` and `vehicle_orientation_labels/标注原则.md`
- **Integration**: Run `test_fence_integration.py` to verify pipeline
- **Performance**: Check GPU/memory usage; may need smaller batch sizes on limited VRAM

---

**Next Action**: User begins annotating `vehicle_orientation_labels/待标注/` images into 4 orientation classes. Agent monitors progress and can begin training P4 model once ~50% of annotations are complete.
