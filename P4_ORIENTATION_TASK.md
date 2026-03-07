# P4 Vehicle Orientation Annotation Task

## Overview

We have completed **P3: Wire Fence Detection** with excellent results (F1=0.9796). 

Now we're moving to **P4: Vehicle Orientation Recognition** — a 4-class classifier to recognize car heading angles.

## What You Need to Do

### Task: Classify 608 ROI Images by Vehicle Orientation

**Location**: `vehicle_orientation_labels/待标注/` — 608 JPG images await classification

**Time Estimate**: ~10-20 hours total (~1-2 minutes per image)

### 4 Orientation Classes

| Class | 中文 | English | Heading | Visual Cue |
|-------|------|---------|---------|-----------|
| `正前方` | Head-on | 0° | Car head toward camera | Symmetric, narrowest, front bumper visible |
| `侧身` | Side profile | 90° | Complete side view | Full car width visible, no front/rear |
| `正后方` | Rear-facing | 180° | Car tail toward camera | Mirror of head-on, rear wing visible |
| `侧后方` | Diagonal | 45°-135° | Quarter rear view | Side + tail/head visible, during overtake |

### How to Annotate

1. **Open** `vehicle_orientation_labels/待标注/` in file explorer
2. **For each image**:
   - Look at the ROI (crop region with the car)
   - Determine which of 4 orientations it matches
   - **Move** the file to the corresponding subfolder:
     - `vehicle_orientation_labels/正前方/`
     - `vehicle_orientation_labels/侧身/`
     - `vehicle_orientation_labels/正后方/`
     - `vehicle_orientation_labels/侧后方/`
3. **If uncertain**: Delete the file (cannot judge orientation clearly)

### Key Guidelines

- **Judge by car's main axis**, not camera angle (overhead/low shot)
- **Ignore background** (track, spectators, other objects)
- **Edge cases**:
  - Extreme head-on (< 10°) → `正前方`
  - Extreme rear (> 170°) → `正后方`
  - Mid-range diagonal → `侧后方`
- **Very unclear/no car?** → Delete

### Detailed Guidelines

See `vehicle_orientation_labels/标注原则.md` (Chinese) for full annotation rules with examples.

## Timeline

### Current Status (Sat Mar 7, 2026)

✅ **P0-P3 Completed**:
- Sharpness detection (P0): Works well
- Composition analysis (P1): Integrated
- Object detection (P2): YOLO F1 car detector
- Wire fence detection (P3): F1=0.9796 (MobileNetV2)

🔄 **P4 In Progress**:
- User annotation phase (608 images, 10-20 hours)
- Data prep ready (folder structure created)

### Next Steps (Agent Workflow)

1. **User completes 50-100 images** → Send sample to agent for preliminary training
2. **Agent trains initial 4-class model** while user continues annotations (parallel)
3. **User finishes remaining 500+ images** → Agent retrains with full dataset
4. **Final evaluation**: P3 vs P3+P4 combined F1 score
5. **Publication**: Results + recommendations

## Expected Impact

- **P3 (Fence veto alone)**: Filters ~1-2% additional images
- **P3+P4 (Orientation veto)**: Expected to filter 5-10% more (poorly-angled shots)
- **Combined**: Estimated **3-4pp improvement in keeper quality** (fewer bad angles)

## File Structure

```
vehicle_orientation_labels/
├── 标注原则.md                     ← Full annotation guidelines (Chinese)
├── 待标注/                         ← YOUR WORK AREA: 608 images to classify
├── 正前方/                         ← (empty, you populate)
├── 侧身/                           ← (empty, you populate)
├── 正后方/                         ← (empty, you populate)
└── 侧后方/                         ← (empty, you populate)
```

## Questions?

If you have questions about orientation classification:
1. Re-read the **4 class definitions** above
2. Check **vehicle_orientation_labels/标注原则.md** for detailed examples
3. When in doubt, use **visual symmetry**:
   - Head-on: Most symmetric
   - Rear: Mirror of head-on
   - Side: Full width, no symmetry
   - Diagonal: Partial side + front/rear

---

**Ready to start?** Open `vehicle_orientation_labels/待标注/` and begin sorting images. 

**Check-in**: After completing 100 images, you can send a sample to verify you're on the right track.

**Estimated timeline**: 2-3 weeks at 1 hour per day = full annotation completion.
