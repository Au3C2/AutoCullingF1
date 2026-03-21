"""Analyze and merge three F1 YOLO datasets into unified 10-team dataset."""
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Dataset definitions ──────────────────────────────────────────────────────
DATASETS = {
    "f1_roboflow": {
        "root": Path("datasets/f1_roboflow"),
        "names": ['Ferrari', 'Mclaren', 'Mercedes', 'Redbull'],
    },
    "f1_teams": {
        "root": Path("datasets/f1_teams"),
        "names": ['Alpine', 'Aston Martin', 'Ferrari', 'Haas', 'Kick Sauber',
                   'McLaren', 'Mercedes', 'Racing Bulls', 'Red Bull', 'Williams'],
    },
    "f1_yolo_cars": {
        "root": Path("datasets/f1_yolo_cars"),
        "names": ['Alpine', 'AstonMartin', 'Ferrai', 'Ferrari', 'Haas',
                   'McLaren', 'Mercedes', 'RedBull', 'Sauber', 'VCARB',
                   'Williams', 'f1car', 'objects'],
    },
}

# ── Unified 10-team label mapping ────────────────────────────────────────────
# Target: 0=Alpine, 1=AstonMartin, 2=Ferrari, 3=Haas, 4=KickSauber,
#         5=McLaren, 6=Mercedes, 7=RacingBulls, 8=RedBull, 9=Williams
UNIFIED_NAMES = [
    'Alpine', 'AstonMartin', 'Ferrari', 'Haas', 'KickSauber',
    'McLaren', 'Mercedes', 'RacingBulls', 'RedBull', 'Williams',
]

# Map (dataset_name, old_class_id) → new_class_id (or None to drop)
LABEL_MAP: dict[tuple[str, int], int | None] = {}

def _build_label_map():
    # f1_roboflow: ['Ferrari', 'Mclaren', 'Mercedes', 'Redbull']
    LABEL_MAP[("f1_roboflow", 0)] = 2   # Ferrari → Ferrari
    LABEL_MAP[("f1_roboflow", 1)] = 5   # Mclaren → McLaren
    LABEL_MAP[("f1_roboflow", 2)] = 6   # Mercedes → Mercedes
    LABEL_MAP[("f1_roboflow", 3)] = 8   # Redbull → RedBull

    # f1_teams: ['Alpine','Aston Martin','Ferrari','Haas','Kick Sauber',
    #            'McLaren','Mercedes','Racing Bulls','Red Bull','Williams']
    LABEL_MAP[("f1_teams", 0)] = 0   # Alpine → Alpine
    LABEL_MAP[("f1_teams", 1)] = 1   # Aston Martin → AstonMartin
    LABEL_MAP[("f1_teams", 2)] = 2   # Ferrari → Ferrari
    LABEL_MAP[("f1_teams", 3)] = 3   # Haas → Haas
    LABEL_MAP[("f1_teams", 4)] = 4   # Kick Sauber → KickSauber
    LABEL_MAP[("f1_teams", 5)] = 5   # McLaren → McLaren
    LABEL_MAP[("f1_teams", 6)] = 6   # Mercedes → Mercedes
    LABEL_MAP[("f1_teams", 7)] = 7   # Racing Bulls → RacingBulls
    LABEL_MAP[("f1_teams", 8)] = 8   # Red Bull → RedBull
    LABEL_MAP[("f1_teams", 9)] = 9   # Williams → Williams

    # f1_yolo_cars: ['Alpine','AstonMartin','Ferrai','Ferrari','Haas',
    #                'McLaren','Mercedes','RedBull','Sauber','VCARB',
    #                'Williams','f1car','objects']
    LABEL_MAP[("f1_yolo_cars", 0)]  = 0    # Alpine → Alpine
    LABEL_MAP[("f1_yolo_cars", 1)]  = 1    # AstonMartin → AstonMartin
    LABEL_MAP[("f1_yolo_cars", 2)]  = 2    # Ferrai (typo) → Ferrari
    LABEL_MAP[("f1_yolo_cars", 3)]  = 2    # Ferrari → Ferrari
    LABEL_MAP[("f1_yolo_cars", 4)]  = 3    # Haas → Haas
    LABEL_MAP[("f1_yolo_cars", 5)]  = 5    # McLaren → McLaren
    LABEL_MAP[("f1_yolo_cars", 6)]  = 6    # Mercedes → Mercedes
    LABEL_MAP[("f1_yolo_cars", 7)]  = 8    # RedBull → RedBull
    LABEL_MAP[("f1_yolo_cars", 8)]  = 4    # Sauber → KickSauber
    LABEL_MAP[("f1_yolo_cars", 9)]  = 7    # VCARB → RacingBulls
    LABEL_MAP[("f1_yolo_cars", 10)] = 9    # Williams → Williams
    LABEL_MAP[("f1_yolo_cars", 11)] = None # f1car → DROP (ambiguous generic)
    LABEL_MAP[("f1_yolo_cars", 12)] = None # objects → DROP (not F1 car)

_build_label_map()


# ── Phase 1: Analyze ─────────────────────────────────────────────────────────
def analyze():
    """Print stats for each dataset."""
    for ds_name, ds_info in DATASETS.items():
        root = ds_info["root"]
        names = ds_info["names"]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"  Classes ({len(names)}): {names}")

        for split in ("train", "valid", "test"):
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"
            if not img_dir.exists():
                print(f"  {split}: NOT FOUND")
                continue

            n_imgs = len(list(img_dir.iterdir()))
            n_lbls = len(list(lbl_dir.iterdir())) if lbl_dir.exists() else 0

            # Count class occurrences
            class_counts: Counter = Counter()
            dropped = 0
            for lbl_file in sorted(lbl_dir.glob("*.txt")):
                for line in lbl_file.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    cls_id = int(line.split()[0])
                    new_id = LABEL_MAP.get((ds_name, cls_id))
                    if new_id is not None:
                        class_counts[UNIFIED_NAMES[new_id]] += 1
                    else:
                        class_counts[f"DROPPED({names[cls_id]})"] += 1
                        dropped += 1

            print(f"  {split}: {n_imgs} images, {n_lbls} labels, {sum(class_counts.values())} boxes")
            for cls_name, cnt in sorted(class_counts.items()):
                print(f"    {cls_name}: {cnt}")


# ── Phase 2: Merge ───────────────────────────────────────────────────────────
def merge(out_root: Path):
    """Merge all three datasets into out_root with unified labels."""
    if out_root.exists():
        print(f"Removing existing {out_root} ...")
        shutil.rmtree(out_root)

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"images": 0, "boxes": 0, "dropped_boxes": 0})

    for ds_name, ds_info in DATASETS.items():
        root = ds_info["root"]
        names = ds_info["names"]

        for split in ("train", "valid", "test"):
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"
            if not img_dir.exists():
                continue

            out_img_dir = out_root / split / "images"
            out_lbl_dir = out_root / split / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file in sorted(img_dir.iterdir()):
                stem = img_file.stem
                suffix = img_file.suffix

                # Unique name: prefix with dataset name to avoid collisions
                new_stem = f"{ds_name}__{stem}"
                new_img = out_img_dir / f"{new_stem}{suffix}"

                # Copy image
                shutil.copy2(img_file, new_img)

                # Remap labels
                lbl_file = lbl_dir / f"{stem}.txt"
                new_lines = []
                if lbl_file.exists():
                    for line in lbl_file.read_text().strip().split("\n"):
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        old_cls = int(parts[0])
                        new_cls = LABEL_MAP.get((ds_name, old_cls))
                        if new_cls is not None:
                            parts[0] = str(new_cls)
                            new_lines.append(" ".join(parts))
                            stats[split]["boxes"] += 1
                        else:
                            stats[split]["dropped_boxes"] += 1

                new_lbl = out_lbl_dir / f"{new_stem}.txt"
                new_lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
                stats[split]["images"] += 1

    # Write data.yaml
    yaml_content = (
        f"train: {(out_root / 'train' / 'images').resolve().as_posix()}\n"
        f"val: {(out_root / 'valid' / 'images').resolve().as_posix()}\n"
        f"test: {(out_root / 'test' / 'images').resolve().as_posix()}\n"
        f"\n"
        f"nc: {len(UNIFIED_NAMES)}\n"
        f"names: {UNIFIED_NAMES}\n"
    )
    (out_root / "data.yaml").write_text(yaml_content)

    print(f"\n{'='*60}")
    print(f"Merged dataset written to {out_root}")
    print(f"Classes ({len(UNIFIED_NAMES)}): {UNIFIED_NAMES}")
    for split in ("train", "valid", "test"):
        s = stats[split]
        print(f"  {split}: {s['images']} images, {s['boxes']} boxes, {s['dropped_boxes']} dropped")

    # Class distribution in merged train set
    print(f"\n--- Merged TRAIN class distribution ---")
    class_counts: Counter = Counter()
    train_lbl_dir = out_root / "train" / "labels"
    for lbl_file in train_lbl_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            cls_id = int(line.split()[0])
            class_counts[UNIFIED_NAMES[cls_id]] += 1
    for name in UNIFIED_NAMES:
        print(f"  {name}: {class_counts.get(name, 0)}")
    print(f"  TOTAL: {sum(class_counts.values())}")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        out = Path("datasets/f1_merged")
        analyze()
        merge(out)
    else:
        analyze()
        print("\n\nRun with --merge to create the merged dataset at datasets/f1_merged/")
