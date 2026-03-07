import os
import argparse
from pathlib import Path
import csv
import cv2
import random

# Use the existing project utilities
from cull_photos import _load_image_rgb
from cull.detector import load_f1_model, detect

def crop_roi(img_rgb, detection, pad_ratio=0.0):
    """Crop the bounding box from the image, converting back to BGR for saving."""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
    
    # Optional padding to give a bit of context
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * pad_ratio), int(bh * pad_ratio)
    
    x1 = max(0, int(x1) - pad_w)
    y1 = max(0, int(y1) - pad_h)
    x2 = min(w, int(x2) + pad_w)
    y2 = min(h, int(y2) + pad_h)
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    roi_rgb = img_rgb[y1:y2, x1:x2]
    return cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv", type=str, default="test_set.csv", help="CSV containing image paths")
    parser.add_argument("--out-dir", type=str, default="p4_data/unlabeled", help="Directory to save ROIs")
    parser.add_argument("--max-samples", type=int, default=1500, help="Maximum number of ROIs to extract")
    parser.add_argument("--f1-model", type=str, default="models/f1_yolov8n.onnx", help="Path to F1 YOLO model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading F1 YOLO model...")
    f1_model = load_f1_model(Path(args.f1_model))
    
    # Find images
    image_paths = []
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'hif_dir' in row and 'filename' in row:
                image_paths.append(Path(row['hif_dir']) / row['filename'])
                
    print(f"Found {len(image_paths)} images to process from {args.csv}")
    
    # Shuffle for a representative sample
    random.seed(42)
    random.shuffle(image_paths)
    
    count = 0
    for i, path in enumerate(image_paths):
        if count >= args.max_samples:
            break
            
        try:
            # We scale down during decode to roughly 1280 to save memory/time
            img_rgb = _load_image_rgb(path, scale_width=1280)
            if img_rgb is None:
                continue
                
            # We don't need COCO fallback for F1 cars here
            dets = detect(img_rgb, f1_model, None, conf=0.3)
            
            # Extract top F1 car only (to avoid background/crowd noise)
            primary_det = None
            for d in dets:
                if d.label == "f1_car":
                    primary_det = d
                    break
                    
            if not primary_det:
                continue
                
            # Add 15% padding to include context and recover parts that YOLO might have cropped tightly
            roi_bgr = crop_roi(img_rgb, primary_det, pad_ratio=0.15)
            if roi_bgr is None or roi_bgr.size == 0:
                continue
                
            # Also filter out absurdly small boxes (likely artifacts or far away)
            h, w = roi_bgr.shape[:2]
            if w < 50 or h < 50:
                continue
                
            out_name = f"{path.parent.name}_{path.stem}_roi.jpg"
            out_path = out_dir / out_name
            
            # Save using cv2.imencode to handle any special chars in path
            success, buffer = cv2.imencode('.jpg', roi_bgr)
            if success:
                with open(out_path, 'wb') as f:
                    f.write(buffer)
                count += 1
                
            if count % 100 == 0:
                print(f"Extracted {count}/{args.max_samples} ROIs...")
                
        except Exception as e:
            continue
            
    print(f"\nDone! Extracted {count} ROIs to {out_dir}")

if __name__ == "__main__":
    main()
