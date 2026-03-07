import csv
import logging
from pathlib import Path
import cv2
import numpy as np

from cull.p4_classifier import get_p4_classifier
from cull.detector import load_f1_model, detect
from cull_photos import _load_image_rgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def main():
    classifier = get_p4_classifier()
    f1_model = load_f1_model(Path('models/f1_yolov8n.onnx'))

    with open('scores_f1_v4_fixed.csv', 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r)
        orig_scores = list(r)

    with open('test_set.csv', 'r', encoding='utf-8') as f:
        test_rows = list(csv.DictReader(f))

    path_map = {r['filename']: Path(r['hif_dir']) / r['filename'] for r in test_rows}
    
    # We'll just run inference on a sample to see distributions
    sample_size = 100
    tp = fp = fn = tn = 0
    
    log.info("Running P4 evaluation on sample...")
    
    kept_preds = []
    rejected_preds = []
    
    count = 0
    for row in orig_scores:
        fname = row[0]
        if fname not in path_map:
            continue
            
        has_arw = row[9] == '1'
        raw_score = float(row[3])
        path = path_map[fname]
        
        # Determine base veto
        s_sharp = float(row[1])
        n_det = int(row[7])
        base_vetoed = n_det == 0 or s_sharp < 0.05 or raw_score < 3.1
        
        # Only evaluate images that passed base veto, otherwise it's just noise
        if base_vetoed:
            continue
            
        try:
            img_rgb = _load_image_rgb(path, scale_width=1280)
            if img_rgb is None: continue
            
            dets = detect(img_rgb, f1_model, None, conf=0.3)
            primary_det = None
            for d in dets:
                if d.label == "f1_car":
                    primary_det = d
                    break
            
            if not primary_det:
                continue
                
            bbox = (primary_det.x1, primary_det.y1, primary_det.x2, primary_det.y2)
            o_str, o_conf, i_pred, i_prob = classifier.predict_roi(img_rgb, bbox)
            
            res = {
                'fname': fname,
                'has_arw': has_arw,
                'raw_score': raw_score,
                'orient': o_str,
                'o_conf': o_conf,
                'integ': i_pred,
                'i_prob': i_prob
            }
            if has_arw:
                kept_preds.append(res)
            else:
                rejected_preds.append(res)
                
            count += 1
            if count >= sample_size:
                break
                
        except Exception as e:
            log.warning(f"Error on {fname}: {e}")
            
    # Analyze
    log.info(f"Evaluated {count} images that passed base vetoes.")
    
    log.info("--- KEPT IMAGES (has_arw=1) ---")
    o_counts = {}
    i_counts = {1:0, 0:0}
    for r in kept_preds:
        o = r['orient']
        o_counts[o] = o_counts.get(o, 0) + 1
        i_counts[r['integ']] += 1
    log.info(f"Orientations: {o_counts}")
    log.info(f"Integrity (1=Full, 0=Cut): {i_counts}")
    
    log.info("--- REJECTED IMAGES (has_arw=0) ---")
    o_counts = {}
    i_counts = {1:0, 0:0}
    for r in rejected_preds:
        o = r['orient']
        o_counts[o] = o_counts.get(o, 0) + 1
        i_counts[r['integ']] += 1
    log.info(f"Orientations: {o_counts}")
    log.info(f"Integrity (1=Full, 0=Cut): {i_counts}")

if __name__ == '__main__':
    main()
