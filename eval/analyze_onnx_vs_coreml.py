import csv
from pathlib import Path

def analyze_diffs(file1, file2):
    onnx_data = {}
    with open(file1, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            onnx_data[row['filename']] = {
                'rating': int(row['rating']),
                'raw': float(row['raw_score']),
                'veto': row['veto_reason']
            }
            
    coreml_data = {}
    with open(file2, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coreml_data[row['filename']] = {
                'rating': int(row['rating']),
                'raw': float(row['raw_score']),
                'veto': row['veto_reason']
            }
            
    mismatches = []
    total = 0
    
    # Common filenames
    filenames = sorted(set(onnx_data.keys()) & set(coreml_data.keys()))
    
    for fname in filenames:
        total += 1
        r1 = onnx_data[fname]['rating']
        r2 = coreml_data[fname]['rating']
        
        if r1 != r2:
            mismatches.append({
                'filename': fname,
                'onnx': r1,
                'coreml': r2,
                'raw_diff': coreml_data[fname]['raw'] - onnx_data[fname]['raw']
            })
            
    print(f"Total images compared: {total}")
    print(f"Total mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\nDetail of Mismatches (Rating differences):")
        print(f"{'Filename':<25} | {'ONNX':<8} | {'CoreML':<8} | {'Raw Diff':<10}")
        print("-" * 60)
        # Show first 20 mismatches
        for m in mismatches[:20]:
            r1_str = "Reject" if m['onnx'] == -1 else f"{m['onnx']} star"
            r2_str = "Reject" if m['coreml'] == -1 else f"{m['coreml']} star"
            print(f"{m['filename']:<25} | {r1_str:<8} | {r2_str:<8} | {m['raw_diff']:+.6f}")
            
        if len(mismatches) > 20:
            print(f"... and {len(mismatches) - 20} more.")
            
        # Summary of shifts
        shifts = {}
        for m in mismatches:
            key = (m['onnx'], m['coreml'])
            shifts[key] = shifts.get(key, 0) + 1
            
        print("\nSummary of Rating Shifts (ONNX -> CoreML):")
        for (r1, r2), count in sorted(shifts.items()):
            r1_s = "Reject" if r1 == -1 else f"{r1}*"
            r2_s = "Reject" if r2 == -1 else f"{r2}*"
            print(f"  {r1_s}  --->  {r2_s} : {count} images")

if __name__ == "__main__":
    analyze_diffs('culling_result_onnx.csv', 'culling_result_coreml.csv')
