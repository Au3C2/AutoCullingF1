import pandas as pd
import os

def main():
    # 1. Load the original scores
    scores_path = 'scores_f1_v4.csv'
    if not os.path.exists(scores_path):
        print(f"Error: {scores_path} not found.")
        return
        
    df_scores = pd.read_csv(scores_path)
    print(f"Original scores loaded: {len(df_scores)} rows")
    
    # 2. Load the error review with feedback
    review_path = 'error_review/error_review.csv'
    if not os.path.exists(review_path):
        print(f"Error: {review_path} not found.")
        return
        
    df_review = pd.read_csv(review_path)
    print(f"Review data loaded: {len(df_review)} rows")
    
    # 3. Define the rules to override labels based on feedback text
    # Looking at the feedback, there are some patterns where the user says "可以给1/2/3/4分" or "能打2/3分"
    # This implies these should probably be KEEP (1) instead of DISCARD (0), turning False Positives into True Positives
    # Also for False Negatives, "可以被拒绝", "需要被拒绝" means they should be DISCARD (0), turning FN into True Negatives
    
    # Create mapping from filename to new target label (-1 means no change)
    label_updates = {}
    
    for _, row in df_review.iterrows():
        if pd.isna(row['feedback']):
            continue
            
        filename = row['filename']
        feedback = str(row['feedback']).lower()
        error_type = row['error_type']
        
        # Default label based on original ground truth logic:
        # group_arw_count > 0 means the photographer kept SOME photos from this burst
        # But for FP/FN in this dataset, it's specific to the image.
        
        new_label = -1 
        
        # False Positives (Model said 1, Truth said 0) -> Should it be 1?
        if error_type == 'FP':
            # User says it can be kept (e.g. "能打X分", "可以给X分", "保留的那张比较好")
            positive_phrases = ['能打', '可以给', '保留', '可被留下']
            if any(phrase in feedback for phrase in positive_phrases):
                new_label = 1 # Update truth to Keep
                
        # False Negatives (Model said 0, Truth said 1) -> Should it be 0?
        elif error_type == 'FN':
            # User says it SHOULD be rejected (e.g. "被拒绝", "拒绝也行")
            negative_phrases = ['被拒绝', '拒绝也行']
            if any(phrase in feedback for phrase in negative_phrases):
                new_label = 0 # Update truth to Discard
                
        if new_label != -1:
            label_updates[filename] = new_label
            
    # Apply updates
    updates_applied = 0
    
    def update_label(row):
        nonlocal updates_applied
        fname = row['filename']
        if fname in label_updates:
            # The column in scores.csv indicating ground truth is 'is_kept' or we need to look at tune_params.py to see how it loads truth
            # tune_params.py uses: df['group_arw_count'] > 0 as a proxy if 'is_kept' isn't there, 
            # wait, let me check tune_params.py logic for loading scores.
            pass # We'll do this in a second pass after checking columns
            
    print(f"Identified {len(label_updates)} label updates based on text rules.")
    
    # Print a few examples
    for i, (fname, label) in enumerate(label_updates.items()):
        if i >= 10: break
        print(f"  {fname} -> {label}")
        
    # We need to see how tune_params.py gets the ground truth.
    # Usually it's read from the CSV directly, let's print the columns of scores_f1_v4.csv
    print(f"\nColumns in {scores_path}: \n{list(df_scores.columns)}")

if __name__ == '__main__':
    main()
