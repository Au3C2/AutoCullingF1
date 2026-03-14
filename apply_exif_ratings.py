import csv
import subprocess
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def update_single_image(img_path, rating):
    """Update a single image's metadata using ExifTool."""
    if not img_path.exists():
        return False, f"File not found: {img_path.name}"

    # Lightroom Metadata logic:
    # Rating: 0-5
    # Pick flag (XMP-xmpDM:Pick):
    #   1  = Picked (Flagged)
    #   0  = No flag
    #   -1 = Rejected
    
    et_rating = max(0, rating)
    pick_flag = 1 if rating > 0 else -1
    
    cmd = [
        "exiftool",
        "-overwrite_original",
        f"-XMP-xmp:Rating={et_rating}",
        f"-XMP-xmpDM:Pick={pick_flag}",
        str(img_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, img_path.name
    except subprocess.CalledProcessError as e:
        return False, f"Error updating {img_path.name}: {e.stderr.decode().strip()}"

def apply_ratings_to_jpg_multi(csv_path, img_dir, max_workers=8):
    csv_path = Path(csv_path)
    img_dir = Path(img_dir)
    
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return

    tasks = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            rating = int(row['rating'])
            img_path = img_dir / filename
            tasks.append((img_path, rating))

    total = len(tasks)
    logger.info(f"Starting batch update: {total} images using {max_workers} workers.")
    
    success_count = 0
    fail_count = 0
    
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(update_single_image, path, rat): path for path, rat in tasks}
        
        for i, future in enumerate(as_completed(future_to_img), 1):
            success, message = future.result()
            if success:
                success_count += 1
                if i % 100 == 0 or i == total:
                    logger.info(f"Progress: {i}/{total} images processed...")
            else:
                fail_count += 1
                logger.warning(message)

    end_time = time.perf_counter()
    duration = end_time - start_time
    
    logger.info("-" * 40)
    logger.info(f"Batch update complete in {duration:.1f}s ({total/duration:.1f} img/s)")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed : {fail_count}")
    logger.info("-" * 40)

if __name__ == "__main__":
    # Settings
    TARGET_CSV = "ysy_photos_latest.csv" 
    IMG_DIR = "/Users/joeylin/Pictures/YSY_PHOTOS/original_photos"
    WORKERS = 12  # Optimized for M-series Mac SSD
    
    apply_ratings_to_jpg_multi(TARGET_CSV, IMG_DIR, max_workers=WORKERS)
