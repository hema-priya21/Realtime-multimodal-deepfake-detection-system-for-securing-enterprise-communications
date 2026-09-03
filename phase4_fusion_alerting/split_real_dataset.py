import random
import shutil
from pathlib import Path

# Explicit project root path
PROJECT_ROOT = Path(r"C:\Users\kslok\OneDrive\Desktop\Realtime-multimodal-deepfake-detection-system-for-securing-enterprise-communications-main")

TRAIN_REAL_DIR = PROJECT_ROOT / "data" / "visual" / "train" / "real"
VAL_REAL_DIR = PROJECT_ROOT / "data" / "visual" / "val" / "real"

VAL_REAL_DIR.mkdir(parents=True, exist_ok=True)

# Find all real images
real_images = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.PNG", "*.JPG"):
    real_images.extend(list(TRAIN_REAL_DIR.glob(ext)))

if not real_images:
    print(f"ERROR: No real images found in {TRAIN_REAL_DIR}")
    exit()

# Cap validation set at 2,000 images to prevent storage bloat and OneDrive sync locks
val_count = min(2000, max(1, int(len(real_images) * 0.2)))
images_to_copy = random.sample(real_images, val_count)

print(f"Found {len(real_images)} real images. Copying {val_count} images to val/real...")

copied_count = 0
for img_path in images_to_copy:
    dest_path = VAL_REAL_DIR / img_path.name
    try:
        if not dest_path.exists():
            shutil.copy(str(img_path), str(dest_path))
        copied_count += 1
    except Exception:
        continue

print(f"Successfully copied {copied_count} real images into val/real!")