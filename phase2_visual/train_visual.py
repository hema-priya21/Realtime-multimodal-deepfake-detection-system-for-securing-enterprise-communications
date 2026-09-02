import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "visual" / "train"
MODEL_PATH = PROJECT_ROOT / "phase2_visual" / "visual_model.pth"

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16  # Increased for higher GPU throughput
EPOCHS = 3
LEARNING_RATE = 0.0005
NUM_WORKERS = min(4, os.cpu_count() or 0)  # Parallel data loading

# ============================================================
# DEVICE & AMP CONFIGURATION
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
    USE_AMP = True
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU"
    USE_AMP = False

# ============================================================
# FORMAT TIME
# ============================================================

def format_time(seconds):
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ============================================================
# PRINT HEADER
# ============================================================

def print_header():
    print()
    print("=" * 80)
    print("           VISUAL DEEPFAKE DETECTION TRAINING")
    print("=" * 80)
    print()
    print("Training device :", DEVICE)

    if torch.cuda.is_available():
        print("GPU             :", GPU_NAME)
        print("CUDA version    :", torch.version.cuda)
        total_memory = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        )
        print(f"GPU memory      : {total_memory:.2f} GB")
        print("Mixed Precision : Enabled (FP16 AMP)")
    else:
        print("GPU             : Not detected")
        print("WARNING         : Training on CPU")

    print()
    print("Dataset         :", DATASET_PATH)
    print("Batch size      :", BATCH_SIZE)
    print("Epochs          :", EPOCHS)
    print("Num Workers     :", NUM_WORKERS)
    print("Image size      :", f"{IMAGE_SIZE} x {IMAGE_SIZE}")
    print("=" * 80)

# ============================================================
# CHECK DATASET
# ============================================================

def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    if not folder.exists():
        return 0
    return sum(
        1 for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    )

def check_dataset():
    print()
    print("=" * 80)
    print("                    DATASET CHECK")
    print("=" * 80)

    if not DATASET_PATH.exists():
        print("\nERROR: Dataset folder not found:\n", DATASET_PATH)
        print("\nRun prepare_dataset.py first.")
        raise SystemExit(1)

    real_folder = DATASET_PATH / "real"
    fake_folder = DATASET_PATH / "fake"

    real_count = count_images(real_folder)
    fake_count = count_images(fake_folder)
    total_count = real_count + fake_count

    print(f"\nREAL images : {real_count:,}")
    print(f"FAKE images : {fake_count:,}")
    print(f"TOTAL images: {total_count:,}\n")

    if real_count == 0:
        print("ERROR: No REAL images found.")
        raise SystemExit(1)

    if fake_count == 0:
        print("ERROR: No FAKE images found.")
        raise SystemExit(1)

    print("Dataset status: READY")
    print("=" * 80)
    return total_count

# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ============================================================
# CREATE MODEL
# ============================================================

def create_model():
    print()
    print("=" * 80)
    print("                    LOADING MODEL")
    print("=" * 80)
    print("\nModel: MobileNetV3 Small")
    print("Loading pretrained weights...")

    try:
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        print("Pretrained weights loaded.")
    except Exception as e:
        print("\nCould not download pretrained weights.")
        print("Using model without pretrained weights.")
        print("Reason:", e)
        model = models.mobilenet_v3_small(weights=None)

    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 2)
    model = model.to(DEVICE)

    print("\nOutput classes:")
    print("0 = fake")
    print("1 = real\n")
    print("Model moved to:", DEVICE)
    print("=" * 80)

    return model

# ============================================================
# GPU MEMORY
# ============================================================

def gpu_memory():
    if not torch.cuda.is_available():
        return "N/A"
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    return f"{allocated:.2f} GB used / {reserved:.2f} GB reserved"

# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    total_batches = len(loader)
    total_images = len(loader.dataset)
    running_loss = 0.0
    correct = 0
    processed_images = 0
    epoch_start = time.time()

    print("\n\n" + "=" * 80)
    print(f"                    EPOCH {epoch}/{EPOCHS}")
    print("=" * 80 + "\n")

    for batch_number, (images, labels) in enumerate(loader, start=1):
        batch_start = time.time()

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision for faster GPU execution
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size_actual = images.size(0)
        processed_images += batch_size_actual
        running_loss += loss.item() * batch_size_actual

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

        current_accuracy = (correct / processed_images) * 100
        percentage = (batch_number / total_batches) * 100
        elapsed = time.time() - epoch_start
        images_per_second = processed_images / elapsed if elapsed > 0 else 0

        remaining_batches = total_batches - batch_number
        average_batch_time = elapsed / batch_number
        eta = average_batch_time * remaining_batches
        batch_time = time.time() - batch_start

        bar_length = 30
        filled = int(bar_length * batch_number / total_batches)
        bar = "#" * filled + "-" * (bar_length - filled)

        print(
            f"\r[{bar}] {percentage:6.2f}% | "
            f"Batch {batch_number}/{total_batches} | "
            f"Images {processed_images:,}/{total_images:,} | "
            f"Loss {loss.item():.4f} | "
            f"Acc {current_accuracy:6.2f}% | "
            f"{images_per_second:6.1f} img/s | "
            f"ETA {format_time(eta)}",
            end="",
            flush=True
        )

        if batch_number % 50 == 0 or batch_number == total_batches:
            print()
            print(f"    Batch time : {batch_time:.2f} sec")
            print(f"    GPU memory : {gpu_memory()}")

    print()
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / processed_images
    epoch_accuracy = (correct / processed_images) * 100

    print("\n" + "-" * 80)
    print(f"EPOCH {epoch} FINISHED")
    print(f"Images processed : {processed_images:,}")
    print(f"Loss             : {epoch_loss:.4f}")
    print(f"Accuracy         : {epoch_accuracy:.2f}%")
    print(f"Epoch time       : {format_time(epoch_time)}")
    print(f"Speed            : {processed_images / epoch_time:.2f} images/sec")
    print("-" * 80)

    return epoch_loss, epoch_accuracy

# ============================================================
# MAIN
# ============================================================

def train_model():
    print_header()
    total_images = check_dataset()

    print("\nLoading ImageFolder dataset...")
    train_dataset = datasets.ImageFolder(
        root=str(DATASET_PATH),
        transform=transform
    )

    print("\nClasses:", train_dataset.classes)
    print("Class mapping:", train_dataset.class_to_idx)
    print("Dataset size:", f"{len(train_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0)
    )

    print("\nTotal batches per epoch:", len(train_loader))
    print("Batch size:", BATCH_SIZE)

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    training_start = time.time()
    best_accuracy = 0.0

    print("\n" + "=" * 80)
    print("                  STARTING TRAINING")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        loss, accuracy = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), str(MODEL_PATH))

            print("\n***** BEST MODEL SAVED *****")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"File: {MODEL_PATH}")
            print("******************************")

    total_time = time.time() - training_start

    print("\n\n" + "=" * 80)
    print("                  TRAINING COMPLETE")
    print("=" * 80 + "\n")
    print(f"Total images      : {total_images:,}")
    print(f"Epochs completed  : {EPOCHS}")
    print(f"Best accuracy     : {best_accuracy:.2f}%")
    print(f"Total time        : {format_time(total_time)}")
    print("\nModel saved to:\n", MODEL_PATH)
    print("\n" + "=" * 80)

if __name__ == "__main__":
    train_model()