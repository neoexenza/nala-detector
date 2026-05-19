#!/usr/bin/env python3
"""
Nala Fine-Tuning Script
Fine-tunes the YOLO model on collected training data.
Run manually: docker compose run --rm training
"""

import os
import sys
import shutil
import random
from pathlib import Path

from ultralytics import YOLO

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = Path("/models")
DATASET_DIR = DATA_DIR / "dataset"

BASE_MODEL = os.environ.get("BASE_MODEL", "/models/yolov8n.pt")
OUTPUT_MODEL = MODELS_DIR / "nala_custom.pt"
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "50"))
IMG_SIZE = int(os.environ.get("TRAIN_IMGSZ", "640"))
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH", "4"))
TRAIN_SPLIT = 0.8


def prepare_dataset():
    """Prepare YOLO-format dataset from collected images."""
    cat_dir = TRAINING_DIR / "cat"
    no_cat_dir = TRAINING_DIR / "no_cat"

    cat_images = list(cat_dir.glob("*.jpg"))
    no_cat_images = list(no_cat_dir.glob("*.jpg"))

    print(f"Cat images: {len(cat_images)}")
    print(f"No-cat images: {len(no_cat_images)}")

    if len(cat_images) < 20:
        print(f"ERROR: Need at least 20 cat images for fine-tuning. "
              f"Currently have {len(cat_images)}. Keep collecting!")
        sys.exit(1)

    # Create YOLO classification dataset structure
    for split in ["train", "val"]:
        for cls in ["cat", "no_cat"]:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Split and copy images
    random.shuffle(cat_images)
    random.shuffle(no_cat_images)

    split_idx_cat = int(len(cat_images) * TRAIN_SPLIT)
    split_idx_nocat = int(len(no_cat_images) * TRAIN_SPLIT)

    for img in cat_images[:split_idx_cat]:
        shutil.copy2(img, DATASET_DIR / "train" / "cat" / img.name)
    for img in cat_images[split_idx_cat:]:
        shutil.copy2(img, DATASET_DIR / "val" / "cat" / img.name)

    for img in no_cat_images[:split_idx_nocat]:
        shutil.copy2(img, DATASET_DIR / "train" / "no_cat" / img.name)
    for img in no_cat_images[split_idx_nocat:]:
        shutil.copy2(img, DATASET_DIR / "val" / "no_cat" / img.name)

    print(f"Dataset prepared:")
    print(f"  Train: {split_idx_cat} cat, {split_idx_nocat} no_cat")
    print(f"  Val: {len(cat_images) - split_idx_cat} cat, "
          f"{len(no_cat_images) - split_idx_nocat} no_cat")

    return DATASET_DIR


def train():
    """Run fine-tuning."""
    print(f"\n{'='*60}")
    print(f"NALA FINE-TUNING")
    print(f"{'='*60}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    dataset_path = prepare_dataset()

    # Load base model
    model = YOLO(BASE_MODEL)

    # Fine-tune as classifier
    results = model.train(
        data=str(dataset_path),
        task="classify",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        project=str(DATA_DIR / "runs"),
        name="nala_finetune",
        exist_ok=True,
        patience=10,
        save=True,
    )

    # Copy best model to models directory
    best_model = DATA_DIR / "runs" / "nala_finetune" / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy2(best_model, OUTPUT_MODEL)
        print(f"\n✅ Custom model saved to: {OUTPUT_MODEL}")
        print("Restart the detector container to load the new model.")
    else:
        print("\n❌ Training completed but no best.pt found.")
        sys.exit(1)


if __name__ == "__main__":
    train()
