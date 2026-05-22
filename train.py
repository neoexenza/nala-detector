#!/usr/bin/env python3
"""
Nala Detector - Fine-tuning Script
Trains a YOLOv8n classification model to distinguish:
  - nala (your cat)
  - other_cat (neighbourhood cats)
  - no_cat (empty terrace, people, foxes, etc.)

Trains from base model on ALL dataset images (dataset/train/{nala,other_cat,no_cat}).
After training, marks all files as "trained" in trained_files.json.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nala-train")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DATASET_DIR = DATA_DIR / "dataset" / "train"
MODEL_DIR = Path("/models")
HISTORY_DIR = MODEL_DIR / "history"
BASE_MODEL = MODEL_DIR / "yolov8n-cls.pt"
CUSTOM_MODEL = MODEL_DIR / "nala_custom.pt"
MANIFEST_PATH = HISTORY_DIR / "manifest.json"
RELOAD_SIGNAL = MODEL_DIR / ".reload"
TRAINED_FILES_PATH = DATA_DIR / "trained_files.json"

EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "50"))
IMGSZ = int(os.environ.get("TRAIN_IMGSZ", "224"))
BATCH = int(os.environ.get("TRAIN_BATCH", "8"))

LABELS = ["nala", "other_cat", "no_cat"]


def load_manifest():
    """Load the model version manifest."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {"versions": [], "next_version": 1}


def save_manifest(manifest):
    """Save manifest to disk."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def get_next_version(manifest):
    """Get the next version number."""
    return manifest.get("next_version", 1)


def load_trained_files():
    """Load set of filenames that have been used in training."""
    if TRAINED_FILES_PATH.exists():
        try:
            with open(TRAINED_FILES_PATH, "r") as f:
                data = json.load(f)
            return set(data.get("files", []))
        except:
            return set()
    return set()


def save_trained_files(files_set):
    """Save trained files manifest."""
    with open(TRAINED_FILES_PATH, "w") as f:
        json.dump({"files": sorted(files_set)}, f, indent=2)


def save_versioned_model(model_path, dataset_stats):
    """Save model to history with version number and metadata."""
    manifest = load_manifest()
    version = get_next_version(manifest)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nala_v{version}_{timestamp}.pt"
    dest = HISTORY_DIR / filename

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, dest)

    entry = {
        "version": version,
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_stats,
        "training_params": {
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH
        },
        "active": True
    }

    # Mark all previous as not active
    for v in manifest["versions"]:
        v["active"] = False

    manifest["versions"].append(entry)
    manifest["next_version"] = version + 1
    save_manifest(manifest)

    logger.info(f"Saved versioned model: {filename} (v{version})")
    return filename


def signal_reload():
    """Write reload signal file for the detector to pick up."""
    RELOAD_SIGNAL.write_text(datetime.now().isoformat())
    logger.info("Wrote reload signal for detector")


def prepare_dataset():
    """Prepare dataset for training using 80/20 train/val split.
    
    Uses images already in dataset/train/{nala,other_cat,no_cat}/.
    Creates a temporary split directory for YOLO training format.
    """
    split_dir = DATA_DIR / "dataset_split"

    # Clean previous split
    if split_dir.exists():
        shutil.rmtree(split_dir)

    train_dir = split_dir / "train"
    val_dir = split_dir / "val"

    # Count and split images per class
    class_counts = {}
    all_files_used = set()

    for label in LABELS:
        src = DATASET_DIR / label
        if not src.exists():
            class_counts[label] = 0
            continue

        images = sorted(src.glob("*.jpg"))
        class_counts[label] = len(images)

        if not images:
            continue

        # Track all files
        for img in images:
            all_files_used.add(img.name)

        split_idx = max(1, int(len(images) * 0.8))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create class directories
        (train_dir / label).mkdir(parents=True, exist_ok=True)
        (val_dir / label).mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_dir / label / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / label / img.name)

        logger.info(f"  {label}: {len(train_images)} train, {len(val_images)} val")

    logger.info(f"Dataset: {class_counts}")
    return split_dir, class_counts, all_files_used


def train():
    """Run fine-tuning."""
    from ultralytics import YOLO

    # Check minimum images
    nala_dir = DATASET_DIR / "nala"
    nala_count = len(list(nala_dir.glob("*.jpg"))) if nala_dir.exists() else 0
    if nala_count < 5:
        logger.error(f"Not enough Nala images ({nala_count}). Need at least 5.")
        sys.exit(1)

    dataset_dir, dataset_stats, all_files_used = prepare_dataset()

    # Use YOLOv8n-cls (classification variant)
    if not BASE_MODEL.exists():
        logger.info("Downloading YOLOv8n-cls base model...")
        model = YOLO("yolov8n-cls.pt")
        # Save it for future runs
        shutil.copy2("yolov8n-cls.pt", BASE_MODEL)
    else:
        model = YOLO(str(BASE_MODEL))

    logger.info(f"Starting training: {EPOCHS} epochs, imgsz={IMGSZ}, batch={BATCH}")
    results = model.train(
        data=str(dataset_dir),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(DATA_DIR / "runs"),
        name="nala_finetune",
        exist_ok=True,
        verbose=True
    )

    # Copy best model to expected location
    best_model = DATA_DIR / "runs" / "nala_finetune" / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy2(best_model, CUSTOM_MODEL)
        logger.info(f"Custom model saved to {CUSTOM_MODEL}")

        # Save versioned copy to history
        save_versioned_model(CUSTOM_MODEL, dataset_stats)

        # Mark all dataset files as trained
        trained_files = load_trained_files()
        trained_files.update(all_files_used)
        save_trained_files(trained_files)
        logger.info(f"Marked {len(all_files_used)} files as trained ({len(trained_files)} total)")

        # Signal detector to hot-reload
        signal_reload()

        # Clean up split directory
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        logger.info("Model deployed and reload signalled.")
    else:
        logger.error("Training completed but no best.pt found!")
        sys.exit(1)


if __name__ == "__main__":
    train()
