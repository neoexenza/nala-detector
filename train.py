#!/usr/bin/env python3
"""
Nala Detector - Fine-tuning Script
Trains a YOLOv8n classification model to distinguish:
  - nala (your cat)
  - other_cat (neighbourhood cats)
  - no_cat (empty terrace, people, foxes, etc.)
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
TRAINING_DIR = DATA_DIR / "training"
MODEL_DIR = Path("/models")
HISTORY_DIR = MODEL_DIR / "history"
BASE_MODEL = MODEL_DIR / "yolov8n-cls.pt"
CUSTOM_MODEL = MODEL_DIR / "nala_custom.pt"
MANIFEST_PATH = HISTORY_DIR / "manifest.json"
RELOAD_SIGNAL = MODEL_DIR / ".reload"

EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "50"))
IMGSZ = int(os.environ.get("TRAIN_IMGSZ", "224"))
BATCH = int(os.environ.get("TRAIN_BATCH", "8"))

# Class mapping: folder names -> labels
CLASSES = {
    "cat": "nala",        # Images confirmed as Nala
    "other_cat": "other_cat",  # Other neighbourhood cats
    "no_cat": "no_cat"    # No cat present
}


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
    """Prepare dataset in YOLO classification format."""
    dataset_dir = DATA_DIR / "dataset"

    # Clean previous dataset
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    # Count images per class
    class_counts = {}
    for folder, label in CLASSES.items():
        src = TRAINING_DIR / folder
        if src.exists():
            images = list(src.glob("*.jpg"))
            class_counts[label] = len(images)
        else:
            class_counts[label] = 0

    logger.info(f"Dataset: {class_counts}")

    # Check minimum images for nala class
    if class_counts.get("nala", 0) < 5:
        logger.error(f"Not enough Nala images ({class_counts.get('nala', 0)}). Need at least 5.")
        sys.exit(1)

    # Split 80/20 train/val for each class
    for folder, label in CLASSES.items():
        src = TRAINING_DIR / folder
        if not src.exists():
            continue

        images = sorted(src.glob("*.jpg"))
        if not images:
            continue

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

    return dataset_dir, class_counts


def train():
    """Run fine-tuning."""
    from ultralytics import YOLO

    dataset_dir, dataset_stats = prepare_dataset()

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

        # Signal detector to hot-reload
        signal_reload()

        logger.info("Model deployed and reload signalled.")
    else:
        logger.error("Training completed but no best.pt found!")
        sys.exit(1)


if __name__ == "__main__":
    train()
