# Nala Cat Detector

Cat detection system using YOLOv8 + MQTT. Designed for Ring camera snapshots delivered via ring-mqtt.

## How it works

1. Ring camera detects motion → publishes snapshot via MQTT
2. Nala detector picks up the snapshot
3. YOLO runs inference — is there a cat?
4. Result published to MQTT result topic
5. Home Assistant (or any MQTT consumer) can trigger notifications

## SQLite Backend

All detection events are stored in `data/nala.db` (WAL mode for concurrent access):

```sql
CREATE TABLE detections (
    id          TEXT PRIMARY KEY,  -- YYYYMMDD_N (incremental per day)
    timestamp   TEXT NOT NULL,     -- ISO 8601
    label       TEXT NOT NULL,     -- nala, other_cat, no_cat
    confidence  REAL,              -- 0.0-1.0 (NULL for no detection)
    frame_path  TEXT,              -- current file location (survives moves)
    in_dataset  INTEGER DEFAULT 0,
    notified    INTEGER DEFAULT 0
);
```

Frame paths are updated when images move between training/dataset folders, so references never break.

## Setup

```bash
cp .env.example .env
# Edit .env with your MQTT broker details
docker compose up -d
```

## Quick commands

```bash
# View logs
docker compose logs -f nala-detector

# Fine-tune via CLI (alternative to web UI)
docker compose run --rm --profile training training

# Restart detector (not needed if hot-reload is working)
docker compose restart nala-detector
```

## Web UI

Access at **http://\<host\>:5151**

The web UI provides a complete interface for managing training data and models.

### Detections (default tab)

- **All events:** Every motion trigger is logged (cat + no_cat), not just positives
- **Modal view:** Click any detection to open a fullscreen modal with the frame image
- **Label & curate:** Three label buttons (Nala / Other Cat / No Cat) + "Move to Dataset" in one click
- **Navigation:** Arrow keys or on-screen buttons to browse detections without closing the modal
- **Smart state:** Already-trained images show as locked; dataset images can be relabelled in-place
- **Pagination:** 30 detections per page with prev/next navigation

### Training Data Management

- **Tabs:** Browse images sorted into Nala / Other Cat / No Cat categories
- **Move:** Select images and move them between categories (e.g. re-classify a mislabelled image)
- **Delete:** Remove bad/blurry images that would hurt training
- **Select All / Deselect All:** Bulk operations for large batches

### Dataset

- **Paginated:** Each label group shows 10 images per page
- **Sections:** Split into "New — pending training" and "Trained" groups
- **Bulk select:** Select across labels for batch delete

### Training

- Click **Start Training** to begin fine-tuning
- Training runs YOLOv8n classification with the current dataset
- Status shown in real-time (in progress / completed / failed)
- Minimum 5 Nala images required to train
- Parameters: 50 epochs, 640px image size, batch size 4

### Model Management (Models tab)

- **Version history:** Every training run saves a versioned model (`nala_v1_YYYYMMDD_HHMMSS.pt`, etc.)
- **Active indicator:** Green badge shows which model is currently deployed
- **Metadata:** Each version shows training date, dataset composition (Nala/other/no_cat counts), and training parameters
- **Deploy:** Click to activate any historical model version — the detector hot-reloads within ~30 seconds
- **Rollback to Generic YOLO:** Revert to the base pre-trained model if a custom model performs poorly
- **Auto-deploy:** After successful training, the new model is automatically deployed and the detector reloads it

### Hot-Reload

The detector checks for model updates every 30 seconds. When a new model is deployed via the web UI (or after training completes), the detector picks it up automatically — no container restart required.

## Data directories

- `data/nala.db` — SQLite database (WAL mode) tracking all detection events
- `data/training/cat/` — confirmed Nala frames (auto-collected + manually sorted)
- `data/training/other_cat/` — other neighbourhood cats (for multi-class training)
- `data/training/no_cat/` — non-cat frames (empty terrace, people, foxes, etc.)
- `data/detections/` — all detection frames (cat + no_cat) with JSON metadata
- `data/dataset/train/` — curated training dataset (nala / other_cat / no_cat)
- `models/yolov8n.pt` — base pre-trained detection model
- `models/yolov8n-cls.pt` — base classification model (downloaded on first training)
- `models/nala_custom.pt` — currently active fine-tuned model
- `models/history/` — versioned model archive with manifest

## Fine-tuning workflow

1. Let the detector run collecting frames (auto-sorted into `no_cat/`)
2. Open the web UI and review images — move Nala frames from `no_cat` → `Nala` tab
3. Delete blurry/useless frames
4. Click **Start Training** in the web UI
5. New model auto-deploys on success; detector hot-reloads
6. Monitor detection accuracy, retrain with more data as needed

Keep all training images between runs — more data improves each successive model.

## Environment variables

See `.env.example` for all available options.

| Variable | Description | Default |
|----------|-------------|---------|
| MQTT_HOST | MQTT broker address | localhost |
| MQTT_PORT | MQTT broker port | 1883 |
| MQTT_USER | MQTT username | — |
| MQTT_PASS | MQTT password | — |
| CONFIDENCE_THRESHOLD | Min confidence to flag detection | 0.4 |
| NOTIFY_COOLDOWN | Seconds between notifications | 300 |
| COLLECT_DATA | Save frames for training | true |
| TRAIN_EPOCHS | Training epochs | 50 |
| TRAIN_IMGSZ | Training image size | 640 |
| TRAIN_BATCH | Training batch size | 4 |

## MQTT topics

| Topic | Direction | Description |
|-------|-----------|-------------|
| `ring/.../snapshot/image` | Subscribe | Incoming camera snapshots |
| `ring/.../motion/state` | Subscribe | Motion events |
| `nala/detection/result` | Publish | JSON detection result |

## Result payload

```json
{
  "detected": true,
  "confidence": 0.87,
  "detections": [{"class": "nala", "confidence": 0.87}],
  "id": "20260522_5",
  "timestamp": "2026-05-22T13:45:00",
  "notify": true
}
```

## Architecture

```
┌──────────────┐     MQTT      ┌────────────────┐     MQTT     ┌─────────────────┐
│ Ring Camera  │───────────────▶│ Nala Detector  │─────────────▶│ Home Assistant  │
│ (via ring-   │  snapshot/     │ (YOLOv8 + hot  │  result/     │ (notifications) │
│  mqtt)       │  motion        │  reload)       │  payload     │                 │
└──────────────┘               └────────────────┘              └─────────────────┘
                                       │
                                       ▼
                               ┌────────────────┐
                               │   Web UI :5151 │
                               │ • Image review │
                               │ • Training     │
                               │ • Model deploy │
                               └────────────────┘
```

## License

MIT
