# Nala Cat Detector

Cat detection system using YOLOv8 + MQTT. Designed for Ring camera snapshots delivered via ring-mqtt.

## How it works

1. Ring camera detects motion → publishes snapshot via MQTT
2. Nala detector picks up the snapshot
3. YOLO runs inference — is there a cat?
4. Result published to MQTT result topic
5. Home Assistant (or any MQTT consumer) can trigger notifications

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

### Training Data Management

- **Tabs:** Browse images sorted into Nala / Other Cat / No Cat categories
- **Move:** Select images and move them between categories (e.g. re-classify a mislabelled image)
- **Delete:** Remove bad/blurry images that would hurt training
- **Select All / Deselect All:** Bulk operations for large batches

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

- `data/training/cat/` — confirmed Nala frames (auto-collected + manually sorted)
- `data/training/other_cat/` — other neighbourhood cats (for multi-class training)
- `data/training/no_cat/` — non-cat frames (empty terrace, people, foxes, etc.)
- `data/detections/` — confirmed cat detections with metadata
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
  "timestamp": "20260519_134500",
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
