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

# Fine-tune (after collecting enough images)
docker compose run --rm --profile training training

# Restart to load new model after training
docker compose restart nala-detector
```

## Data directories

- `data/training/cat/` — auto-collected cat frames
- `data/training/no_cat/` — auto-collected non-cat frames
- `data/detections/` — confirmed cat detections with metadata
- `models/yolov8n.pt` — base pre-trained model (downloaded at build)
- `models/nala_custom.pt` — fine-tuned model (created after training)

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
  "detections": [{"confidence": 0.87, "bbox": [100, 200, 300, 400]}],
  "timestamp": "20260519_134500",
  "notify": true
}
```

## Fine-tuning workflow

1. Let the detector run for 1-2 weeks collecting frames
2. Review `data/training/cat/` — delete anything that is not your cat
3. `data/training/no_cat/` should contain only non-cat images (usually fine as-is)
4. Run: `docker compose run --rm --profile training training`
5. Restart: `docker compose restart nala-detector`
6. Repeat as needed for better accuracy

Minimum 20 cat images required. 50-100 gives solid results.

## License

MIT
