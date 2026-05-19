#!/usr/bin/env python3
"""
Nala Cat Detector
Subscribes to Ring camera motion snapshots via MQTT,
runs YOLO inference, and publishes cat detection results.
"""

import os
import io
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import paho.mqtt.client as mqtt
from PIL import Image
from ultralytics import YOLO

# Configuration from environment
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("MQTT_USER", "mqttuser")
MQTT_PASS = os.environ.get("MQTT_PASS", "hassmqtt")

# MQTT topics
SNAPSHOT_TOPIC = os.environ.get(
    "SNAPSHOT_TOPIC",
    "ring/0edbd740-1b1f-48f4-a72d-512697d5764f/camera/ac9fc38ba649/snapshot/image"
)
MOTION_TOPIC = os.environ.get(
    "MOTION_TOPIC",
    "ring/0edbd740-1b1f-48f4-a72d-512697d5764f/camera/ac9fc38ba649/motion/state"
)
RESULT_TOPIC = os.environ.get("RESULT_TOPIC", "nala/detection/result")
SNAPSHOT_REQUEST_TOPIC = os.environ.get(
    "SNAPSHOT_REQUEST_TOPIC",
    "ring/0edbd740-1b1f-48f4-a72d-512697d5764f/camera/ac9fc38ba649/take_snapshot/command"
)

# Detection settings
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.4"))
CAT_CLASS_ID = 15  # COCO class ID for 'cat'
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/yolov8n.pt")
CUSTOM_MODEL_PATH = os.environ.get("CUSTOM_MODEL_PATH", "/models/nala_custom.pt")

# Data collection for fine-tuning
COLLECT_DATA = os.environ.get("COLLECT_DATA", "true").lower() == "true"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
TRAINING_DIR = DATA_DIR / "training"
DETECTIONS_DIR = DATA_DIR / "detections"

# Cooldown to avoid spam (seconds)
NOTIFY_COOLDOWN = int(os.environ.get("NOTIFY_COOLDOWN", "300"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("nala")


class NalaDetector:
    def __init__(self):
        self.last_notify_time = 0
        self.model = self._load_model()
        self.mqtt_client = None
        self._waiting_for_snapshot = False

        # Ensure data directories exist
        if COLLECT_DATA:
            (TRAINING_DIR / "cat").mkdir(parents=True, exist_ok=True)
            (TRAINING_DIR / "no_cat").mkdir(parents=True, exist_ok=True)
            DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Load custom model if available, otherwise fall back to pre-trained."""
        if os.path.exists(CUSTOM_MODEL_PATH):
            logger.info(f"Loading custom model: {CUSTOM_MODEL_PATH}")
            return YOLO(CUSTOM_MODEL_PATH)
        else:
            logger.info(f"Loading pre-trained model: {MODEL_PATH}")
            return YOLO(MODEL_PATH)

    def reload_model(self):
        """Hot-reload model (call after fine-tuning)."""
        self.model = self._load_model()
        logger.info("Model reloaded")

    def detect_cat(self, image_bytes):
        """Run inference on image bytes. Returns (is_cat, confidence, details)."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            results = self.model(image, verbose=False)

            cat_detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == CAT_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
                        cat_detections.append({
                            "confidence": round(conf, 3),
                            "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
                        })

            is_cat = len(cat_detections) > 0
            max_conf = max(d["confidence"] for d in cat_detections) if is_cat else 0.0

            return is_cat, max_conf, cat_detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return False, 0.0, []

    def save_training_data(self, image_bytes, is_cat, timestamp):
        """Save frame for future fine-tuning."""
        if not COLLECT_DATA:
            return

        subdir = "cat" if is_cat else "no_cat"
        filename = f"{timestamp}.jpg"
        filepath = TRAINING_DIR / subdir / filename

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        logger.debug(f"Saved training frame: {filepath}")

    def save_detection(self, image_bytes, details, timestamp):
        """Save detected cat image for review."""
        filepath = DETECTIONS_DIR / f"{timestamp}.jpg"
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        meta_path = DETECTIONS_DIR / f"{timestamp}.json"
        with open(meta_path, "w") as f:
            json.dump(details, f)

    def should_notify(self):
        """Check cooldown to avoid spamming."""
        now = time.time()
        if now - self.last_notify_time >= NOTIFY_COOLDOWN:
            self.last_notify_time = now
            return True
        return False

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connected callback."""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            client.subscribe(SNAPSHOT_TOPIC)
            client.subscribe(MOTION_TOPIC)
            logger.info(f"Subscribed to: {SNAPSHOT_TOPIC}")
            logger.info(f"Subscribed to: {MOTION_TOPIC}")
        else:
            logger.error(f"MQTT connection failed: rc={rc}")

    def on_motion(self, client, userdata, msg):
        """Handle motion event — request a snapshot."""
        payload = msg.payload.decode("utf-8", errors="ignore").strip()
        if payload.upper() == "ON":
            logger.info("Motion detected, requesting snapshot...")
            self._waiting_for_snapshot = True
            client.publish(SNAPSHOT_REQUEST_TOPIC, "ON")

    def on_snapshot(self, client, userdata, msg):
        """Handle incoming snapshot image."""
        image_bytes = msg.payload
        if len(image_bytes) < 1000:
            logger.warning(f"Received too-small payload ({len(image_bytes)} bytes), skipping")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Received snapshot ({len(image_bytes)} bytes), running detection...")

        is_cat, confidence, details = self.detect_cat(image_bytes)

        # Save for training regardless of result
        self.save_training_data(image_bytes, is_cat, timestamp)

        if is_cat:
            logger.info(f"🐱 CAT DETECTED! Confidence: {confidence:.1%}")
            self.save_detection(image_bytes, details, timestamp)

            # Publish result to MQTT
            result_payload = json.dumps({
                "detected": True,
                "confidence": confidence,
                "detections": details,
                "timestamp": timestamp,
                "notify": self.should_notify()
            })
            client.publish(RESULT_TOPIC, result_payload, retain=True)
        else:
            logger.info("No cat detected")
            result_payload = json.dumps({
                "detected": False,
                "timestamp": timestamp
            })
            client.publish(RESULT_TOPIC, result_payload, retain=True)

        self._waiting_for_snapshot = False

    def on_message(self, client, userdata, msg):
        """Route messages to appropriate handler."""
        if msg.topic == MOTION_TOPIC:
            self.on_motion(client, userdata, msg)
        elif msg.topic == SNAPSHOT_TOPIC:
            self.on_snapshot(client, userdata, msg)

    def run(self):
        """Main loop."""
        logger.info("Starting Nala Cat Detector...")
        logger.info(f"MQTT: {MQTT_HOST}:{MQTT_PORT}")
        logger.info(f"Model: {CUSTOM_MODEL_PATH if os.path.exists(CUSTOM_MODEL_PATH) else MODEL_PATH}")
        logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        logger.info(f"Notify cooldown: {NOTIFY_COOLDOWN}s")
        logger.info(f"Collecting training data: {COLLECT_DATA}")

        self.mqtt_client = mqtt.Client(
            client_id="nala-detector",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        self.mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

        while True:
            try:
                self.mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
                self.mqtt_client.loop_forever()
            except Exception as e:
                logger.error(f"Connection error: {e}, retrying in 10s...")
                time.sleep(10)


if __name__ == "__main__":
    detector = NalaDetector()
    detector.run()
