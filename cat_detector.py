#!/usr/bin/env python3
"""
Nala Cat Detector
Subscribes to Ring camera motion snapshots via MQTT,
runs YOLO inference, and publishes cat detection results.
Now with SQLite for persistent event tracking.
"""

import os
import io
import json
import time
import logging
import sqlite3
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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

# RTSP stream for frame grabbing during motion
RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://192.168.0.200:8554/ac9fc38ba649_live"
)

# Detection settings
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.4"))
CAT_CLASS_ID = 15  # COCO class ID for 'cat'
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/yolov8n.pt")
CUSTOM_MODEL_PATH = os.environ.get("CUSTOM_MODEL_PATH", "/models/nala_custom.pt")
USE_CLASSIFICATION = False  # Set True when custom model is a classifier

# Model hot-reload
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
RELOAD_SIGNAL = MODEL_DIR / ".reload"
RELOAD_CHECK_INTERVAL = int(os.environ.get("RELOAD_CHECK_INTERVAL", "30"))

# Data collection for fine-tuning
COLLECT_DATA = os.environ.get("COLLECT_DATA", "true").lower() == "true"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
TRAINING_DIR = DATA_DIR / "training"
DETECTIONS_DIR = DATA_DIR / "detections"
DB_PATH = DATA_DIR / "nala.db"

# Cooldown to avoid spam (seconds)
NOTIFY_COOLDOWN = int(os.environ.get("NOTIFY_COOLDOWN", "300"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("nala")


def init_db():
    """Initialize SQLite database with WAL mode."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          TEXT PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            label       TEXT NOT NULL,
            confidence  REAL,
            frame_path  TEXT,
            in_dataset  INTEGER DEFAULT 0,
            notified    INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_det_label ON detections(label)")
    conn.commit()
    conn.close()


def get_next_id(conn):
    """Get the next incremental ID for today: YYYYMMDD_N."""
    today = datetime.now().strftime("%Y%m%d")
    rows = conn.execute(
        "SELECT id FROM detections WHERE id LIKE ?",
        (f"{today}_%",)
    ).fetchall()
    if rows:
        max_seq = 0
        for row in rows:
            try:
                seq = int(row[0].split("_", 1)[1])
                if seq > max_seq:
                    max_seq = seq
            except (ValueError, IndexError):
                pass
        return f"{today}_{max_seq + 1}"
    else:
        return f"{today}_1"


class NalaDetector:
    def __init__(self):
        self.last_notify_time = 0
        self.model = self._load_model()
        self.mqtt_client = None
        self._reload_thread = None
        self._db_lock = threading.Lock()

        # Ensure data directories exist
        if COLLECT_DATA:
            (TRAINING_DIR / "cat").mkdir(parents=True, exist_ok=True)
            (TRAINING_DIR / "no_cat").mkdir(parents=True, exist_ok=True)
            DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize database
        init_db()

    def _load_model(self):
        """Load custom model if available, otherwise fall back to pre-trained."""
        global USE_CLASSIFICATION
        if os.path.exists(CUSTOM_MODEL_PATH):
            logger.info(f"Loading custom classification model: {CUSTOM_MODEL_PATH}")
            USE_CLASSIFICATION = True
            return YOLO(CUSTOM_MODEL_PATH)
        else:
            logger.info(f"Loading pre-trained detection model: {MODEL_PATH}")
            USE_CLASSIFICATION = False
            return YOLO(MODEL_PATH)

    def reload_model(self):
        """Hot-reload model (call after fine-tuning)."""
        self.model = self._load_model()
        logger.info("Model reloaded successfully")

    def _start_reload_watcher(self):
        """Start background thread to watch for reload signal."""
        def watcher():
            logger.info(f"Model reload watcher started (checking every {RELOAD_CHECK_INTERVAL}s)")
            while True:
                try:
                    time.sleep(RELOAD_CHECK_INTERVAL)
                    if RELOAD_SIGNAL.exists():
                        logger.info("Reload signal detected! Reloading model...")
                        try:
                            self.reload_model()
                            RELOAD_SIGNAL.unlink()
                            logger.info("Reload signal consumed")
                        except Exception as e:
                            logger.error(f"Model reload failed: {e}")
                except Exception as e:
                    logger.error(f"Reload watcher error: {e}")
                    time.sleep(10)

        self._reload_thread = threading.Thread(target=watcher, daemon=True)
        self._reload_thread.start()

    def detect_cat(self, image_bytes):
        """Run inference on image bytes. Returns (is_nala, confidence, details)."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            results = self.model(image, verbose=False)

            if USE_CLASSIFICATION:
                for result in results:
                    probs = result.probs
                    top_class = result.names[probs.top1]
                    top_conf = float(probs.top1conf)

                    if top_class == "nala" and top_conf >= CONFIDENCE_THRESHOLD:
                        return True, top_conf, [{"class": "nala", "confidence": round(top_conf, 3)}]
                    else:
                        return False, 0.0, [{"class": top_class, "confidence": round(top_conf, 3)}]
            else:
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

    def save_training_data(self, image_bytes, is_cat, det_id):
        """Save frame for future fine-tuning."""
        if not COLLECT_DATA:
            return

        subdir = "cat" if is_cat else "no_cat"
        filename = f"{det_id}.jpg"
        filepath = TRAINING_DIR / subdir / filename

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        logger.debug(f"Saved training frame: {filepath}")

    def save_detection_frame(self, image_bytes, det_id):
        """Save frame to detections directory."""
        filepath = DETECTIONS_DIR / f"{det_id}.jpg"
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return str(filepath)

    def save_to_db(self, det_id, timestamp_iso, label, confidence, frame_path, notified):
        """Insert detection event into SQLite."""
        with self._db_lock:
            conn = sqlite3.connect(str(DB_PATH), timeout=10)
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO detections
                       (id, timestamp, label, confidence, frame_path, in_dataset, notified)
                       VALUES (?, ?, ?, ?, ?, 0, ?)""",
                    (det_id, timestamp_iso, label, confidence, frame_path, 1 if notified else 0)
                )
                conn.commit()
            finally:
                conn.close()

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

    def _grab_rtsp_frame(self, client):
        """Grab a frame from the RTSP live stream using ffmpeg."""
        time.sleep(3)
        logger.info(f"Grabbing frame from RTSP stream...")
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-rtsp_transport", "tcp",
                    "-i", RTSP_URL,
                    "-frames:v", "1",
                    "-f", "image2",
                    "-q:v", "2",
                    "/tmp/motion_frame.jpg"
                ],
                capture_output=True, timeout=10
            )
            if result.returncode == 0 and os.path.exists("/tmp/motion_frame.jpg"):
                with open("/tmp/motion_frame.jpg", "rb") as f:
                    image_bytes = f.read()
                if len(image_bytes) > 1000:
                    logger.info(f"Got RTSP frame ({len(image_bytes)} bytes)")
                    self._process_frame(client, image_bytes)
                    return
            logger.warning(f"RTSP frame grab failed (rc={result.returncode})")
        except subprocess.TimeoutExpired:
            logger.warning("RTSP frame grab timed out (camera may not be streaming)")
        except Exception as e:
            logger.error(f"RTSP error: {e}")

        # Fallback: request on-demand snapshot after recording finishes
        logger.info("Falling back to on-demand snapshot (waiting for recording to end)...")
        time.sleep(5)
        client.publish(SNAPSHOT_REQUEST_TOPIC, "PRESS")

    def _process_frame(self, client, image_bytes):
        """Run detection on a frame and publish results."""
        # Generate incremental ID
        with self._db_lock:
            conn = sqlite3.connect(str(DB_PATH), timeout=10)
            det_id = get_next_id(conn)
            conn.close()

        now = datetime.now()
        timestamp_iso = now.isoformat()

        is_cat, confidence, details = self.detect_cat(image_bytes)

        # Determine label
        if is_cat:
            label = "nala"
            if details and isinstance(details[0], dict) and "class" in details[0]:
                label = details[0]["class"]
        else:
            label = "no_cat"
            if details and isinstance(details[0], dict) and "class" in details[0]:
                label = details[0]["class"]

        # Save frame (ALL events, not just positives)
        frame_path = self.save_detection_frame(image_bytes, det_id)

        # Save training data
        self.save_training_data(image_bytes, is_cat, det_id)

        # Determine notification
        notified = False
        if is_cat:
            logger.info(f"CAT DETECTED! Confidence: {confidence:.1%} [ID: {det_id}]")
            notified = self.should_notify()
            result_payload = json.dumps({
                "detected": True,
                "confidence": confidence,
                "detections": details,
                "id": det_id,
                "timestamp": timestamp_iso,
                "notify": notified
            })
            client.publish(RESULT_TOPIC, result_payload, retain=True)
        else:
            logger.info(f"No cat detected [ID: {det_id}]")
            result_payload = json.dumps({
                "detected": False,
                "id": det_id,
                "timestamp": timestamp_iso
            })
            client.publish(RESULT_TOPIC, result_payload, retain=True)

        # Save to SQLite — always store confidence (even for below-threshold detections)
        self.save_to_db(det_id, timestamp_iso, label, confidence if confidence else None, frame_path, notified)

        # Also save legacy JSON for backward compat
        meta_path = DETECTIONS_DIR / f"{det_id}.json"
        with open(meta_path, "w") as f:
            json.dump(details if details else [{"class": label, "confidence": confidence}], f)

    def on_motion(self, client, userdata, msg):
        """Handle motion event - grab frame from RTSP live stream."""
        payload = msg.payload.decode("utf-8", errors="ignore").strip()
        if payload.upper() == "ON":
            logger.info("Motion detected! Grabbing live frame...")
            t = threading.Thread(
                target=self._grab_rtsp_frame,
                args=(client,),
                daemon=True
            )
            t.start()

    def on_snapshot(self, client, userdata, msg):
        """Handle incoming snapshot image from MQTT (interval or on-demand)."""
        image_bytes = msg.payload
        if len(image_bytes) < 1000:
            logger.warning(f"Received too-small payload ({len(image_bytes)} bytes), skipping")
            return

        logger.info(f"Received MQTT snapshot ({len(image_bytes)} bytes)")
        self._process_frame(client, image_bytes)

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
        logger.info(f"RTSP: {RTSP_URL}")
        logger.info(f"Model: {CUSTOM_MODEL_PATH if os.path.exists(CUSTOM_MODEL_PATH) else MODEL_PATH}")
        logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        logger.info(f"Notify cooldown: {NOTIFY_COOLDOWN}s")
        logger.info(f"Collecting training data: {COLLECT_DATA}")
        logger.info(f"SQLite DB: {DB_PATH}")

        self._start_reload_watcher()

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
