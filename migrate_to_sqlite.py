#!/usr/bin/env python3
"""
One-time migration: read all existing JSON+JPG pairs from /data/detections/
and insert them into SQLite with proper incremental IDs.
Original files are kept in place.
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "nala.db"
DETECTIONS_DIR = DATA_DIR / "detections"


def init_db(conn):
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_det_ts    ON detections(timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_det_label ON detections(label)"
    )
    conn.commit()


def parse_meta(meta):
    """Extract (label, confidence) from the stored JSON blob."""
    if isinstance(meta, list) and meta:
        entry = meta[0]
    elif isinstance(meta, dict):
        entry = meta
    else:
        return "unknown", None

    label = entry.get("class", "unknown")
    conf  = entry.get("confidence", 0.0)

    # COCO "cat" → normalise to "nala"
    if label == "cat":
        label = "nala"

    return label, float(conf) if conf else None


def main():
    if not DETECTIONS_DIR.exists():
        print("No detections directory — nothing to migrate.")
        return

    json_files = sorted(DETECTIONS_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found — nothing to migrate.")
        return

    print(f"Found {len(json_files)} JSON records in {DETECTIONS_DIR}")

    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    init_db(conn)

    # Build a map of already-present IDs to avoid dupes on re-run
    existing_ids = {r[0] for r in conn.execute("SELECT id FROM detections")}

    # Find the highest sequence already stored per date
    day_seq: dict[str, int] = defaultdict(int)
    for eid in existing_ids:
        parts = eid.split("_", 1)
        if len(parts) == 2:
            try:
                day_seq[parts[0]] = max(day_seq[parts[0]], int(parts[1]))
            except ValueError:
                pass

    inserted = skipped = 0

    for jf in json_files:
        stem = jf.stem  # e.g. "20260519_132255"

        # Parse the timestamp embedded in the filename
        try:
            dt = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        except ValueError:
            print(f"  SKIP {jf.name}: cannot parse timestamp")
            skipped += 1
            continue

        date_str  = dt.strftime("%Y%m%d")
        ts_iso    = dt.isoformat()

        # Read metadata
        try:
            with open(jf) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"  SKIP {jf.name}: {e}")
            skipped += 1
            continue

        label, confidence = parse_meta(meta)

        # Frame path — may or may not exist
        jpg = DETECTIONS_DIR / f"{stem}.jpg"
        frame_path = str(jpg) if jpg.exists() else None

        # Allocate the next incremental ID for this day
        day_seq[date_str] += 1
        det_id = f"{date_str}_{day_seq[date_str]}"

        # Skip if somehow already present
        if det_id in existing_ids:
            skipped += 1
            continue

        conn.execute(
            """INSERT OR IGNORE INTO detections
               (id, timestamp, label, confidence, frame_path, in_dataset, notified)
               VALUES (?, ?, ?, ?, ?, 0, 0)""",
            (det_id, ts_iso, label, confidence, frame_path),
        )
        existing_ids.add(det_id)
        inserted += 1

    conn.commit()
    conn.close()
    print(f"Done — {inserted} inserted, {skipped} skipped.  DB: {DB_PATH}")


if __name__ == "__main__":
    main()
