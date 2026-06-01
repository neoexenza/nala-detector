#!/usr/bin/env python3
"""
Nala detector cleanup script.
Deletes detection frames and DB entries older than 30 days,
UNLESS the entry has in_dataset=1 (promoted to training set).
Run daily via cron.
"""

import sqlite3
import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = "/popcorn/komodo/stacks/nala/data/nala.db"
DETECTIONS_DIR = Path("/popcorn/komodo/stacks/nala/data/detections")
CUTOFF_DAYS = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("nala-cleanup")

def main():
    cutoff = datetime.now(timezone.utc) - timedelta(days=CUTOFF_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")
    log.info(f"Cutoff: {cutoff_str} (deleting entries older than {CUTOFF_DAYS} days, in_dataset=0)")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Find DB entries to delete (old + not in dataset)
    cur.execute("""
        SELECT id, frame_path FROM detections
        WHERE timestamp < ? AND in_dataset = 0
    """, (cutoff_str,))
    rows = cur.fetchall()
    log.info(f"DB entries to delete: {len(rows)}")

    deleted_files = 0
    missing_files = 0

    for row_id, frame_path in rows:
        # Delete the frame file if it exists
        if frame_path:
            fp = Path(frame_path)
            if fp.exists():
                fp.unlink()
                deleted_files += 1
            else:
                missing_files += 1
        # Also delete the .json sidecar in detections dir (named by id)
        json_path = DETECTIONS_DIR / f"{row_id}.json"
        if json_path.exists():
            json_path.unlink()
        # Also delete a .jpg in detections dir matching the id
        jpg_path = DETECTIONS_DIR / f"{row_id}.jpg"
        if jpg_path.exists():
            jpg_path.unlink()

    # 2. Delete the DB entries
    cur.execute("""
        DELETE FROM detections
        WHERE timestamp < ? AND in_dataset = 0
    """, (cutoff_str,))
    deleted_rows = cur.rowcount
    conn.commit()
    conn.close()

    log.info(f"Deleted {deleted_rows} DB rows, {deleted_files} frame files ({missing_files} already missing)")

    # 3. Sweep for orphan files in detections dir older than cutoff
    # (files with no DB entry at all, filename starts with YYYYMMDD)
    orphans = 0
    for f in DETECTIONS_DIR.iterdir():
        if not f.is_file():
            continue
        try:
            # filenames: YYYYMMDD_HHMMSS.ext or YYYYMMDD_N.ext
            date_str = f.stem[:8]
            file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            if file_date < cutoff:
                f.unlink()
                orphans += 1
        except (ValueError, IndexError):
            pass  # skip files we can't parse

    if orphans:
        log.info(f"Deleted {orphans} orphan files from detections dir")

    log.info("Cleanup complete.")

if __name__ == "__main__":
    main()
