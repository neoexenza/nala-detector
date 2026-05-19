"""
Nala Detector - Training Data Web UI
Simple web interface to review/delete training images and kick off fine-tuning.
"""

import os
import json
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_file

app = Flask(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/data")
CAT_DIR = os.path.join(DATA_DIR, "training", "cat")
NO_CAT_DIR = os.path.join(DATA_DIR, "training", "no_cat")

training_status = {"running": False, "last_result": None}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Nala Detector</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 1rem; }
h1 { margin-bottom: 0.5rem; }
.stats { color: #888; margin-bottom: 1rem; }
.tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
.tab { padding: 0.5rem 1rem; background: #16213e; border: 1px solid #0f3460; border-radius: 4px; cursor: pointer; color: #eee; }
.tab.active { background: #0f3460; border-color: #e94560; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 0.5rem; }
.card { position: relative; border-radius: 4px; overflow: hidden; border: 2px solid transparent; }
.card.selected { border-color: #e94560; }
.card img { width: 100%; aspect-ratio: 4/3; object-fit: cover; cursor: pointer; }
.card .name { font-size: 0.7rem; padding: 2px 4px; background: rgba(0,0,0,0.7); position: absolute; bottom: 0; left: 0; right: 0; white-space: nowrap; overflow: hidden; }
.actions { position: sticky; bottom: 0; background: #1a1a2e; padding: 1rem 0; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
.btn { padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem; }
.btn-danger { background: #e94560; color: #fff; }
.btn-primary { background: #0f3460; color: #fff; border: 1px solid #e94560; }
.btn-move { background: #533483; color: #fff; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.status { padding: 0.5rem; background: #16213e; border-radius: 4px; margin-bottom: 1rem; }
.count { background: #0f3460; padding: 2px 6px; border-radius: 10px; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>🐱 Nala Detector</h1>
<div class="stats" id="stats"></div>
<div class="status" id="training-status" style="display:none"></div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('cat')">Cat <span class="count" id="cat-count">0</span></div>
  <div class="tab" onclick="switchTab('no_cat')">No Cat <span class="count" id="no-cat-count">0</span></div>
</div>

<div class="grid" id="grid"></div>

<div class="actions">
  <button class="btn btn-danger" onclick="deleteSelected()">Delete Selected</button>
  <button class="btn btn-move" onclick="moveSelected()">Move to <span id="move-target">no_cat</span></button>
  <button class="btn btn-primary" onclick="selectAll()">Select All</button>
  <button class="btn btn-primary" onclick="deselectAll()">Deselect All</button>
  <button class="btn btn-primary" onclick="startTraining()" id="train-btn">Start Training</button>
</div>

<script>
let currentTab = 'cat';
let selected = new Set();
let images = [];

async function load() {
  const r = await fetch(`/api/images/${currentTab}`);
  images = await r.json();
  document.getElementById('cat-count').textContent = currentTab === 'cat' ? images.length : document.getElementById('cat-count').textContent;
  document.getElementById('no-cat-count').textContent = currentTab === 'no_cat' ? images.length : document.getElementById('no-cat-count').textContent;
  // load both counts
  const rc = await fetch('/api/stats');
  const stats = await rc.json();
  document.getElementById('cat-count').textContent = stats.cat;
  document.getElementById('no-cat-count').textContent = stats.no_cat;
  document.getElementById('stats').textContent = `${stats.cat} cat images, ${stats.no_cat} no-cat images, ${stats.detections} detections logged`;
  render();
  checkTraining();
}

function render() {
  const grid = document.getElementById('grid');
  grid.innerHTML = images.map(img => `
    <div class="card ${selected.has(img) ? 'selected' : ''}" onclick="toggle('${img}')">
      <img src="/api/image/${currentTab}/${img}" loading="lazy">
      <div class="name">${img}</div>
    </div>
  `).join('');
  document.getElementById('move-target').textContent = currentTab === 'cat' ? 'no_cat' : 'cat';
}

function toggle(img) { selected.has(img) ? selected.delete(img) : selected.add(img); render(); }
function selectAll() { images.forEach(i => selected.add(i)); render(); }
function deselectAll() { selected.clear(); render(); }

function switchTab(tab) {
  currentTab = tab;
  selected.clear();
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab')[tab === 'cat' ? 0 : 1].classList.add('active');
  load();
}

async function deleteSelected() {
  if (!selected.size) return;
  if (!confirm(`Delete ${selected.size} images?`)) return;
  await fetch('/api/delete', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({folder: currentTab, files: [...selected]}) });
  selected.clear();
  load();
}

async function moveSelected() {
  if (!selected.size) return;
  const target = currentTab === 'cat' ? 'no_cat' : 'cat';
  await fetch('/api/move', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({from: currentTab, to: target, files: [...selected]}) });
  selected.clear();
  load();
}

async function startTraining() {
  if (!confirm('Start fine-tuning? This may take a while.')) return;
  await fetch('/api/train', { method: 'POST' });
  checkTraining();
}

async function checkTraining() {
  const r = await fetch('/api/training-status');
  const s = await r.json();
  const el = document.getElementById('training-status');
  if (s.running) { el.style.display = 'block'; el.textContent = '⏳ Training in progress...'; document.getElementById('train-btn').disabled = true; }
  else if (s.last_result) { el.style.display = 'block'; el.textContent = '✅ ' + s.last_result; document.getElementById('train-btn').disabled = false; }
  else { el.style.display = 'none'; document.getElementById('train-btn').disabled = false; }
}

load();
setInterval(checkTraining, 10000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/stats")
def stats():
    cat_count = len(list(Path(CAT_DIR).glob("*.jpg"))) if os.path.exists(CAT_DIR) else 0
    no_cat_count = len(list(Path(NO_CAT_DIR).glob("*.jpg"))) if os.path.exists(NO_CAT_DIR) else 0
    det_dir = os.path.join(DATA_DIR, "detections")
    det_count = len(list(Path(det_dir).glob("*.json"))) if os.path.exists(det_dir) else 0
    return jsonify({"cat": cat_count, "no_cat": no_cat_count, "detections": det_count})

@app.route("/api/images/<folder>")
def list_images(folder):
    if folder not in ("cat", "no_cat"):
        return jsonify([])
    dir_path = CAT_DIR if folder == "cat" else NO_CAT_DIR
    if not os.path.exists(dir_path):
        return jsonify([])
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".jpg")], reverse=True)
    return jsonify(files)

@app.route("/api/image/<folder>/<filename>")
def get_image(folder, filename):
    if folder not in ("cat", "no_cat"):
        return "not found", 404
    dir_path = CAT_DIR if folder == "cat" else NO_CAT_DIR
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath) or ".." in filename:
        return "not found", 404
    return send_file(filepath, mimetype="image/jpeg")

@app.route("/api/delete", methods=["POST"])
def delete_images():
    data = request.json
    folder = data.get("folder")
    files = data.get("files", [])
    if folder not in ("cat", "no_cat"):
        return jsonify({"error": "bad folder"}), 400
    dir_path = CAT_DIR if folder == "cat" else NO_CAT_DIR
    deleted = 0
    for f in files:
        if ".." in f:
            continue
        path = os.path.join(dir_path, f)
        if os.path.exists(path):
            os.remove(path)
            deleted += 1
    return jsonify({"deleted": deleted})

@app.route("/api/move", methods=["POST"])
def move_images():
    data = request.json
    src_folder = data.get("from")
    dst_folder = data.get("to")
    files = data.get("files", [])
    if src_folder not in ("cat", "no_cat") or dst_folder not in ("cat", "no_cat"):
        return jsonify({"error": "bad folder"}), 400
    src_dir = CAT_DIR if src_folder == "cat" else NO_CAT_DIR
    dst_dir = CAT_DIR if dst_folder == "cat" else NO_CAT_DIR
    os.makedirs(dst_dir, exist_ok=True)
    moved = 0
    for f in files:
        if ".." in f:
            continue
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(src):
            os.rename(src, dst)
            moved += 1
    return jsonify({"moved": moved})

@app.route("/api/train", methods=["POST"])
def start_training():
    if training_status["running"]:
        return jsonify({"error": "already running"}), 409
    training_status["running"] = True
    training_status["last_result"] = None
    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return jsonify({"started": True})

@app.route("/api/training-status")
def get_training_status():
    return jsonify(training_status)

def run_training():
    try:
        result = subprocess.run(
            ["python3", "/app/train.py"],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            training_status["last_result"] = "Training completed successfully"
        else:
            training_status["last_result"] = f"Training failed: {result.stderr[-200:]}"
    except Exception as e:
        training_status["last_result"] = f"Training error: {str(e)}"
    finally:
        training_status["running"] = False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
