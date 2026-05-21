"""
Nala Detector - Training Data Web UI
Simple web interface to review/delete training images, kick off fine-tuning,
and manage model versions.
"""

import os
import json
import shutil
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, send_file

app = Flask(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODEL_DIR = "/models"
HISTORY_DIR = os.path.join(MODEL_DIR, "history")
MANIFEST_PATH = os.path.join(HISTORY_DIR, "manifest.json")
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "nala_custom.pt")
RELOAD_SIGNAL = os.path.join(MODEL_DIR, ".reload")

FOLDERS = {
    "cat": os.path.join(DATA_DIR, "training", "cat"),
    "other_cat": os.path.join(DATA_DIR, "training", "other_cat"),
    "no_cat": os.path.join(DATA_DIR, "training", "no_cat"),
}

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
.tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
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
.btn-success { background: #2d6a4f; color: #fff; }
.btn-warning { background: #b8860b; color: #fff; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-small { padding: 0.3rem 0.6rem; font-size: 0.8rem; background: #16213e; border: 1px solid #0f3460; color: #eee; }
.sort-bar { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem; }
.sort-bar label { color: #888; font-size: 0.85rem; }
.status { padding: 0.5rem; background: #16213e; border-radius: 4px; margin-bottom: 1rem; }
.count { background: #0f3460; padding: 2px 6px; border-radius: 10px; font-size: 0.8rem; }

/* Models tab styles */
.models-container { display: none; }
.models-container.visible { display: block; }
.images-container { display: block; }
.images-container.hidden { display: none; }
.model-list { display: flex; flex-direction: column; gap: 0.75rem; }
.model-card { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; }
.model-card.active { border-color: #2d6a4f; background: #1a2e3e; }
.model-info { flex: 1; min-width: 200px; }
.model-info h3 { font-size: 1rem; margin-bottom: 0.25rem; color: #eee; }
.model-info .meta { font-size: 0.8rem; color: #888; }
.model-info .meta span { margin-right: 1rem; }
.model-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: bold; }
.badge-active { background: #2d6a4f; color: #fff; }
.badge-inactive { background: #333; color: #888; }
.model-actions { display: flex; gap: 0.5rem; }
.no-models { color: #888; text-align: center; padding: 2rem; }

/* Detections tab styles */
.detections-container { display: none; }
.detections-container.visible { display: block; }
.det-list { display: flex; flex-direction: column; gap: 0.25rem; }
.det-row { display: flex; align-items: center; gap: 1rem; padding: 0.6rem 0.8rem; background: #16213e; border: 1px solid #0f3460; border-radius: 4px; cursor: pointer; transition: background 0.15s; }
.det-row:hover { background: #1a2e3e; border-color: #e94560; }
.det-date { font-size: 0.85rem; color: #ccc; min-width: 140px; }
.det-label { font-size: 0.85rem; font-weight: bold; min-width: 80px; }
.det-label.nala { color: #e94560; }
.det-label.other_cat { color: #533483; }
.det-label.no_cat { color: #888; }
.det-confidence { font-size: 0.85rem; color: #2d6a4f; min-width: 50px; }
.det-pagination { display: flex; align-items: center; gap: 1rem; margin-top: 1rem; justify-content: center; }
.det-pagination .btn { min-width: 80px; }
.det-page-info { color: #888; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>🐱 Nala Detector</h1>
<div class="stats" id="stats"></div>
<div class="status" id="training-status" style="display:none"></div>

<div class="tabs">
  <div class="tab active" data-tab="cat" onclick="switchTab('cat')">Nala <span class="count" id="cat-count">0</span></div>
  <div class="tab" data-tab="other_cat" onclick="switchTab('other_cat')">Other Cat <span class="count" id="other-cat-count">0</span></div>
  <div class="tab" data-tab="no_cat" onclick="switchTab('no_cat')">No Cat <span class="count" id="no-cat-count">0</span></div>
  <div class="tab" data-tab="detections" onclick="switchTab('detections')">Detections <span class="count" id="detections-count">0</span></div>
  <div class="tab" data-tab="models" onclick="switchTab('models')">Models <span class="count" id="models-count">0</span></div>
</div>

<div class="images-container" id="images-container">
  <div class="sort-bar">
    <label>Sort: </label>
    <button class="btn btn-small" id="sort-btn" onclick="toggleSort()">Newest first</button>
  </div>
  <div class="grid" id="grid"></div>
  <div class="actions">
    <button class="btn btn-danger" onclick="deleteSelected()">Delete Selected</button>
    <span id="move-buttons"></span>
    <button class="btn btn-primary" onclick="selectAll()">Select All</button>
    <button class="btn btn-primary" onclick="deselectAll()">Deselect All</button>
    <button class="btn btn-primary" onclick="startTraining()" id="train-btn">Start Training</button>
  </div>
</div>

<div class="detections-container" id="detections-container">
  <div class="det-list" id="det-list"></div>
  <div class="det-pagination">
    <button class="btn btn-primary" id="det-prev" onclick="detPrev()">← Prev</button>
    <span class="det-page-info" id="det-page-info">Page 1</span>
    <button class="btn btn-primary" id="det-next" onclick="detNext()">Next →</button>
  </div>
</div>

<div class="models-container" id="models-container">
  <div class="model-list" id="model-list"></div>
  <div class="actions" style="margin-top: 1rem;">
    <button class="btn btn-warning" onclick="rollbackGeneric()">Rollback to Generic YOLO</button>
  </div>
</div>

<script>
let currentTab = 'cat';
let selected = new Set();
let images = [];
let sortOrder = localStorage.getItem('nala-sort') || 'newest';
let detPage = 1;
const DET_PAGE_SIZE = 30;
let detections = [];

async function load() {
  if (currentTab === 'models') { loadModels(); return; }
  if (currentTab === 'detections') { loadDetections(); return; }
  const r = await fetch(`/api/images/${currentTab}?sort=${sortOrder}`);
  images = await r.json();
  const rc = await fetch('/api/stats');
  const stats = await rc.json();
  document.getElementById('cat-count').textContent = stats.cat;
  document.getElementById('other-cat-count').textContent = stats.other_cat;
  document.getElementById('no-cat-count').textContent = stats.no_cat;
  document.getElementById('detections-count').textContent = stats.detections;
  document.getElementById('stats').textContent = `${stats.cat} Nala, ${stats.other_cat} other cats, ${stats.no_cat} no-cat, ${stats.detections} detections`;
  render();
  checkTraining();
  loadModelsCount();
}

async function loadDetections() {
  const r = await fetch('/api/detections?sort=' + sortOrder);
  detections = await r.json();
  const rc = await fetch('/api/stats');
  const stats = await rc.json();
  document.getElementById('detections-count').textContent = stats.detections;
  document.getElementById('cat-count').textContent = stats.cat;
  document.getElementById('other-cat-count').textContent = stats.other_cat;
  document.getElementById('no-cat-count').textContent = stats.no_cat;
  document.getElementById('stats').textContent = `${stats.cat} Nala, ${stats.other_cat} other cats, ${stats.no_cat} no-cat, ${stats.detections} detections`;
  renderDetections();
}

function renderDetections() {
  const total = detections.length;
  const totalPages = Math.max(1, Math.ceil(total / DET_PAGE_SIZE));
  if (detPage > totalPages) detPage = totalPages;
  const start = (detPage - 1) * DET_PAGE_SIZE;
  const page = detections.slice(start, start + DET_PAGE_SIZE);

  const list = document.getElementById('det-list');
  if (total === 0) {
    list.innerHTML = '<div class="no-models">No detections recorded yet.</div>';
  } else {
    list.innerHTML = page.map(d => {
      const labelClass = d.label === 'nala' ? 'nala' : d.label === 'other_cat' ? 'other_cat' : 'no_cat';
      const labelText = d.label === 'nala' ? 'Nala' : d.label === 'other_cat' ? 'Other Cat' : d.label;
      return `<div class="det-row" onclick="goToDetection('${d.file}', '${d.label}')">
        <span class="det-date">${d.datetime}</span>
        <span class="det-label ${labelClass}">${labelText}</span>
        <span class="det-confidence">${d.confidence}%</span>
      </div>`;
    }).join('');
  }

  document.getElementById('det-page-info').textContent = `Page ${detPage} / ${totalPages} (${total} total)`;
  document.getElementById('det-prev').disabled = detPage <= 1;
  document.getElementById('det-next').disabled = detPage >= totalPages;
}

function detPrev() { if (detPage > 1) { detPage--; renderDetections(); } }
function detNext() { const totalPages = Math.ceil(detections.length / DET_PAGE_SIZE); if (detPage < totalPages) { detPage++; renderDetections(); } }

function goToDetection(file, label) {
  // Map label to tab
  let tab = 'cat';
  if (label === 'other_cat') tab = 'other_cat';
  else if (label === 'no_cat') tab = 'no_cat';
  else tab = 'cat';  // nala -> cat tab

  // Switch to that tab and select the file
  currentTab = tab;
  selected.clear();
  selected.add(file);
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.getElementById('images-container').classList.remove('hidden');
  document.getElementById('detections-container').classList.remove('visible');
  document.getElementById('models-container').classList.remove('visible');
  load();
}

async function loadModelsCount() {
  try {
    const r = await fetch('/api/models');
    const data = await r.json();
    document.getElementById('models-count').textContent = data.versions ? data.versions.length : 0;
  } catch(e) {}
}

async function loadModels() {
  const r = await fetch('/api/models');
  const data = await r.json();
  const list = document.getElementById('model-list');
  document.getElementById('models-count').textContent = data.versions ? data.versions.length : 0;

  if (!data.versions || data.versions.length === 0) {
    list.innerHTML = '<div class="no-models">No trained models yet. Train your first model from the training data tabs.</div>';
    return;
  }

  const versions = [...data.versions].sort((a, b) => b.version - a.version);
  list.innerHTML = versions.map(v => {
    const date = new Date(v.timestamp).toLocaleString();
    const dataset = v.dataset || {};
    const datasetStr = Object.entries(dataset).map(([k, c]) => `${k}: ${c}`).join(', ');
    const params = v.training_params || {};
    const paramsStr = `${params.epochs || '?'}ep, ${params.imgsz || '?'}px, batch ${params.batch || '?'}`;
    const isActive = v.active;
    return `
      <div class="model-card ${isActive ? 'active' : ''}">
        <div class="model-info">
          <h3>v${v.version} <span class="model-badge ${isActive ? 'badge-active' : 'badge-inactive'}">${isActive ? 'ACTIVE' : 'inactive'}</span></h3>
          <div class="meta">
            <span>📅 ${date}</span>
            <span>📊 ${datasetStr || 'N/A'}</span><br>
            <span>⚙️ ${paramsStr}</span>
            <span>📁 ${v.filename}</span>
          </div>
        </div>
        <div class="model-actions">
          ${isActive ? '' : `<button class="btn btn-success" onclick="deployModel('${v.filename}')">Deploy</button>`}
        </div>
      </div>`;
  }).join('');
}

function render() {
  const grid = document.getElementById('grid');
  grid.innerHTML = images.map(img => `
    <div class="card ${selected.has(img) ? 'selected' : ''}" onclick="toggle('${img}')">
      <img src="/api/image/${currentTab}/${img}" loading="lazy">
      <div class="name">${img}</div>
    </div>
  `).join('');
  const folders = ['cat', 'other_cat', 'no_cat'].filter(f => f !== currentTab);
  document.getElementById('move-buttons').innerHTML = folders.map(f =>
    `<button class="btn btn-move" onclick="moveSelected('${f}')">${f === 'cat' ? 'Nala' : f === 'other_cat' ? 'Other Cat' : 'No Cat'}</button>`
  ).join('');
  document.getElementById('sort-btn').textContent = sortOrder === 'newest' ? 'Newest first' : 'Oldest first';

  // Scroll to selected item if any
  if (selected.size === 1) {
    setTimeout(() => {
      const sel = document.querySelector('.card.selected');
      if (sel) sel.scrollIntoView({behavior: 'smooth', block: 'center'});
    }, 100);
  }
}

function toggleSort() {
  sortOrder = sortOrder === 'newest' ? 'oldest' : 'newest';
  localStorage.setItem('nala-sort', sortOrder);
  load();
}

function toggle(img) { selected.has(img) ? selected.delete(img) : selected.add(img); render(); }
function selectAll() { images.forEach(i => selected.add(i)); render(); }
function deselectAll() { selected.clear(); render(); }

function switchTab(tab) {
  currentTab = tab;
  selected.clear();
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  if (tab === 'models') {
    document.getElementById('images-container').classList.add('hidden');
    document.getElementById('detections-container').classList.remove('visible');
    document.getElementById('models-container').classList.add('visible');
    loadModels();
  } else if (tab === 'detections') {
    document.getElementById('images-container').classList.add('hidden');
    document.getElementById('models-container').classList.remove('visible');
    document.getElementById('detections-container').classList.add('visible');
    loadDetections();
  } else {
    document.getElementById('images-container').classList.remove('hidden');
    document.getElementById('detections-container').classList.remove('visible');
    document.getElementById('models-container').classList.remove('visible');
    load();
  }
}

async function deleteSelected() {
  if (!selected.size) return;
  if (!confirm(`Delete ${selected.size} images?`)) return;
  await fetch('/api/delete', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({folder: currentTab, files: [...selected]}) });
  selected.clear();
  load();
}

async function moveSelected(target) {
  if (!selected.size) return;
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

async function deployModel(filename) {
  if (!confirm(`Deploy model ${filename}?`)) return;
  const r = await fetch('/api/deploy', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({filename: filename}) });
  const result = await r.json();
  if (result.success) { alert('Model deployed! Detector will reload within 30s.'); loadModels(); }
  else { alert('Deploy failed: ' + (result.error || 'unknown error')); }
}

async function rollbackGeneric() {
  if (!confirm('Rollback to generic YOLOv8n?')) return;
  const r = await fetch('/api/rollback-generic', { method: 'POST' });
  const result = await r.json();
  if (result.success) { alert('Rolled back to generic YOLO. Detector will reload within 30s.'); loadModels(); }
  else { alert('Rollback failed: ' + (result.error || 'unknown error')); }
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
    cat_count = len(list(Path(FOLDERS["cat"]).glob("*.jpg"))) if os.path.exists(FOLDERS["cat"]) else 0
    other_cat_count = len(list(Path(FOLDERS["other_cat"]).glob("*.jpg"))) if os.path.exists(FOLDERS["other_cat"]) else 0
    no_cat_count = len(list(Path(FOLDERS["no_cat"]).glob("*.jpg"))) if os.path.exists(FOLDERS["no_cat"]) else 0
    det_dir = os.path.join(DATA_DIR, "detections")
    det_count = len(list(Path(det_dir).glob("*.json"))) if os.path.exists(det_dir) else 0
    return jsonify({"cat": cat_count, "other_cat": other_cat_count, "no_cat": no_cat_count, "detections": det_count})

@app.route("/api/detections")
def list_detections():
    """Return paginated detection log with metadata from .json files."""
    sort_order = request.args.get('sort', 'newest')
    det_dir = os.path.join(DATA_DIR, "detections")
    if not os.path.exists(det_dir):
        return jsonify([])

    json_files = sorted(
        [f for f in os.listdir(det_dir) if f.endswith(".json")],
        reverse=(sort_order == 'newest')
    )

    results = []
    for jf in json_files:
        json_path = os.path.join(det_dir, jf)
        img_file = jf.replace('.json', '.jpg')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            # data is a list like [{"class": "nala", "confidence": 0.98}]
            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
            else:
                entry = data
            label = entry.get("class", "unknown")
            confidence = round(entry.get("confidence", 0) * 100, 1)
        except:
            label = "unknown"
            confidence = 0

        # Parse datetime from filename: 20260521_035735.json
        basename = jf.replace('.json', '')
        try:
            dt = datetime.strptime(basename, "%Y%m%d_%H%M%S")
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            dt_str = basename

        results.append({
            "file": img_file,
            "datetime": dt_str,
            "label": label,
            "confidence": confidence
        })

    return jsonify(results)

@app.route("/api/images/<folder>")
def list_images(folder):
    sort_order = request.args.get('sort', 'newest')
    if folder not in FOLDERS:
        return jsonify([])
    dir_path = FOLDERS[folder]
    if not os.path.exists(dir_path):
        return jsonify([])
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".jpg")], reverse=(sort_order == 'newest'))
    return jsonify(files)

@app.route("/api/image/<folder>/<filename>")
def get_image(folder, filename):
    if folder == 'detections':
        dir_path = os.path.join(DATA_DIR, "detections")
    elif folder in FOLDERS:
        dir_path = FOLDERS[folder]
    else:
        return "not found", 404
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath) or ".." in filename:
        return "not found", 404
    return send_file(filepath, mimetype="image/jpeg")

@app.route("/api/delete", methods=["POST"])
def delete_images():
    data = request.json
    folder = data.get("folder")
    files = data.get("files", [])
    if folder in FOLDERS:
        dir_path = FOLDERS[folder]
    else:
        return jsonify({"error": "bad folder"}), 400
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
    if src_folder not in FOLDERS:
        return jsonify({"error": "bad source folder"}), 400
    if dst_folder not in FOLDERS:
        return jsonify({"error": "bad destination folder"}), 400
    src_dir = FOLDERS[src_folder]
    dst_dir = FOLDERS[dst_folder]
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

@app.route("/api/models")
def get_models():
    """Return model version manifest."""
    if not os.path.exists(MANIFEST_PATH):
        return jsonify({"versions": [], "next_version": 1})
    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
        return jsonify(manifest)
    except Exception as e:
        return jsonify({"versions": [], "error": str(e)})

@app.route("/api/deploy", methods=["POST"])
def deploy_model():
    """Deploy a specific model version."""
    data = request.json
    filename = data.get("filename")
    if not filename or ".." in filename:
        return jsonify({"success": False, "error": "invalid filename"}), 400

    source = os.path.join(HISTORY_DIR, filename)
    if not os.path.exists(source):
        return jsonify({"success": False, "error": "model file not found"}), 404

    try:
        shutil.copy2(source, CUSTOM_MODEL_PATH)
        if os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
            for v in manifest["versions"]:
                v["active"] = (v["filename"] == filename)
            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2)
        with open(RELOAD_SIGNAL, "w") as f:
            f.write(datetime.now().isoformat())
        return jsonify({"success": True, "deployed": filename})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/rollback-generic", methods=["POST"])
def rollback_generic():
    """Remove custom model so detector falls back to generic YOLO."""
    try:
        if os.path.exists(CUSTOM_MODEL_PATH):
            os.remove(CUSTOM_MODEL_PATH)
        if os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
            for v in manifest["versions"]:
                v["active"] = False
            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2)
        with open(RELOAD_SIGNAL, "w") as f:
            f.write(datetime.now().isoformat())
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def run_training():
    try:
        result = subprocess.run(
            ["python3", "/app/train.py"],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            training_status["last_result"] = "Training completed successfully — model auto-deployed"
        else:
            training_status["last_result"] = f"Training failed: {result.stderr[-200:]}"
    except Exception as e:
        training_status["last_result"] = f"Training error: {str(e)}"
    finally:
        training_status["running"] = False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
