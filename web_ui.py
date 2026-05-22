"""
Nala Detector - Training Data Web UI
Simple web interface to review/delete training images, kick off fine-tuning,
and manage model versions.
"""

import os
import json
import sqlite3
import shutil
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, send_file

app = Flask(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "nala.db")
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

FOLDER_TO_LABEL = {
    "cat": "nala",
    "other_cat": "other_cat",
    "no_cat": "no_cat",
}

LABEL_TO_FOLDER = {v: k for k, v in FOLDER_TO_LABEL.items()}

DATASET_DIR = os.path.join(DATA_DIR, "dataset", "train")
TRAINED_FILES_PATH = os.path.join(DATA_DIR, "trained_files.json")

training_status = {"running": False, "last_result": None}


def get_trained_files():
    """Load set of filenames that have been used in training."""
    if not os.path.exists(TRAINED_FILES_PATH):
        return set()
    try:
        with open(TRAINED_FILES_PATH, "r") as f:
            data = json.load(f)
        return set(data.get("files", []))
    except:
        return set()


def save_trained_files(files_set):
    """Save trained files manifest."""
    with open(TRAINED_FILES_PATH, "w") as f:
        json.dump({"files": sorted(files_set)}, f, indent=2)


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
.stats { color: #888; margin-bottom: 1rem; font-size: 0.9rem; }
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
.hidden { display: none !important; }

/* Models */
.models-container { display: none; }
.models-container.visible { display: block; }
.model-list { display: flex; flex-direction: column; gap: 0.75rem; }
.model-card { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; }
.model-card.active { border-color: #2d6a4f; background: #1a2e3e; }
.model-info { flex: 1; min-width: 200px; }
.model-info h3 { font-size: 1rem; margin-bottom: 0.25rem; }
.model-info .meta { font-size: 0.8rem; color: #888; }
.model-info .meta span { margin-right: 1rem; }
.model-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: bold; }
.badge-active { background: #2d6a4f; color: #fff; }
.badge-inactive { background: #333; color: #888; }
.model-actions { display: flex; gap: 0.5rem; }
.no-items { color: #888; text-align: center; padding: 2rem; }

/* Detections */
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
.det-page-info { color: #888; font-size: 0.85rem; }

/* Dataset */
.dataset-container { display: none; }
.dataset-container.visible { display: block; }
.dataset-section { margin-bottom: 2rem; }
.dataset-section h2 { font-size: 1.1rem; margin-bottom: 0.75rem; padding-bottom: 0.25rem; border-bottom: 1px solid #0f3460; }
.dataset-section h2.new-header { color: #2d6a4f; }
.dataset-section h2.trained-header { color: #888; }
.dataset-group { margin-bottom: 1rem; }
.dataset-group h3 { font-size: 0.9rem; margin-bottom: 0.5rem; }
.dataset-group h3.nala { color: #e94560; }
.dataset-group h3.other_cat { color: #533483; }
.dataset-group h3.no_cat { color: #888; }
</style>
</head>
<body>
<h1>🐱 Nala Detector</h1>
<div class="stats" id="stats"></div>
<div class="status" id="training-status" style="display:none"></div>

<div class="tabs">
  <div class="tab active" data-tab="detections" onclick="switchTab('detections')">Detections <span class="count" id="detections-count">0</span></div>
  <div class="tab" data-tab="cat" onclick="switchTab('cat')">Nala <span class="count" id="cat-count">0</span></div>
  <div class="tab" data-tab="other_cat" onclick="switchTab('other_cat')">Other Cat <span class="count" id="other-cat-count">0</span></div>
  <div class="tab" data-tab="no_cat" onclick="switchTab('no_cat')">No Cat <span class="count" id="no-cat-count">0</span></div>
  <div class="tab" data-tab="dataset" onclick="switchTab('dataset')">Dataset <span class="count" id="dataset-count">0</span></div>
  <div class="tab" data-tab="models" onclick="switchTab('models')">Models <span class="count" id="models-count">0</span></div>
</div>

<!-- Label tabs content -->
<div class="images-container hidden" id="images-container">
  <div class="sort-bar">
    <label>Sort: </label>
    <button class="btn btn-small" id="sort-btn" onclick="toggleSort()">Newest first</button>
  </div>
  <div class="grid" id="grid"></div>
  <div class="actions" id="label-actions">
    <button class="btn btn-danger" onclick="deleteSelected()">Delete</button>
    <button class="btn btn-success" onclick="moveToDataset()">Move to Dataset</button>
    <span id="relabel-buttons"></span>
    <button class="btn btn-primary" onclick="selectAll()">Select All</button>
    <button class="btn btn-primary" onclick="deselectAll()">Deselect All</button>
  </div>
</div>

<!-- Detections tab content -->
<div class="detections-container visible" id="detections-container">
  <div class="det-list" id="det-list"></div>
  <div class="det-pagination">
    <button class="btn btn-primary" id="det-prev" onclick="detPrev()">&larr; Prev</button>
    <span class="det-page-info" id="det-page-info">Page 1</span>
    <button class="btn btn-primary" id="det-next" onclick="detNext()">Next &rarr;</button>
  </div>
</div>

<!-- Dataset tab content -->
<div class="dataset-container" id="dataset-container">
  <div class="actions">
    <button class="btn btn-success" onclick="startTraining()" id="train-btn">Start Training</button>
    <button class="btn btn-danger" onclick="deleteDatasetSelected()">Delete Selected</button>
    <button class="btn btn-primary" onclick="selectAllDataset()">Select All</button>
    <button class="btn btn-primary" onclick="deselectAllDataset()">Deselect All</button>
  </div>
  <div id="dataset-content"></div>
</div>

<!-- Models tab content -->
<div class="models-container" id="models-container">
  <div class="model-list" id="model-list"></div>
  <div class="actions" style="margin-top: 1rem;">
    <button class="btn btn-warning" onclick="rollbackGeneric()">Rollback to Generic YOLO</button>
  </div>
</div>

<script>
let currentTab = 'detections';
let selected = new Set();
let images = [];
let sortOrder = localStorage.getItem('nala-sort') || 'newest';
let detPage = 1;
const DET_PAGE_SIZE = 30;
let detections = [];
let datasetData = null;
let datasetSelected = new Set();

const LABEL_NAMES = {cat: 'Nala', other_cat: 'Other Cat', no_cat: 'No Cat'};
const LABEL_TARGETS = {cat: ['other_cat', 'no_cat'], other_cat: ['cat', 'no_cat'], no_cat: ['cat', 'other_cat']};
const FOLDER_TO_LABEL_MAP = {cat: 'nala', other_cat: 'other_cat', no_cat: 'no_cat'};

function switchTab(tab) {
  currentTab = tab;
  selected.clear();
  datasetSelected.clear();
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.getElementById('images-container').classList.add('hidden');
  document.getElementById('detections-container').classList.remove('visible');
  document.getElementById('dataset-container').classList.remove('visible');
  document.getElementById('models-container').classList.remove('visible');
  if (tab === 'detections') {
    document.getElementById('detections-container').classList.add('visible');
  } else if (tab === 'dataset') {
    document.getElementById('dataset-container').classList.add('visible');
  } else if (tab === 'models') {
    document.getElementById('models-container').classList.add('visible');
  } else {
    document.getElementById('images-container').classList.remove('hidden');
  }
  load();
}

async function load() {
  if (currentTab === 'models') { loadModels(); return; }
  if (currentTab === 'detections') { loadDetections(); return; }
  if (currentTab === 'dataset') { loadDataset(); return; }
  // Label tabs
  const r = await fetch('/api/images/' + currentTab + '?sort=' + sortOrder);
  images = await r.json();
  await refreshStats();
  render();
  renderRelabelButtons();
  checkTraining();
}

async function refreshStats() {
  const rc = await fetch('/api/stats');
  const stats = await rc.json();
  document.getElementById('cat-count').textContent = stats.new_cat;
  document.getElementById('other-cat-count').textContent = stats.new_other_cat;
  document.getElementById('no-cat-count').textContent = stats.new_no_cat;
  document.getElementById('detections-count').textContent = stats.detections;
  document.getElementById('dataset-count').textContent = stats.dataset_total;
  document.getElementById('stats').textContent =
    stats.new_cat + ' new Nala, ' + stats.new_other_cat + ' new other cats, ' + stats.new_no_cat + ' new no-cat | Dataset: ' + stats.dataset_new + ' new + ' + stats.dataset_trained + ' trained | Detections: ' + stats.detections;
}

function render() {
  const grid = document.getElementById('grid');
  grid.innerHTML = images.map(f => {
    const sel = selected.has(f) ? 'selected' : '';
    return `<div class="card ${sel}" onclick="toggle('${f}')"><img src="/api/image/${currentTab}/${f}" loading="lazy"><div class="name">${f}</div></div>`;
  }).join('');
}

function renderRelabelButtons() {
  const span = document.getElementById('relabel-buttons');
  const targets = LABEL_TARGETS[currentTab] || [];
  span.innerHTML = targets.map(t =>
    `<button class="btn btn-move" onclick="relabelToDataset('${t}')">&rarr; ${LABEL_NAMES[t]}</button>`
  ).join(' ');
}

function toggle(f) { selected.has(f) ? selected.delete(f) : selected.add(f); render(); }
function selectAll() { images.forEach(f => selected.add(f)); render(); }
function deselectAll() { selected.clear(); render(); }

function toggleSort() {
  sortOrder = sortOrder === 'newest' ? 'oldest' : 'newest';
  localStorage.setItem('nala-sort', sortOrder);
  document.getElementById('sort-btn').textContent = sortOrder === 'newest' ? 'Newest first' : 'Oldest first';
  load();
}

async function deleteSelected() {
  if (selected.size === 0) return;
  if (!confirm('Delete ' + selected.size + ' image(s)?')) return;
  await fetch('/api/delete', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({folder: currentTab, files: [...selected]}) });
  selected.clear();
  load();
}

async function moveToDataset() {
  if (selected.size === 0) { alert('Select images first'); return; }
  await fetch('/api/move-to-dataset', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({from: currentTab, files: [...selected]}) });
  selected.clear();
  load();
}

async function relabelToDataset(toLabel) {
  if (selected.size === 0) { alert('Select images first'); return; }
  await fetch('/api/relabel-to-dataset', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({from: currentTab, to_label: toLabel, files: [...selected]}) });
  selected.clear();
  load();
}

// --- Detections ---
async function loadDetections() {
  const r = await fetch('/api/detections?sort=' + sortOrder);
  detections = await r.json();
  await refreshStats();
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
    list.innerHTML = '<div class="no-items">No detections recorded yet.</div>';
  } else {
    list.innerHTML = page.map(d => {
      const lc = d.label === 'nala' ? 'nala' : d.label === 'other_cat' ? 'other_cat' : 'no_cat';
      const lt = d.label === 'nala' ? 'Nala' : d.label === 'other_cat' ? 'Other Cat' : d.label;
      return `<div class="det-row" onclick="goToDetection('${d.file}','${d.label}')"><span class="det-date">${d.datetime}</span><span class="det-label ${lc}">${lt}</span><span class="det-confidence">${d.confidence}%</span></div>`;
    }).join('');
  }
  document.getElementById('det-page-info').textContent = 'Page ' + detPage + ' / ' + totalPages + ' (' + total + ' total)';
  document.getElementById('det-prev').disabled = detPage <= 1;
  document.getElementById('det-next').disabled = detPage >= totalPages;
}

function detPrev() { if (detPage > 1) { detPage--; renderDetections(); } }
function detNext() { if (detPage < Math.ceil(detections.length / DET_PAGE_SIZE)) { detPage++; renderDetections(); } }

function goToDetection(file, label) {
  let tab = 'cat';
  if (label === 'other_cat') tab = 'other_cat';
  else if (label === 'no_cat') tab = 'no_cat';
  currentTab = tab;
  selected.clear();
  selected.add(file);
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.getElementById('images-container').classList.remove('hidden');
  document.getElementById('detections-container').classList.remove('visible');
  document.getElementById('dataset-container').classList.remove('visible');
  document.getElementById('models-container').classList.remove('visible');
  load();
}

// --- Dataset ---
async function loadDataset() {
  const r = await fetch('/api/dataset');
  datasetData = await r.json();
  await refreshStats();
  renderDataset();
  checkTraining();
}

function renderDataset() {
  const container = document.getElementById('dataset-content');
  if (!datasetData) { container.innerHTML = ''; return; }

  let html = '';

  // New section
  const newTotal = (datasetData.new.nala||[]).length + (datasetData.new.other_cat||[]).length + (datasetData.new.no_cat||[]).length;
  html += '<div class="dataset-section"><h2 class="new-header">New — pending training (' + newTotal + ')</h2>';
  if (newTotal === 0) {
    html += '<div class="no-items">No new images pending training.</div>';
  } else {
    html += renderDatasetGroup(datasetData.new, 'new');
  }
  html += '</div>';

  // Trained section
  const trainedTotal = (datasetData.trained.nala||[]).length + (datasetData.trained.other_cat||[]).length + (datasetData.trained.no_cat||[]).length;
  html += '<div class="dataset-section"><h2 class="trained-header">Trained (' + trainedTotal + ')</h2>';
  if (trainedTotal === 0) {
    html += '<div class="no-items">No trained images yet. Run training to populate.</div>';
  } else {
    html += renderDatasetGroup(datasetData.trained, 'trained');
  }
  html += '</div>';

  container.innerHTML = html;
}

function renderDatasetGroup(group, prefix) {
  let html = '';
  for (const label of ['nala', 'other_cat', 'no_cat']) {
    const files = group[label] || [];
    if (files.length === 0) continue;
    const labelName = label === 'nala' ? 'Nala' : label === 'other_cat' ? 'Other Cat' : 'No Cat';
    html += '<div class="dataset-group"><h3 class="' + label + '">' + labelName + ' (' + files.length + ')</h3><div class="grid">';
    html += files.map(f => {
      const key = label + '/' + f;
      const sel = datasetSelected.has(key) ? 'selected' : '';
      return `<div class="card ${sel}" onclick="toggleDatasetItem('${key}')"><img src="/api/dataset-image/${label}/${f}" loading="lazy"><div class="name">${f}</div></div>`;
    }).join('');
    html += '</div></div>';
  }
  return html;
}

function toggleDatasetItem(key) { datasetSelected.has(key) ? datasetSelected.delete(key) : datasetSelected.add(key); renderDataset(); }
function selectAllDataset() {
  if (!datasetData) return;
  for (const section of ['new', 'trained']) {
    for (const label of ['nala', 'other_cat', 'no_cat']) {
      (datasetData[section][label] || []).forEach(f => datasetSelected.add(label + '/' + f));
    }
  }
  renderDataset();
}
function deselectAllDataset() { datasetSelected.clear(); renderDataset(); }

async function deleteDatasetSelected() {
  if (datasetSelected.size === 0) return;
  if (!confirm('Delete ' + datasetSelected.size + ' image(s) from dataset?')) return;
  const items = [...datasetSelected].map(k => { const [l, f] = k.split('/'); return {label: l, file: f}; });
  await fetch('/api/dataset/delete', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({items}) });
  datasetSelected.clear();
  loadDataset();
}

// --- Training ---
async function startTraining() {
  if (!confirm('Start training on all dataset images?')) return;
  const r = await fetch('/api/train', { method: 'POST' });
  const d = await r.json();
  if (d.error) { alert(d.error); return; }
  checkTraining();
}

async function checkTraining() {
  const r = await fetch('/api/training-status');
  const d = await r.json();
  const el = document.getElementById('training-status');
  if (d.running) {
    el.style.display = 'block';
    el.textContent = '⏳ Training in progress...';
    el.style.borderColor = '#b8860b';
    setTimeout(checkTraining, 5000);
  } else if (d.last_result) {
    el.style.display = 'block';
    el.textContent = d.last_result;
    el.style.borderColor = d.last_result.includes('success') ? '#2d6a4f' : '#e94560';
  } else {
    el.style.display = 'none';
  }
  document.getElementById('train-btn').disabled = d.running;
}

// --- Models ---
async function loadModels() {
  const r = await fetch('/api/models');
  const data = await r.json();
  const list = document.getElementById('model-list');
  document.getElementById('models-count').textContent = data.versions ? data.versions.length : 0;
  await refreshStats();
  if (!data.versions || data.versions.length === 0) {
    list.innerHTML = '<div class="no-items">No trained models yet.</div>';
    return;
  }
  list.innerHTML = data.versions.slice().reverse().map(v => {
    const date = new Date(v.timestamp).toLocaleString();
    const ds = v.dataset || {};
    const dsStr = Object.entries(ds).map(([k,c]) => k + ': ' + c).join(', ');
    const p = v.training_params || {};
    const pStr = p.epochs + 'ep, ' + p.imgsz + 'px, batch ' + p.batch;
    const badge = v.active ? '<span class="model-badge badge-active">Active</span>' : '<span class="model-badge badge-inactive">Inactive</span>';
    const deployBtn = v.active ? '' : `<button class="btn btn-success" onclick="deployModel('${v.filename}')">Deploy</button>`;
    return `<div class="model-card ${v.active ? 'active' : ''}"><div class="model-info"><h3>v${v.version} ${badge}</h3><div class="meta"><span>${date}</span><span>${dsStr}</span><span>${pStr}</span></div></div><div class="model-actions">${deployBtn}</div></div>`;
  }).join('');
}

async function deployModel(filename) {
  if (!confirm('Deploy this model?')) return;
  await fetch('/api/deploy', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({filename}) });
  loadModels();
}

async function rollbackGeneric() {
  if (!confirm('Rollback to generic YOLO? Custom model will be removed.')) return;
  await fetch('/api/rollback-generic', { method: 'POST' });
  loadModels();
}

// Init
load();
</script>
</body>
</html>
"""

# ============ BACKEND API ============

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/stats")
def get_stats():
    # Count images in training folders (pre-dataset, pending review)
    new_cat = len([f for f in os.listdir(FOLDERS["cat"]) if f.endswith(".jpg")]) if os.path.exists(FOLDERS["cat"]) else 0
    new_other = len([f for f in os.listdir(FOLDERS["other_cat"]) if f.endswith(".jpg")]) if os.path.exists(FOLDERS["other_cat"]) else 0
    new_no = len([f for f in os.listdir(FOLDERS["no_cat"]) if f.endswith(".jpg")]) if os.path.exists(FOLDERS["no_cat"]) else 0

    # Count dataset images
    trained_files = get_trained_files()
    dataset_new = 0
    dataset_trained = 0
    for label in ["nala", "other_cat", "no_cat"]:
        label_dir = os.path.join(DATASET_DIR, label)
        if os.path.exists(label_dir):
            for f in os.listdir(label_dir):
                if f.endswith(".jpg"):
                    if f in trained_files:
                        dataset_trained += 1
                    else:
                        dataset_new += 1

    # Detections (from SQLite)
    det_count = 0
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5)
            det_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            conn.close()
        except:
            pass

    return jsonify({
        "new_cat": new_cat,
        "new_other_cat": new_other,
        "new_no_cat": new_no,
        "dataset_new": dataset_new,
        "dataset_trained": dataset_trained,
        "dataset_total": dataset_new + dataset_trained,
        "detections": det_count,
    })


@app.route("/api/images/<folder>")
def list_images(folder):
    """List images in a training folder (pending review)."""
    sort_order = request.args.get('sort', 'newest')
    if folder not in FOLDERS:
        return jsonify([])
    dir_path = FOLDERS[folder]
    if not os.path.exists(dir_path):
        return jsonify([])
    files = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]
    files.sort(reverse=(sort_order == 'newest'))
    return jsonify(files)


@app.route("/api/image/<folder>/<filename>")
def get_image(folder, filename):
    """Serve an image from training folder."""
    if folder not in FOLDERS:
        return "not found", 404
    if ".." in filename:
        return "invalid", 400
    filepath = os.path.join(FOLDERS[folder], filename)
    if not os.path.exists(filepath):
        return "not found", 404
    return send_file(filepath, mimetype="image/jpeg")


@app.route("/api/delete", methods=["POST"])
def delete_images():
    """Delete images from a training folder."""
    data = request.json
    folder = data.get("folder")
    files = data.get("files", [])
    if folder not in FOLDERS:
        return jsonify({"error": "bad folder"}), 400
    deleted = 0
    for f in files:
        if ".." in f:
            continue
        path = os.path.join(FOLDERS[folder], f)
        if os.path.exists(path):
            os.remove(path)
            deleted += 1
    return jsonify({"deleted": deleted})


@app.route("/api/move-to-dataset", methods=["POST"])
def move_to_dataset():
    """Move images from training folder to dataset under the same label."""
    data = request.json
    folder = data.get("from")
    files = data.get("files", [])
    if folder not in FOLDERS:
        return jsonify({"error": "bad folder"}), 400
    label = FOLDER_TO_LABEL[folder]
    dst_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)
    moved = 0
    for f in files:
        if ".." in f:
            continue
        src = os.path.join(FOLDERS[folder], f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved += 1
            # Update SQLite if DB exists
            if os.path.exists(DB_PATH):
                try:
                    det_id = f.replace(".jpg", "")
                    conn = sqlite3.connect(DB_PATH, timeout=5)
                    conn.execute(
                        "UPDATE detections SET frame_path = ?, in_dataset = 1 WHERE id = ?",
                        (dst, det_id)
                    )
                    conn.commit()
                    conn.close()
                except:
                    pass
    return jsonify({"moved": moved})


@app.route("/api/relabel-to-dataset", methods=["POST"])
def relabel_to_dataset():
    """Move images from training folder to dataset under a DIFFERENT label."""
    data = request.json
    folder = data.get("from")
    to_label = data.get("to_label")
    files = data.get("files", [])
    if folder not in FOLDERS:
        return jsonify({"error": "bad source folder"}), 400
    # to_label should be one of the training folder names (cat, other_cat, no_cat)
    # Map to dataset label
    if to_label not in FOLDER_TO_LABEL:
        return jsonify({"error": "bad target label"}), 400
    dst_label = FOLDER_TO_LABEL[to_label]
    dst_dir = os.path.join(DATASET_DIR, dst_label)
    os.makedirs(dst_dir, exist_ok=True)
    moved = 0
    for f in files:
        if ".." in f:
            continue
        src = os.path.join(FOLDERS[folder], f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved += 1
            # Update SQLite if DB exists
            if os.path.exists(DB_PATH):
                try:
                    det_id = f.replace(".jpg", "")
                    conn = sqlite3.connect(DB_PATH, timeout=5)
                    conn.execute(
                        "UPDATE detections SET frame_path = ?, in_dataset = 1 WHERE id = ?",
                        (dst, det_id)
                    )
                    conn.commit()
                    conn.close()
                except:
                    pass
    return jsonify({"moved": moved})


@app.route("/api/dataset")
def get_dataset():
    """Return dataset images split into new and trained."""
    trained_files = get_trained_files()
    result = {"new": {"nala": [], "other_cat": [], "no_cat": []}, "trained": {"nala": [], "other_cat": [], "no_cat": []}}
    for label in ["nala", "other_cat", "no_cat"]:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.exists(label_dir):
            continue
        files = sorted([f for f in os.listdir(label_dir) if f.endswith(".jpg")], reverse=True)
        for f in files:
            if f in trained_files:
                result["trained"][label].append(f)
            else:
                result["new"][label].append(f)
    return jsonify(result)


@app.route("/api/dataset-image/<label>/<filename>")
def get_dataset_image(label, filename):
    """Serve an image from dataset."""
    if label not in ["nala", "other_cat", "no_cat"]:
        return "not found", 404
    if ".." in filename:
        return "invalid", 400
    filepath = os.path.join(DATASET_DIR, label, filename)
    if not os.path.exists(filepath):
        return "not found", 404
    return send_file(filepath, mimetype="image/jpeg")


@app.route("/api/dataset/delete", methods=["POST"])
def delete_dataset_images():
    """Delete images from dataset."""
    data = request.json
    items = data.get("items", [])
    deleted = 0
    trained_files = get_trained_files()
    for item in items:
        label = item.get("label")
        filename = item.get("file")
        if not label or not filename or ".." in filename:
            continue
        if label not in ["nala", "other_cat", "no_cat"]:
            continue
        filepath = os.path.join(DATASET_DIR, label, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            trained_files.discard(filename)
            deleted += 1
    save_trained_files(trained_files)
    return jsonify({"deleted": deleted})


@app.route("/api/detections")
def list_detections():
    """Return detection log from SQLite (all events including no_cat)."""
    sort_order = request.args.get('sort', 'newest')
    order = "DESC" if sort_order == 'newest' else "ASC"

    if not os.path.exists(DB_PATH):
        return jsonify([])

    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT id, timestamp, label, confidence, frame_path FROM detections ORDER BY timestamp {order}"
        ).fetchall()
        conn.close()
    except Exception:
        return jsonify([])

    results = []
    for row in rows:
        det_id = row["id"]
        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(row["timestamp"])
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            dt_str = row["timestamp"]

        confidence = round(row["confidence"] * 100, 1) if row["confidence"] else 0

        results.append({
            "id": det_id,
            "file": f"{det_id}.jpg",
            "datetime": dt_str,
            "label": row["label"],
            "confidence": confidence,
        })

    return jsonify(results)


@app.route("/api/detection-image/<filename>")
def get_detection_image(filename):
    """Serve a detection image."""
    if ".." in filename:
        return "invalid", 400
    filepath = os.path.join(DATA_DIR, "detections", filename)
    if not os.path.exists(filepath):
        return "not found", 404
    return send_file(filepath, mimetype="image/jpeg")




@app.route("/api/detection-image-by-id/<det_id>")
def get_detection_image_by_id(det_id):
    """Serve a detection image by its database ID."""
    if ".." in det_id:
        return "invalid", 400
    if not os.path.exists(DB_PATH):
        return "not found", 404
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        row = conn.execute("SELECT frame_path FROM detections WHERE id = ?", (det_id,)).fetchone()
        conn.close()
        if row and row[0] and os.path.exists(row[0]):
            return send_file(row[0], mimetype="image/jpeg")
    except:
        pass
    # Fallback to detections dir
    filepath = os.path.join(DATA_DIR, "detections", f"{det_id}.jpg")
    if os.path.exists(filepath):
        return send_file(filepath, mimetype="image/jpeg")
    return "not found", 404

@app.route("/api/train", methods=["POST"])
def start_training():
    if training_status["running"]:
        return jsonify({"error": "Training already in progress"}), 409
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
            training_status["last_result"] = f"Training failed: {result.stderr[-500:]}"
    except Exception as e:
        training_status["last_result"] = f"Training error: {str(e)}"
    finally:
        training_status["running"] = False


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
