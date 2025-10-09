let selectedImage = null;
let chartInstance = null;
let frameChartInstance = null;
let liveChartInstance = null;
let datasetFrames = [];
let liveTimer = null;
let liveWs = null;

function selectImage(name) {
  selectedImage = name;
  document.getElementById('selected-image').innerText = name;
}

async function loadModel(name) {
  const form = new FormData();
  form.append('name', name);
  const res = await fetch('/load_model', { method: 'POST', body: form });
  const data = await res.json();
  console.log('Model:', data);
}

async function selectModelFolder() {
  const folder = document.getElementById('modelFolder').value.trim();
  if (!folder) return;
  const form = new FormData();
  form.append('folder_path', folder);
  const res = await fetch('/select_model_folder', { method: 'POST', body: form });
  const data = await res.json();
  console.log('Plugin Model:', data);
}

async function preprocessSelected(operation) {
  if (!selectedImage) {
    alert('Select an image first.');
    return;
  }
  const form = new FormData();
  form.append('image_name', selectedImage);
  form.append('operation', operation);
  const res = await fetch('/preprocess', { method: 'POST', body: form });
  const data = await res.json();
  if (data.ok) {
    document.getElementById('overlay').src = data.processed_path;
  }
}

async function runInference(name) {
  selectedImage = name;
  document.getElementById('selected-image').innerText = name;
  const form = new FormData();
  form.append('image_name', name);
  const res = await fetch('/inference', { method: 'POST', body: form });
  const data = await res.json();
  if (!data.ok) {
    alert(data.error || 'Inference failed');
    return;
  }
  document.getElementById('overlay').src = data.overlay_url;
  updateChart(data.stats);
  updateStatsText(data.stats);
}

function updateChart(stats) {
  const ctx = document.getElementById('statsChart');
  const areas = stats.areas || [];
  const labels = areas.map((_, i) => `Crystal ${i+1}`);
  const dataset = {
    label: 'Area',
    data: areas,
    borderColor: '#5b9cff',
    backgroundColor: '#5b9cff66'
  };
  if (chartInstance) chartInstance.destroy();
  chartInstance = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [dataset] },
    options: { responsive: true, plugins: { legend: { labels: { color: '#e6e6e6' } } }, scales: { x: { ticks: { color: '#e6e6e6' } }, y: { ticks: { color: '#e6e6e6' } } } }
  });
}

function updateStatsText(stats) {
  const el = document.getElementById('statsText');
  el.innerHTML = `
    <div>Count: <strong>${stats.count}</strong></div>
    <div>Mean area: <strong>${stats.mean_area.toFixed(2)}</strong></div>
    <div>Mean aspect ratio: <strong>${stats.mean_aspect_ratio.toFixed(2)}</strong></div>
  `;
}

async function ingestDataset() {
  const folder = document.getElementById('datasetFolder').value.trim();
  if (!folder) return;
  const form = new FormData();
  form.append('dataset_path', folder);
  const res = await fetch('/ingest_dataset', { method: 'POST', body: form });
  const data = await res.json();
  if (!data.ok) { alert(data.error || 'Failed to ingest dataset'); return; }
  const framesRes = await fetch('/dataset_frames');
  const framesData = await framesRes.json();
  datasetFrames = framesData.frames || [];
  const slider = document.getElementById('timeSlider');
  slider.min = 0;
  slider.max = Math.max(0, datasetFrames.length - 1);
  slider.value = 0;
  document.getElementById('timeLabel').innerText = `t = ${datasetFrames[0]?.time ?? 0}`;
  if (datasetFrames.length) onTimeChange(0);
}

async function onTimeChange(idx) {
  const i = parseInt(idx, 10);
  const frame = datasetFrames[i];
  if (!frame) return;
  document.getElementById('timeLabel').innerText = `t = ${frame.time}`;
  const res = await fetch(`/frame_stats?frame_name=${encodeURIComponent(frame.name)}`);
  const data = await res.json();
  if (!data.ok) { return; }
  document.getElementById('frameOverlay').src = data.overlay_url;
  updateFrameChart(data.stats);
  updateFrameStatsText(data.stats);
}

function updateFrameChart(stats) {
  const ctx = document.getElementById('frameChart');
  const areas = stats.areas || [];
  const labels = areas.map((_, i) => `Crystal ${i+1}`);
  const dataset = { label: 'Area', data: areas, borderColor: '#9cf', backgroundColor: '#9cf6' };
  if (frameChartInstance) frameChartInstance.destroy();
  frameChartInstance = new Chart(ctx, { type: 'bar', data: { labels, datasets: [dataset] }, options: { responsive: true, plugins: { legend: { labels: { color: '#e6e6e6' } } }, scales: { x: { ticks: { color: '#e6e6e6' } }, y: { ticks: { color: '#e6e6e6' } } } } });
}

function updateFrameStatsText(stats) {
  const el = document.getElementById('frameStats');
  el.innerHTML = `
    <div>Count: <strong>${stats.count}</strong></div>
    <div>Mean area: <strong>${stats.mean_area.toFixed(2)}</strong></div>
    <div>Mean aspect ratio: <strong>${stats.mean_aspect_ratio.toFixed(2)}</strong></div>
  `;
}

async function sendLiveFrame() {
  const fileEl = document.getElementById('liveFile');
  const tsEl = document.getElementById('liveTs');
  if (!fileEl.files.length) { alert('Pick a frame image'); return; }
  const form = new FormData();
  form.append('file', fileEl.files[0]);
  form.append('timestamp', tsEl.value || '0');
  const res = await fetch('/stream_frame', { method: 'POST', body: form });
  const data = await res.json();
  if (!data.ok) alert('Failed to send frame');
}

async function pollLive() {
  const res = await fetch('/live_stats');
  const data = await res.json();
  if (!data.ok || !data.last) return;
  const last = data.last;
  if (last.overlay_url) document.getElementById('liveOverlay').src = last.overlay_url;
  updateLiveChart(last.stats || { areas: [] });
  updateLiveStatsText(last.stats || { count: 0, mean_area: 0, mean_aspect_ratio: 0 });
}

function startLivePolling() {
  // Prefer WebSocket, fallback to polling
  if (liveWs || liveTimer) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  try {
    liveWs = new WebSocket(`${proto}://${location.host}/ws/live`);
    liveWs.onopen = () => {
      console.log('Live WebSocket connected');
    };
    liveWs.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        if (!payload.ok || !payload.last) return;
        const last = payload.last;
        if (last.overlay_url) document.getElementById('liveOverlay').src = last.overlay_url;
        updateLiveChart(last.stats || { areas: [] });
        updateLiveStatsText(last.stats || { count: 0, mean_area: 0, mean_aspect_ratio: 0 });
      } catch (e) {
        console.warn('WS parse error:', e);
      }
    };
    liveWs.onclose = () => {
      console.log('Live WebSocket closed');
      liveWs = null;
      if (!liveTimer) {
        liveTimer = setInterval(pollLive, 500);
      }
    };
    liveWs.onerror = () => {
      console.warn('Live WebSocket error; falling back to polling');
      try { liveWs.close(); } catch {}
      liveWs = null;
      if (!liveTimer) {
        liveTimer = setInterval(pollLive, 500);
      }
    };
  } catch (e) {
    console.warn('WebSocket init failed; using polling');
    liveTimer = setInterval(pollLive, 500);
  }
}

function stopLivePolling() {
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  if (liveWs) { try { liveWs.close(); } catch {} liveWs = null; }
}

function updateLiveChart(stats) {
  const ctx = document.getElementById('liveChart');
  const areas = stats.areas || [];
  const labels = areas.map((_, i) => `Crystal ${i+1}`);
  const dataset = { label: 'Area', data: areas, borderColor: '#5fffbf', backgroundColor: '#5fffbf66' };
  if (liveChartInstance) liveChartInstance.destroy();
  liveChartInstance = new Chart(ctx, { type: 'bar', data: { labels, datasets: [dataset] }, options: { responsive: true, plugins: { legend: { labels: { color: '#e6e6e6' } } }, scales: { x: { ticks: { color: '#e6e6e6' } }, y: { ticks: { color: '#e6e6e6' } } } } });
}

function updateLiveStatsText(stats) {
  const el = document.getElementById('liveStats');
  el.innerHTML = `
    <div>Count: <strong>${stats.count}</strong></div>
    <div>Mean area: <strong>${stats.mean_area.toFixed(2)}</strong></div>
    <div>Mean aspect ratio: <strong>${stats.mean_aspect_ratio.toFixed(2)}</strong></div>
  `;
}