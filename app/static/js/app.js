let selectedImage = null;
let chartInstance = null;
let frameChartInstance = null;
let liveChartInstance = null;
let datasetFrames = [];
let liveTimer = null;
let liveWs = null;

// ===== Preprocess UI State =====
let preprocImg = null; // original image element
let preprocCanvasOrig = null;
let preprocCanvasProc = null;
let compareChart = null;
let preprocParams = { desaturate: 0, invert: false, gradient_strength: 0, clahe: false, equalize: false };
let preprocModel = null; // independent model selection for Preprocess tab
let preprocLoadingEl = null; // loading overlay element for preprocess actions

function showAlert(type, message) {
  // Create alert element
  const alertDiv = document.createElement('div');
  alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
  alertDiv.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;
  
  // Find or create alerts container
  let alertsContainer = document.getElementById('alerts-container');
  if (!alertsContainer) {
    alertsContainer = document.createElement('div');
    alertsContainer.id = 'alerts-container';
    alertsContainer.style.position = 'fixed';
    alertsContainer.style.top = '20px';
    alertsContainer.style.right = '20px';
    alertsContainer.style.zIndex = '9999';
    alertsContainer.style.maxWidth = '400px';
    document.body.appendChild(alertsContainer);
  }
  
  // Add alert to container
  alertsContainer.appendChild(alertDiv);
  
  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    if (alertDiv.parentNode) {
      alertDiv.remove();
    }
  }, 5000);
}

function selectImage(name) {
  selectedImage = name;
  document.getElementById('selected-image').innerText = name;
  const badge = document.getElementById('preproc-selected-image');
  if (badge) badge.textContent = name;
  setupPreprocPreview(name);
  // Don't change the results display state when just selecting an image
}

// Note: loadModel function is now defined later in the file with dynamic model support

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

function setupPreprocPreview(imageName) {
  const empty = document.getElementById('preproc-empty');
  const preview = document.getElementById('preproc-preview');
  preprocCanvasOrig = document.getElementById('preprocOriginalCanvas');
  preprocCanvasProc = document.getElementById('preprocProcessedCanvas');
  if (!imageName || !preprocCanvasOrig || !preprocCanvasProc) return;
  if (empty) empty.style.display = 'none';
  if (preview) preview.style.display = 'block';
  preprocImg = new Image();
  preprocImg.crossOrigin = 'anonymous';
  preprocImg.onload = () => {
    const parentW = preprocCanvasOrig.parentElement ? preprocCanvasOrig.parentElement.clientWidth : 0;
    const maxW = parentW ? Math.max(200, parentW - 20) : (preprocCanvasOrig.clientWidth || 512);
    const scale = Math.min(1, maxW / preprocImg.width);
    const w = Math.round(preprocImg.width * scale);
    const h = Math.round(preprocImg.height * scale);
    preprocCanvasOrig.width = w; preprocCanvasOrig.height = h;
    preprocCanvasProc.width = w; preprocCanvasProc.height = h;
    drawPreprocOriginal();
    drawPreprocProcessed();
    wirePreprocControls();
  };
  // Use /get_image to avoid static mount issues in certain environments
  preprocImg.src = `/get_image?name=${encodeURIComponent(imageName)}`;
}

function wirePreprocControls() {
  const desat = document.getElementById('preprocDesat');
  const grad = document.getElementById('preprocGrad');
  const invert = document.getElementById('preprocInvert');
  const clahe = document.getElementById('preprocClahe');
  const equalize = document.getElementById('preprocEqualize');
  const desatLabel = document.getElementById('preprocDesatLabel');
  const gradLabel = document.getElementById('preprocGradLabel');
  if (!desat || !grad || !invert || !clahe || !equalize) return;
  const redraw = () => { drawPreprocProcessed(); };
  desat.oninput = () => { preprocParams.desaturate = parseFloat(desat.value) / 100.0; desatLabel.textContent = `${desat.value}%`; redraw(); };
  grad.oninput = () => { preprocParams.gradient_strength = parseFloat(grad.value) / 100.0; gradLabel.textContent = `${grad.value}%`; redraw(); };
  invert.onchange = () => { preprocParams.invert = invert.checked; redraw(); };
  // CLAHE/Equalize require backend processing to reflect accurately; request preview
  clahe.onchange = () => { preprocParams.clahe = clahe.checked; requestPreprocPreview(); };
  equalize.onchange = () => { preprocParams.equalize = equalize.checked; requestPreprocPreview(); };
  const btnCompare = document.getElementById('btnRunCompare');
  const btnSave = document.getElementById('btnSavePreprocessed');
  if (btnCompare) btnCompare.onclick = runInferenceCompare;
  if (btnSave) btnSave.onclick = savePreprocessedImage;
}

function drawPreprocOriginal() {
  if (!preprocImg || !preprocCanvasOrig) return;
  const ctx = preprocCanvasOrig.getContext('2d');
  ctx.clearRect(0, 0, preprocCanvasOrig.width, preprocCanvasOrig.height);
  ctx.drawImage(preprocImg, 0, 0, preprocCanvasOrig.width, preprocCanvasOrig.height);
}

function drawPreprocProcessed() {
  if (!preprocImg || !preprocCanvasProc) return;
  const w = preprocCanvasProc.width, h = preprocCanvasProc.height;
  const ctx = preprocCanvasProc.getContext('2d');
  ctx.drawImage(preprocImg, 0, 0, w, h);
  const imgData = ctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  const des = preprocParams.desaturate || 0;
  const inv = preprocParams.invert || false;
  const gradStr = preprocParams.gradient_strength || 0;
  let gradArr = null;
  if (gradStr > 0) {
    const gray = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const r = d[i], g = d[i+1], b = d[i+2];
        gray[y*w+x] = 0.299*r + 0.587*g + 0.114*b;
      }
    }
    const sobelX = [[-1,0,1],[-2,0,2],[-1,0,1]];
    const sobelY = [[-1,-2,-1],[0,0,0],[1,2,1]];
    gradArr = new Float32Array(w * h);
    for (let y = 1; y < h-1; y++) {
      for (let x = 1; x < w-1; x++) {
        let gx=0, gy=0;
        for (let ky=-1; ky<=1; ky++) {
          for (let kx=-1; kx<=1; kx++) {
            const val = gray[(y+ky)*w+(x+kx)];
            gx += val * sobelX[ky+1][kx+1];
            gy += val * sobelY[ky+1][kx+1];
          }
        }
        const mag = Math.sqrt(gx*gx + gy*gy);
        gradArr[y*w+x] = mag;
      }
    }
    let max = 1e-6;
    for (let i=0;i<gradArr.length;i++) if (gradArr[i]>max) max = gradArr[i];
    for (let i=0;i<gradArr.length;i++) gradArr[i] = gradArr[i]/max*255;
  }
  for (let y=0;y<h;y++) {
    for (let x=0;x<w;x++) {
      const i = (y*w + x) * 4;
      let r = d[i], g = d[i+1], b = d[i+2];
      if (des > 0) {
        const gray = 0.299*r + 0.587*g + 0.114*b;
        r = r*(1-des) + gray*des;
        g = g*(1-des) + gray*des;
        b = b*(1-des) + gray*des;
      }
      if (gradArr) {
        const gmag = gradArr[y*w + x];
        r = Math.min(255, r + gmag*gradStr);
        g = Math.min(255, g + gmag*gradStr);
        b = Math.min(255, b + gmag*gradStr);
      }
      if (inv) {
        r = 255 - r; g = 255 - g; b = 255 - b;
      }
      d[i] = r; d[i+1] = g; d[i+2] = b;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

// Backend-driven preview for CLAHE/Equalize effects
let previewDebounce = null;
async function requestPreprocPreview() {
  if (!selectedImage || !preprocCanvasProc) return;
  // Show small loading spinner over processed canvas
  showPreprocLoading('Applying preprocessing...');
  if (previewDebounce) clearTimeout(previewDebounce);
  previewDebounce = setTimeout(async () => {
    try {
      const form = new FormData();
      form.append('image_name', selectedImage);
      form.append('pipeline', getPreprocPipeline());
      const res = await fetch('/preproc_preview', { method: 'POST', body: form });
      const data = await res.json();
      if (data.ok && data.overlay_b64) {
        const imgEl = new Image();
        imgEl.onload = () => {
          // Draw scaled to canvas to avoid resizing (performance)
          const ctx = preprocCanvasProc.getContext('2d');
          ctx.clearRect(0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
          ctx.drawImage(imgEl, 0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
        };
        imgEl.src = data.overlay_b64;
      }
    } catch (e) {
      console.error('Preproc preview failed', e);
    } finally {
      hidePreprocLoading();
    }
  }, 150);
}

function getPreprocPipeline() {
  return JSON.stringify({
    desaturate: preprocParams.desaturate,
    invert: preprocParams.invert,
    gradient_strength: preprocParams.gradient_strength,
    clahe: preprocParams.clahe,
    equalize: preprocParams.equalize
  });
}

async function runInferenceCompare() {
  if (!selectedImage) { showAlert('danger', 'Select an image first'); return; }
  if (!preprocModel) {
    showAlert('danger', 'Select a model for the Preprocess tab before running inference');
    return;
  }
  // Show loading overlay similar to inference tab
  showPreprocLoading('Running inference on processed image...');
  const form = new FormData();
  form.append('image_name', selectedImage);
  form.append('pipeline', getPreprocPipeline());
  form.append('model_folder', preprocModel.folder || preprocModel.id || '');
  let data = null;
  try {
    const res = await fetch('/inference_compare_preproc', { method: 'POST', body: form });
    data = await res.json();
  } catch (e) {
    hidePreprocLoading();
    showAlert('danger', 'Preprocess inference failed: ' + e.message);
    return;
  }
  hidePreprocLoading();
  if (!data || !data.ok) { showAlert('danger', (data && data.error) || 'Preprocess inference failed'); return; }
  updateCompareChart(data.original?.stats || {}, data.processed?.stats || {});
  updateCompareStatsText(data.original?.stats || {}, data.processed?.stats || {});
  // Replace canvases with inference overlays
  if (data.original?.overlay_b64) {
    const im1 = new Image(); im1.onload = () => {
      const ctx = preprocCanvasOrig.getContext('2d');
      ctx.clearRect(0, 0, preprocCanvasOrig.width, preprocCanvasOrig.height);
      ctx.drawImage(im1, 0, 0, preprocCanvasOrig.width, preprocCanvasOrig.height);
    }; im1.src = data.original.overlay_b64;
  }
  if (data.processed?.overlay_b64) {
    const im2 = new Image(); im2.onload = () => {
      const ctx = preprocCanvasProc.getContext('2d');
      ctx.clearRect(0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
      ctx.drawImage(im2, 0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
    }; im2.src = data.processed.overlay_b64;
  }
}

function updateCompareChart(statsOrig, statsProc) {
  const orig = statsOrig || {};
  const proc = statsProc || {};
  renderComparisonChart('compareChartLen', orig.lengths || [], proc.lengths || [], 'Length (px)');
  renderComparisonChart('compareChartWid', orig.widths || [], proc.widths || [], 'Width (px)');
  renderComparisonChart('compareChartAR', orig.aspect_ratios || [], proc.aspect_ratios || [], 'Aspect Ratio');
}

function updateCompareStatsText(statsOrig, statsProc) {
  const el = document.getElementById('compareStatsText');
  if (!el) return;
  const fmt = (v) => (v!=null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
  const o = statsOrig || {};
  const p = statsProc || {};
  el.innerHTML = `
    <div class="text-light">Original → Count: <strong>${o.count || 0}</strong>, Mean length: <strong>${fmt(o.mean_length)}</strong>, Mean width: <strong>${fmt(o.mean_width)}</strong>, AR: <strong>${fmt(o.mean_aspect_ratio)}</strong>${o.mean_confidence!=null ? `, Mean confidence: <strong>${fmt((o.mean_confidence||0)*100)}%</strong>` : ''}</div>
    <div class="text-light">Processed → Count: <strong>${p.count || 0}</strong>, Mean length: <strong>${fmt(p.mean_length)}</strong>, Mean width: <strong>${fmt(p.mean_width)}</strong>, AR: <strong>${fmt(p.mean_aspect_ratio)}</strong>${p.mean_confidence!=null ? `, Mean confidence: <strong>${fmt((p.mean_confidence||0)*100)}%</strong>` : ''}</div>
  `;
}

function updateLiveChart(stats) {
  const lengths = stats.lengths || [];
  const widths = stats.widths || [];
  const ars = stats.aspect_ratios || [];
  renderBarChart('liveChartLen', lengths, 'Length (px)', '#5fffbf');
  renderBarChart('liveChartWid', widths, 'Width (px)', '#9cf');
  renderBarChart('liveChartAR', ars, 'Aspect Ratio', '#f59e0b');
}

function updateLiveStatsText(stats) {
  const el = document.getElementById('liveStats');
  if (!el) return;
  const fmt = (v) => (v!=null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
  el.innerHTML = `
    <div>Count: <strong>${stats.count || 0}</strong></div>
    <div>Mean length: <strong>${fmt(stats.mean_length)}</strong></div>
    <div>Mean width: <strong>${fmt(stats.mean_width)}</strong></div>
    <div>Mean aspect ratio: <strong>${fmt(stats.mean_aspect_ratio)}</strong></div>
  `;
}

function displayInferenceResults(data) {
  console.log('Inference results:', data);
  
  // Get UI elements
  const resultsDisplay = document.getElementById('results-display');
  const emptyState = document.getElementById('empty-state');
  const overlayImg = document.getElementById('overlay');
  const modelNameSpan = document.getElementById('results-model-name');
  const statsText = document.getElementById('statsText');
  
  // Show results and hide empty state
  if (resultsDisplay) resultsDisplay.style.display = 'block';
  if (emptyState) emptyState.style.display = 'none';
  
  // Display overlay image
  if (overlayImg && data.overlay_url) {
    overlayImg.src = data.overlay_url;
    overlayImg.alt = `Inference results for ${data.image}`;
  }
  
  // Display model name
  if (modelNameSpan && data.model_info) {
    modelNameSpan.textContent = data.model_info.name || 'Unknown Model';
  }
  
  // Display statistics
  if (statsText && data.stats) {
    const fmt = (v) => (v!=null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
    let statsHtml = '<h6>Detection Statistics:</h6>';
    if (data.stats.count !== undefined) {
      statsHtml += `<p><strong>Detections:</strong> ${data.stats.count}</p>`;
    }
    if (data.stats.mean_length !== undefined) {
      statsHtml += `<p><strong>Mean Length:</strong> ${fmt(data.stats.mean_length)} px</p>`;
    }
    if (data.stats.mean_width !== undefined) {
      statsHtml += `<p><strong>Mean Width:</strong> ${fmt(data.stats.mean_width)} px</p>`;
    }
    if (data.stats.mean_aspect_ratio !== undefined) {
      statsHtml += `<p><strong>Mean Aspect Ratio:</strong> ${fmt(data.stats.mean_aspect_ratio)}</p>`;
    }
    if (data.stats.mean_confidence !== undefined) {
      statsHtml += `<p><strong>Mean Confidence:</strong> ${(Number(data.stats.mean_confidence) * 100).toFixed(1)}%</p>`;
    }
    statsText.innerHTML = statsHtml;
  }
  
  // Update charts (length, width, aspect ratio)
  const s = data.stats || {};
  renderBarChart('statsChartLen', s.lengths || [], 'Length (px)', '#5b9cff');
  renderBarChart('statsChartWid', s.widths || [], 'Width (px)', '#9cf');
  renderBarChart('statsChartAR', s.aspect_ratios || [], 'Aspect Ratio', '#f59e0b');
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
  
  // Show frame results and hide empty state
  showFrameResults();
  document.getElementById('frameOverlay').src = data.overlay_url;
  updateFrameChart(data.stats);
  updateFrameStatsText(data.stats);
}

function showFrameResults() {
  document.getElementById('frame-empty-state').style.display = 'none';
  document.getElementById('frame-results-display').style.display = 'block';
}

function showFrameEmptyState() {
  document.getElementById('frame-results-display').style.display = 'none';
  document.getElementById('frame-empty-state').style.display = 'block';
}

function updateFrameChart(stats) {
  const lengths = stats.lengths || [];
  const widths = stats.widths || [];
  const ars = stats.aspect_ratios || [];
  renderHistogramChart('frameChartLen', lengths, 'Length (px)', '#5b9cff');
  renderHistogramChart('frameChartWid', widths, 'Width (px)', '#9cf');
  renderHistogramChart('frameChartAR', ars, 'Aspect Ratio', '#f59e0b');
}

function updateFrameStatsText(stats) {
  const el = document.getElementById('frameStats');
  const fmt = (v) => (v!=null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
  el.innerHTML = `
    <div>Count: <strong>${stats.count || 0}</strong></div>
    <div>Mean length: <strong>${fmt(stats.mean_length)}</strong></div>
    <div>Mean width: <strong>${fmt(stats.mean_width)}</strong></div>
    <div>Mean aspect ratio: <strong>${fmt(stats.mean_aspect_ratio)}</strong></div>
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
  if (last.overlay_url) {
    showLiveResults();
    document.getElementById('liveOverlay').src = last.overlay_url;
  }
  updateLiveChart(last.stats || {});
  updateLiveStatsText(last.stats || {});
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
        if (last.overlay_url) {
          showLiveResults();
          document.getElementById('liveOverlay').src = last.overlay_url;
        }
        updateLiveChart(last.stats || {});
        updateLiveStatsText(last.stats || {});
      } catch (e) {
        console.warn('WS parse error:', e);
      }
    };
    liveWs.onclose = () => {
      console.log('Live WebSocket closed');
      liveWs = null;
      if (!liveTimer) {
        if (liveTimer) clearInterval(liveTimer);
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

function showLiveResults() {
  document.getElementById('live-empty-state').style.display = 'none';
  document.getElementById('live-results-display').style.display = 'block';
}

function showLiveEmptyState() {
  document.getElementById('live-results-display').style.display = 'none';
  document.getElementById('live-empty-state').style.display = 'block';
}

// Legacy updateLiveChart removed (replaced by multi-chart implementation above)

// Legacy updateLiveStatsText removed (replaced by multi-metric implementation above)

// Load available models and create buttons
async function loadAvailableModels() {
  try {
    const res = await fetch('/available_models');
    const data = await res.json();
    if (data.ok && data.models && data.models.length > 0) {
      createModelButtons(data.models);
    } else {
      showNoModelsMessage();
    }
  } catch (error) {
    console.error('Failed to load available models:', error);
    showNoModelsMessage();
  }
}

function showNoModelsMessage() {
  const container = document.getElementById('model-buttons-container');
  container.innerHTML = '<div class="text-warning"><i class="bi bi-exclamation-triangle"></i> No models found. Please check the models folder.</div>';
}

function selectModel(model) {
  // Update UI to show selected model
  const buttons = document.querySelectorAll('#model-buttons-container button');
  buttons.forEach(btn => {
    // Reset all buttons to outline style
    btn.classList.remove('active', 'btn-primary', 'btn-outline-primary');
    btn.classList.add('btn-outline-primary');
    
    // Highlight the selected button
    if (btn.textContent.trim() === model.name) {
      btn.classList.remove('btn-outline-primary');
      btn.classList.add('btn-primary', 'active');
    }
  });
  
  // Load the model from the models folder
  loadModel(model.id);
}

// Global variable to track current model
let currentModel = null;

function loadModel(modelId) {
  const formData = new FormData();
  formData.append('folder_path', modelId);
  
  fetch('/select_model_folder', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.ok) {
      currentModel = data.model;
      updateCurrentModelIndicator(data.model.name);
      showAlert('success', `Model "${data.model.name}" loaded successfully`);
    } else {
      showAlert('danger', `Failed to load model: ${data.error || 'Unknown error'}`);
    }
  })
  .catch(error => {
    console.error('Error loading model:', error);
    showAlert('danger', `Error loading model: ${error.message}`);
  });
}

function updateCurrentModelIndicator(modelName) {
  const indicator = document.getElementById('current-model-name');
  if (indicator) {
    indicator.textContent = modelName;
    indicator.classList.remove('text-muted');
    indicator.classList.add('text-success');
  }
}

// Legacy duplicate displayInferenceResults removed (see consolidated implementation later in file)

// Charts registry and helpers for multi-chart rendering
const charts = {};
function destroyChart(id) {
  if (charts[id]) { try { charts[id].destroy(); } catch {} delete charts[id]; }
}
function renderBarChart(canvasId, data, label, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const hasData = Array.isArray(data) && data.length > 0;
  const labels = hasData ? data.map((_, i) => `#${i+1}`) : ['No data'];
  const dataset = {
    label,
    data: hasData ? data : [0],
    backgroundColor: color,
    borderColor: color
  };
  destroyChart(canvasId);
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [dataset] },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#e6e6e6' } },
        y: { ticks: { color: '#e6e6e6' } }
      }
    }
  });
}

// ===== Histogram helpers (new) =====
const DEFAULT_HIST_BINS = 10;
function computeHistogram(data, bins = DEFAULT_HIST_BINS, range = null) {
  const arr = Array.isArray(data) ? data.filter(v => typeof v === 'number' && isFinite(v)) : [];
  if (arr.length === 0) {
    return { counts: [0], labels: ['No data'], min: 0, max: 0, width: 0 };
  }
  const min = range ? range[0] : Math.min(...arr);
  const max = range ? range[1] : Math.max(...arr);
  if (max === min) {
    return { counts: [arr.length], labels: [`${min.toFixed(2)}`], min, max, width: 0 };
  }
  const width = (max - min) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of arr) {
    let idx = Math.floor((v - min) / width);
    if (idx < 0) idx = 0;
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  }
  const labels = Array.from({ length: bins }, (_, i) => {
    const a = min + i * width;
    const b = a + width;
    return `${a.toFixed(1)}–${b.toFixed(1)}`;
  });
  return { counts, labels, min, max, width };
}

function renderHistogramChart(canvasId, data, label, color, bins = DEFAULT_HIST_BINS) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const hist = computeHistogram(data, bins);
  destroyChart(canvasId);
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels: hist.labels, datasets: [{ label: `${label} count`, data: hist.counts, backgroundColor: color, borderColor: color }] },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: '#e6e6e6' } }, y: { ticks: { color: '#e6e6e6' } } }
    }
  });
}

function renderComparisonChart(canvasId, origData, procData, label) {
  // Updated: render binned histograms for original vs processed
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const all = [...(Array.isArray(origData) ? origData : []), ...(Array.isArray(procData) ? procData : [])].filter(v => typeof v === 'number' && isFinite(v));
  const hasAny = all.length > 0;
  const range = hasAny ? [Math.min(...all), Math.max(...all)] : [0, 0];
  const bins = DEFAULT_HIST_BINS;
  const hOrig = computeHistogram(origData || [], bins, range);
  const hProc = computeHistogram(procData || [], bins, range);
  destroyChart(canvasId);
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hasAny ? hOrig.labels : ['No data'],
      datasets: [
        { label: `Original ${label}`, data: hasAny ? hOrig.counts : [0], backgroundColor: '#999999', borderColor: '#999999' },
        { label: `Processed ${label}`, data: hasAny ? hProc.counts : [0], backgroundColor: '#5b9cff66', borderColor: '#5b9cff' }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#e6e6e6' } } },
      scales: { x: { ticks: { color: '#e6e6e6' } }, y: { ticks: { color: '#e6e6e6' } } }
    }
  });
}

function createModelButtons(models) {
  const container = document.getElementById('model-buttons-container');
  if (!container) {
    console.error('Model buttons container not found');
    return;
  }
  
  // Clear existing content
  container.innerHTML = '';
  
  if (!models || models.length === 0) {
    showNoModelsMessage();
    return;
  }
  
  models.forEach(model => {
    const button = document.createElement('button');
    button.className = 'btn btn-outline-primary me-2 mb-2';
    button.textContent = model.name;
    
    button.addEventListener('click', () => selectModel(model));
    container.appendChild(button);
  });
}

// Preprocess tab model list
async function loadPreprocModels() {
  try {
    const res = await fetch('/available_models');
    const data = await res.json();
    const container = document.getElementById('preproc-model-buttons');
    if (!container) return;
    container.innerHTML = '';
    if (!data.ok || !data.models || data.models.length === 0) {
      container.innerHTML = '<div class="text-warning small"><i class="bi bi-exclamation-triangle"></i> No models found.</div>';
      return;
    }
    data.models.forEach(model => {
      const btn = document.createElement('button');
      btn.className = 'btn btn-outline-secondary btn-sm me-2 mb-2';
      btn.textContent = model.name;
      btn.addEventListener('click', () => {
        preprocModel = model;
        // Toggle button styles
        container.querySelectorAll('button').forEach(b => { b.classList.remove('active'); b.classList.remove('btn-secondary'); b.classList.add('btn-outline-secondary'); });
        btn.classList.add('active');
        btn.classList.remove('btn-outline-secondary');
        btn.classList.add('btn-secondary');
      });
      container.appendChild(btn);
    });
  } catch (e) { console.error('Failed to load preprocess models', e); }
}



// Image zoom modal functionality
function showImageInModal(imageSrc, title = 'Image Preview', isCanvas = false, canvasElement = null) {
  const modalEl = document.getElementById('imageZoomModal');
  if (!modalEl) {
    console.warn('imageZoomModal element not found in DOM');
    if (typeof showAlert === 'function') showAlert('danger', 'Zoom modal not found. Please reload the page.');
    return;
  }
  if (typeof bootstrap === 'undefined' || !bootstrap.Modal) {
    console.warn('Bootstrap Modal not available');
    if (typeof showAlert === 'function') showAlert('danger', 'Bootstrap JS not loaded. Check base.html includes.');
    return;
  }
  const modal = new bootstrap.Modal(modalEl, { keyboard: true });
  const modalTitle = document.getElementById('imageZoomModalTitle');
  const modalImg = document.getElementById('imageZoomImg');
  const modalCanvas = document.getElementById('imageZoomCanvas');
  const spinner = document.getElementById('imageZoomSpinner');
  const viewport = document.getElementById('imageZoomViewport');
  
  // Reset zoom state
  resetZoomState(viewport);
  
  // Set title
  if (modalTitle) modalTitle.textContent = title;
  
  // Hide both img and canvas initially
  if (modalImg) modalImg.style.display = 'none';
  if (modalCanvas) modalCanvas.style.display = 'none';
  if (spinner) spinner.style.display = 'block';
  
  if (isCanvas && canvasElement && modalCanvas) {
    // Copy canvas content to modal canvas
    const ctx = modalCanvas.getContext('2d');
    modalCanvas.width = canvasElement.width;
    modalCanvas.height = canvasElement.height;
    ctx.drawImage(canvasElement, 0, 0);
    
    if (spinner) spinner.style.display = 'none';
    modalCanvas.style.display = 'block';
    enableInteractiveZoom(viewport, modalCanvas);
  } else if (modalImg) {
    // Load image
    modalImg.onload = function() {
      if (spinner) spinner.style.display = 'none';
      modalImg.style.display = 'block';
      enableInteractiveZoom(viewport, modalImg);
    };
    modalImg.onerror = function() {
      if (spinner) spinner.style.display = 'none';
      modalImg.style.display = 'block';
      modalImg.alt = 'Failed to load image';
    };
    modalImg.src = imageSrc;
  }
  
  modal.show();
}

// Interactive zoom behavior (pan + wheel/pinch zoom) with aspect ratio preserved
function enableInteractiveZoom(viewportEl, contentEl) {
  if (!viewportEl || !contentEl) return;
  viewportEl.style.position = 'relative';
  viewportEl.style.overflow = 'hidden';
  // Enforce viewport constraints so images never exceed screen bounds
  viewportEl.style.maxHeight = '85vh';
  viewportEl.style.maxWidth = '90vw';
  viewportEl.style.touchAction = 'none';
  contentEl.style.transformOrigin = '0 0';
  contentEl.style.willChange = 'transform';
  contentEl.style.userSelect = 'none';

  const state = viewportEl._zoomState || { scale: 1, x: 0, y: 0 };
  viewportEl._zoomState = state;

  function apply() {
    contentEl.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
  }

  function clampPosition() {
    const rect = contentEl.getBoundingClientRect();
    const vp = viewportEl.getBoundingClientRect();
    const maxX = Math.max(0, rect.width - vp.width);
    const maxY = Math.max(0, rect.height - vp.height);
    state.x = Math.min(Math.max(state.x, -maxX), 0);
    state.y = Math.min(Math.max(state.y, -maxY), 0);
  }

  // Wheel zoom
  viewportEl.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1.1 : 0.9;
    const prev = state.scale;
    state.scale = Math.min(Math.max(state.scale * delta, 1), 10);
    const rect = contentEl.getBoundingClientRect();
    const vp = viewportEl.getBoundingClientRect();
    const cx = e.clientX - vp.left - state.x;
    const cy = e.clientY - vp.top - state.y;
    state.x -= (cx / prev) * (state.scale - prev);
    state.y -= (cy / prev) * (state.scale - prev);
    clampPosition();
    apply();
  }, { passive: false });

  // Drag to pan
  let dragging = false, lastX = 0, lastY = 0;
  viewportEl.addEventListener('pointerdown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; viewportEl.setPointerCapture(e.pointerId); });
  viewportEl.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX; const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    state.x += dx; state.y += dy;
    clampPosition();
    apply();
  });
  viewportEl.addEventListener('pointerup', () => { dragging = false; });
  viewportEl.addEventListener('pointercancel', () => { dragging = false; });

  // Pinch zoom
  let pinchDist = 0;
  viewportEl.addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) {
      pinchDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
    }
  }, { passive: true });
  viewportEl.addEventListener('touchmove', (e) => {
    if (e.touches.length === 2 && pinchDist) {
      e.preventDefault();
      const newDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      const delta = newDist / pinchDist;
      pinchDist = newDist;
      state.scale = Math.min(Math.max(state.scale * delta, 1), 10);
      clampPosition();
      apply();
    }
  }, { passive: false });

  // Double-click to reset
  viewportEl.addEventListener('dblclick', () => { state.scale = 1; state.x = 0; state.y = 0; apply(); });

  apply();
}

function resetZoomState(viewportEl) {
  if (!viewportEl) return;
  viewportEl._zoomState = { scale: 1, x: 0, y: 0 };
}

// Helper: compose a canvas that contains the base image plus a visible overlay canvas
function composeImageWithOverlay(imageEl, overlayCanvas) {
  if (!imageEl || !overlayCanvas) return null;
  // If overlay is hidden, we don't compose
  const isHidden = overlayCanvas.style && overlayCanvas.style.display === 'none';
  if (isHidden) return null;
  const w = overlayCanvas.width || imageEl.clientWidth || 512;
  const h = overlayCanvas.height || imageEl.clientHeight || 512;
  const cnv = document.createElement('canvas');
  cnv.width = w; cnv.height = h;
  const ctx = cnv.getContext('2d');
  // Compute object-fit contain placement for the image so it matches the overlay drawing
  const natW = imageEl.naturalWidth || w;
  const natH = imageEl.naturalHeight || h;
  const arImg = natW / natH;
  const arBox = w / h;
  let dispW, dispH, offX, offY;
  if (arImg > arBox) { dispW = w; dispH = w / arImg; offX = 0; offY = (h - dispH) / 2; }
  else { dispH = h; dispW = h * arImg; offY = 0; offX = (w - dispW) / 2; }
  ctx.clearRect(0,0,w,h);
  ctx.drawImage(imageEl, 0, 0, natW, natH, offX, offY, dispW, dispH);
  // Draw the overlay canvas on top (it already has proper scaling/offset)
  try { ctx.drawImage(overlayCanvas, 0, 0); } catch {}
  return cnv;
}

function setupImageClickHandlers() {
  // Handle clickable images
  document.addEventListener('click', function(e) {
    // Special handling for synthetic previews: request high-res image before opening
    if (e.target.id && e.target.id.startsWith('synth-prev-')) {
      const rid = e.target.id.replace('synth-prev-', '');
      if (window.openSynthHighResModal) {
        e.preventDefault();
        window.openSynthHighResModal(rid);
        return;
      }
    }

    if (e.target.classList.contains('clickable-image')) {
      const title = e.target.alt || 'Image Preview';
      // Try composing with any visible overlay canvas in the same container
      let overlayCanvas = null;
      // Prefer an overlay canvas inside the same positioned container
      const parent = e.target.parentElement;
      if (parent) {
        overlayCanvas = parent.querySelector('canvas.image-overlay-canvas');
      }
      // If none found, search a bit wider within the same row/card
      if (!overlayCanvas) {
        let p = e.target.closest('.position-relative, .card, .row, .col');
        if (p) overlayCanvas = p.querySelector('canvas.image-overlay-canvas');
      }
      // For synth rows, skip composing overlay to preserve high-res clarity (low-res overlay would downscale)
      if (e.target.id && e.target.id.startsWith('synth-prev-')) {
        overlayCanvas = null;
      }
      const composite = overlayCanvas ? composeImageWithOverlay(e.target, overlayCanvas) : null;
      if (composite) {
        showImageInModal(null, title, true, composite);
      } else {
        showImageInModal(e.target.src, title);
      }
    }
    
    if (e.target.classList.contains('clickable-canvas')) {
      const title = e.target.id.includes('Original') ? 'Original Image' : 
                   e.target.id.includes('Processed') ? 'Processed Image' : 'Canvas Preview';
      showImageInModal(null, title, true, e.target);
    }
  });
}

// Initialize the page state
document.addEventListener('DOMContentLoaded', function() {
  // Setup image click handlers FIRST
  try { setupImageClickHandlers(); } catch (e) { console.warn('Failed to setup click handlers', e); }

  // Show empty states initially for all tabs (guarded)
  try { if (typeof showEmptyState === 'function') showEmptyState(); } catch(e) { console.warn('showEmptyState missing'); }
  try { if (typeof showFrameEmptyState === 'function') showFrameEmptyState(); } catch(e) { console.warn('showFrameEmptyState missing'); }
  try { if (typeof showLiveEmptyState === 'function') showLiveEmptyState(); } catch(e) { console.warn('showLiveEmptyState missing'); }
  
  // Load available models, guarded
  setTimeout(() => {
    try { if (typeof loadAvailableModels === 'function') loadAvailableModels(); } catch(e) { console.warn('loadAvailableModels missing'); }
    try { if (typeof loadPreprocModels === 'function') loadPreprocModels(); } catch(e) { console.warn('loadPreprocModels missing'); }
  }, 100);
});

// Simple loading overlay for preprocess tab
function ensurePreprocLoadingEl() {
  if (preprocLoadingEl) return preprocLoadingEl;
  preprocLoadingEl = document.createElement('div');
  preprocLoadingEl.id = 'preproc-loading-overlay';
  preprocLoadingEl.style.position = 'fixed';
  preprocLoadingEl.style.top = '0';
  preprocLoadingEl.style.left = '0';
  preprocLoadingEl.style.right = '0';
  preprocLoadingEl.style.bottom = '0';
  preprocLoadingEl.style.background = 'rgba(0,0,0,0.4)';
  preprocLoadingEl.style.display = 'none';
  preprocLoadingEl.style.zIndex = '1050';
  preprocLoadingEl.style.alignItems = 'center';
  preprocLoadingEl.style.justifyContent = 'center';
  preprocLoadingEl.style.color = '#fff';
  preprocLoadingEl.style.fontSize = '1rem';
  preprocLoadingEl.innerHTML = '<div class="text-center"><div class="spinner-border text-light mb-3" role="status"><span class="visually-hidden">Loading...</span></div><div id="preproc-loading-text">Processing...</div></div>';
  document.body.appendChild(preprocLoadingEl);
  return preprocLoadingEl;
}

function showPreprocLoading(text) {
  const el = ensurePreprocLoadingEl();
  const txt = document.getElementById('preproc-loading-text');
  if (txt) txt.textContent = text || 'Processing...';
  el.style.display = 'flex';
}

function hidePreprocLoading() {
  const el = ensurePreprocLoadingEl();
  el.style.display = 'none';
}

// Synthesis tab logic moved to synth.js to avoid duplication and global variable conflicts.

// ===== INFERENCE FUNCTIONS =====
async function runInference(imageName) {
  if (!imageName) {
    showAlert('warning', 'No image selected for inference.');
    return;
  }

  // Show loading interface
  showLoadingInterface(imageName);

  try {
    const formData = new FormData();
    formData.append('image_name', imageName);

    const response = await fetch('/inference', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.ok) {
      displayInferenceResults(data);
      showAlert('success', `Inference completed for ${imageName}`);
    } else {
      showAlert('danger', `Inference failed: ${data.error || 'Unknown error'}`);
      hideLoadingInterface();
    }
  } catch (error) {
    console.error('Inference error:', error);
    showAlert('danger', `Inference failed: ${error.message}`);
    hideLoadingInterface();
  }
}

function showLoadingInterface(imageName) {
  const loadingInterface = document.getElementById('loading-interface');
  const resultsDisplay = document.getElementById('results-display');
  const emptyState = document.getElementById('empty-state');
  const loadingImageName = document.getElementById('loading-image-name');
  const loadingModelName = document.getElementById('loading-model-name');

  if (loadingInterface) loadingInterface.style.display = 'block';
  if (resultsDisplay) resultsDisplay.style.display = 'none';
  if (emptyState) emptyState.style.display = 'none';
  if (loadingImageName) loadingImageName.textContent = imageName;
  if (loadingModelName) {
    const currentModel = document.getElementById('current-model-name');
    loadingModelName.textContent = currentModel ? currentModel.textContent : 'Unknown';
  }
}

function hideLoadingInterface() {
  const loadingInterface = document.getElementById('loading-interface');
  if (loadingInterface) loadingInterface.style.display = 'none';
}

function displayInferenceResults(data) {
  // Hide loading interface
  hideLoadingInterface();

  // Hide empty state and show results display
  const emptyState = document.getElementById('empty-state');
  const resultsDisplay = document.getElementById('results-display');
  if (emptyState) emptyState.style.display = 'none';
  if (resultsDisplay) resultsDisplay.style.display = 'block';

  // Update overlay image
  const overlayImg = document.getElementById('overlay');
  if (overlayImg && data.overlay_url) {
    overlayImg.src = data.overlay_url;
  }

  // Update model info
  const modelInfo = document.getElementById('results-model-name');
  if (modelInfo && data.model_info) {
    modelInfo.textContent = data.model_info.name || 'Unknown';
  }

  // Update statistics
  if (data.stats) {
    updateStatsDisplay(data.stats);
    updateStatsCharts(data.stats);
  }

  console.log('Inference results:', data);
}

function updateStatsDisplay(stats) {
  const statsText = document.getElementById('statsText');
  if (statsText) {
    const fmt = (v) => (v != null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
    statsText.innerHTML = `
      <div>Count: <strong>${stats.count || 0}</strong></div>
      <div>Mean length: <strong>${fmt(stats.mean_length)}</strong></div>
      <div>Mean width: <strong>${fmt(stats.mean_width)}</strong></div>
      <div>Mean aspect ratio: <strong>${fmt(stats.mean_aspect_ratio)}</strong></div>
    `;
  }
}

function updateStatsCharts(stats) {
  const lengths = stats.lengths || [];
  const widths = stats.widths || [];
  const aspectRatios = stats.aspect_ratios || [];

  // Render binned histograms instead of per-crystal bars
  renderHistogramChart('statsChartLen', lengths, 'Length (px)', '#5b9cff');
  renderHistogramChart('statsChartWid', widths, 'Width (px)', '#9cf');
  renderHistogramChart('statsChartAR', aspectRatios, 'Aspect Ratio', '#f59e0b');
}