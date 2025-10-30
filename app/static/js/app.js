let selectedImage = null;
let chartInstance = null;
let frameChartInstance = null;
let liveChartInstance = null;
let datasetFrames = [];
let liveTimer = null;
let liveWs = null;
let outputsModel = null; // selected model for Outputs tab
let outputsCurrentPresetName = null; // track current preset name in Outputs
let preprocCurrentPresetName = null; // track current preset name in Preprocess
// Outputs batch cache for drilldown
let outputsBatchSummary = null;
let outputsBatchPerImage = [];
// Track CSV buttons and enable/disable based on data availability
function outputsSetCsvButtonsEnabled(enabled) {
  const btn1 = document.getElementById('btnOutputsCsvSummary');
  const btn2 = document.getElementById('btnOutputsCsvPerImage');
  [btn1, btn2].forEach(btn => { if (btn) btn.disabled = !enabled; });
}

function outputsExportSummaryCSV() {
  try {
    const summary = outputsBatchSummary;
    if (!summary || !summary.times || !summary.stats_by_time) {
      showAlert('warning', 'No summary data to export. Run batch first.');
      return;
    }
    const times = summary.times;
    const map = summary.stats_by_time || {};
    const rows = [];
    // Header row
    rows.push(['time','mean_length','std_length','mean_width','std_width','mean_aspect_ratio','std_aspect_ratio','count_avg']);
    times.forEach(t => {
      const st = getStatsForTime(map, t) || {};
      rows.push([
        t,
        st.mean_length ?? '',
        st.std_length ?? '',
        st.mean_width ?? '',
        st.std_width ?? '',
        st.mean_aspect_ratio ?? '',
        st.std_aspect_ratio ?? '',
        st.count_avg ?? ''
      ]);
    });
    const csv = rows.map(r => r.map(v => formatCsvCell(v)).join(',')).join('\n');
    triggerCsvDownload(csv, 'outputs_summary.csv');
  } catch (e) {
    console.error('outputsExportSummaryCSV failed', e);
    showAlert('danger', 'CSV export failed: ' + e.message);
  }
}

function outputsExportPerImageCSV() {
  try {
    const items = outputsBatchPerImage || [];
    if (!items.length) {
      showAlert('warning', 'No per-image data to export. Run batch first.');
      return;
    }
    const rows = [];
    rows.push(['filename','time','overlay_url','count','mean_length','mean_width','mean_aspect_ratio']);
    items.forEach(e => {
      const s = e.stats || {};
      const timeVal = e.time ?? e.timestamp ?? '';
      const name = e.name || e.filename || e.file || e.path || e.image || e.stem || '';
      rows.push([
        name,
        timeVal,
        e.overlay_url || '',
        s.count ?? '',
        s.mean_length ?? '',
        s.mean_width ?? '',
        s.mean_aspect_ratio ?? ''
      ]);
    });
    const csv = rows.map(r => r.map(v => formatCsvCell(v)).join(',')).join('\n');
    triggerCsvDownload(csv, 'outputs_per_image.csv');
  } catch (e) {
    console.error('outputsExportPerImageCSV failed', e);
    showAlert('danger', 'CSV export failed: ' + e.message);
  }
}

function formatCsvCell(v) {
  if (v==null) return '';
  const s = String(v);
  const needsQuotes = /[",\n]/.test(s);
  const escaped = s.replace(/"/g, '""');
  return needsQuotes ? `"${escaped}"` : escaped;
}

function triggerCsvDownload(csvText, filename) {
  const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}

// ===== Preprocess UI State =====
let preprocImg = null; // original image element
let preprocCanvasOrig = null;
let preprocCanvasProc = null;
let compareChart = null;
let preprocParams = { desaturate: 0, invert: false, gradient_strength: 0, clahe: false, equalize: false, clahe_clip_limit: 2.0, clahe_tile_grid: 8 };
let preprocModel = null; // independent model selection for Preprocess tab
let preprocLoadingEl = null; // loading overlay element for preprocess actions
let preprocBaseImg = null; // server-processed base image for client-side ops
let preprocPreviewCache = new Map(); // cache of server-preprocessed base images keyed by image+pipeline
let preprocWaitingForBase = false; // guards applying client ops until server base is ready

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
  // Clear cached preprocessed base when switching images
  preprocBaseImg = null;
  preprocWaitingForBase = false;
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
  desat.oninput = () => { preprocParams.desaturate = parseFloat(desat.value) / 100.0; if (desatLabel) desatLabel.textContent = `${desat.value}%`; redraw(); };
  grad.oninput = () => { preprocParams.gradient_strength = parseFloat(grad.value) / 100.0; if (gradLabel) gradLabel.textContent = `${grad.value}%`; redraw(); };
  invert.onchange = () => { preprocParams.invert = invert.checked; redraw(); };
  // CLAHE/Equalize require backend processing to reflect accurately; request preview
  clahe.onchange = () => { preprocParams.clahe = clahe.checked; requestPreprocPreview(); };
  equalize.onchange = () => { preprocParams.equalize = equalize.checked; requestPreprocPreview(); };
  // Optional: CLAHE tunables controls (if present)
  const clipEl = document.getElementById('preprocClaheClip');
  const gridEl = document.getElementById('preprocClaheGrid');
  const clipLbl = document.getElementById('preprocClaheClipLabel');
  const gridLbl = document.getElementById('preprocClaheGridLabel');
  if (clipEl) clipEl.oninput = () => { preprocParams.clahe_clip_limit = parseFloat(clipEl.value) || 2.0; if (clipLbl) clipLbl.textContent = preprocParams.clahe_clip_limit.toFixed(1); if (preprocParams.clahe) requestPreprocPreview(); };
  if (gridEl) gridEl.oninput = () => { preprocParams.clahe_tile_grid = parseInt(gridEl.value) || 8; if (gridLbl) gridLbl.textContent = preprocParams.clahe_tile_grid; if (preprocParams.clahe) requestPreprocPreview(); };
  const btnCompare = document.getElementById('btnRunCompare');
  const btnSave = document.getElementById('btnSavePreprocessed');
  if (btnCompare) btnCompare.onclick = runInferenceCompare;
  if (btnSave) btnSave.onclick = savePreprocessedImage;
  // Preset save for Preprocess tab
  const btnPreprocSavePreset = document.getElementById('preprocSavePresetBtn');
  if (btnPreprocSavePreset) btnPreprocSavePreset.onclick = preprocSavePresetPrompt;
  const btnPreprocSaveInCurrent = document.getElementById('preprocSaveInCurrentPresetBtn');
  if (btnPreprocSaveInCurrent) btnPreprocSaveInCurrent.onclick = preprocSaveInCurrentPreset;
}

function preprocCollectPipelineObj() {
  return {
    desaturate: preprocParams.desaturate || 0,
    invert: !!preprocParams.invert,
    gradient_strength: preprocParams.gradient_strength || 0,
    clahe: !!preprocParams.clahe,
    equalize: !!preprocParams.equalize,
    clahe_clip_limit: preprocParams.clahe_clip_limit,
    clahe_tile_grid: preprocParams.clahe_tile_grid,
  };
}

function preprocApplyPipelineToControls(cfg) {
  const desat = document.getElementById('preprocDesat');
  const grad = document.getElementById('preprocGrad');
  const invert = document.getElementById('preprocInvert');
  const clahe = document.getElementById('preprocClahe');
  const equalize = document.getElementById('preprocEqualize');
  const desatLabel = document.getElementById('preprocDesatLabel');
  const gradLabel = document.getElementById('preprocGradLabel');
  if (typeof cfg.desaturate === 'number') { preprocParams.desaturate = cfg.desaturate; if (desat) desat.value = Math.round(cfg.desaturate * 100); if (desatLabel && desat) desatLabel.textContent = `${desat.value}%`; }
  if (typeof cfg.gradient_strength === 'number') { preprocParams.gradient_strength = cfg.gradient_strength; if (grad) grad.value = Math.round(cfg.gradient_strength * 100); if (gradLabel && grad) gradLabel.textContent = `${grad.value}%`; }
  if (typeof cfg.invert === 'boolean') { preprocParams.invert = cfg.invert; if (invert) invert.checked = cfg.invert; }
  if (typeof cfg.clahe === 'boolean') { preprocParams.clahe = cfg.clahe; if (clahe) clahe.checked = cfg.clahe; }
  if (typeof cfg.equalize === 'boolean') { preprocParams.equalize = cfg.equalize; if (equalize) equalize.checked = cfg.equalize; }
  // CLAHE tunables
  const clipEl = document.getElementById('preprocClaheClip');
  const gridEl = document.getElementById('preprocClaheGrid');
  const clipLbl = document.getElementById('preprocClaheClipLabel');
  const gridLbl = document.getElementById('preprocClaheGridLabel');
  if (typeof cfg.clahe_clip_limit === 'number') { preprocParams.clahe_clip_limit = cfg.clahe_clip_limit; if (clipEl) clipEl.value = String(cfg.clahe_clip_limit); if (clipLbl) clipLbl.textContent = preprocParams.clahe_clip_limit.toFixed(1); }
  if (typeof cfg.clahe_tile_grid === 'number') { preprocParams.clahe_tile_grid = cfg.clahe_tile_grid; if (gridEl) gridEl.value = String(cfg.clahe_tile_grid); if (gridLbl) gridLbl.textContent = String(preprocParams.clahe_tile_grid); }
  // Redraw processed preview or request backend if CLAHE/equalize are active
  if (preprocParams.clahe || preprocParams.equalize) requestPreprocPreview(); else drawPreprocProcessed();
}

async function preprocLoadPresetsList() {
  try {
    const res = await fetch('/preproc_presets');
    const data = await res.json();
    const menu = document.getElementById('preproc-presets-menu');
    if (!menu) return;
    menu.innerHTML = '';
    if (!data.ok || !data.presets || data.presets.length === 0) {
      menu.innerHTML = '<li><span class="dropdown-item text-muted">No presets</span></li>';
      return;
    }
    data.presets.forEach(name => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.className = 'dropdown-item';
      a.textContent = name;
      a.href = '#';
      a.onclick = (e) => { e.preventDefault(); preprocLoadPreset(name); };
      li.appendChild(a);
      menu.appendChild(li);
    });
  } catch (e) { console.error('Failed to load preprocess presets', e); }
}

async function preprocLoadPreset(name) {
  try {
    const res = await fetch(`/preproc_get_preset?name=${encodeURIComponent(name)}`);
    const data = await res.json();
    if (data.ok && data.pipeline) {
      preprocApplyPipelineToControls(data.pipeline);
      preprocCurrentPresetName = data.name || name;
      const lbl = document.getElementById('preprocCurrentPresetName');
      if (lbl) lbl.textContent = preprocCurrentPresetName;
      showAlert('success', `Loaded preprocess preset "${data.name || name}"`);
    } else {
      showAlert('danger', `Failed to load preset: ${data.error || 'Unknown error'}`);
    }
  } catch (e) {
    showAlert('danger', 'Failed to load preset: ' + e.message);
  }
}

async function preprocSavePresetPrompt() {
  const cfg = preprocCollectPipelineObj();
  const name = prompt('Preset name');
  if (!name) return;
  try {
    const res = await fetch('/preproc_save_preset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, pipeline: cfg })
    });
    const data = await res.json();
    if (data.ok) {
      showAlert('success', `Preset saved as ${data.name}`);
      preprocCurrentPresetName = data.name || name;
      const lbl = document.getElementById('preprocCurrentPresetName');
      if (lbl) lbl.textContent = preprocCurrentPresetName;
      preprocLoadPresetsList();
    } else {
      showAlert('danger', `Failed to save preset: ${data.error || 'Unknown error'}`);
    }
  } catch (e) {
    showAlert('danger', 'Preset save failed: ' + e.message);
  }
}

async function preprocSaveInCurrentPreset() {
  const cfg = preprocCollectPipelineObj();
  let name = preprocCurrentPresetName;
  if (!name) {
    // If no current preset, ask for a name
    name = prompt('No current preset loaded. Enter a name to save as current:');
    if (!name) return;
  }
  try {
    const res = await fetch('/preproc_save_preset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, pipeline: cfg })
    });
    const data = await res.json();
    if (data.ok) {
      showAlert('success', `Preset "${data.name || name}" updated`);
      preprocCurrentPresetName = data.name || name;
      const lbl = document.getElementById('preprocCurrentPresetName');
      if (lbl) lbl.textContent = preprocCurrentPresetName;
      preprocLoadPresetsList();
    } else {
      showAlert('danger', `Failed to save in current: ${data.error || 'Unknown error'}`);
    }
  } catch (e) {
    showAlert('danger', 'Save in current failed: ' + e.message);
  }
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
  // If server-side contrast ops are active and we don't yet have a base, request and wait
  if ((preprocParams.clahe || preprocParams.equalize) && !preprocBaseImg) {
    try { showPreprocLoading('Preparing preview...'); } catch {}
    preprocWaitingForBase = true;
    requestPreprocPreview();
    return;
  }
  // Source is server-processed base if available and CLAHE/equalize active; else original
  if ((preprocParams.clahe || preprocParams.equalize) && preprocBaseImg) {
    const base = new Image();
    base.onload = () => {
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(base, 0, 0, w, h);
      applyClientSideOps(ctx, w, h);
    };
    base.src = preprocBaseImg;
    return;
  }
  ctx.drawImage(preprocImg, 0, 0, w, h);
  applyClientSideOps(ctx, w, h);
}

function applyClientSideOps(ctx, w, h) {
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
let previewAbortController = null;
let previewOverlayGuard = null;
async function requestPreprocPreview() {
  if (!selectedImage || !preprocCanvasProc) return;
  const previewKey = `${selectedImage}|${getPreprocPreviewPipeline()}`;
  // Cancel any scheduled preview and any in-flight request
  if (previewDebounce) { clearTimeout(previewDebounce); previewDebounce = null; }
  if (previewAbortController) { try { previewAbortController.abort(); } catch {} previewAbortController = null; }
  // If we have cached base image for this pipeline, draw immediately and skip network
  const cached = preprocPreviewCache.get(previewKey);
  if (cached) {
    preprocWaitingForBase = false;
    preprocBaseImg = cached;
    const imgEl = new Image();
    imgEl.onload = () => {
      const ctx = preprocCanvasProc.getContext('2d');
      ctx.clearRect(0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
      ctx.drawImage(imgEl, 0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
      // Apply client-side modifications on top of server-processed base
      applyClientSideOps(ctx, preprocCanvasProc.width, preprocCanvasProc.height);
    };
    imgEl.src = cached;
    return;
  }
  // Debounce actual request
  previewDebounce = setTimeout(async () => {
    showPreprocLoading('Applying preprocessing...');
    if (previewOverlayGuard) { clearTimeout(previewOverlayGuard); }
    previewOverlayGuard = setTimeout(() => { try { hidePreprocLoading(); } catch {} }, 10000);
    try {
      previewAbortController = new AbortController();
      const form = new FormData();
      form.append('image_name', selectedImage);
      const pipelineStr = getPreprocPreviewPipeline();
      form.append('pipeline', pipelineStr);
      const res = await fetch('/preproc_preview', { method: 'POST', body: form, signal: previewAbortController.signal });
      const data = await res.json();
      if (data.ok && data.overlay_b64) {
        // cache
        preprocPreviewCache.set(previewKey, data.overlay_b64);
        preprocBaseImg = data.overlay_b64;
        const imgEl = new Image();
        imgEl.onload = () => {
          const ctx = preprocCanvasProc.getContext('2d');
          ctx.clearRect(0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
          ctx.drawImage(imgEl, 0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
          // Apply client-side modifications on top of server-processed base
          applyClientSideOps(ctx, preprocCanvasProc.width, preprocCanvasProc.height);
        };
        imgEl.src = data.overlay_b64;
        preprocWaitingForBase = false;
      } else if (data && !data.ok) {
        console.error('Preproc preview error:', data.error);
        showAlert('danger', 'Preproc preview failed: ' + (data.error || 'Unknown error'));
        preprocWaitingForBase = false; // unblock UI; optionally fall back to original image
      }
    } catch (e) {
      if (!(e && e.name === 'AbortError')) {
        console.error('Preproc preview failed', e);
        showAlert('danger', 'Preproc preview failed: ' + e.message);
        preprocWaitingForBase = false; // unblock UI; optionally fall back to original image
      }
    } finally {
      if (previewOverlayGuard) { clearTimeout(previewOverlayGuard); previewOverlayGuard = null; }
      hidePreprocLoading();
      previewAbortController = null;
    }
  }, 200);
}

function getPreprocPreviewPipeline() {
  return JSON.stringify({
    clahe: preprocParams.clahe,
    equalize: preprocParams.equalize,
    clahe_clip_limit: preprocParams.clahe_clip_limit,
    clahe_tile_grid: preprocParams.clahe_tile_grid
  });
}

function getPreprocPipeline() {
  return JSON.stringify({
    desaturate: preprocParams.desaturate,
    invert: preprocParams.invert,
    gradient_strength: preprocParams.gradient_strength,
    clahe: preprocParams.clahe,
    equalize: preprocParams.equalize,
    clahe_clip_limit: preprocParams.clahe_clip_limit,
    clahe_tile_grid: preprocParams.clahe_tile_grid
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

// Save the current image with the preprocessing pipeline applied (full-resolution)
async function savePreprocessedImage() {
  if (!selectedImage) {
    showAlert('danger', 'Select an image first');
    return;
  }
  // Optional desired filename input (if present in the UI)
  const desiredEl = document.getElementById('preprocFilenameInput') || document.getElementById('preprocDesiredName');
  const desiredName = desiredEl ? desiredEl.value.trim() : '';

  const form = new FormData();
  form.append('image_name', selectedImage);
  form.append('pipeline', getPreprocPipeline());
  if (desiredName) form.append('desired_name', desiredName);

  try {
    showPreprocLoading('Saving full-resolution image...');
    const res = await fetch('/save_preprocessed', { method: 'POST', body: form });
    const data = await res.json();
    hidePreprocLoading();
    if (data.ok) {
      const fname = data.filename || desiredName || selectedImage;
      showAlert('success', `Saved preprocessed image as ${fname}`);
      // If a link element exists, update it
      const linkEl = document.getElementById('preprocSavedLink');
      if (linkEl && data.saved_url) {
        linkEl.href = data.saved_url;
        linkEl.textContent = 'Open saved image';
        linkEl.target = '_blank';
        linkEl.rel = 'noopener noreferrer';
        linkEl.style.display = 'inline';
      }
    } else {
      showAlert('danger', `Save failed: ${data.error || 'Unknown error'}`);
    }
  } catch (e) {
    hidePreprocLoading();
    console.error('save_preprocessed error', e);
    showAlert('danger', 'Save failed: ' + e.message);
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
  // Stabilize canvas sizing to avoid responsive feedback loops causing endless expansion
  try {
    const parent = ctx.parentElement;
    if (parent) {
      if (!parent.style.position) parent.style.position = 'relative';
      parent.style.minHeight = parent.style.minHeight || '260px';
    }
    ctx.style.width = '100%';
    ctx.style.height = ctx.style.height || '220px';
  } catch {}
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
      maintainAspectRatio: false,
      devicePixelRatio: 2,
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

// Improve default chart appearance and resolution globally (guarded)
(function setupChartDefaults(){
  try {
    if (typeof Chart !== 'undefined' && Chart.defaults) {
      Chart.defaults.color = '#e6e6e6';
      Chart.defaults.font.size = 13; // larger base font
      Chart.defaults.plugins = Chart.defaults.plugins || {};
      Chart.defaults.plugins.legend = Chart.defaults.plugins.legend || {};
      Chart.defaults.plugins.legend.labels = Chart.defaults.plugins.legend.labels || {};
      Chart.defaults.plugins.legend.labels.color = '#e6e6e6';
    }
  } catch(e) { /* ignore */ }
})();
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
  // Stabilize canvas and container sizing to prevent infinite growth due to responsive recalculations
  try {
    const parent = ctx.parentElement;
    if (parent) {
      if (!parent.style.position) parent.style.position = 'relative';
      parent.style.minHeight = parent.style.minHeight || '260px';
    }
    ctx.style.width = '100%';
    ctx.style.height = ctx.style.height || '220px';
  } catch {}
  const hist = computeHistogram(data, bins);
  destroyChart(canvasId);
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels: hist.labels, datasets: [{ label: `${label} count`, data: hist.counts, backgroundColor: color, borderColor: color, borderWidth: 1 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      devicePixelRatio: 2,
      layout: { padding: { top: 6, right: 6, bottom: 6, left: 6 } },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#e6e6e6', font: { size: 12 } },
          grid: { color: 'rgba(230,230,230,0.08)' }
        },
        y: {
          ticks: { color: '#e6e6e6', font: { size: 12 } },
          grid: { color: 'rgba(230,230,230,0.08)' }
        }
      }
    }
  });
}

function renderComparisonChart(canvasId, origData, procData, label) {
  // Updated: render binned histograms for original vs processed
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  // Stabilize sizing similar to other charts
  try {
    const parent = ctx.parentElement;
    if (parent) {
      if (!parent.style.position) parent.style.position = 'relative';
      parent.style.minHeight = parent.style.minHeight || '260px';
    }
    ctx.style.width = '100%';
    ctx.style.height = ctx.style.height || '220px';
  } catch {}
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
      maintainAspectRatio: false,
      devicePixelRatio: 2,
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
    try { if (typeof loadOutputsModels === 'function') loadOutputsModels(); } catch(e) { console.warn('loadOutputsModels missing'); }
    try { if (typeof outputsLoadPresetsList === 'function') outputsLoadPresetsList(); } catch(e) { console.warn('outputsLoadPresetsList missing'); }
    try { if (typeof preprocLoadPresetsList === 'function') preprocLoadPresetsList(); } catch(e) { console.warn('preprocLoadPresetsList missing'); }
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
    // Prevent broken image icon; show message and keep previous if load fails
    overlayImg.onerror = () => {
      showAlert('danger', 'Failed to load inference overlay image');
    };
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

// ===== OUTPUTS TAB LOGIC =====
async function loadOutputsModels() {
  try {
    const res = await fetch('/available_models');
    const data = await res.json();
    const container = document.getElementById('outputs-model-buttons');
    if (!container) return;
    container.innerHTML = '';
    if (!data.ok || !data.models || data.models.length === 0) {
      container.innerHTML = '<div class="text-warning small"><i class="bi bi-exclamation-triangle"></i> No models found.</div>';
      return;
    }
    data.models.forEach(model => {
      const btn = document.createElement('button');
      btn.className = 'btn btn-outline-secondary btn-sm me-2 mb-2';
      // Ensure clicking a model DOES NOT submit the surrounding form
      // (inside forms, button default type is "submit")
      btn.type = 'button';
      btn.textContent = model.name;
      btn.addEventListener('click', () => {
        outputsModel = model;
        // Toggle styles
        container.querySelectorAll('button').forEach(b => { b.classList.remove('active'); b.classList.remove('btn-secondary'); b.classList.add('btn-outline-secondary'); });
        btn.classList.add('active');
        btn.classList.remove('btn-outline-secondary');
        btn.classList.add('btn-secondary');
      });
      container.appendChild(btn);
    });
  } catch (e) { console.error('Failed to load outputs models', e); }
}



function outputsCollectPipelineObj() {
  const desat = document.getElementById('outputsDesat');
  const grad = document.getElementById('outputsGrad');
  const invert = document.getElementById('outputsInvert');
  const clahe = document.getElementById('outputsClahe');
  const equalize = document.getElementById('outputsEqualize');
  const clip = document.getElementById('outputsClaheClip');
  const grid = document.getElementById('outputsClaheGrid');
  return {
    desaturate: desat ? (parseFloat(desat.value) / 100.0) : 0,
    gradient_strength: grad ? (parseFloat(grad.value) / 100.0) : 0,
    invert: invert ? !!invert.checked : false,
    clahe: clahe ? !!clahe.checked : false,
    equalize: equalize ? !!equalize.checked : false,
    clahe_clip_limit: clip ? parseFloat(clip.value) : 2.0,
    clahe_tile_grid: grid ? parseInt(grid.value) : 8,
  };
}

function outputsApplyPipelineToControls(cfg) {
  // cfg fields: desaturate [0..1], gradient_strength [0..1], invert, clahe, equalize
  const desat = document.getElementById('outputsDesat');
  const grad = document.getElementById('outputsGrad');
  const invert = document.getElementById('outputsInvert');
  const clahe = document.getElementById('outputsClahe');
  const equalize = document.getElementById('outputsEqualize');
  const desLbl = document.getElementById('outputsDesatLabel');
  const gradLbl = document.getElementById('outputsGradLabel');
  if (desat && typeof cfg.desaturate === 'number') { desat.value = Math.round(cfg.desaturate * 100); if (desLbl) desLbl.textContent = `${desat.value}%`; }
  if (grad && typeof cfg.gradient_strength === 'number') { grad.value = Math.round(cfg.gradient_strength * 100); if (gradLbl) gradLbl.textContent = `${grad.value}%`; }
  if (invert != null && typeof cfg.invert === 'boolean') invert.checked = cfg.invert;
  if (clahe != null && typeof cfg.clahe === 'boolean') clahe.checked = cfg.clahe;
  if (equalize != null && typeof cfg.equalize === 'boolean') equalize.checked = cfg.equalize;
  // CLAHE tunables
  const clip = document.getElementById('outputsClaheClip');
  const grid = document.getElementById('outputsClaheGrid');
  const clipLbl = document.getElementById('outputsClaheClipLabel');
  const gridLbl = document.getElementById('outputsClaheGridLabel');
  if (clip && typeof cfg.clahe_clip_limit === 'number') { clip.value = String(cfg.clahe_clip_limit); if (clipLbl) clipLbl.textContent = String(cfg.clahe_clip_limit); }
  if (grid && typeof cfg.clahe_tile_grid === 'number') { grid.value = String(cfg.clahe_tile_grid); if (gridLbl) gridLbl.textContent = String(cfg.clahe_tile_grid); }
}

async function outputsLoadPresetsList() {
  try {
    const res = await fetch('/preproc_presets');
    const data = await res.json();
    const menu = document.getElementById('outputs-presets-menu');
    if (!menu) return;
    menu.innerHTML = '';
    // Always provide a None option to clear current selection
    {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.className = 'dropdown-item';
      a.textContent = 'None';
      a.href = '#';
  a.onclick = (e) => {
    e.preventDefault();
    outputsCurrentPresetName = null;
    // Reset Outputs preprocess controls to defaults when clearing preset
    try {
      outputsApplyPipelineToControls({
        desaturate: 0,
        gradient_strength: 0,
        invert: false,
        clahe: false,
        equalize: false,
        clahe_clip_limit: 2.0,
        clahe_tile_grid: 8,
      });
    } catch (err) { console.warn('Failed to reset outputs controls', err); }
    const lbl = document.getElementById('outputsCurrentPresetName');
    if (lbl) lbl.textContent = 'None';
    showAlert('info', 'Outputs preset cleared');
  };
      li.appendChild(a);
      menu.appendChild(li);
      const hr = document.createElement('li');
      hr.innerHTML = '<hr class="dropdown-divider">';
      menu.appendChild(hr);
    }
    if (!data.ok || !data.presets || data.presets.length === 0) {
      const li = document.createElement('li');
      li.innerHTML = '<span class="dropdown-item text-muted">No presets</span>';
      menu.appendChild(li);
      return;
    }
    data.presets.forEach(name => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.className = 'dropdown-item';
      a.textContent = name;
      a.href = '#';
      a.onclick = (e) => { e.preventDefault(); outputsLoadPreset(name); };
      li.appendChild(a);
      menu.appendChild(li);
    });
  } catch (e) { console.error('Failed to load presets', e); }
}

async function outputsLoadPreset(name) {
  try {
    const res = await fetch(`/preproc_get_preset?name=${encodeURIComponent(name)}`);
    const data = await res.json();
    if (data.ok && data.pipeline) {
      outputsApplyPipelineToControls(data.pipeline);
      outputsCurrentPresetName = data.name || name;
      const lbl = document.getElementById('outputsCurrentPresetName');
      if (lbl) lbl.textContent = outputsCurrentPresetName;
      showAlert('success', `Loaded preset "${data.name || name}"`);
    } else {
      showAlert('danger', `Failed to load preset: ${data.error || 'Unknown error'}`);
    }
  } catch (e) {
    showAlert('danger', 'Failed to load preset: ' + e.message);
  }
}

async function outputsSaveInCurrentPreset() {
  // Saving presets in Outputs is intentionally disabled.
  showAlert('info', 'Saving in current preset is disabled in Outputs. Use Load Preset to apply configurations.');
}

async function outputsUploadFolder() {
  const input = document.getElementById('outputsFolderUpload');
  if (!input || !input.files || input.files.length === 0) {
    showAlert('warning', 'Please select a folder to upload.');
    return;
  }
  const files = Array.from(input.files);
  const form = new FormData();
  const rels = [];
  // Build FormData robustly: append under multiple common keys to be safe
  files.forEach((f, i) => {
    // Append under a single, consistent key to avoid duplicate uploads
    form.append('files', f);
    // webkitRelativePath preserves subfolder structure from the chosen directory
    const rel = f.webkitRelativePath || f.name;
    rels.push(rel);
  });
  // Send both array and object forms; backend accepts either
  form.append('paths_json', JSON.stringify(rels));
  form.append('paths_json_obj', JSON.stringify({ filenames: rels, count: rels.length }));

  // Client-side debug logs to help diagnose upload issues
  try {
    console.log('[outputsUploadFolder] Selected files:', files.length);
    console.log('[outputsUploadFolder] Example names:', files.slice(0, 5).map(f => ({ name: f.name, rel: f.webkitRelativePath })));
    // Log a few FormData keys
    const fdPreview = [];
    for (const [k, v] of form.entries()) {
      if (fdPreview.length >= 10) break; // avoid huge logs
      fdPreview.push({ key: k, type: (v && v.constructor && v.constructor.name) || typeof v });
    }
    console.log('[outputsUploadFolder] FormData preview:', fdPreview);
  } catch (e) {
    console.warn('[outputsUploadFolder] Debug logging failed:', e);
  }
  try {
    // Show persistent spinner + message until upload completes
    const statusEl = document.getElementById('outputs-upload-status');
    const statusText = document.getElementById('outputs-upload-status-text');
    if (statusText) statusText.textContent = 'Uploading dataset... sending files to the server';
    if (statusEl) statusEl.style.display = 'block';
    showAlert('info', 'Uploading folder...');
    const res = await fetch('/outputs_upload_folder', {
      method: 'POST',
      body: form,
    });
    const data = await res.json();
    if (data.ok && data.dataset_path) {
      const ds = document.getElementById('outputsDatasetFolder');
      // Prefer dataset_path_final if server detected a single top-level subfolder
      let pathToUse = data.dataset_path_final || data.dataset_path;
      // If not provided, derive common top-level from client-side webkitRelativePath
      if (!data.dataset_path_final) {
        try {
          const tops = new Set();
          for (const f of files) {
            const rel = (f.webkitRelativePath || f.name).replace(/^\/+/, '').replace(/\\/g, '/');
            const parts = rel.split('/').filter(p => p && p !== '.' && p !== '..');
            if (parts.length > 1) tops.add(parts[0]);
          }
          if (tops.size === 1) {
            const only = [...tops][0];
            pathToUse = `${data.dataset_path}/${only}`;
          }
        } catch (e) { /* ignore */ }
      }
      if (ds) ds.value = pathToUse;
      const savedCount = (typeof data.nonzero_saved === 'number') ? data.nonzero_saved : (data.saved || files.length);
      showAlert('success', `Folder uploaded (${savedCount} files). Dataset path set.`);
    } else {
      // Show extra server-provided debug info if available
      const dbg = data && data.debug ? `\nServer debug: keys=${JSON.stringify(data.debug.keys)}; value_types=${JSON.stringify(data.debug.value_types)}; paths_json_len=${data.debug.paths_json_len}` : '';
      console.error('[outputsUploadFolder] Upload failed response:', data);
      showAlert('danger', `Upload failed: ${data.error || 'Unknown error'}${dbg}`);
    }
  } catch (e) {
    showAlert('danger', 'Upload failed: ' + e.message);
  } finally {
    // Hide spinner/message only after response or error
    const statusEl = document.getElementById('outputs-upload-status');
    if (statusEl) statusEl.style.display = 'none';
  }
}

async function outputsSavePresetPrompt() {
  // Saving presets in Outputs is intentionally disabled.
  showAlert('info', 'Saving presets is disabled in Outputs. Use Load Preset to apply saved configurations.');
}

function renderOutputsLineChart(canvasId, labels, dataArr, label, color, filenameMap) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  // Guard against Chart.js responsive reflow loops: lock canvas height per our CSS
  try {
    const parent = ctx.parentElement;
    if (parent) {
      parent.style.minHeight = '260px';
    }
    // Freeze the canvas device pixel ratio scaling once per render
    ctx.style.height = ctx.style.height || '220px';
    ctx.style.width = '100%';
  } catch {}
  destroyChart(canvasId);
  charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels.map(t => `${t}`),
      datasets: [{
        label,
        data: dataArr,
        borderColor: color,
        backgroundColor: color + '22',
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 3.5,
        pointHoverRadius: 6.5,
        pointHitRadius: 9,
        pointBackgroundColor: color
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      devicePixelRatio: 2,
      plugins: {
        legend: { labels: { color: '#e6e6e6', font: { size: 13 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0]?.label ? `t = ${items[0].label}` : '',
            afterBody: (items) => {
              const idx = items[0]?.dataIndex ?? 0;
              const t = labels[idx];
              const names = getNamesForTime(filenameMap, t);
              if (!names || names.length === 0) return '';
              const maxList = 6;
              const shown = names.slice(0, maxList).join(', ');
              return `Files (${names.length}): ${shown}${names.length>maxList?' …':''}`;
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#e6e6e6', font: { size: 12 } }, grid: { color: 'rgba(230,230,230,0.08)' } },
        y: { ticks: { color: '#e6e6e6', font: { size: 12 } }, grid: { color: 'rgba(230,230,230,0.08)' } }
      },
      onClick: (evt, elems) => {
        const chart = charts[canvasId];
        const points = chart.getElementsAtEventForMode(evt, 'nearest', { intersect: false }, true);
        if (!points || points.length === 0) return;
        const idx = points[0].index;
        const t = labels[idx];
        // Reset drilldown visuals when changing time selection
        try { if (typeof outputsResetDrilldown === 'function') outputsResetDrilldown(true); } catch {}
        const names = getNamesForTime(filenameMap, t);
        outputsShowFilenameListForTime(t, names || []);
      }
    }
  });
}

function renderOutputsCharts(summary) {
  if (!summary || !summary.times || summary.times.length === 0) {
    showAlert('warning', 'No images found in dataset or empty results.');
    return;
  }
  const times = summary.times;
  const map = summary.stats_by_time || {};
  const filenames = summary.filename_map || {};
  function collect(metricName) {
    return times.map(t => {
      const st = getStatsForTime(map, t);
      return st && st[metricName] != null ? Number(st[metricName]) : 0;
    });
  }
  renderOutputsLineChart('outputsChartMeanLen', times, collect('mean_length'), 'Mean Length', '#5b9cff', filenames);
  renderOutputsLineChart('outputsChartStdLen', times, collect('std_length'), 'Std Length', '#5b9cff', filenames);
  renderOutputsLineChart('outputsChartMeanWid', times, collect('mean_width'), 'Mean Width', '#9cf', filenames);
  renderOutputsLineChart('outputsChartStdWid', times, collect('std_width'), 'Std Width', '#9cf', filenames);
  renderOutputsLineChart('outputsChartMeanAR', times, collect('mean_aspect_ratio'), 'Mean Aspect Ratio', '#f59e0b', filenames);
  renderOutputsLineChart('outputsChartStdAR', times, collect('std_aspect_ratio'), 'Std Aspect Ratio', '#f59e0b', filenames);
  // Single crystal count plot (average only)
  renderOutputsLineChart('outputsChartCountAvg', times, collect('count_avg'), 'Crystal Count', '#22c55e', filenames);
  // Reset drilldown empty state
  const dd = document.getElementById('outputs-drilldown');
  const empty = document.getElementById('outputs-drilldown-empty');
  if (dd && empty) { dd.style.display = 'none'; empty.style.display = 'block'; }
  // Enable CSV buttons now that data exists
  outputsSetCsvButtonsEnabled(true);
}

function outputsShowFilenameListForTime(timeVal, names) {
  const dd = document.getElementById('outputs-drilldown');
  const empty = document.getElementById('outputs-drilldown-empty');
  if (!dd) return;
  // Ensure a container exists for the list
  let list = document.getElementById('outputs-filenames-list');
  if (!list) {
    list = document.createElement('div');
    list.id = 'outputs-filenames-list';
    list.className = 'mb-3';
    const heading = document.createElement('div');
    heading.className = 'text-muted';
    heading.textContent = 'Files at t = ' + timeVal + ':';
    list.appendChild(heading);
    const ul = document.createElement('div');
    ul.className = 'list-group list-group-flush';
    list.appendChild(ul);
    dd.insertBefore(list, dd.firstChild);
  }
  // Update heading with the newly selected time value
  const headingEl = list.querySelector('.text-muted');
  if (headingEl) headingEl.textContent = 'Files at t = ' + timeVal + ':';
  // Reset drilldown visuals when switching times; keep the filename list visible
  try { if (typeof outputsResetDrilldown === 'function') outputsResetDrilldown(true); } catch {}
  const ul = list.querySelector('.list-group');
  ul.innerHTML = '';
  if (!names || names.length === 0) {
    ul.innerHTML = '<div class="list-group-item text-muted">No files</div>';
  } else {
    names.forEach(n => {
      const btn = document.createElement('button');
      btn.className = 'list-group-item list-group-item-action';
      btn.textContent = n;
      btn.title = 'Show detailed stats';
      btn.onclick = () => outputsShowPerImage(n);
      ul.appendChild(btn);
    });
  }
  if (empty) empty.style.display = 'none';
  dd.style.display = 'block';
}

function outputsFindPerImage(name) {
  if (!outputsBatchPerImage || outputsBatchPerImage.length === 0) return null;
  const target = normalizeName(name);
  for (const e of outputsBatchPerImage) {
    const candidates = [e.name, e.filename, e.file, e.path, e.image, e.stem].filter(Boolean);
    for (const c of candidates) {
      if (normalizeName(c) === target) return e;
    }
  }
  return null;
}

function outputsShowPerImage(name) {
  const entry = outputsFindPerImage(name);
  const dd = document.getElementById('outputs-drilldown');
  const empty = document.getElementById('outputs-drilldown-empty');
  if (!entry) { showAlert('danger', 'Image not found in batch results: ' + name); return; }
  // Show panel
  if (empty) empty.style.display = 'none';
  if (dd) dd.style.display = 'block';
  // Overlay
  const img = document.getElementById('outputsDrillOverlay');
  const ph = document.getElementById('outputsDrillPlaceholder');
  if (img) {
    // Robust loader that avoids spurious alerts during rapid time switches and retries once with cache busting.
    // Also falls back to alternate candidate URLs constructed from batch summary when available.
    const overlayUrl = entry.overlay_url;
    const candidates = [];
    if (overlayUrl) candidates.push(overlayUrl);
    // Absolute form for relative URLs (e.g., "/static/..." -> "http(s)://host/static/...")
    try {
      if (overlayUrl && overlayUrl.startsWith('/')) {
        const abs = new URL(overlayUrl, window.location.origin).href;
        candidates.push(abs);
      }
    } catch {}
    // Construct a fallback using filename_map from summary to determine time key
    try {
      if (typeof outputsBatchSummary === 'object' && outputsBatchSummary && outputsBatchSummary.filename_map) {
        const target = normalizeName(name);
        let tKey = null;
        for (const [k, arr] of Object.entries(outputsBatchSummary.filename_map)) {
          if (!Array.isArray(arr)) continue;
          for (const a of arr) {
            if (normalizeName(a) === target) { tKey = k; break; }
          }
          if (tKey != null) break;
        }
        if (tKey != null) {
          const stem = target.replace(/\.[^.]+$/, '');
          const built = `/static/results/outputs/${tKey}/${stem}_overlay.png`;
          candidates.push(built);
          // Also absolute variant
          try { candidates.push(new URL(built, window.location.origin).href); } catch {}
        }
      }
    } catch {}
    let retriedCacheBust = false;
    let candidateIndex = 0;
    const normalizePath = (u) => {
      try { const a = document.createElement('a'); a.href = u; return a.pathname || u; } catch { return u; }
    };
    const tryNext = () => {
      if (candidateIndex >= candidates.length) {
        img.style.display = 'none';
        if (ph) ph.style.display = 'block';
        showAlert('danger', 'Failed to load overlay image for ' + name);
        return;
      }
      const url = candidates[candidateIndex++];
      img.dataset.requestedSrc = url;
      img.src = url;
    };
    img.onerror = () => {
      // Ignore stale errors if a newer src was requested
      const currentReq = img.dataset.requestedSrc || '';
      const isForCurrent = normalizePath(img.src) === normalizePath(currentReq) || img.src.endsWith(currentReq);
      if (!isForCurrent) return;
      // One cache-busting retry per candidate before moving on
      if (!retriedCacheBust) {
        retriedCacheBust = true;
        const bust = (img.dataset.requestedSrc || img.src) + ((img.dataset.requestedSrc || img.src).includes('?') ? '&' : '?') + 'nocache=' + Date.now();
        img.dataset.requestedSrc = bust;
        img.src = bust;
        return;
      }
      retriedCacheBust = false;
      tryNext();
    };
    img.onload = () => { retriedCacheBust = false; };
    tryNext();
    img.alt = 'Overlay for ' + name;
    img.classList.add('clickable-image');
    img.style.display = 'block';
  }
  if (ph) ph.style.display = 'none';
  // Stats and histograms
  const s = entry.stats || {};
  const fmt = (v) => (v!=null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
  const statsEl = document.getElementById('outputsDrillStats');
  if (statsEl) {
    statsEl.innerHTML = `
      <div>Count: <strong>${s.count || 0}</strong></div>
      <div>Mean length: <strong>${fmt(s.mean_length)}</strong></div>
      <div>Mean width: <strong>${fmt(s.mean_width)}</strong></div>
      <div>Mean aspect ratio: <strong>${fmt(s.mean_aspect_ratio)}</strong></div>
    `;
  }
  renderHistogramChart('outputsDrillLen', s.lengths || [], 'Length (px)', '#5b9cff');
  renderHistogramChart('outputsDrillWid', s.widths || [], 'Width (px)', '#9cf');
  renderHistogramChart('outputsDrillAR', s.aspect_ratios || [], 'Aspect Ratio', '#f59e0b');
}

// Reset drilldown visuals and stats. If keepList is true, preserve the filename list while hiding overlay/stats.
function outputsResetDrilldown(keepList=false) {
  const dd = document.getElementById('outputs-drilldown');
  const empty = document.getElementById('outputs-drilldown-empty');
  const img = document.getElementById('outputsDrillOverlay');
  const ph = document.getElementById('outputsDrillPlaceholder');
  if (img) {
    try { img.onerror = null; img.onload = null; } catch {}
    try { img.removeAttribute('data-requested-src'); } catch {}
    img.src = '';
    img.alt = '';
    img.style.display = 'none';
  }
  if (ph) { ph.style.display = 'block'; }
  // Clear stats text
  const statsEl = document.getElementById('outputsDrillStats');
  if (statsEl) statsEl.innerHTML = '';
  // Destroy or clear charts if available
  try {
    if (typeof destroyChart === 'function') {
      destroyChart('outputsDrillLen');
      destroyChart('outputsDrillWid');
      destroyChart('outputsDrillAR');
    } else {
      // Fallback: clear canvases
      ['outputsDrillLen','outputsDrillWid','outputsDrillAR'].forEach(id => {
        const c = document.getElementById(id);
        if (c && c.getContext) c.getContext('2d').clearRect(0,0,c.width,c.height);
      });
    }
  } catch {}
  if (!keepList) {
    // Hide drilldown panel and show empty message
    if (dd) dd.style.display = 'none';
    if (empty) empty.style.display = 'block';
  }
}

async function runOutputsBatch() {
  const folderEl = document.getElementById('outputsDatasetFolder');
  const datasetPath = folderEl ? folderEl.value.trim() : '';
  if (!datasetPath) { showAlert('danger', 'Please enter a dataset folder path'); return; }
  if (!outputsModel) { showAlert('danger', 'Please select a model for Outputs'); return; }
  const cfg = outputsCollectPipelineObj();
  const form = new FormData();
  form.append('dataset_path', datasetPath);
  form.append('pipeline', JSON.stringify(cfg));
  form.append('model_folder', outputsModel.folder || outputsModel.id || '');
  try {
    // Show status area and progress bar
    const statusEl = document.getElementById('outputs-upload-status');
    const statusText = document.getElementById('outputs-upload-status-text');
    const progBox = document.getElementById('outputs-progress');
    const progBar = document.getElementById('outputs-progress-bar');
    const progTxt = document.getElementById('outputs-progress-text');
    const progCnt = document.getElementById('outputs-progress-count');
    const progTot = document.getElementById('outputs-progress-total');
    if (statusEl) statusEl.style.display = 'block';
    if (progBox) progBox.style.display = 'block';
    if (statusText) statusText.textContent = 'Running batch…';
    if (progBar) { progBar.style.width = '0%'; progBar.setAttribute('aria-valuenow', '0'); progBar.textContent = '0%'; }
    if (progTxt) progTxt.textContent = 'Initializing…';
    if (progCnt) progCnt.textContent = '0';
    if (progTot) progTot.textContent = '0';
    showAlert('info', 'Batch started. Tracking progress…');

    // Start async job
    const startRes = await fetch('/outputs_run_batch_start', { method: 'POST', body: form });
    const startData = await startRes.json();
    if (!startData.ok) { showAlert('danger', startData.error || 'Failed to start batch'); return; }
    const jobId = startData.job_id;

    // Poll status until finished
    const pollIntervalMs = 800;
    let finished = false;
    while (!finished) {
      await new Promise(r => setTimeout(r, pollIntervalMs));
      const stRes = await fetch(`/outputs_run_batch_status?job_id=${encodeURIComponent(jobId)}`);
      const st = await stRes.json();
      if (!st.ok) {
        showAlert('danger', st.error || 'Status error');
        break;
      }
      const percent = Number(st.percent || 0);
      const processed = st.processed || 0;
      const total = st.total || 0;
      if (progBar) { const pct = Math.max(0, Math.min(100, percent)); progBar.style.width = pct + '%'; progBar.setAttribute('aria-valuenow', String(pct)); progBar.textContent = pct.toFixed(0) + '%'; }
      if (progTxt) progTxt.textContent = st.message || 'Processing…';
      if (progCnt) progCnt.textContent = String(processed);
      if (progTot) progTot.textContent = String(total);
      finished = st.status === 'finished';
      if (st.status === 'error') {
        showAlert('danger', st.message || 'Batch failed');
        break;
      }
    }

    if (finished) {
      const resRes = await fetch(`/outputs_run_batch_result?job_id=${encodeURIComponent(jobId)}`);
      const data = await resRes.json();
      if (!data.ok) { showAlert('danger', data.error || 'Failed to fetch results'); return; }
      outputsBatchSummary = data.summary || null;
      outputsBatchPerImage = data.per_image || [];
      renderOutputsCharts(outputsBatchSummary);
      outputsSetCsvButtonsEnabled(!!outputsBatchSummary);
      showAlert('success', 'Batch completed');
    }
  } catch (e) {
    console.error('outputs_run_batch failed', e);
    showAlert('danger', 'Batch failed: ' + e.message);
  } finally {
    // Hide spinner/message only after the batch completes or errors
    const statusEl2 = document.getElementById('outputs-upload-status');
    const progBox2 = document.getElementById('outputs-progress');
    if (progBox2) progBox2.style.display = 'none';
    if (statusEl2) statusEl2.style.display = 'none';
  }
}

// ===== Robust time-key lookups for Outputs =====
function canonicalTimeKey(t) {
  // Prefer exact string key, else use rounded to 6 decimals (to match JSON serialization of floats)
  if (typeof t === 'string') return t;
  if (Number.isInteger(t)) return String(t);
  return Number(t).toFixed(6); // stable representation
}

function getStatsForTime(map, t) {
  if (!map) return null;
  // Try multiple candidates
  const k1 = t;
  const k2 = `${t}`;
  const k3 = canonicalTimeKey(t);
  const st = map[k1] || map[k2] || map[k3];
  if (st) return st;
  // Fallback: find nearest numeric key within small epsilon
  const tv = typeof t === 'number' ? t : parseFloat(t);
  if (!isFinite(tv)) return null;
  let best = null, bestDist = 1e9;
  for (const key of Object.keys(map)) {
    const kv = parseFloat(key);
    if (!isFinite(kv)) continue;
    const d = Math.abs(kv - tv);
    if (d < bestDist) { bestDist = d; best = map[key]; }
  }
  return best;
}

function getNamesForTime(filenameMap, t) {
  if (!filenameMap) return [];
  const k1 = t;
  const k2 = `${t}`;
  const k3 = canonicalTimeKey(t);
  const names = filenameMap[k1] || filenameMap[k2] || filenameMap[k3];
  if (Array.isArray(names)) return names;
  // Fallback try nearest numeric key
  const tv = typeof t === 'number' ? t : parseFloat(t);
  let bestKey = null, bestDist = 1e9;
  for (const key of Object.keys(filenameMap)) {
    const kv = parseFloat(key);
    if (!isFinite(kv)) continue;
    const d = Math.abs(kv - tv);
    if (d < bestDist) { bestDist = d; bestKey = key; }
  }
  const fallback = bestKey ? filenameMap[bestKey] : [];
  return Array.isArray(fallback) ? fallback : [];
}

function normalizeName(n) {
  if (!n) return '';
  let s = String(n);
  // remove query params if any
  s = s.split('?')[0];
  // strip directories
  const parts = s.split(/[\\\/]/);
  s = parts[parts.length - 1];
  // optionally strip extension
  s = s.toLowerCase();
  return s;
}

// ===== Page Initialization & Tab Wiring =====
// Load models and presets when the Outputs tab is shown, and once on page load
document.addEventListener('DOMContentLoaded', () => {
  try {
    // Pre-load models/presets for Outputs so the UI is ready when user switches
    loadOutputsModels();
    outputsLoadPresetsList();
  } catch (e) { console.warn('Failed initial Outputs preload', e); }
  try {
    // Preprocess presets menu initial population
    if (typeof preprocLoadPresetsList === 'function') preprocLoadPresetsList();
  } catch (e) { console.warn('Failed initial Preprocess presets preload', e); }
  // Wire tab show events to refresh data when user navigates
  const mainTabs = document.getElementById('mainTabs');
  if (mainTabs) {
    mainTabs.addEventListener('shown.bs.tab', (evt) => {
      const target = evt.target && evt.target.getAttribute('data-bs-target');
      if (target === '#outputs') {
        try { loadOutputsModels(); } catch {}
        try { outputsLoadPresetsList(); } catch {}
      } else if (target === '#preprocess') {
        try { if (typeof preprocLoadPresetsList === 'function') preprocLoadPresetsList(); } catch {}
        try { if (typeof loadPreprocModels === 'function') loadPreprocModels(); } catch {}
      }
    });
  }
});

// Populate Preprocess tab model buttons independently from Inference
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
      btn.type = 'button';
      btn.textContent = model.name;
      btn.addEventListener('click', () => {
        preprocModel = model;
        container.querySelectorAll('button').forEach(b => { b.classList.remove('active'); b.classList.remove('btn-secondary'); b.classList.add('btn-outline-secondary'); });
        btn.classList.add('active');
        btn.classList.remove('btn-outline-secondary');
        btn.classList.add('btn-secondary');
      });
      container.appendChild(btn);
    });
  } catch (e) { console.error('Failed to load preprocess models', e); }
}