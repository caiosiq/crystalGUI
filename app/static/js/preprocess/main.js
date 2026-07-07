/* preprocess/main.js */

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
  clearPreprocInferenceLock();
  abortPreprocPreview();
}

function preprocHasActivePipeline() {
  return (preprocParams.desaturate > 0)
    || !!preprocParams.invert
    || (preprocParams.gradient_strength > 0)
    || !!preprocParams.clahe
    || !!preprocParams.equalize;
}

function preprocNeedsServerPreview() {
  // CLAHE/equalize must be rendered server-side; other ops stay client-side for live preview.
  return !!(preprocParams.clahe || preprocParams.equalize);
}

function ensureClientProcessedPreview() {
  return new Promise((resolve) => {
    if (!preprocImg || !preprocCanvasProc) { resolve(); return; }
    const w = preprocCanvasProc.width;
    const h = preprocCanvasProc.height;
    const ctx = preprocCanvasProc.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(preprocImg, 0, 0, w, h);
    if (preprocHasActivePipeline() && !preprocNeedsServerPreview()) {
      applyClientSideOps(ctx, w, h);
    }
    resolve();
  });
}

function clearPreprocInferenceLock() {
  preprocInferenceLocked = false;
  preprocInferenceOverlays = { originalDetections: [], processedBase: null, processedDetections: [] };
}

function abortPreprocPreview() {
  if (previewDebounce) { clearTimeout(previewDebounce); previewDebounce = null; }
  if (previewAbortController) { try { previewAbortController.abort(); } catch {} previewAbortController = null; }
  if (previewOverlayGuard) { clearTimeout(previewOverlayGuard); previewOverlayGuard = null; }
}

function drawObbsOnCanvas(ctx, detections, scaleX, scaleY) {
  if (!detections || !detections.length) return;
  ctx.save();
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2;
  for (const d of detections) {
    const cx = (d.x || 0) * scaleX;
    const cy = (d.y || 0) * scaleY;
    const bw = Math.max((d.w || 0) * scaleX, 1);
    const bh = Math.max((d.h || 0) * scaleY, 1);
    const angle = d.angle || 0;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.strokeRect(-bw / 2, -bh / 2, bw, bh);
    ctx.restore();
  }
  ctx.restore();
}

function renderPreprocToCanvas(canvas, kind, showObbs, drawW, drawH) {
  return new Promise((resolve) => {
    if (!canvas || !preprocImg || preprocImg.naturalWidth <= 0) { resolve(); return; }
    const ctx = canvas.getContext('2d');
    const scaleX = drawW / preprocImg.naturalWidth;
    const scaleY = drawH / preprocImg.naturalHeight;
    const detections = kind === 'original'
      ? (preprocInferenceOverlays.originalDetections || [])
      : (preprocInferenceOverlays.processedDetections || []);

    const finish = (baseImg) => {
      ctx.clearRect(0, 0, drawW, drawH);
      if (baseImg) ctx.drawImage(baseImg, 0, 0, drawW, drawH);
      if (showObbs && detections.length) drawObbsOnCanvas(ctx, detections, scaleX, scaleY);
      resolve();
    };

    if (kind === 'original') {
      finish(preprocImg);
      return;
    }

    const baseB64 = preprocInferenceOverlays.processedBase;
    if (!baseB64) { finish(null); return; }
    const img = new Image();
    img.onload = () => finish(img);
    img.onerror = () => finish(null);
    img.src = baseB64;
  });
}

function drawProcessedCanvasWithDetections(baseB64, detections, showObbs = true) {
  if (!preprocCanvasProc || !baseB64) return Promise.resolve();
  preprocInferenceOverlays.processedBase = baseB64;
  preprocInferenceOverlays.processedDetections = detections || [];
  return renderPreprocToCanvas(
    preprocCanvasProc,
    'processed',
    showObbs,
    preprocCanvasProc.width,
    preprocCanvasProc.height
  );
}

function drawPreprocInferenceOverlays(showObbs = true) {
  if (!preprocCanvasOrig || !preprocCanvasProc) return;
  renderPreprocToCanvas(
    preprocCanvasOrig,
    'original',
    showObbs,
    preprocCanvasOrig.width,
    preprocCanvasOrig.height
  );
  if (preprocInferenceOverlays.processedBase) {
    drawProcessedCanvasWithDetections(
      preprocInferenceOverlays.processedBase,
      preprocInferenceOverlays.processedDetections,
      showObbs
    );
  }
}

async function fetchProcessedPreviewB64() {
  preprocSyncParamsFromControls();
  if (preprocNeedsServerPreview()) {
    const previewKey = `${selectedImage}|${getPreprocPipeline()}`;
    const cached = preprocPreviewCache.get(previewKey);
    if (cached) {
      preprocBaseImg = cached;
      return cached;
    }
    const form = new FormData();
    form.append('image_name', selectedImage);
    form.append('pipeline', getPreprocPipeline());
    const res = await fetch('/preproc_preview', { method: 'POST', body: form });
    const data = await res.json();
    if (!data.ok || !data.overlay_b64) {
      throw new Error((data && data.error) || 'Failed to fetch processed preview');
    }
    preprocPreviewCache.set(previewKey, data.overlay_b64);
    preprocBaseImg = data.overlay_b64;
    return data.overlay_b64;
  }
  await ensureClientProcessedPreview();
  return preprocCanvasProc.toDataURL('image/jpeg', 0.92);
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
  preprocSyncParamsFromControls();
  const redraw = () => {
    clearPreprocInferenceLock();
    preprocBaseImg = null;
    if (preprocNeedsServerPreview()) requestPreprocPreview();
    else drawPreprocProcessed();
  };
  desat.oninput = () => { preprocParams.desaturate = parseFloat(desat.value) / 100.0; if (desatLabel) desatLabel.textContent = `${desat.value}%`; redraw(); };
  grad.oninput = () => { preprocParams.gradient_strength = parseFloat(grad.value) / 100.0; if (gradLabel) gradLabel.textContent = `${grad.value}%`; redraw(); };
  invert.onchange = () => { preprocParams.invert = invert.checked; redraw(); };
  // CLAHE/Equalize require backend processing to match inference
  const serverPreview = () => {
    preprocBaseImg = null;
    clearPreprocInferenceLock();
    if (preprocNeedsServerPreview()) requestPreprocPreview();
    else drawPreprocProcessed();
  };
  clahe.onchange = () => { preprocParams.clahe = clahe.checked; serverPreview(); };
  equalize.onchange = () => { preprocParams.equalize = equalize.checked; serverPreview(); };
  // Optional: CLAHE tunables controls (if present)
  const clipEl = document.getElementById('preprocClaheClip');
  const gridEl = document.getElementById('preprocClaheGrid');
  const clipLbl = document.getElementById('preprocClaheClipLabel');
  const gridLbl = document.getElementById('preprocClaheGridLabel');
  if (clipEl) clipEl.oninput = () => { preprocParams.clahe_clip_limit = parseFloat(clipEl.value) || 2.0; if (clipLbl) clipLbl.textContent = preprocParams.clahe_clip_limit.toFixed(1); if (preprocParams.clahe) { clearPreprocInferenceLock(); requestPreprocPreview(); } };
  if (gridEl) gridEl.oninput = () => { preprocParams.clahe_tile_grid = parseInt(gridEl.value) || 8; if (gridLbl) gridLbl.textContent = preprocParams.clahe_tile_grid; if (preprocParams.clahe) { clearPreprocInferenceLock(); requestPreprocPreview(); } };
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
  clearPreprocInferenceLock();
  preprocBaseImg = null;
  if (preprocNeedsServerPreview()) requestPreprocPreview(); else drawPreprocProcessed();
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
  if (preprocInferenceLocked) {
    drawPreprocInferenceOverlays();
    return;
  }
  const w = preprocCanvasProc.width, h = preprocCanvasProc.height;
  const ctx = preprocCanvasProc.getContext('2d');
  // Server-side CLAHE/Equalize: full pipeline preview from backend (matches inference)
  if (preprocNeedsServerPreview()) {
    if (!preprocBaseImg) {
      try { showPreprocLoading('Preparing preview...'); } catch {}
      preprocWaitingForBase = true;
      requestPreprocPreview();
      return;
    }
    const base = new Image();
    base.onload = () => {
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(base, 0, 0, w, h);
    };
    base.src = preprocBaseImg;
    return;
  }
  // Client-side transforms (desaturate / invert / gradient) for responsive preview
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(preprocImg, 0, 0, w, h);
  if (preprocHasActivePipeline()) applyClientSideOps(ctx, w, h);
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

async function requestPreprocPreview() {
  if (!selectedImage || !preprocCanvasProc) return;
  if (preprocInferenceLocked) return;
  const previewKey = `${selectedImage}|${getPreprocPipeline()}`;
  const requestSeq = ++preprocPreviewRequestSeq;
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
      if (requestSeq !== preprocPreviewRequestSeq || preprocInferenceLocked) return;
      const ctx = preprocCanvasProc.getContext('2d');
      ctx.clearRect(0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
      ctx.drawImage(imgEl, 0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
    };
    imgEl.src = cached;
    return;
  }
  // Debounce actual request
  previewDebounce = setTimeout(async () => {
    if (preprocInferenceLocked) return;
    showPreprocLoading('Applying preprocessing...');
    if (previewOverlayGuard) { clearTimeout(previewOverlayGuard); }
    previewOverlayGuard = setTimeout(() => { try { hidePreprocLoading(); } catch {} }, 10000);
    try {
      previewAbortController = new AbortController();
      const form = new FormData();
      form.append('image_name', selectedImage);
      form.append('pipeline', getPreprocPipeline());
      const res = await fetch('/preproc_preview', { method: 'POST', body: form, signal: previewAbortController.signal });
      const data = await res.json();
      if (requestSeq !== preprocPreviewRequestSeq || preprocInferenceLocked) return;
      if (data.ok && data.overlay_b64) {
        preprocPreviewCache.set(previewKey, data.overlay_b64);
        preprocBaseImg = data.overlay_b64;
        const imgEl = new Image();
        imgEl.onload = () => {
          if (requestSeq !== preprocPreviewRequestSeq || preprocInferenceLocked) return;
          const ctx = preprocCanvasProc.getContext('2d');
          ctx.clearRect(0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
          ctx.drawImage(imgEl, 0, 0, preprocCanvasProc.width, preprocCanvasProc.height);
        };
        imgEl.src = data.overlay_b64;
        preprocWaitingForBase = false;
      } else if (data && !data.ok) {
        console.error('Preproc preview error:', data.error);
        showAlert('danger', 'Preproc preview failed: ' + (data.error || 'Unknown error'));
        preprocWaitingForBase = false;
      }
    } catch (e) {
      if (!(e && e.name === 'AbortError')) {
        console.error('Preproc preview failed', e);
        showAlert('danger', 'Preproc preview failed: ' + e.message);
        preprocWaitingForBase = false;
      }
    } finally {
      if (previewOverlayGuard) { clearTimeout(previewOverlayGuard); previewOverlayGuard = null; }
      hidePreprocLoading();
      previewAbortController = null;
    }
  }, 200);
}

function preprocSyncParamsFromControls() {
  const desat = document.getElementById('preprocDesat');
  const grad = document.getElementById('preprocGrad');
  const invert = document.getElementById('preprocInvert');
  const clahe = document.getElementById('preprocClahe');
  const equalize = document.getElementById('preprocEqualize');
  const clipEl = document.getElementById('preprocClaheClip');
  const gridEl = document.getElementById('preprocClaheGrid');
  if (desat) preprocParams.desaturate = parseFloat(desat.value) / 100.0;
  if (grad) preprocParams.gradient_strength = parseFloat(grad.value) / 100.0;
  if (invert) preprocParams.invert = invert.checked;
  if (clahe) preprocParams.clahe = clahe.checked;
  if (equalize) preprocParams.equalize = equalize.checked;
  if (clipEl) preprocParams.clahe_clip_limit = parseFloat(clipEl.value) || 2.0;
  if (gridEl) preprocParams.clahe_tile_grid = parseInt(gridEl.value, 10) || 8;
}

function getPreprocPreviewPipeline() {
  // Kept for compatibility; preview now uses the same full pipeline as inference.
  return getPreprocPipeline();
}

function getPreprocPipeline() {
  preprocSyncParamsFromControls();
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
  abortPreprocPreview();
  preprocPreviewRequestSeq += 1;
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
  try {
    const processedBase = await fetchProcessedPreviewB64();
    preprocInferenceLocked = true;
    preprocInferenceOverlays = {
      originalDetections: data.original?.detections || [],
      processedBase,
      processedDetections: data.processed?.detections || [],
    };
    drawPreprocInferenceOverlays();
  } catch (e) {
    console.error('Failed to render processed inference overlay', e);
    showAlert('danger', 'Inference finished but failed to render processed overlay: ' + e.message);
  }
}

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
