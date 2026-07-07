/* outputs/main.js */

function outputsSetCsvButtonsEnabled(enabled) {
  const btn1 = document.getElementById('btnOutputsCsvSummary');
  const btn2 = document.getElementById('btnOutputsCsvPerImage');
  const btnJson = document.getElementById('btnOutputsJsonFull');
  const btnCharts = document.getElementById('btnOutputsCharts');
  [btn1, btn2, btnJson, btnCharts].forEach(btn => { if (btn) btn.disabled = !enabled; });
}

function outputsChartsExportBasename() {
  const ds = outputsSelectedDataset?.display_name || outputsSelectedDataset?.name || 'dataset';
  const safe = String(ds).replace(/[^\w\-_.]+/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');
  return `outputs_charts_${safe || 'dataset'}`;
}

function outputsLoadImageFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to render chart image'));
    img.src = dataUrl;
  });
}

async function outputsBuildChartsCompositeCanvas() {
  if (!outputsBatchSummary || !outputsBatchSummary.times?.length) {
    throw new Error('No chart data available. Run batch first.');
  }
  const cellW = 500;
  const cellH = 220;
  const titleH = 26;
  const pad = 24;
  const headerH = 52;
  const pairRows = OUTPUTS_EVOLUTION_CHART_SPECS.filter(s => !s.fullWidth);
  const fullRows = OUTPUTS_EVOLUTION_CHART_SPECS.filter(s => s.fullWidth);
  const pairRowCount = Math.ceil(pairRows.length / 2);
  const totalW = pad * 2 + cellW * 2 + pad;
  const totalH = pad + headerH + pad
    + pairRowCount * (titleH + cellH + pad)
    + fullRows.length * (titleH + cellH + pad)
    + pad;

  const canvas = document.createElement('canvas');
  canvas.width = totalW;
  canvas.height = totalH;
  const g = canvas.getContext('2d');
  g.fillStyle = '#1f1f1f';
  g.fillRect(0, 0, totalW, totalH);

  const dsLabel = outputsSelectedDataset?.display_name || outputsSelectedDataset?.name || 'Dataset';
  const unit = outputsBatchSummary.time_unit || outputsTimeUnit || 'min';
  g.fillStyle = '#f0f0f0';
  g.font = 'bold 20px sans-serif';
  g.fillText('Outputs over time', pad, pad + 22);
  g.fillStyle = '#b8b8b8';
  g.font = '13px sans-serif';
  g.fillText(`${dsLabel} · time (${unit})`, pad, pad + 42);

  let y = pad + headerH + pad;
  const drawChartCell = async (spec, x, width) => {
    const chart = charts[spec.id];
    if (!chart) throw new Error(`Chart not ready: ${spec.title}`);
    g.fillStyle = '#d8d8d8';
    g.font = '600 14px sans-serif';
    g.fillText(spec.title, x, y + 18);
    const img = await outputsLoadImageFromDataUrl(chart.toBase64Image('image/png', 1));
    const plotY = y + titleH;
    const plotH = cellH;
    const plotW = width;
    const scale = Math.min(plotW / img.width, plotH / img.height);
    const drawW = img.width * scale;
    const drawH = img.height * scale;
    const drawX = x + (plotW - drawW) / 2;
    const drawY = plotY + (plotH - drawH) / 2;
    g.fillStyle = '#2a2a2a';
    g.fillRect(x, plotY, plotW, plotH);
    g.drawImage(img, drawX, drawY, drawW, drawH);
  };

  for (let i = 0; i < pairRows.length; i += 2) {
    await drawChartCell(pairRows[i], pad, cellW);
    if (pairRows[i + 1]) {
      await drawChartCell(pairRows[i + 1], pad + cellW + pad, cellW);
    }
    y += titleH + cellH + pad;
  }
  for (const spec of fullRows) {
    await drawChartCell(spec, pad, totalW - pad * 2);
    y += titleH + cellH + pad;
  }
  return canvas;
}

async function outputsDownloadChartsPng() {
  try {
    const canvas = await outputsBuildChartsCompositeCanvas();
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG export failed'))), 'image/png');
    });
    triggerBlobDownload(blob, `${outputsChartsExportBasename()}.png`);
    showAlert('success', 'Charts saved as PNG');
  } catch (e) {
    console.error('outputsDownloadChartsPng failed', e);
    showAlert('danger', 'Chart export failed: ' + e.message);
  }
}

async function outputsDownloadChartsJpg() {
  try {
    const canvas = await outputsBuildChartsCompositeCanvas();
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('JPG export failed'))), 'image/jpeg', 0.92);
    });
    triggerBlobDownload(blob, `${outputsChartsExportBasename()}.jpg`);
    showAlert('success', 'Charts saved as JPG');
  } catch (e) {
    console.error('outputsDownloadChartsJpg failed', e);
    showAlert('danger', 'Chart export failed: ' + e.message);
  }
}

async function outputsDownloadChartsPdf() {
  try {
    if (!window.jspdf?.jsPDF) {
      showAlert('danger', 'PDF library not loaded. Try PNG export or refresh the page.');
      return;
    }
    const canvas = await outputsBuildChartsCompositeCanvas();
    const dataUrl = canvas.toDataURL('image/png');
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    const margin = 8;
    const maxW = pageW - margin * 2;
    const maxH = pageH - margin * 2;
    const aspect = canvas.height / canvas.width;
    let drawW = maxW;
    let drawH = drawW * aspect;
    if (drawH > maxH) {
      drawH = maxH;
      drawW = drawH / aspect;
    }
    const x = (pageW - drawW) / 2;
    const y = margin;
    pdf.addImage(dataUrl, 'PNG', x, y, drawW, drawH);
    pdf.save(`${outputsChartsExportBasename()}.pdf`);
    showAlert('success', 'Charts saved as PDF');
  } catch (e) {
    console.error('outputsDownloadChartsPdf failed', e);
    showAlert('danger', 'PDF export failed: ' + e.message);
  }
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

function outputsExportFullJSON() {
  try {
    if (!outputsBatchSummary || !outputsBatchPerImage?.length) {
      showAlert('warning', 'No batch data to export. Run batch first.');
      return;
    }
    const ds = outputsSelectedDataset?.display_name || outputsSelectedDataset?.name || 'dataset';
    const payload = {
      meta: {
        dataset: ds,
        dataset_path: outputsSelectedDatasetPath || '',
        exported_at: new Date().toISOString(),
        units: {
          lengths: 'px',
          widths: 'px',
          supervisor_reference_units: 'um',
        },
        scale: outputsGetScaleMeta(),
      },
      summary: outputsBatchSummary,
      per_image: outputsBatchPerImage,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const safe = String(ds).replace(/[^\w\-_.]+/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '') || 'dataset';
    const a = document.createElement('a');
    a.href = url;
    a.download = `outputs_full_${safe}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showAlert('success', 'Full JSON exported — use scripts/compare_psd.py to compare with supervisor Excel');
  } catch (e) {
    console.error('outputsExportFullJSON failed', e);
    showAlert('danger', 'JSON export failed: ' + e.message);
  }
}

function outputsGetScaleMeta() {
  const path = outputsSelectedDatasetPath || '';
  const sc = path ? outputsScaleByDataset[path] : null;
  if (!sc || !sc.umPerPx) return null;
  return {
    um_per_px: sc.umPerPx,
    reference_um: sc.umValue,
    line_length_px: sc.linePx,
    sample_image: sc.sampleName || null,
    method: 'line_on_image',
  };
}

function outputsGetUmPerPx() {
  const path = outputsSelectedDatasetPath || '';
  const sc = path ? outputsScaleByDataset[path] : null;
  return sc && sc.umPerPx > 0 ? sc.umPerPx : null;
}

function outputsScaleUpdateStatus(msg, tone = 'warning') {
  const el = document.getElementById('outputsScaleStatus');
  if (!el) return;
  el.textContent = msg;
  el.className = `small text-${tone}`;
}

function outputsScaleLineLengthPx(points) {
  if (!points || points.length < 2) return 0;
  const dx = points[1].x - points[0].x;
  const dy = points[1].y - points[0].y;
  return Math.hypot(dx, dy);
}

function outputsScaleDraw() {
  const canvas = document.getElementById('outputsScaleCanvas');
  const img = document.getElementById('outputsScaleImg');
  if (!canvas || !img || !canvas.getContext) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!outputsScaleLinePoints.length) return;

  const toDisp = (p) => ({
    x: (p.x / img.naturalWidth) * canvas.width,
    y: (p.y / img.naturalHeight) * canvas.height,
  });

  const pts = outputsScaleLinePoints.map(toDisp);
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  if (pts.length > 1) ctx.lineTo(pts[1].x, pts[1].y);
  ctx.stroke();

  pts.forEach((p, i) => {
    ctx.fillStyle = i === 0 ? '#22d3ee' : '#f59e0b';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function outputsScaleSyncCanvas() {
  const img = document.getElementById('outputsScaleImg');
  const canvas = document.getElementById('outputsScaleCanvas');
  if (!img || !canvas || !img.naturalWidth) return;
  const w = img.clientWidth;
  const h = img.clientHeight;
  canvas.width = Math.max(1, Math.round(w));
  canvas.height = Math.max(1, Math.round(h));
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  outputsScaleDraw();
}

function outputsScaleClearLine() {
  outputsScaleLinePoints = [];
  outputsScaleDraw();
  const umInput = document.getElementById('outputsScaleUmValue');
  if (umInput && !outputsGetUmPerPx()) umInput.value = '';
  outputsScaleUpdateStatus('Click two points on the image to draw a calibration line.');
}

function outputsScaleEventToImageCoords(evt) {
  const canvas = document.getElementById('outputsScaleCanvas');
  const img = document.getElementById('outputsScaleImg');
  const rect = canvas.getBoundingClientRect();
  const dispX = evt.clientX - rect.left;
  const dispY = evt.clientY - rect.top;
  const x = (dispX / canvas.width) * img.naturalWidth;
  const y = (dispY / canvas.height) * img.naturalHeight;
  return { x, y };
}

function outputsScaleOnCanvasClick(evt) {
  if (!outputsSelectedDatasetPath) return;
  const p = outputsScaleEventToImageCoords(evt);
  if (outputsScaleLinePoints.length >= 2) outputsScaleLinePoints = [];
  outputsScaleLinePoints.push(p);
  outputsScaleDraw();
  if (outputsScaleLinePoints.length === 1) {
    outputsScaleUpdateStatus('Click the second endpoint of the calibration line.');
  } else {
    const pxLen = outputsScaleLineLengthPx(outputsScaleLinePoints);
    outputsScaleUpdateStatus(`Line drawn: ${pxLen.toFixed(1)} px. Enter the real length in µm and click Apply scale.`, 'info');
  }
}

function outputsScaleApply() {
  const path = outputsSelectedDatasetPath;
  if (!path) { showAlert('warning', 'Select a dataset first.'); return; }
  if (outputsScaleLinePoints.length < 2) {
    showAlert('warning', 'Draw a calibration line first (two clicks on the image).');
    return;
  }
  const umInput = document.getElementById('outputsScaleUmValue');
  const umValue = umInput ? parseFloat(umInput.value) : NaN;
  if (!isFinite(umValue) || umValue <= 0) {
    showAlert('warning', 'Enter a positive length in µm for the calibration line.');
    return;
  }
  const linePx = outputsScaleLineLengthPx(outputsScaleLinePoints);
  if (linePx <= 0) {
    showAlert('warning', 'Calibration line is too short.');
    return;
  }
  const umPerPx = umValue / linePx;
  const sampleNameEl = document.getElementById('outputsScaleSampleName');
  outputsScaleByDataset[path] = {
    umPerPx,
    umValue,
    linePx,
    p1: outputsScaleLinePoints[0],
    p2: outputsScaleLinePoints[1],
    sampleName: sampleNameEl ? sampleNameEl.textContent.replace(/^Sample:\s*/, '') : '',
    sampleIndex: outputsScaleSampleIndex,
  };
  outputsScaleUpdateStatus(
    `Scale set: ${umPerPx.toFixed(6)} µm/px (${umValue} µm over ${linePx.toFixed(1)} px). You can run the batch.`,
    'success'
  );
  showAlert('success', `Scale calibrated: ${umPerPx.toFixed(4)} µm/px`);
}

async function outputsScaleLoadSample(index = 0) {
  const path = outputsSelectedDatasetPath;
  if (!path) return;
  try {
    const res = await fetch(`/outputs/dataset_sample?dataset_path=${encodeURIComponent(path)}&index=${index}`);
    const data = await res.json();
    if (!data.ok) {
      outputsScaleUpdateStatus(data.error || 'Failed to load sample image', 'danger');
      return;
    }
    outputsScaleSampleIndex = data.index;
    const img = document.getElementById('outputsScaleImg');
    const nameEl = document.getElementById('outputsScaleSampleName');
    if (nameEl) nameEl.textContent = `Sample: ${data.name} (${data.index + 1}/${data.total})`;
    if (!img) return;

    const saved = outputsScaleByDataset[path];
    if (saved && saved.sampleIndex === data.index && saved.p1 && saved.p2) {
      outputsScaleLinePoints = [saved.p1, saved.p2];
      const umInput = document.getElementById('outputsScaleUmValue');
      if (umInput && saved.umValue) umInput.value = String(saved.umValue);
    } else {
      outputsScaleLinePoints = [];
      const umInput = document.getElementById('outputsScaleUmValue');
      if (umInput && !saved?.umPerPx) umInput.value = '';
    }

    img.onload = () => {
      outputsScaleSyncCanvas();
      if (outputsGetUmPerPx()) {
        const sc = outputsScaleByDataset[path];
        outputsScaleUpdateStatus(
          `Scale active: ${sc.umPerPx.toFixed(6)} µm/px (${sc.umValue} µm / ${sc.linePx.toFixed(1)} px)`,
          'success'
        );
      } else if (outputsScaleLinePoints.length < 2) {
        outputsScaleUpdateStatus('Click two points on the image to draw a calibration line.');
      }
    };
    img.src = data.image_url;
  } catch (e) {
    console.error('outputsScaleLoadSample failed', e);
    outputsScaleUpdateStatus('Failed to load sample image', 'danger');
  }
}

function outputsScalePrevSample() {
  outputsScaleLoadSample(Math.max(0, outputsScaleSampleIndex - 1));
}

function outputsScaleNextSample() {
  outputsScaleLoadSample(outputsScaleSampleIndex + 1);
}

function outputsScaleInitForDataset() {
  const panel = document.getElementById('outputsScalePanel');
  const path = outputsSelectedDatasetPath;
  if (!panel) return;
  if (!path) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';

  const canvas = document.getElementById('outputsScaleCanvas');
  if (canvas && !canvas.dataset.bound) {
    canvas.dataset.bound = '1';
    canvas.addEventListener('click', outputsScaleOnCanvasClick);
    window.addEventListener('resize', () => outputsScaleSyncCanvas());
  }

  const saved = outputsScaleByDataset[path];
  outputsScaleSampleIndex = saved?.sampleIndex || 0;
  outputsScaleLinePoints = saved?.p1 && saved?.p2 ? [saved.p1, saved.p2] : [];
  const umInput = document.getElementById('outputsScaleUmValue');
  if (umInput) umInput.value = saved?.umValue ? String(saved.umValue) : '';

  outputsScaleLoadSample(outputsScaleSampleIndex);
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
      outputsSelectedDatasetPath = pathToUse;
      await fetchOutputsDatasets();
      const uploaded = outputsUploadedDatasets.find(d => d.path === pathToUse);
      if (uploaded) selectOutputsDataset(uploaded);
      const savedCount = (typeof data.nonzero_saved === 'number') ? data.nonzero_saved : (data.saved || files.length);
      showAlert('success', `Folder uploaded (${savedCount} files). Dataset selected.`);
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
    heading.textContent = 'Files at t = ' + formatTimeLabel(timeVal) + ':';
    list.appendChild(heading);
    const ul = document.createElement('div');
    ul.className = 'list-group list-group-flush';
    list.appendChild(ul);
    dd.insertBefore(list, dd.firstChild);
  }
  // Update heading with the newly selected time value
  const headingEl = list.querySelector('.text-muted');
  if (headingEl) headingEl.textContent = 'Files at t = ' + formatTimeLabel(timeVal) + ':';
  // Reset drilldown visuals when switching times; keep the filename list visible
  try { if (typeof outputsResetDrilldown === 'function') outputsResetDrilldown(true); } catch {}
  const ul = list.querySelector('.list-group');
  ul.innerHTML = '';
  if (!names || names.length === 0) {
    ul.innerHTML = '<div class="list-group-item text-muted">No files</div>';
  } else {
    names.forEach(n => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'list-group-item list-group-item-action outputs-drill-file';
      btn.dataset.fileName = n;
      btn.textContent = n;
      btn.title = 'Show detailed stats';
      btn.onclick = () => outputsShowPerImage(n);
      if (outputsDrillSelectedName && normalizeName(outputsDrillSelectedName) === normalizeName(n)) {
        btn.classList.add('active');
      }
      ul.appendChild(btn);
    });
  }
  if (empty) empty.style.display = 'none';
  dd.style.display = 'block';
}

function outputsFindPerImage(name) {
  if (!outputsBatchPerImage || outputsBatchPerImage.length === 0) return null;
  const target = normalizeName(name);
  const targetBase = target.split('/').pop();
  for (const e of outputsBatchPerImage) {
    const candidates = [e.name, e.filename, e.file, e.path, e.image, e.stem].filter(Boolean);
    for (const c of candidates) {
      if (normalizeName(c) === target) return e;
    }
  }
  // Legacy batches may only store basenames; only accept that when target has no folder.
  if (!target.includes('/')) {
    for (const e of outputsBatchPerImage) {
      const candidates = [e.name, e.filename, e.file, e.path, e.image, e.stem].filter(Boolean);
      for (const c of candidates) {
        const norm = normalizeName(c);
        if (norm === target || norm.split('/').pop() === targetBase) return e;
      }
    }
  }
  return null;
}

function outputsOverlayCandidates(entry, name) {
  const candidates = [];
  const push = (url) => {
    if (url && !candidates.includes(url)) candidates.push(url);
  };
  if (entry?.overlay_url) push(entry.overlay_url);
  const stem = overlayStemForDatasetImage(name);
  const legacyStem = normalizeName(name).replace(/\//g, '__');
  if (typeof outputsBatchSummary === 'object' && outputsBatchSummary?.filename_map) {
    const target = normalizeName(name);
    for (const [k, arr] of Object.entries(outputsBatchSummary.filename_map)) {
      if (!Array.isArray(arr)) continue;
      if (!arr.some((a) => normalizeName(a) === target)) continue;
      push(`/static/results/outputs/${k}/${stem}_overlay.png`);
      push(`/static/results/outputs/${k}/${legacyStem}_overlay.png`);
      push(`/static/results/outputs/${k}/${legacyStem}.jpg_overlay.png`);
      break;
    }
  }
  return candidates;
}

function outputsLoadDrillOverlay(img, ph, entry, name) {
  if (!img) return;
  img.onerror = null;
  img.onload = null;
  img.style.display = 'none';
  const candidates = outputsOverlayCandidates(entry, name);
  if (!candidates.length) {
    if (ph) ph.style.display = 'block';
    return;
  }
  let idx = 0;
  const loadAt = (i) => {
    if (i >= candidates.length) {
      img.style.display = 'none';
      if (ph) ph.style.display = 'block';
      showAlert('danger', 'Failed to load overlay for ' + name);
      return;
    }
    const base = encodeStaticPath(candidates[i]);
    const bust = base + (base.includes('?') ? '&' : '?') + 'v=' + encodeURIComponent(normalizeName(name));
    img.dataset.requestedSrc = bust;
    img.src = bust;
  };
  img.onload = () => {
    if (img.dataset.requestedSrc && !sameResolvedUrl(img.src, img.dataset.requestedSrc)) return;
    img.style.display = 'block';
    if (ph) ph.style.display = 'none';
  };
  img.onerror = () => {
    if (img.dataset.requestedSrc && !sameResolvedUrl(img.src, img.dataset.requestedSrc)) return;
    idx += 1;
    loadAt(idx);
  };
  loadAt(idx);
}

function outputsMarkDrillFileSelected(name) {
  outputsDrillSelectedName = name || '';
  document.querySelectorAll('#outputs-filenames-list .outputs-drill-file').forEach((btn) => {
    const active = normalizeName(btn.dataset.fileName || btn.textContent) === normalizeName(name);
    btn.classList.toggle('active', active);
  });
}

function outputsShowPerImage(name) {
  const entry = outputsFindPerImage(name);
  const dd = document.getElementById('outputs-drilldown');
  const empty = document.getElementById('outputs-drilldown-empty');
  if (!entry) { showAlert('danger', 'Image not found in batch results: ' + name); return; }
  outputsMarkDrillFileSelected(name);
  if (empty) empty.style.display = 'none';
  if (dd) dd.style.display = 'block';
  const img = document.getElementById('outputsDrillOverlay');
  const ph = document.getElementById('outputsDrillPlaceholder');
  outputsLoadDrillOverlay(img, ph, entry, name);
  if (img) {
    img.alt = 'Overlay for ' + name;
    img.classList.add('clickable-image');
  }
  const s = entry.stats || {};
  const fmt = (v) => (v!=null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
  const statsEl = document.getElementById('outputsDrillStats');
  if (statsEl) {
    statsEl.innerHTML = `
      <div class="small text-info mb-1">${name}</div>
      <div>Count: <strong>${s.count || 0}</strong></div>
      <div>Mean length: <strong>${fmt(s.mean_length)}</strong></div>
      <div>Mean width: <strong>${fmt(s.mean_width)}</strong></div>
      <div>Mean aspect ratio: <strong>${fmt(s.mean_aspect_ratio)}</strong></div>
    `;
  }
  renderDrilldownHistogramChart('outputsDrillLen', s.lengths || [], 'Length (px)', '#5b9cff');
  renderDrilldownHistogramChart('outputsDrillWid', s.widths || [], 'Width (px)', '#9cf');
  renderDrilldownHistogramChart('outputsDrillAR', s.aspect_ratios || [], 'Aspect Ratio', '#f59e0b', {
    range: [0.1, 2.0],
    bins: 10,
    labelDecimals: 1,
  });
}

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
    outputsDrillSelectedName = '';
    // Hide drilldown panel and show empty message
    if (dd) dd.style.display = 'none';
    if (empty) empty.style.display = 'block';
  }
}
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

function renderOutputsDatasetsList(datasets) {
  const list = document.getElementById('outputsDatasetsList');
  if (!list) return;
  list.innerHTML = '';

  const query = (document.getElementById('outputsDatasetSearch')?.value || '').trim().toLowerCase();
  const filtered = query
    ? datasets.filter(ds => {
        const label = (ds.display_name || ds.name || '').toLowerCase();
        return label.includes(query) || (ds.path || '').toLowerCase().includes(query);
      })
    : datasets;

  if (!filtered.length) {
    list.innerHTML = '<div class="list-group-item bg-transparent text-muted py-2">No uploaded datasets found</div>';
    return;
  }

  filtered.forEach(ds => {
    const item = document.createElement('button');
    item.type = 'button';
    const isActive = outputsSelectedDatasetPath && outputsSelectedDatasetPath === ds.path;
    item.className = `list-group-item list-group-item-action bg-transparent text-start py-2${isActive ? ' active' : ''}`;
    item.onclick = () => selectOutputsDataset(ds);
    const label = ds.display_name || ds.name;
    item.innerHTML = `
      <div class="d-flex justify-content-between align-items-center">
        <span class="fw-semibold text-truncate" title="${label}">${label}</span>
        <span class="badge bg-secondary">${ds.image_count || 0}</span>
      </div>
      <div class="small text-muted text-truncate" style="font-size:0.7rem;">${ds.path}</div>
    `;
    list.appendChild(item);
  });
}

function selectOutputsDataset(ds) {
  outputsSelectedDataset = ds;
  outputsSelectedDatasetPath = ds.path;
  renderOutputsDatasetsList(outputsUploadedDatasets);
  outputsScaleInitForDataset();
}

async function fetchOutputsDatasets() {
  try {
    const res = await fetch('/outputs/datasets');
    const data = await res.json();
    if (!data.ok || !data.datasets) return;
    outputsUploadedDatasets = data.datasets;

    if (outputsSelectedDatasetPath) {
      const match = outputsUploadedDatasets.find(d => d.path === outputsSelectedDatasetPath);
      if (match) {
        outputsSelectedDataset = match;
      } else {
        outputsSelectedDataset = null;
        outputsSelectedDatasetPath = '';
      }
    }

    renderOutputsDatasetsList(outputsUploadedDatasets);
  } catch (e) {
    console.error('Failed to fetch outputs datasets', e);
  }
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
      labels: labels.map(t => formatTimeLabel(t)),
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
  outputsTimeUnit = summary.time_unit || outputsTimeUnit || 'min';
  const map = summary.stats_by_time || {};
  const filenames = summary.filename_map || {};
  function collect(metricName) {
    return times.map(t => {
      const st = getStatsForTime(map, t);
      return st && st[metricName] != null ? Number(st[metricName]) : 0;
    });
  }
  const umPerPx = outputsGetUmPerPx();
  const lenScale = (arr) => umPerPx ? arr.map(v => v * umPerPx) : arr;
  const lenUnit = umPerPx ? 'µm' : 'px';
  renderOutputsLineChart('outputsChartMeanLen', times, lenScale(collect('mean_length')), `Mean Length (${lenUnit})`, '#5b9cff', filenames);
  renderOutputsLineChart('outputsChartStdLen', times, lenScale(collect('std_length')), `Std Length (${lenUnit})`, '#5b9cff', filenames);
  renderOutputsLineChart('outputsChartMeanWid', times, lenScale(collect('mean_width')), `Mean Width (${lenUnit})`, '#9cf', filenames);
  renderOutputsLineChart('outputsChartStdWid', times, lenScale(collect('std_width')), `Std Width (${lenUnit})`, '#9cf', filenames);
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

async function runOutputsBatch() {
  const datasetPath = outputsSelectedDatasetPath || '';
  if (!datasetPath) { showAlert('danger', 'Please select a dataset from the list'); return; }
  if (!outputsModel) { showAlert('danger', 'Please select a model for Outputs'); return; }
  if (!outputsGetUmPerPx()) {
    showAlert('warning', 'Set the image scale first: draw a line on the sample image and enter its length in µm.');
    const panel = document.getElementById('outputsScalePanel');
    if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return;
  }
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
