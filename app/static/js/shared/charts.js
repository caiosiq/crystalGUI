/* shared/charts.js */

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
  // Stabilize canvas and container sizing to prevent infinite growth due to responsive recalculations
  try {
    const parent = ctx.parentElement;
    if (parent && !parent.classList.contains('outputs-drill-hist')) {
      if (!parent.style.position) parent.style.position = 'relative';
      parent.style.minHeight = parent.style.minHeight || '260px';
    }
    ctx.style.width = '100%';
    if (!parent || !parent.classList.contains('outputs-drill-hist')) {
      ctx.style.height = ctx.style.height || '220px';
    }
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

function renderDrilldownHistogramChart(canvasId, data, label, color, options = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const bins = options.bins ?? 8;
  const range = options.range ?? null;
  const labelDecimals = options.labelDecimals ?? 0;
  const hist = computeHistogram(data, bins, range);
  const shortLabels = hist.labels.map((lbl, i) => {
    if (!hist.width) return lbl;
    const mid = hist.min + (i + 0.5) * hist.width;
    if (labelDecimals > 0) return mid.toFixed(labelDecimals);
    return Number.isInteger(mid) ? String(mid) : mid.toFixed(0);
  });
  destroyChart(canvasId);
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: shortLabels,
      datasets: [{ label: `${label} count`, data: hist.counts, backgroundColor: color, borderColor: color, borderWidth: 1 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      devicePixelRatio: window.devicePixelRatio || 1,
      layout: { padding: { top: 4, right: 4, bottom: 2, left: 2 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const idx = items[0]?.dataIndex ?? 0;
              return hist.labels[idx] || '';
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: '#e6e6e6',
            font: { size: 10 },
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 6
          },
          grid: { color: 'rgba(230,230,230,0.08)' }
        },
        y: {
          ticks: { color: '#e6e6e6', font: { size: 10 }, maxTicksLimit: 4 },
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
