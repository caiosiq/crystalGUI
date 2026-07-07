/* live/main.js */

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
  document.getElementById('timeLabel').innerText = `t = ${formatTimeLabel(datasetFrames[0]?.time ?? 0)}`;
  if (datasetFrames.length) onTimeChange(0);
}

async function onTimeChange(idx) {
  const i = parseInt(idx, 10);
  const frame = datasetFrames[i];
  if (!frame) return;
  document.getElementById('timeLabel').innerText = `t = ${formatTimeLabel(frame.time)}`;
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
