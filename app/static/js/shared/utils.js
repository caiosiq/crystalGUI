/* shared/utils.js */

function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
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

function cacheBustStaticUrl(url) {
  if (!url) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}t=${Date.now()}`;
}

function formatTimeLabel(t, unit = null) {
  const u = unit || outputsTimeUnit || 'min';
  const v = Number(t);
  if (!isFinite(v)) return String(t);
  if (u) {
    return Number.isInteger(v) ? `${v} ${u}` : `${v} ${u}`;
  }
  return String(v);
}

function sameResolvedUrl(a, b) {
  if (!a || !b) return a === b;
  try {
    return new URL(a, window.location.origin).href === new URL(b, window.location.origin).href;
  } catch {
    return a === b;
  }
}

function encodeStaticPath(url) {
  if (!url) return url;
  const qIdx = url.indexOf('?');
  const path = qIdx >= 0 ? url.slice(0, qIdx) : url;
  const query = qIdx >= 0 ? url.slice(qIdx) : '';
  const encoded = path.split('/').map((seg, i) => (i === 0 || !seg) ? seg : encodeURIComponent(seg)).join('/');
  return encoded + query;
}

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
  let s = String(n).split('?')[0].replace(/\\/g, '/').toLowerCase();
  return s;
}

function overlayStemForDatasetImage(relName) {
  return String(relName || '').replace(/\\/g, '/').replace(/\//g, '__').replace(/\.[^./]+$/, '');
}
