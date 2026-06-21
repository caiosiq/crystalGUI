// Server Communication Layer

export async function uploadTargetImage(file) {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch('/upload_target', { method: 'POST', body: fd });
    return await res.json();
}

export async function generatePreview(payload) {
    const res = await fetch('/synth_preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    return await res.json();
}

// Presets
export async function fetchPresets() {
    const res = await fetch('/synth_presets');
    return await res.json();
}

export async function fetchPreset(name) {
    const res = await fetch(`/synth_get_preset?name=${encodeURIComponent(name)}`);
    return await res.json();
}

export async function deletePreset(name) {
    const res = await fetch(`/synth_delete_preset/${encodeURIComponent(name)}`, { method: 'DELETE' });
    return await res.json();
}

export async function savePreset(name, config) {
    const res = await fetch('/synth_save_preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, config })
    });
    return await res.json();
}

// Batch Jobs
export async function fetchBatchDefaults() {
    const res = await fetch('/synth_batch_defaults');
    return await res.json();
}

export async function submitBatchJob(payload) {
    const res = await fetch('/synth_batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    return await res.json();
}

export async function fetchJobs() {
    const res = await fetch('/synth_jobs');
    return await res.json();
}

export async function deleteJob(jobId) {
    const res = await fetch(`/synth_delete_job/${jobId}`, { method: 'DELETE' });
    // Returns 204 or 200, assume json or ok
    if (res.ok) return { ok: true }; // Sometimes delete returns no content
    // Try json just in case
    try { return await res.json(); } catch(e) { return { ok: false, error: res.statusText }; }
}

// Optimization
export async function fetchOptimizationParams() {
    const res = await fetch('/calibration/params');
    return await res.json();
}

export async function startOptimization(formData) {
    const res = await fetch('/calibration/start', { method: 'POST', body: formData });
    return await res.json();
}

export async function stopOptimization(jobId) {
    const res = await fetch(`/calibration/stop/${jobId}`, { method: 'POST' });
    return await res.json();
}

export async function fetchOptimizationStatus(jobId) {
    const res = await fetch(`/calibration/status/${jobId}`);
    return await res.json();
}

export async function saveSyntheticTarget(config, seed) {
    const res = await fetch('/synth_save_target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config, seed })
    });
    return await res.json();
}

export async function computeLoss(formData) {
    const res = await fetch('/calibration/compute_loss', { method: 'POST', body: formData });
    return await res.json();
}
