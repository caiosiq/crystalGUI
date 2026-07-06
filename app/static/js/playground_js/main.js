import { debounce, showToast, updateLabelFor, syncRangeLabels, updateLambdaTLabels, updateTextureTypeHelp } from './utils.js?v=9';
import { getConfig, applyConfigToUI } from './config.js?v=12';
import * as api from './api.js?v=6';
import * as ui from './ui.js?v=5';
import * as render from './render.js?v=6';
import * as opt from './optimization.js?v=5';

const BATCH_BASE_DIR_KEY = 'osog_batch_base_dir';

// ------------------------------------------------------------------
// Global State
// ------------------------------------------------------------------

let currentSeed = null;
let currentStageT = 0.5;
let targetImageFilename = null;
let debounceTimer = null;

function getCurrentStageT() {
    const slider = document.getElementById('stageTSlider');
    if (!slider) return currentStageT;
    return parseInt(slider.value, 10) / 100;
}

function updateStageTLabel() {
    currentStageT = getCurrentStageT();
    const lbl = document.getElementById('stageTLabel');
    if (lbl) lbl.textContent = currentStageT.toFixed(2);
}

// ------------------------------------------------------------------
// Core Logic
// ------------------------------------------------------------------

async function regenerate(forcedConfig = null) {
    const status = document.getElementById('statusText');
    if(status) status.textContent = 'Generating...';
    
    try {
        const synthConfig = forcedConfig || getConfig();
        const mergeOn = !!(synthConfig.physics && synthConfig.physics.label_merge && synthConfig.physics.label_merge.enable);
        const showRaw = !!document.getElementById('synLabelMergeShowRaw')?.checked;
        
        // 1. Generate Synthetic (Left)
        const stageT = getCurrentStageT();
        const p1 = api.generatePreview({
            t: stageT,
            config: synthConfig,
            return_heads: true,
            return_obbs: true,
            return_obbs_raw: mergeOn && showRaw,
            seed: currentSeed
        });
        
        // 2. Generate GT (Right) if in Validate Mode
        let p2 = Promise.resolve(null);
        if (ui.isCompareMode || document.body.classList.contains('validate-mode')) {
             // Apply Overrides from Optimization module
             const gtConfig = opt.applyOverrides(synthConfig, opt.gtOverrides);
             p2 = api.generatePreview({
                 t: stageT,
                 config: gtConfig,
                 return_heads: true,
                 return_obbs: false,
                 seed: currentSeed
             });
        }

        const [data1, data2] = await Promise.all([p1, p2]);
        
        // Handle Main Image
        if (data1.ok) {
            handlePreviewData(data1, 'mainImage', false);
            
            if(status) {
                status.textContent = `Ready (${data1.width}x${data1.height})`;
                status.classList.remove('text-danger');
                status.classList.add('text-success');
            }
            ui.updateMetrics(data1);
            
            if (data1.seed_used !== undefined) currentSeed = data1.seed_used;
        } else {
            throw new Error(data1.error);
        }
        
        // Handle GT Image
        if (data2 && data2.ok) {
            // For GT, we typically just show the optical head on the right
            const img = document.getElementById('compareImageDisplay');
            if(img) {
                img.src = data2.image_b64;
                img.style.display = 'block';
            }
        }

    } catch (e) {
        showError(e.message, e.stack);
    }
}

function handlePreviewData(data, targetImgId, isGt) {
    // Determine image source
    let imgSrc = data.image_b64;
    
    // If it's the main view, we support head switching
    if (!isGt) {
        // Update hidden head images
         if (data.heads) {
            if (data.heads.optical) setSrc('img-optical', data.heads.optical);
            else if (data.image_b64) setSrc('img-optical', data.image_b64);
            
            if (data.heads.height) setSrc('img-height', data.heads.height);
            if (data.heads.depth) setSrc('img-depth', data.heads.depth);
            if (data.heads.mask) setSrc('img-mask', data.heads.mask);
            
            if (data.heads.brightfield) {
                setSrc('img-brightfield', data.heads.brightfield);
                document.getElementById('thumb-brightfield').style.display = 'block';
            } else {
                document.getElementById('thumb-brightfield').style.display = 'none';
            }
        }
        
        // Get active head
        const activeThumb = document.querySelector('.head-thumb.active');
        let activeType = 'optical';
        if (activeThumb) activeType = activeThumb.id.replace('thumb-', '');
        
        if (activeType !== '3d') {
             const activeSrcEl = document.getElementById(`img-${activeType}`);
             if (activeSrcEl && activeSrcEl.src) imgSrc = activeSrcEl.src;
        }
        
        // Update Main Image
        const img = document.getElementById(targetImgId);
        if(img) {
            img.src = imgSrc;
            img.style.display = 'block';
        }
        
        // OBBs & 3D
        if (data.obbs) {
            const showRaw = !!document.getElementById('synLabelMergeShowRaw')?.checked;
            render.drawObbs(data.obbs, data.width, data.height, {
                rawObbs: data.obbs_raw || null,
                showRaw: showRaw && !!(data.obbs_raw && data.obbs_raw.length),
            });
            render.update3DScene(data.obbs, data.width, data.height);
            const countEl = document.getElementById('obbCountLabel');
            if (countEl) {
                const nMerged = data.obbs.length;
                const nRaw = data.obbs_raw ? data.obbs_raw.length : nMerged;
                countEl.textContent = data.obbs_raw
                    ? `Labels: ${nMerged} merged (${nRaw} raw)`
                    : `Labels: ${nMerged}`;
            }
        }
    }
}

function setSrc(id, src) {
    const el = document.getElementById(id);
    if(el) el.src = src;
}

function scheduleRegenerate() {
    if (opt.isOptimizing) return; 
    
    const status = document.getElementById('statusText');
    if(status) status.textContent = 'Changed...';
    
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        regenerate();
        // Also refresh optimization param list values if in validate mode
        if(document.body.classList.contains('validate-mode')) {
             opt.updateOptimizationListValues();
             opt.updateGTTunerValues();
        }
    }, 50); 
}

function showError(msg, traceback) {
    const status = document.getElementById('statusText');
    if(status) {
        status.textContent = 'Error!';
        status.classList.add('text-danger');
    }
    
    document.getElementById('errorMsg').textContent = msg || "Unknown Error";
    document.getElementById('errorTrace').textContent = traceback || "No traceback available.";
    
    const modalEl = document.getElementById('errorModal');
    if(modalEl && window.bootstrap) {
        const modal = new bootstrap.Modal(modalEl);
        modal.show();
    }
    console.error(msg, traceback);
}

// ------------------------------------------------------------------
// Initialization
// ------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    
    const stageSlider = document.getElementById('stageTSlider');
    if (stageSlider) {
        stageSlider.addEventListener('input', () => {
            updateStageTLabel();
            scheduleRegenerate();
        });
        updateStageTLabel();
    }

    // 1. Attach listeners to all inputs for live update
    document.querySelectorAll('input, select').forEach(el => {
        // Skip GT Tuner inputs (handled by optimization.js)
        if (el.classList.contains('gt-input')) return;
        if (el.id === 'stageTSlider') return;
        
        if (el.type === 'range' || el.type === 'number' || el.type === 'checkbox' || el.tagName === 'SELECT') {
            el.addEventListener('input', (e) => {
                // Don't trigger if it's an optimization parameter checkbox
                if(e.target.classList.contains('opt-param-chk')) return;

                if (e.target.id === 'synOpticsMode' || e.target.id === 'synDofEnable') {
                    ui.updateOpticsControls(document.getElementById('synOpticsMode').value);
                }
                if (['synTextureType', 'synRoughness', 'synGrainSize', 'synAnisotropy', 'synOpticsMode'].includes(e.target.id)) {
                    updateTextureTypeHelp();
                }
                
                // Update label immediately
                updateLabelFor(e.target.id, e.target.value);
                // Schedule regenerate
                scheduleRegenerate();
            });
        }
    });
    
    // 2. Buttons
    document.getElementById('btnRegenerate').onclick = () => {
        currentSeed = null; // Force new seed
        regenerate();
    };
    
    document.getElementById('btnValidateMode').onclick = () => {
        ui.toggleComparisonMode(() => render.onWindowResize()); // Reuse comparison mode logic or custom?
        // Wait, toggleValidateMode in original playground.js did BOTH UI toggle AND buildGTTuner.
        // We split it.
        // Let's implement specific logic here.
        const isVal = document.body.classList.contains('validate-mode');
        if (!isVal) {
            // Enable
            document.body.classList.add('validate-mode');
            document.getElementById('btnValidateMode').classList.add('active', 'btn-info');
            document.getElementById('btnValidateMode').classList.remove('btn-outline-info');
            
            document.getElementById('gtSidebar').style.display = 'flex';
            document.getElementById('compareContainer').style.display = 'flex';
            
            opt.buildGTTuner();
            regenerate();
        } else {
            // Disable
            document.body.classList.remove('validate-mode');
            document.getElementById('btnValidateMode').classList.remove('active', 'btn-info');
            document.getElementById('btnValidateMode').classList.add('btn-outline-info');
            
            document.getElementById('gtSidebar').style.display = 'none';
            document.getElementById('compareContainer').style.display = 'none';
            
            regenerate();
        }
    };
    
    document.getElementById('btnCompareMode').onclick = () => ui.toggleComparisonMode(() => render.onWindowResize());
    document.getElementById('btnObb').onclick = ui.toggleObb;
    
    // Head Switching
    ['optical', 'height', 'depth', 'mask', 'brightfield', '3d'].forEach(type => {
        const thumb = document.getElementById(`thumb-${type}`);
        if(thumb) {
            thumb.onclick = () => ui.switchHead(type, {
                on3DInit: render.init3DViewer,
                onResize: render.onWindowResize
            });
        }
    });
    
    // 3. Optimization
    opt.initOptimization({
        onRegenerate: (config) => regenerate(config),
        onUpdateUI: (config) => applyConfigToUI(config)
    });
    
    document.getElementById('btnStartOpt').onclick = () => opt.startOptimization(targetImageFilename);
    document.getElementById('btnStopOpt').onclick = opt.stopOptimization;
    document.getElementById('btnApplyOpt').onclick = () => {
        document.getElementById('btnApplyOpt').classList.add('d-none');
        regenerate();
    };
    
    // 4. Drag Drop
    ui.setupDragDrop(async (file) => {
        const res = await api.uploadTargetImage(file);
        if(res.ok) {
            targetImageFilename = res.filename;
            const badge = document.getElementById('targetImgName');
            if(badge) {
                badge.textContent = res.filename;
                badge.classList.remove('bg-secondary');
                badge.classList.add('bg-success');
            }
            ui.setTargetImageState(true);
        } else {
            showToast("Upload failed: " + res.error);
        }
    });
    
    // 5. Presets (Simplified)
    loadPresets();
    initBatchOutputFields();
    
    // 6. Loss Evaluation
    const evalBtn = document.querySelector('#lossResults')?.previousElementSibling?.querySelector('button');
    if(evalBtn) evalBtn.onclick = () => opt.evaluateLoss(targetImageFilename);

    // 7. Initial Load
    opt.loadOptimizationParams();
    syncRangeLabels();
    updateTextureTypeHelp();
    regenerate();
    
    // Spacebar
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
            e.preventDefault();
            currentSeed = null; 
            regenerate();
        }
    });
});

// Helper for Presets (can be moved to presets.js if needed, but small enough)
async function loadPresets() {
    const data = await api.fetchPresets();
    const sel = document.getElementById('presetSelector');
    if (data.ok && sel) {
        sel.innerHTML = '<option value="" disabled selected>Select Preset...</option>';
        data.presets.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            sel.appendChild(opt);
        });
    }
}

// Global scope hacks for onclick in HTML? 
// The HTML uses onclick="switchSidebarTab(...)".
// ES6 modules are not global. We need to attach these to window or rewrite HTML listeners.
// Since we are rewriting, let's attach to window for compatibility with existing HTML onclicks.
window.switchSidebarTab = ui.switchSidebarTab;
window.switchHead = (type) => ui.switchHead(type, { on3DInit: render.init3DViewer, onResize: render.onWindowResize });
window.toggleObb = ui.toggleObb;
window.debugExport = async () => {
    const btn = document.getElementById('btnDebugExport');
    const orig = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = 'Exporting...'; }
    try {
        const data = await api.debugExport({
            t: getCurrentStageT(),
            config: getConfig(),
            seed: currentSeed,
        });
        if (data.ok) {
            const a = data.analysis || {};
            console.log('[Debug Export] folder:', data.folder);
            console.log('[Debug Export] analysis:', a);
            showToast(`Saved to ${data.folder}`);
            // Also surface the key finding inline for quick reading.
            const msg = `raw=${a.raw_count}, merged=${a.merged_count}, `
                + `pairs>=thr=${a.pairs_overlap_ge_threshold} `
                + `(same_gid=${a['  of_those_same_group_id']}, diff_gid=${a['  of_those_diff_group_id']}), `
                + `group_filter=${a.merge_by_group_id}`;
            console.log('[Debug Export] summary:', msg);
        } else {
            showToast('Debug export failed: ' + (data.error || 'unknown'));
        }
    } catch (e) {
        showToast('Debug export error: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = orig; }
    }
};
window.toggleComparisonMode = () => ui.toggleComparisonMode(() => render.onWindowResize());
window.toggleValidateMode = () => document.getElementById('btnValidateMode').click(); // Delegate
window.loadSelectedPreset = async () => {
    const sel = document.getElementById('presetSelector');
    if(sel.value) {
        const data = await api.fetchPreset(sel.value);
        if(data.ok) {
            applyConfigToUI(data.config);
            showToast(`Loaded ${sel.value}`);
            const nameEl = document.getElementById('batchDatasetName');
            if (nameEl) nameEl.value = sel.value;
            updateBatchPathPreview();
            // Update Optimization UI to reflect loaded config
            opt.updateOptimizationListValues();
            if(document.body.classList.contains('validate-mode')) {
                opt.updateGTTunerValues();
            }
            regenerate();
        }
    }
};
window.savePresetPrompt = () => {
    const name = prompt("Preset Name:");
    if(name) {
        api.savePreset(name, getConfig()).then(d => {
            if(d.ok) { showToast("Saved"); loadPresets(); }
            else showToast(d.error);
        });
    }
};
window.deleteSelectedPreset = async () => {
    const sel = document.getElementById('presetSelector');
    if(sel.value && confirm("Delete?")) {
        await api.deletePreset(sel.value);
        loadPresets();
    }
};
window.submitBatchJob = async () => {
    const count = parseInt(document.getElementById('batchCount').value) || 100;
    const tasks = parseInt(document.getElementById('batchTasks').value) || 4;
    const baseDir = document.getElementById('batchBaseDir')?.value.trim() || '';
    const datasetName = document.getElementById('batchDatasetName')?.value.trim() || '';
    const password = document.getElementById('batchPassword').value.trim();
    const presetName = document.getElementById('presetSelector').value || "custom";

    if (!password) {
        showToast("Error: Password required");
        return;
    }
    if (!baseDir) {
        showToast("Error: Output root directory is required");
        return;
    }

    localStorage.setItem(BATCH_BASE_DIR_KEY, baseDir);

    const payload = {
        config: getConfig(),
        n_images: count,
        n_tasks: tasks,
        password: password,
        preset_name: presetName,
        base_dir: baseDir,
        dataset_name: datasetName || presetName,
    };

    const data = await api.submitBatchJob(payload);
    if (data.ok) {
        const dest = data.out_dir ? `\n${data.out_dir}` : '';
        showToast(`Job ${data.job_id} submitted.${dest}`);
        refreshJobs();
    } else {
        showToast(data.error);
    }
};

function sanitizeBatchName(name, fallback = 'custom') {
    const safe = String(name || '').trim().replace(/[^\w\-_.]/g, '_');
    return safe || fallback;
}

function updateBatchPathPreview() {
    const preview = document.getElementById('batchPathPreview');
    const baseEl = document.getElementById('batchBaseDir');
    const nameEl = document.getElementById('batchDatasetName');
    if (!preview || !baseEl) return;

    const base = baseEl.value.trim().replace(/\/+$/, '');
    const presetName = document.getElementById('presetSelector')?.value || 'custom';
    const name = sanitizeBatchName(nameEl?.value || presetName, sanitizeBatchName(presetName));
    if (!base) {
        preview.textContent = 'Set an output root directory to see the final path.';
        return;
    }
    preview.textContent = `${base}/${name}_YYYY_MM_DD_HH_MM`;
}

async function initBatchOutputFields() {
    const baseEl = document.getElementById('batchBaseDir');
    const nameEl = document.getElementById('batchDatasetName');
    if (!baseEl) return;

    const saved = localStorage.getItem(BATCH_BASE_DIR_KEY);
    if (saved) {
        baseEl.value = saved;
    } else {
        try {
            const data = await api.fetchBatchDefaults();
            if (data.ok && data.batch_root_dir) {
                baseEl.value = data.batch_root_dir;
            }
        } catch (e) {
            console.warn('Could not load batch defaults', e);
        }
    }

    const onBatchFieldChange = () => {
        updateBatchPathPreview();
        const base = baseEl.value.trim();
        if (base) localStorage.setItem(BATCH_BASE_DIR_KEY, base);
    };

    baseEl.addEventListener('input', onBatchFieldChange);
    if (nameEl) nameEl.addEventListener('input', updateBatchPathPreview);
    document.getElementById('presetSelector')?.addEventListener('change', updateBatchPathPreview);

    for (const id of ['batchLambdaMin', 'batchLambdaMax']) {
        document.getElementById(id)?.addEventListener('input', updateLambdaTLabels);
        document.getElementById(id)?.addEventListener('change', updateLambdaTLabels);
    }

    updateBatchPathPreview();
    updateLambdaTLabels();
};
// Jobs
async function refreshJobs() {
    const data = await api.fetchJobs();
    const tbody = document.getElementById('jobsTableBody');
    if (data.ok && tbody) {
        tbody.innerHTML = '';
        data.jobs.forEach(job => {
            const tr = document.createElement('tr');
            const shortId = job.job_id.length > 8 ? job.job_id.substring(0,8) : job.job_id;
            let statusColor = 'text-warning';
            if (job.status === 'completed') statusColor = 'text-success';
            if (job.status === 'error') statusColor = 'text-danger';
            
            tr.innerHTML = `
                <td>
                    <span title="${job.job_id}">${shortId}</span>
                    <br><span class="text-muted" style="font-size:0.65rem;">${job.out_dir}</span>
                </td>
                <td class="${statusColor}">${job.status}</td>
                <td>${job.progress ? job.progress.toFixed(0) : 0}%</td>
                <td>
                    <button class="btn btn-sm btn-link text-danger p-0" onclick="deleteJob('${job.job_id}')">
                        <i class="bi bi-x-circle"></i>
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }
}
window.deleteJob = async (id) => {
    if(confirm("Delete Job?")) {
        await api.deleteJob(id);
        refreshJobs();
    }
};
window.evaluateLoss = () => opt.evaluateLoss(targetImageFilename);
window.startOptimization = () => opt.startOptimization(targetImageFilename);
window.stopOptimization = opt.stopOptimization;
window.applyOptimizationResults = () => document.getElementById('btnApplyOpt').click();
window.confirmExitValidateMode = () => { /* handled by toggle */ };
window.clearRefImage = ui.clearRefImage;
window.toggleFloatingDashboard = () => {
    const content = document.getElementById('optFloatingContent');
    const icon = document.getElementById('optFloatingIcon');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.className = 'bi bi-chevron-down';
    } else {
        content.style.display = 'none';
        icon.className = 'bi bi-chevron-up';
    }
};

// Start jobs loop
setInterval(refreshJobs, 5000);
