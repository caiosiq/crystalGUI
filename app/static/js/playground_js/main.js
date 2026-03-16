import { debounce, showToast, updateLabelFor } from './utils.js';
import { getConfig, applyConfigToUI } from './config.js';
import * as api from './api.js';
import * as ui from './ui.js';
import * as render from './render.js';
import * as opt from './optimization.js';

// ------------------------------------------------------------------
// Global State
// ------------------------------------------------------------------

let currentSeed = null;
let targetImageFilename = null;
let debounceTimer = null;

// ------------------------------------------------------------------
// Core Logic
// ------------------------------------------------------------------

async function regenerate(forcedConfig = null) {
    const status = document.getElementById('statusText');
    if(status) status.textContent = 'Generating...';
    
    try {
        const synthConfig = forcedConfig || getConfig();
        
        // 1. Generate Synthetic (Left)
        // We pass 'mainImage' ID logic to UI, but here we just get data.
        const p1 = api.generatePreview({
            t: 0.5,
            config: synthConfig,
            return_heads: true,
            return_obbs: true, // Only main view needs OBBs usually
            seed: currentSeed
        });
        
        // 2. Generate GT (Right) if in Validate Mode
        let p2 = Promise.resolve(null);
        if (ui.isCompareMode || document.body.classList.contains('validate-mode')) {
             // Apply Overrides from Optimization module
             const gtConfig = opt.applyOverrides(synthConfig, opt.gtOverrides);
             p2 = api.generatePreview({
                 t: 0.5,
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
            render.drawObbs(data.obbs, data.width, data.height);
            render.update3DScene(data.obbs, data.width, data.height);
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
    
    // 1. Attach listeners to all inputs for live update
    document.querySelectorAll('input, select').forEach(el => {
        // Skip GT Tuner inputs (handled by optimization.js)
        if (el.classList.contains('gt-input')) return;
        
        if (el.type === 'range' || el.type === 'number' || el.type === 'checkbox' || el.tagName === 'SELECT') {
            el.addEventListener('input', (e) => {
                // Don't trigger if it's an optimization parameter checkbox
                if(e.target.classList.contains('opt-param-chk')) return;

                // Special handling for Optics Mode change
                if (e.target.id === 'synOpticsMode' || e.target.id === 'synDofEnable') {
                    ui.updateOpticsControls(document.getElementById('synOpticsMode').value);
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
    
    // 6. Loss Evaluation
    const evalBtn = document.querySelector('#lossResults')?.previousElementSibling?.querySelector('button');
    if(evalBtn) evalBtn.onclick = () => opt.evaluateLoss(targetImageFilename);

    // 7. Initial Load
    opt.loadOptimizationParams();
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
window.toggleComparisonMode = () => ui.toggleComparisonMode(() => render.onWindowResize());
window.toggleValidateMode = () => document.getElementById('btnValidateMode').click(); // Delegate
window.loadSelectedPreset = async () => {
    const sel = document.getElementById('presetSelector');
    if(sel.value) {
        const data = await api.fetchPreset(sel.value);
        if(data.ok) {
            applyConfigToUI(data.config);
            showToast(`Loaded ${sel.value}`);
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
    const outDir = document.getElementById('batchOutDir').value.trim();
    const password = document.getElementById('batchPassword').value.trim();
    const presetName = document.getElementById('presetSelector').value || "custom";
    
    if (!password) {
        showToast("Error: Password required");
        return;
    }

    const payload = { 
        config: getConfig(), 
        n_images: count, 
        n_tasks: tasks,
        password: password,
        preset_name: presetName
    };
    if(outDir) payload.out_dir = outDir;
    
    const data = await api.submitBatchJob(payload);
    if(data.ok) { 
        showToast(`Job ${data.job_id}`); 
        refreshJobs(); 
        // Clear password for security? Or keep for convenience? 
        // Let's keep it for convenience if they submit multiple jobs.
    }
    else showToast(data.error);
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
