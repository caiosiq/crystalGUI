import { getVal, formatVal, getDisplayName, showToast } from './utils.js?v=4';
import { getConfig, applyConfigToUI } from './config.js?v=4';
import * as api from './api.js?v=4';

// State
export let isOptimizing = false;
export let optimizationRules = {};
export let paramLinks = {}; // true = linked, false = overridden
export let gtOverrides = {};
let optimizationJobId = null;
let optimizationTimer = null;
let lossChart = null;
let lossData = { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: 'rgb(255, 99, 132)', tension: 0.1 }] };

let paramChart = null;
let paramData = { labels: [], datasets: [] };

// Callbacks
let onRegenerate = () => console.log("Regenerate not set");
let onUpdateUI = () => console.log("UpdateUI not set");

export function initOptimization(callbacks) {
    if (callbacks.onRegenerate) onRegenerate = callbacks.onRegenerate;
    if (callbacks.onUpdateUI) onUpdateUI = callbacks.onUpdateUI;

    // Attach tab listeners
    const tabEls = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabEls.forEach(tab => {
        tab.addEventListener('shown.bs.tab', (event) => {
            if (event.target.id === 'tab-loss-btn' && lossChart) lossChart.resize();
            if (event.target.id === 'tab-params-btn' && paramChart) paramChart.resize();
        });
    });
}

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------

const PARAM_GROUPS = {
    'Morphology': [
        'physics.rod_specs.count_range',
        'physics.rod_specs.length_range',
        'physics.rod_specs.aspect_range',
        'physics.rod_specs.ragged_p',
        'physics.rod_specs.polarity_p',
        'physics.rod_specs.inclusions'
    ],
    'Surface / Texture': [
        'physics.rod_specs.surf_roughness',
        'physics.rod_specs.grain_size',
        'physics.rod_specs.anisotropy',
        'physics.rod_specs.anisotropy_angle_deg'
    ],
    'Agglomeration': [
        'physics.fused.p1',
        'physics.fused.sintering_strength'
    ],
    'Dynamics': [
        'physics.flow_shear_rate',
        'physics.sedimentation_strength'
    ],
    'Optics (Physical)': [
        'optics.polarizer_angle_deg',
        'optics.shadow_gain',
        'optics.focus_z',
        'optics.aperture'
    ],
    'Sensor / Renderer': [
        'sensor.bg_noise_std',
        'sensor.blur_sigma',
        'sensor.vignette_strength',
        'sensor.chromatic_aberration_strength',
        'sensor.spectral_dispersion_strength'
    ],
    'Diffraction': [
        'sensor.diffraction_spikes_intensity',
        'sensor.diffraction_spikes_length',
        'sensor.diffraction_spikes_angle_deg',
        'sensor.diffraction_spikes_threshold'
    ],
    'Depth of Field': [
        'sensor.focus_z',
        'sensor.aperture'
    ],
    'Fouling': [
        'sensor.fouling_prob',
        'sensor.fouling_opacity'
    ],
    'Distractors': [
        'sensor.distractor_blur_sigma',
        'sensor.distractor_opacity',
        'sensor.distractor_anisotropy'
    ]
};

function getParamGroup(key) {
    for (const [group, keys] of Object.entries(PARAM_GROUPS)) {
        if (keys.includes(key)) return group;
    }
    return 'Other';
}

// ------------------------------------------------------------------
// Params & Linking
// ------------------------------------------------------------------

export async function loadOptimizationParams(skipFetch = false) {
    const list = document.getElementById('optParamsList');
    
    if(Object.keys(optimizationRules).length === 0 && !skipFetch) {
        if(list) list.innerHTML = '<div class="text-center text-muted small mt-2">Loading params...</div>';
        try {
            const data = await api.fetchOptimizationParams();
            if(data.ok && data.params) {
                optimizationRules = data.params;
                initParamLinks();
            }
        } catch(e) {
            console.error(e);
            return;
        }
    }
    
    if(Object.keys(optimizationRules).length > 0 && list) {
        list.innerHTML = '';
        const currentConfig = getConfig();
        
        // Group parameters
        const groups = {};
        Object.keys(optimizationRules).forEach(key => {
            const g = getParamGroup(key);
            if (!groups[g]) groups[g] = [];
            groups[g].push(key);
        });
        
        const orderedGroups = Object.keys(PARAM_GROUPS).filter(g => groups[g]);
        if (groups['Other']) orderedGroups.push('Other');
        
        orderedGroups.forEach(groupName => {
            const groupHeader = document.createElement('div');
            groupHeader.className = 'text-uppercase text-secondary fw-bold small mt-2 mb-1 px-1 border-bottom border-dark';
            groupHeader.style.fontSize = '0.7rem';
            groupHeader.textContent = groupName;
            list.appendChild(groupHeader);
            
            groups[groupName].sort().forEach(key => {
                const meta = optimizationRules[key];
                const div = document.createElement('div');
                div.className = 'form-check small d-flex justify-content-between align-items-center mb-1';
                div.id = `opt_item_${key}`;
                
                const val = getParamValue(currentConfig, key, meta);
                let valStr = formatVal(val);
                
                div.innerHTML = `
                    <div>
                    <input class="form-check-input opt-param-chk" type="checkbox" value="${key}" id="chk_opt_${key}">
                    <label class="form-check-label" for="chk_opt_${key}" title="${meta.description || ''}">
                        ${getDisplayName(key)}
                    </label>
                    ${meta.description ? `<i class="bi bi-info-circle ms-1 text-muted" style="font-size: 0.75rem;" title="${meta.description}\nBounds: ${meta.bounds ? `[${meta.bounds.join(', ')}]` : 'N/A'}"></i>` : ''}
                </div>
                    <span id="badge_opt_${key}" class="badge bg-secondary bg-opacity-50" style="font-weight:normal; font-family:monospace;">${valStr}</span>
                `;
                list.appendChild(div);
            });
        });
    }
}

function initParamLinks() {
    Object.keys(optimizationRules).forEach(key => {
        if (paramLinks[key] === undefined) {
            paramLinks[key] = true;
        }
    });
}

export function updateOptimizationListValues(forcedConfig = null) {
    const currentConfig = forcedConfig || getConfig();
    
    Object.keys(optimizationRules).forEach(key => {
        const meta = optimizationRules[key];
        const val = getParamValue(currentConfig, key, meta);
        const badge = document.getElementById(`badge_opt_${key}`);
        if(badge) badge.textContent = formatVal(val);
        
        // Removed auto-checking logic to respect user's manual selection
        // const chk = document.getElementById(`chk_opt_${key}`);
        // if (chk && paramLinks[key] === false) {
        //     if (!chk.checked) chk.checked = true;
        // }
    });
}

function getParamValue(config, key, meta) {
    let lookupPath = key;
    let index = null;
    if (meta && meta.target_attr) {
        lookupPath = meta.target_attr[0];
        index = meta.target_attr[1];
    }
    const parts = lookupPath.split('.');
    let curr = config;
    for(let p of parts) {
        if(curr === undefined) return undefined;
        curr = curr[p];
    }
    if (index !== null && Array.isArray(curr)) return curr[index];
    
    // Fallback
    if (curr === undefined || curr === 0) {
            if (key === 'sensor.focus_z' && config.optics) return config.optics.focus_z;
            if (key === 'sensor.aperture' && config.optics) return config.optics.aperture;
    }
    return curr;
}

// ------------------------------------------------------------------
// GT Tuner
// ------------------------------------------------------------------

export async function buildGTTuner() {
    const container = document.getElementById('gtControlsContent');
    if (!container) return;

    if(Object.keys(optimizationRules).length === 0) {
        await loadOptimizationParams();
    }
    
    const params = optimizationRules;
    if(Object.keys(params).length > 0) {
        container.innerHTML = '';
        
        const groups = {};
        Object.keys(params).forEach(key => {
            const g = getParamGroup(key);
            if (!groups[g]) groups[g] = [];
            groups[g].push(key);
        });
        
        const orderedGroups = Object.keys(PARAM_GROUPS).filter(g => groups[g]);
        if (groups['Other']) orderedGroups.push('Other');
        
        orderedGroups.forEach(groupName => {
            const groupHeader = document.createElement('div');
            groupHeader.className = 'text-uppercase text-secondary fw-bold small mt-3 mb-2 px-1 border-bottom border-dark';
            groupHeader.textContent = groupName;
            container.appendChild(groupHeader);
            
            groups[groupName].forEach(key => {
                createGTTunerItem(container, key, params[key]);
            });
        });
        
        updateGTTunerValues();
    } else {
        container.innerHTML = '<div class="text-danger small">Failed to load params</div>';
    }
}

function createGTTunerItem(container, key, meta) {
    const div = document.createElement('div');
    div.className = 'mb-2 pb-2 border-bottom border-secondary gt-tuner-item';
    div.id = `gt_item_${key}`;
    div.style.display = 'none';
    
    const header = document.createElement('div');
    header.className = 'd-flex justify-content-between align-items-center mb-1';
    
    const labelGroup = document.createElement('div');
    labelGroup.className = 'd-flex align-items-center gap-2';
    
    const linkBtn = document.createElement('button');
    linkBtn.className = 'btn btn-sm btn-link p-0 text-decoration-none';
    linkBtn.id = `btn_link_${key}`;
    linkBtn.title = "Link/Unlink from Synthetic";
    linkBtn.onclick = () => toggleParamLink(key);
    linkBtn.innerHTML = paramLinks[key] ? '<i class="bi bi-link text-success"></i>' : '<i class="bi bi-link-45deg text-warning"></i>';
    
    const label = document.createElement('label');
    label.className = 'form-label small m-0';
    label.textContent = getDisplayName(key);
    
    labelGroup.appendChild(linkBtn);
    labelGroup.appendChild(label);
    
    const valSpan = document.createElement('span');
    valSpan.className = 'badge bg-dark border border-secondary text-light';
    valSpan.id = `gt_val_${key}`;
    valSpan.textContent = '-';
    
    header.appendChild(labelGroup);
    header.appendChild(valSpan);
    div.appendChild(header);
    
    let inputContainer = document.createElement('div');
    const currentConfig = getConfig();
    let val = getParamValue(currentConfig, key, meta);
    
    if (val === undefined) {
         if (meta.bounds) val = meta.bounds[0];
         else val = 0;
    }

    if (Array.isArray(val)) {
        inputContainer.className = 'd-flex gap-1 align-items-center';
        
        const minInput = document.createElement('input');
        minInput.type = 'number';
        minInput.className = 'form-control form-control-sm gt-input';
        minInput.dataset.key = key;
        minInput.dataset.index = 0;
        minInput.step = (meta.type === 'int') ? 1 : 0.1;
        minInput.placeholder = 'Min';
        minInput.onchange = (e) => handleGTChange(key, parseFloat(e.target.value), 0);
        
        const sep = document.createElement('span');
        sep.className = 'text-muted small';
        sep.textContent = '-';
        
        const maxInput = document.createElement('input');
        maxInput.type = 'number';
        maxInput.className = 'form-control form-control-sm gt-input';
        maxInput.dataset.key = key;
        maxInput.dataset.index = 1;
        maxInput.step = (meta.type === 'int') ? 1 : 0.1;
        maxInput.placeholder = 'Max';
        maxInput.onchange = (e) => handleGTChange(key, parseFloat(e.target.value), 1);
        
        inputContainer.appendChild(minInput);
        inputContainer.appendChild(sep);
        inputContainer.appendChild(maxInput);
        
    } else if (typeof val === 'boolean' || meta.type === 'bool') {
        inputContainer.className = 'form-check form-switch';
        const chk = document.createElement('input');
        chk.className = 'form-check-input gt-input';
        chk.type = 'checkbox';
        chk.dataset.key = key;
        chk.id = `gt_chk_${key}`;
        chk.onchange = (e) => handleGTChange(key, e.target.checked);
        inputContainer.appendChild(chk);
        
    } else {
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'form-range gt-input';
        slider.dataset.key = key;
        slider.id = `gt_slider_${key}`;
        
        let min = meta.bounds ? meta.bounds[0] : 0;
        let max = meta.bounds ? meta.bounds[1] : 100;
        let step = (max - min) / 100;
        if(meta.type === 'int') { step = 1; min = Math.floor(min); max = Math.ceil(max); }
        
        slider.min = min;
        slider.max = max;
        slider.step = step;
        slider.oninput = (e) => {
            document.getElementById(`gt_val_${key}`).textContent = parseFloat(e.target.value).toFixed(2);
        };
        slider.onchange = (e) => handleGTChange(key, parseFloat(e.target.value));
        inputContainer.appendChild(slider);
    }
    
    div.appendChild(inputContainer);
    container.appendChild(div);
}

export function updateGTTunerValues() {
    const synthConfig = getConfig();
    const gtConfig = applyOverrides(synthConfig, gtOverrides);
    
    Object.keys(optimizationRules).forEach(key => {
        const meta = optimizationRules[key];
        const val = getParamValue(gtConfig, key, meta);
        
        const itemDiv = document.getElementById(`gt_item_${key}`);
        
        if (val === undefined) {
             if(itemDiv) {
                 itemDiv.style.order = '9999';
                 itemDiv.style.display = 'block';
                 itemDiv.classList.add('opacity-50');
                 itemDiv.querySelectorAll('input').forEach(i => i.disabled = true);
                 const badge = document.getElementById(`gt_val_${key}`);
                 if(badge) badge.textContent = "Inactive";
             }
             return;
        }
        
        if(itemDiv) {
            itemDiv.style.display = 'block';
            itemDiv.style.order = '0';
            itemDiv.classList.remove('opacity-50');
            itemDiv.querySelectorAll('input').forEach(i => i.disabled = false);
        }

        const badge = document.getElementById(`gt_val_${key}`);
        if(badge) badge.textContent = formatVal(val);
        
        if (Array.isArray(val)) {
            const inputs = document.querySelectorAll(`input.gt-input[data-key="${key}"]`);
            inputs.forEach(inp => {
                const idx = parseInt(inp.dataset.index);
                if (inp.value != val[idx]) inp.value = val[idx];
            });
        } else if (typeof val === 'boolean') {
            const chk = document.getElementById(`gt_chk_${key}`);
            if(chk && chk.checked !== val) chk.checked = val;
        } else {
            const slider = document.getElementById(`gt_slider_${key}`);
            if(slider && document.activeElement !== slider) {
                if (Math.abs(parseFloat(slider.value) - val) > 0.001) slider.value = val;
            }
        }
        
        const linkBtn = document.getElementById(`btn_link_${key}`);
        if(linkBtn) {
            linkBtn.innerHTML = paramLinks[key] ? '<i class="bi bi-link text-success"></i>' : '<i class="bi bi-link-45deg text-warning"></i>';
        }
    });
}

function toggleParamLink(key) {
    if (paramLinks[key]) {
        paramLinks[key] = false;
        const currentConfig = getConfig();
        const meta = optimizationRules[key];
        const val = getParamValue(currentConfig, key, meta);
        if(Array.isArray(val)) gtOverrides[key] = [...val];
        else gtOverrides[key] = val;
        
        // Auto-check optimization when first unlinking
        const chk = document.getElementById(`chk_opt_${key}`);
        if(chk) chk.checked = true;

    } else {
        paramLinks[key] = true;
        delete gtOverrides[key];
    }
    updateGTTunerValues();
    updateOptimizationListValues();
    onRegenerate();
}

function handleGTChange(key, value, index = null) {
    const wasLinked = paramLinks[key];
    
    if (paramLinks[key]) {
        paramLinks[key] = false;
    }
    
    if (index !== null) {
        if (!gtOverrides[key]) {
             const currentConfig = getConfig();
             const meta = optimizationRules[key];
             const val = getParamValue(currentConfig, key, meta);
             gtOverrides[key] = [...val];
        }
        gtOverrides[key][index] = value;
    } else {
        gtOverrides[key] = value;
    }
    
    const badge = document.getElementById(`gt_val_${key}`);
    if(badge) {
        if (index !== null) badge.textContent = formatVal(gtOverrides[key]);
        else badge.textContent = formatVal(value);
    }
    
    const linkBtn = document.getElementById(`btn_link_${key}`);
    if(linkBtn) linkBtn.innerHTML = '<i class="bi bi-link-45deg text-warning"></i>';

    // If it was linked before this change, auto-check optimization
    if (wasLinked) {
         const chk = document.getElementById(`chk_opt_${key}`);
         if(chk) chk.checked = true;
    }
    
    updateOptimizationListValues();
    onRegenerate();
}

export function applyOverrides(config, overrides) {
    const clone = JSON.parse(JSON.stringify(config));
    
    Object.keys(overrides).forEach(key => {
        if (paramLinks[key] === false) { 
            let val = overrides[key];
            let path = key;
            let index = null;
            
            if (optimizationRules[key] && optimizationRules[key].target_attr) {
                 path = optimizationRules[key].target_attr[0];
                 index = optimizationRules[key].target_attr[1];
            }
            
            const parts = path.split('.');
            let current = clone;
            
            for(let i=0; i<parts.length-1; i++) {
                if(current[parts[i]] === undefined) current[parts[i]] = {};
                current = current[parts[i]];
            }
            
            const lastPart = parts[parts.length-1];
            
            if (index !== null) {
                if (Array.isArray(current[lastPart])) {
                    current[lastPart][index] = val;
                }
            } else {
                current[lastPart] = val;
                if (path === 'optics.focus_z' && clone.sensor) clone.sensor.focus_z = val;
                if (path === 'optics.aperture' && clone.sensor) clone.sensor.aperture = val;
            }
        }
    });
    return clone;
}

// ------------------------------------------------------------------
// Optimization Control
// ------------------------------------------------------------------

export function resetOptUI() {
    document.getElementById('btnStartOpt').disabled = false;
    document.getElementById('btnStartOpt').classList.remove('d-none');
    document.getElementById('btnStopOpt').classList.add('d-none');
    document.getElementById('optStatus').textContent = 'Ready';
    document.getElementById('optProgressBar').style.width = '0%';
}

function getRandomColor() {
    const r = Math.floor(Math.random() * 200);
    const g = Math.floor(Math.random() * 200);
    const b = Math.floor(Math.random() * 200);
    return `rgb(${r}, ${g}, ${b})`;
}

export async function startOptimization(targetImageFilename) {
    // If no file uploaded, try to use synthetic GT (Synthetic-to-Synthetic)
    if(!targetImageFilename) {
        // We assume the user wants to optimize against the current GT Tuner settings
        const confirmMsg = "No target file uploaded. Do you want to optimize against the current Synthetic GT configuration?";
        if(!confirm(confirmMsg)) return;

        const btn = document.getElementById('btnStartOpt');
        btn.disabled = true;
        document.getElementById('optStatus').textContent = 'Generating synthetic target...';
        
        try {
            const currentConfig = getConfig();
            const gtConfig = applyOverrides(currentConfig, gtOverrides);
            // Use a random seed for the target to ensure we are matching the "distribution" or specific instance
            // Ideally we should allow user to pick seed, but random is fine for "Validate Mode" usually
            const seed = Math.floor(Math.random() * 2000000000);
            
            const res = await api.saveSyntheticTarget(gtConfig, seed);
            if(res.ok && res.filename) {
                targetImageFilename = res.filename;
                showToast("Generated synthetic target: " + targetImageFilename);
            } else {
                throw new Error(res.error || "Unknown error generating target");
            }
        } catch(e) {
            alert("Failed to generate synthetic target: " + e.message);
            resetOptUI();
            return;
        }
    }
    
    const selected = [];
    document.querySelectorAll('.opt-param-chk:checked').forEach(c => selected.push(c.value));
    
    if(selected.length === 0) {
        alert("Select at least one parameter to optimize.");
        return;
    }
    
    document.getElementById('btnStartOpt').disabled = true;
    document.getElementById('btnStartOpt').classList.add('d-none');
    document.getElementById('btnStopOpt').classList.remove('d-none');
    document.getElementById('optStatus').textContent = 'Starting...';
    document.getElementById('optProgressBar').style.width = '0%';
    resetLossChart(); 
    resetParamChart();

    // Setup Param Chart Datasets
    const currentConfig = getConfig();
    const gtConfig = applyOverrides(currentConfig, gtOverrides);
    
    selected.forEach(key => {
        const color = getRandomColor();
        const displayName = getDisplayName(key);
        
        // 1. Optimization Curve (Solid)
        paramData.datasets.push({
            label: displayName,
            data: [],
            borderColor: color,
            tension: 0.1,
            fill: false,
            _key: key
        });
        
        // 2. GT Line (Dashed)
        const meta = optimizationRules[key];
        const gtVal = getParamValue(gtConfig, key, meta);
        
        if (gtVal !== undefined && gtVal !== null) {
             paramData.datasets.push({
                label: `${displayName} (GT)`,
                data: [], 
                borderColor: color,
                borderDash: [5, 5],
                pointRadius: 0,
                tension: 0,
                fill: false,
                _isGT: true,
                _gtVal: gtVal
            });
        }
    });

    isOptimizing = true; 
    
    const config = getConfig();
    const maxSteps = parseInt(document.getElementById('optMaxSteps').value) || 200;
    const lr = parseFloat(document.getElementById('optLR').value) || 0.05;
    
    const fd = new FormData();
    fd.append('target_image_name', targetImageFilename);
    fd.append('initial_config', JSON.stringify(config));
    fd.append('selected_params', JSON.stringify(selected));
    fd.append('max_steps', maxSteps);
    fd.append('learning_rate', lr);
    
    try {
        const data = await api.startOptimization(fd);
        if(data.ok) {
            optimizationJobId = data.job_id;
            document.getElementById('optStatus').textContent = 'Running...';
            optimizationTimer = setInterval(checkOptimizationStatus, 1000);
        } else {
            alert("Failed to start: " + data.error);
            resetOptUI();
        }
    } catch(e) {
        alert("Error: " + e.message);
        resetOptUI();
    }
}

export function stopOptimization() {
    if(optimizationJobId) {
        api.stopOptimization(optimizationJobId);
        document.getElementById('optStatus').textContent = 'Stopping...';
    }
}

async function checkOptimizationStatus() {
    if(!optimizationJobId) return;
    
    try {
        const data = await api.fetchOptimizationStatus(optimizationJobId);
        
        if(data.ok && data.status) {
            const s = data.status;
            const state = s.status; 
            const step = s.step || 0;
            const max_steps = s.max_steps || 200;
            const current_loss = s.loss;
            const current_config = s.current_params; 
            
            const pct = max_steps > 0 ? (step / max_steps) * 100 : 0;
            document.getElementById('optProgressBar').style.width = `${pct}%`;
            document.getElementById('optStatus').textContent = `${state} (${step}/${max_steps})`;
            
            if(current_loss !== undefined && current_loss !== 0 && current_loss !== null) {
                document.getElementById('optLoss').textContent = `Loss: ${current_loss.toFixed(4)}`;
                
                const dashboard = document.getElementById('optFloatingDashboard');
                if (dashboard.style.display === 'none') {
                    dashboard.style.display = 'block';
                    if (lossChart) lossChart.resize();
                    if (paramChart) paramChart.resize();
                }
                
                if(!lossChart) initLossChart();
                if(!paramChart) initParamChart();
                
                const lastStep = lossData.labels[lossData.labels.length - 1];
                if (lastStep !== step) {
                    lossData.labels.push(step);
                    lossData.datasets[0].data.push(current_loss);
                    lossChart.update();

                    // Update Param Chart
                    if (paramData.labels.length < lossData.labels.length) {
                         paramData.labels.push(step);
                    }
                    
                    paramData.datasets.forEach(ds => {
                        if (ds._isGT) {
                            ds.data.push(ds._gtVal);
                        } else {
                            // The key might be flattened in current_config or nested?
                            // api.py usually returns flat dict for current_params
                            const val = current_config[ds._key];
                            if (val !== undefined) {
                                 ds.data.push(val);
                            } else {
                                // Fallback or hold last value
                                const last = ds.data.length > 0 ? ds.data[ds.data.length-1] : 0;
                                ds.data.push(last);
                            }
                        }
                    });
                    paramChart.update();
                }
            }
            
            // Live Feedback
            if (current_config && Object.keys(current_config).length > 0) {
                  const fullConfig = getConfig(); 
                  const mergedConfig = JSON.parse(JSON.stringify(fullConfig));
                  
                  // Force update mergedConfig with current_config params
                  // (Logic similar to applyOverrides but ignores paramLinks)
                  Object.keys(current_config).forEach(key => {
                      let val = current_config[key];
                      let path = key;
                      let index = null;
                      if (optimizationRules[key] && optimizationRules[key].target_attr) {
                           path = optimizationRules[key].target_attr[0];
                           index = optimizationRules[key].target_attr[1];
                      }
                      const parts = path.split('.');
                      let curr = mergedConfig;
                      for(let i=0; i<parts.length-1; i++) {
                          if(curr[parts[i]] === undefined) curr[parts[i]] = {};
                          curr = curr[parts[i]];
                      }
                      const lastPart = parts[parts.length-1];
                      if (index !== null) {
                          if (Array.isArray(curr[lastPart])) curr[lastPart][index] = val;
                      } else {
                          curr[lastPart] = val;
                          if (path === 'optics.focus_z' && mergedConfig.sensor) mergedConfig.sensor.focus_z = val;
                          if (path === 'optics.aperture' && mergedConfig.sensor) mergedConfig.sensor.aperture = val;
                      }
                  });
                  
                  // Update UI with merged config
                  onUpdateUI(mergedConfig);
                  
                  if (step % 5 === 0 || step === 1) {
                      updateOptimizationListValues(mergedConfig);
                      onRegenerate(mergedConfig); // Pass config to avoid re-reading from UI
                  }
             }
            
            if(['finished', 'stopped', 'error', 'failed'].includes(state)) {
                clearInterval(optimizationTimer);
                optimizationTimer = null;
                optimizationJobId = null;
                isOptimizing = false; 
                
                document.getElementById('btnStopOpt').classList.add('d-none');
                document.getElementById('btnStartOpt').disabled = false;
                document.getElementById('btnStartOpt').classList.remove('d-none');
                
                if(state === 'finished' || state === 'stopped') {
                     document.getElementById('btnApplyOpt').classList.remove('d-none');
                     
                     if(current_config && Object.keys(current_config).length > 0) {
                         const fullConfig = getConfig();
                         const finalConfig = applyOverrides(fullConfig, current_config);
                         applyConfigToUI(finalConfig);
                         onRegenerate();
                         showToast(`Optimization ${state}! Final config applied.`);
                     }
                } else if (state === 'error') {
                     alert("Optimization failed: " + (s.error || 'Unknown error'));
                }
            }
        }
    } catch(e) {
        console.error("Poll error", e);
    }
}

export async function evaluateLoss(targetImageFilename) {
     if(!targetImageFilename) {
         // We assume the user wants to optimize against the current GT Tuner settings
         const confirmMsg = "No target file uploaded. Do you want to calculate loss against the current Synthetic GT configuration?";
         if(!confirm(confirmMsg)) return;
 
         const btn = document.querySelector('#lossResults').previousElementSibling.querySelector('button');
         btn.disabled = true;
         btn.textContent = 'Gen GT...';
         
         try {
             const currentConfig = getConfig();
             const gtConfig = applyOverrides(currentConfig, gtOverrides);
             const seed = Math.floor(Math.random() * 2000000000);
             
             const res = await api.saveSyntheticTarget(gtConfig, seed);
             if(res.ok && res.filename) {
                 targetImageFilename = res.filename;
                 showToast("Generated synthetic target: " + targetImageFilename);
             } else {
                 throw new Error(res.error || "Unknown error generating target");
             }
         } catch(e) {
             alert("Failed to generate synthetic target: " + e.message);
             btn.disabled = false;
             btn.textContent = 'Calculate';
             return;
         }
    }
    
    const btn = document.querySelector('#lossResults').previousElementSibling.querySelector('button');
    btn.disabled = true;
    btn.textContent = '...';
    
    const config = getConfig();
    const samples = parseInt(document.getElementById('lossSamples').value) || 1;
    
    const fd = new FormData();
    fd.append('target_image_name', targetImageFilename);
    fd.append('current_config', JSON.stringify(config));
    fd.append('n_samples', samples);
    
    try {
        const data = await api.computeLoss(fd);
        const resDiv = document.getElementById('lossResults');
        if(data.ok || data.mean_loss !== undefined) { 
            resDiv.innerHTML = `
                <div>Mean Loss: <b>${data.mean_loss?.toFixed(4) || 'N/A'}</b></div>
                <div>Std Dev: ${data.std_loss?.toFixed(4) || 'N/A'}</div>
            `;
        } else {
             resDiv.innerHTML = `<div class="text-danger">${data.error || 'Error'}</div>`;
        }
    } catch(e) {
         document.getElementById('lossResults').innerHTML = `<div class="text-danger">${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Calculate';
    }
}

// Chart
function initLossChart() {
    const ctx = document.getElementById('optLossChart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: lossData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { title: { display: true, text: 'Step' } },
                y: { title: { display: true, text: 'Loss' }, beginAtZero: false }
            }
        }
    });
}

function initParamChart() {
    const canvas = document.getElementById('optParamsChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    paramChart = new Chart(ctx, {
        type: 'line',
        data: paramData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { title: { display: true, text: 'Step' } },
                y: { title: { display: true, text: 'Value' } }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        boxWidth: 10,
                        font: { size: 10 }
                    }
                }
            }
        }
    });
}

export function resetLossChart() {
    lossData.labels = [];
    lossData.datasets[0].data = [];
    if(lossChart) lossChart.update();
}

export function resetParamChart() {
    paramData.labels = [];
    paramData.datasets = [];
    if(paramChart) paramChart.update();
}
