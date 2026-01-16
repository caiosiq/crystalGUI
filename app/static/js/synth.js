// ===================== Synthesis (Data Generator) Tab =====================
// Wrap in IIFE to avoid leaking identifiers into global scope
(function() {
  // State for synthesis rows (closure-scoped)
  let synthRows = [];
  let synthRowCounter = 0;
  // Track the currently selected/loaded preset name. Defaults to 'standard'.
  let currentPresetName = 'standard';

  // Helper to apply config to form (handles both flat and nested structures)
  function applyConfigToForm(cfg) {
    const set = (id, val) => { const el = document.getElementById(id); if (el != null && val != null) el.value = val; };
    const setChecked = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };
    
    // Helper to safely access nested properties
    const get = (obj, path, def) => {
      return path.split('.').reduce((o, k) => (o && o[k] !== undefined) ? o[k] : undefined, obj) ?? (obj[path] ?? def);
    };

    // Canvas
    set('synWidth', get(cfg, 'canvas.width', cfg.width));
    set('synHeight', get(cfg, 'canvas.height', cfg.height));
    setChecked('synUseGpu', get(cfg, 'canvas.use_gpu', cfg.use_gpu));

    // Sensor
    const bg = get(cfg, 'sensor.bg_gray_range', cfg.bg_gray_range);
    if (Array.isArray(bg) && bg.length === 2) { set('synBgMin', bg[0]); set('synBgMax', bg[1]); }
    
    const tiltPtp = get(cfg, 'sensor.tilt_ptp', cfg.tilt_ptp);
    if (Array.isArray(tiltPtp) && tiltPtp.length === 2) { set('synTiltMin', tiltPtp[0]); set('synTiltMax', tiltPtp[1]); }
    
    setChecked('synTiltEnable', get(cfg, 'sensor.tilt_enable', cfg.tilt_enable));
    set('synVignette', get(cfg, 'sensor.vignette_strength', cfg.vignette_strength));
    set('synBgNoise', get(cfg, 'sensor.bg_noise_std', cfg.bg_noise_std));
    set('synBlur', get(cfg, 'sensor.blur_sigma', cfg.blur_sigma));
    
    // Sensor - Scalebar
    setChecked('synScaleEnable', get(cfg, 'sensor.scalebar.enable', cfg.scalebar_enable));
    set('synScaleProb', get(cfg, 'sensor.scalebar.prob', cfg.scalebar_prob));
    
    const scLen = get(cfg, 'sensor.scalebar.len_px', cfg.scalebar_len_px);
    if (Array.isArray(scLen) && scLen.length === 2) { set('synScaleLenLo', scLen[0]); set('synScaleLenHi', scLen[1]); }
    
    const scThick = get(cfg, 'sensor.scalebar.thick_px', cfg.scalebar_thick_px);
    if (Array.isArray(scThick) && scThick.length === 2) { set('synScaleThickLo', scThick[0]); set('synScaleThickHi', scThick[1]); }
    
    set('synScaleMargin', get(cfg, 'sensor.scalebar.margin_px', cfg.scalebar_margin_px));

    // Optics
    set('synOpticsMode', get(cfg, 'optics.mode', 'dic'));
    set('synPolarizerAngle', get(cfg, 'optics.polarizer_angle_deg', 0));
    set('synTaper', get(cfg, 'optics.taper_strength', cfg.taper_strength));
    
    const shGain = get(cfg, 'optics.shadow_gain', cfg.shadow_gain);
    if (Array.isArray(shGain) && shGain.length === 2) { set('synShGainMin', shGain[0]); set('synShGainMax', shGain[1]); }
    
    const shWidth = get(cfg, 'optics.shadow_width_mult', cfg.shadow_width_mult);
    if (Array.isArray(shWidth) && shWidth.length === 2) { set('synShWidthMin', shWidth[0]); set('synShWidthMax', shWidth[1]); }
    
    const shBias = get(cfg, 'optics.shadow_bias', cfg.shadow_bias);
    if (Array.isArray(shBias) && shBias.length === 2) { set('synShBiasMin', shBias[0]); set('synShBiasMax', shBias[1]); }
    
    const shOffset = get(cfg, 'optics.shadow_offset_px', cfg.shadow_offset_px);
    if (Array.isArray(shOffset) && shOffset.length === 2) { set('synShOffsetMin', shOffset[0]); set('synShOffsetMax', shOffset[1]); }

    // Physics - Rods
    setChecked('synRodsEnable', get(cfg, 'physics.rods.enable', cfg.rods_enable) ?? true);
    
    const nRods = get(cfg, 'physics.rods.n_rods_rng_lo_hi', cfg.n_rods_rng_lo_hi);
    if (Array.isArray(nRods)) {
      if (nRods.length === 3) {
        set('synNrodsLo', nRods[0]); set('synNrodsHiT0', nRods[1]); set('synNrodsHi', nRods[2]);
      } else if (nRods.length === 2) {
        set('synNrodsLo', nRods[0]); set('synNrodsHi', nRods[1]);
        const t0El = document.getElementById('synNrodsHiT0'); if (t0El) t0El.value = nRods[1];
      }
    }
    
    const lenPx = get(cfg, 'physics.rods.rod_len_px_lo_hi', cfg.rod_len_px_lo_hi);
    if (Array.isArray(lenPx)) {
      if (lenPx.length === 3) {
        set('synLenLo', lenPx[0]); set('synLenHiT0', lenPx[1]); set('synLenHi', lenPx[2]);
      } else if (lenPx.length === 2) {
        set('synLenLo', lenPx[0]); set('synLenHi', lenPx[1]);
        const t0El = document.getElementById('synLenHiT0'); if (t0El) t0El.value = lenPx[1];
      }
    }
    
    const ar = get(cfg, 'physics.rods.rod_aspect_lo_hi', cfg.rod_aspect_lo_hi);
    if (Array.isArray(ar)) {
      if (ar.length === 3) {
        set('synArLo', ar[0]); set('synArHiT0', ar[1]); set('synArHi', ar[2]);
      } else if (ar.length === 2) {
        set('synArLo', ar[0]); set('synArHi', ar[1]);
        const t0El = document.getElementById('synArHiT0'); if (t0El) t0El.value = ar[1];
      }
    }
    
    const delta = get(cfg, 'physics.rods.rod_delta_rng', cfg.rod_delta_rng);
    if (Array.isArray(delta)) {
      if (delta.length === 3) {
        set('synRodDeltaMin', delta[0]); set('synRodDeltaMaxT0', delta[1]); set('synRodDeltaMax', delta[2]);
      } else if (delta.length === 2) {
        set('synRodDeltaMin', delta[0]); set('synRodDeltaMax', delta[1]);
        const t0El = document.getElementById('synRodDeltaMaxT0'); if (t0El) t0El.value = delta[1];
      }
    }

    // Physics - Ghosts
    setChecked('synGhostEnable', get(cfg, 'physics.ghosts.enable', cfg.ghost_enable));
    set('synGhostFraction', get(cfg, 'physics.ghosts.fraction', cfg.ghost_fraction));
    set('synGhostGainMult', get(cfg, 'physics.ghosts.gain_mult', cfg.ghost_gain_mult));
    set('synGhostBlur', get(cfg, 'physics.ghosts.blur_sigma', cfg.ghost_blur_sigma));
    // Note: ghost_noise_std removed from new config or unused? It was in old config. Ignoring if not in new.
    set('synGhostCurv', get(cfg, 'physics.ghosts.curvature', cfg.ghost_curvature));
    
    const ghDelta = get(cfg, 'physics.ghosts.delta_rng', cfg.ghost_delta_rng);
    if (Array.isArray(ghDelta) && ghDelta.length === 2) { set('synGhostDeltaMin', ghDelta[0]); set('synGhostDeltaMax', ghDelta[1]); }

    // Physics - Debris
    set('synDebrisRate', get(cfg, 'physics.debris.rate', cfg.debris_rate) || 0);
    
    const debDelta = get(cfg, 'physics.debris.int_delta', cfg.debris_int_delta);
    if (Array.isArray(debDelta) && debDelta.length === 2) { set('synDebrisDeltaMin', debDelta[0]); set('synDebrisDeltaMax', debDelta[1]); }
    
    set('synDebrisDashProb', get(cfg, 'physics.debris.dash_prob', cfg.debris_dash_prob) || 0.15);
    
    const debSize = get(cfg, 'physics.debris.size_px', cfg.debris_size_px);
    if (Array.isArray(debSize) && debSize.length === 2) { set('synDebrisSizeMin', debSize[0]); set('synDebrisSizeMax', debSize[1]); }

    // Physics - Global (Batch)
    const stLambda = get(cfg, 'physics.stage_lambda_range', cfg.stage_lambda_range);
    if (Array.isArray(stLambda) && stLambda.length === 2) { set('synLambdaMin', stLambda[0]); set('synLambdaMax', stLambda[1]); }
    
    // Sync labels
    if (typeof syncAllSliderLabels === 'function') syncAllSliderLabels();
  }

// Initialize defaults from backend on DOM ready
  document.addEventListener('DOMContentLoaded', () => {
    // Check GPU availability to show/hide toggle
    fetch('/system_info')
      .then(r => r.json())
      .then(data => {
        if (data.ok && data.gpu_available) {
          const el = document.getElementById('synUseGpuContainer');
          if (el) el.style.display = 'block';
        }
      })
      .catch(() => {});

    fetch('/synth_default_config')
      .then(r => r.json())
      .then(data => {
        if (!data || !data.ok || !data.config) return;
        applyConfigToForm(data.config);
      })
      .catch(() => {});
  });
  // Sync all numeric slider labels with their current values
  function syncAllSliderLabels() {
    const pairs = [
      ['synVignette','synVignetteLbl'],
      ['synBgNoise','synBgNoiseLbl'],
      ['synTaper','synTaperLbl'],
      ['synGhostFraction','synGhostFractionLbl'],
      ['synGhostGainMult','synGhostGainLbl'],
      ['synGhostCurv','synGhostCurvLbl'],
      ['synGhostNoise','synGhostNoiseLbl'],
      ['synGhostBlur','synGhostBlurLbl'],
      ['synScaleProb','synScaleProbLbl'],
      ['synDebrisRate','synDebrisRateLbl'],
      ['synDebrisDashProb','synDebrisDashProbLbl'],
      ['synObbPct','synObbPctLbl'],
      ['synJpegQ','synJpegQLbl'],
    ];
    pairs.forEach(([id,lbl]) => {
      const el = document.getElementById(id);
      const sp = document.getElementById(lbl);
      if (el && sp) {
        const v = el.value;
        sp.textContent = (id==='synObbPct') ? `${v}%` : v;
      }
    });
  }
  // After defaults load, assume we are on the 'standard' preset
  // currentPresetName remains 'standard' unless a named preset is loaded via the modal.
  // Ensure labels match slider positions on initial load.
  document.addEventListener('DOMContentLoaded', () => { setTimeout(syncAllSliderLabels, 100); });

  function getSynthConfigFromForm() {
    const num = (id, def=0) => { const el = document.getElementById(id); if (!el) return def; const v = parseFloat(el.value); return isNaN(v) ? def : v; };
    const intv = (id, def=0) => { const el = document.getElementById(id); if (!el) return def; const v = parseInt(el.value, 10); return isNaN(v) ? def : v; };
    const checked = (id, def=false) => { const el = document.getElementById(id); return el ? !!el.checked : def; };
    const n_lo = intv('synNrodsLo', 150);
    const n_hi_t0 = intv('synNrodsHiT0', 600);
    const n_hi_t1 = intv('synNrodsHi', 600);
    const len_lo = num('synLenLo', 30);
    const len_hi_t0 = num('synLenHiT0', 380);
    const len_hi_t1 = num('synLenHi', 380);
    const ar_lo = num('synArLo', 0.02);
    const ar_hi_t0 = num('synArHiT0', 0.3);
    const ar_hi_t1 = num('synArHi', 0.3);
    
    // Construct nested config structure
    const cfg = {
      canvas: {
        width: Math.max(16, Math.floor(num('synWidth', 1024))),
        height: Math.max(16, Math.floor(num('synHeight', 768))),
        use_gpu: checked('synUseGpu', false),
      },
      sensor: {
        bg_gray_range: [Math.max(0, Math.floor(num('synBgMin', 90))), Math.min(255, Math.floor(num('synBgMax', 97)))],
        tilt_ptp: [num('synTiltMin', 12), num('synTiltMax', 15)],
        tilt_enable: checked('synTiltEnable', true),
        vignette_strength: Math.max(0, Math.min(1, num('synVignette', 0))),
        bg_noise_std: Math.max(0, num('synBgNoise', 0)),
        blur_sigma: num('synBlur', 0),
        // Relief field (if inputs exist in UI, currently mostly defaults in JS or hardcoded?
        // Ah, relief field inputs exist in HTML but were missing in JS getSynthConfigFromForm previously?
        // Let's check HTML. Yes, synReliefEnable etc exist.
        relief_field_enable: checked('synReliefEnable', true),
        relief_field_sigma_px: [num('synReliefSigmaMin', 20), num('synReliefSigmaMax', 20)],
        relief_field_gain: [num('synReliefGainMin', -0.5), num('synReliefGainMax', 0)],
        relief_field_dir_deg: [num('synReliefDirMin', 0), num('synReliefDirMax', 0)],
        relief_field_extra_blur: num('synReliefExtraBlur', 0),
        
        scalebar: {
          enable: checked('synScaleEnable', true),
          prob: Math.max(0, Math.min(1, num('synScaleProb', 0.5))),
          len_px: [intv('synScaleLenLo', 80), intv('synScaleLenHi', 240)],
          thick_px: [intv('synScaleThickLo', 2), intv('synScaleThickHi', 12)],
          margin_px: intv('synScaleMargin', 24),
        }
      },
      optics: {
        taper_strength: Math.max(0, Math.min(1, num('synTaper', 0.45))),
        shadow_gain: [num('synShGainMin', 6), num('synShGainMax', 12)],
        shadow_width_mult: [num('synShWidthMin', 0.02), num('synShWidthMax', 0.05)],
        shadow_bias: [num('synShBiasMin', 0.05), num('synShBiasMax', 0.12)],
        shadow_offset_px: [num('synShOffsetMin', 0.05), num('synShOffsetMax', 0.25)],
        rod_halo_sigma: num('synRodHaloSigma', 3.2),
        rod_halo_gain: num('synRodHaloGain', 0.4),
      },
      physics: {
        rods: {
          enable: checked('synRodsEnable', true),
          prob_plate: num('synProbPlate', 0.0),
          prob_cube: num('synProbCube', 0.0),
          prob_sphere: num('synProbSphere', 0.0),
          enable_3d: checked('synEnable3d', false),
          n_rods_rng_lo_hi: [n_lo, n_hi_t0, n_hi_t1],
          rod_len_px_lo_hi: [len_lo, len_hi_t0, len_hi_t1],
          rod_aspect_lo_hi: [ar_lo, ar_hi_t0, ar_hi_t1],
          rod_delta_rng: [num('synRodDeltaMin', -12), num('synRodDeltaMaxT0', 0), num('synRodDeltaMax', 0)],
        },
        ghosts: {
          enable: checked('synGhostEnable', false),
          fraction: Math.max(0, Math.min(3, num('synGhostFraction', 0.2))),
          gain_mult: Math.max(0, Math.min(1, num('synGhostGainMult', 0.5))),
          blur_sigma: Math.max(0, num('synGhostBlur', 0)),
          curvature: Math.max(0, Math.min(1, num('synGhostCurv', 0))),
          delta_rng: [num('synGhostDeltaMin', -3), num('synGhostDeltaMax', 0)],
          width_jit_amp: num('synGhostWidthJit', 0.25),
          edge_jit_amp: num('synGhostEdgeJit', 0.25),
          offset_jit_amp: num('synGhostOffsetJit', 0.39),
          curve_kappa_range: [num('synGhostCurveKMin', -0.125), num('synGhostCurveKMax', 0.125)],
          ragged_p: num('synGhostRaggedP', 0.9),
          ragged_corr: num('synGhostRaggedCorr', 0.9),
          mult_mix: num('synGhostMultMix', 0.9),
        },
        debris: {
          rate: Math.max(0, Math.min(1, num('synDebrisRate', 0))),
          int_delta: [num('synDebrisDeltaMin', -6), num('synDebrisDeltaMax', 6)],
          dash_prob: Math.max(0, Math.min(1, num('synDebrisDashProb', 0.15))),
          size_px: [intv('synDebrisSizeMin', 1), intv('synDebrisSizeMax', 3)],
        },
        fused: {
          enable: checked('synFusedEnable', true),
          p0: num('synFusedP0', 0.0001),
          p1: num('synFusedP1', 0.003),
        },
        stage_lambda_range: [num('synLambdaMin', 0.1), num('synLambdaMax', 10.0)],
      }
    };
    
    // Sort ranges where applicable
    cfg.sensor.bg_gray_range.sort((a,b)=>a-b);
    cfg.sensor.tilt_ptp.sort((a,b)=>a-b);
    cfg.optics.shadow_gain.sort((a,b)=>a-b);
    cfg.optics.shadow_width_mult.sort((a,b)=>a-b);
    cfg.optics.shadow_bias.sort((a,b)=>a-b);
    cfg.optics.shadow_offset_px.sort((a,b)=>a-b);
    
    const sort2Only = (arr) => { if (Array.isArray(arr) && arr.length === 2) arr.sort((a,b)=>a-b); };
    sort2Only(cfg.physics.rods.n_rods_rng_lo_hi);
    sort2Only(cfg.physics.rods.rod_len_px_lo_hi);
    sort2Only(cfg.physics.rods.rod_aspect_lo_hi);
    sort2Only(cfg.physics.rods.rod_delta_rng);
    
    cfg.physics.ghosts.delta_rng.sort((a,b)=>a-b);
    cfg.sensor.scalebar.len_px.sort((a,b)=>a-b);
    cfg.sensor.scalebar.thick_px.sort((a,b)=>a-b);
    cfg.physics.debris.int_delta.sort((a,b)=>a-b);
    cfg.physics.debris.size_px.sort((a,b)=>a-b);
    
    return cfg;
  }

  function addSynthRowFromFile() {
    const input = document.getElementById('synthAddRowInput');
    if (!input || !input.files || !input.files[0]) {
      if (typeof showAlert === 'function') showAlert('warning', 'Please choose an image first.');
      return;
    }
    const file = input.files[0];
    const url = URL.createObjectURL(file);
    addSynthRow(url, file.name);
    input.value = '';
  }

  function addSynthRow(referenceUrl, label) {
    const container = document.getElementById('synth-rows');
    if (!container) { if (typeof showAlert === 'function') showAlert('danger', 'Synthesis workspace not found'); return; }
    // Make reference URL robust: if a bare image name is passed, convert to backendserved URL
    let refUrl = referenceUrl;
    if (typeof refUrl === 'string') {
      const startsWithProto = refUrl.startsWith('http://') || refUrl.startsWith('https://');
      const startsWithSlash = refUrl.startsWith('/');
      if (!startsWithProto && !startsWithSlash) {
        refUrl = `/get_image?name=${encodeURIComponent(refUrl)}`;
      }
    }
    // Determine a friendly label
    let displayLabel = label;
    if (!displayLabel) {
      if (typeof referenceUrl === 'string') {
        const startsWithProto = referenceUrl.startsWith('http://') || referenceUrl.startsWith('https://');
        const startsWithSlash = referenceUrl.startsWith('/');
        if (!startsWithProto && !startsWithSlash) {
          // Bare name was passed in; use it directly
          displayLabel = referenceUrl;
        } else if (referenceUrl.startsWith('/get_image')) {
          try {
            const u = new URL(referenceUrl, window.location.origin);
            const nm = u.searchParams.get('name');
            displayLabel = nm ? decodeURIComponent(nm) : referenceUrl;
          } catch {
            displayLabel = referenceUrl;
          }
        } else {
          displayLabel = referenceUrl;
        }
      } else {
        displayLabel = 'reference';
      }
    }
    const id = `row${++synthRowCounter}`;
    const t0 = 0.5;
    synthRows.push({ id, t: t0, referenceUrl: refUrl });

  const row = document.createElement('div');
  // Add gutter to avoid any overlap between boxes and the right-side controls
  row.className = 'row g-3 align-items-start border rounded p-2';
  row.dataset.rowId = id;
  row.innerHTML = `
    <div class="col-md-5 d-flex flex-column align-items-center">
      <div class="mb-1 text-muted small text-center" style="width: 260px;">Reference</div>
      <div class="position-relative" style="width: 260px; height: 260px; overflow: hidden;">
        <img src="${refUrl}" alt="reference" class="rounded border synth-ref clickable-image" data-row-id="${id}"
             style="object-fit: contain; width: 100%; height: 100%; display: none; cursor: pointer;"
             loading="eager" decoding="async" fetchpriority="high" title="Click to zoom" />
      </div>
      <div class="small text-muted mt-1 text-truncate text-center" style="width: 260px;">${displayLabel}</div>
    </div>
    <div class="col-md-5 d-flex flex-column align-items-center">
      <div class="mb-1 text-muted small text-center" style="width: 260px;">Synthetic Preview</div>
      <div class="position-relative" style="width: 260px; height: 260px; overflow: hidden;">
        <img id="synth-prev-${id}" class="rounded border clickable-image" alt="synthetic preview"
             style="object-fit: contain; width: 100%; height: 100%; display: none; cursor: pointer;" title="Click to zoom" loading="eager" decoding="async" fetchpriority="high" />
        <canvas id="synth-obb-${id}" class="position-absolute top-0 start-0 image-overlay-canvas" width="260" height="260" style="pointer-events: none; display: none;"></canvas>
        <div id="synth-loading-${id}" class="position-absolute top-50 start-50 translate-middle text-center" style="display:none;">
          <div class="spinner-border text-info" role="status" style="width: 2.2rem; height: 2.2rem;"></div>
          <div class="small text-muted mt-2" id="synth-loading-text-${id}"></div>
        </div>
      </div>
      <div class="small text-muted mt-1 text-center" id="synth-name-${id}" style="width: 260px;">t=0.50.jpg</div>
    </div>
    <div class="col-md-2 d-flex flex-column align-items-stretch">
      <label class="form-label mb-1 text-nowrap">t <span class="text-muted small">[0,1]</span></label>
      <div class="input-group input-group-sm mb-2">
        <input id="synth-t-${id}" type="range" min="0" max="1" step="0.01" value="${t0}" class="form-range flex-grow-1" oninput="updateSynthT('${id}', this.value)" />
        <span class="input-group-text" id="synth-t-label-${id}">${t0.toFixed(2)}</span>
      </div>
      <div class="d-grid gap-1 mt-1">
        <button class="btn btn-outline-primary btn-sm" title="Toggle OBB" onclick="toggleObbRow('${id}')">OBB</button>
      </div>
      <div class="d-flex gap-1 mt-2">
        <button class="btn btn-success btn-sm" title="Regenerate" onclick="regenerateSynthRow('${id}')"><i class="bi bi-arrow-clockwise"></i></button>
        <button class="btn btn-outline-danger btn-sm" title="Remove" onclick="removeSynthRow('${id}')"><i class="bi bi-trash"></i></button>
      </div>
    </div>
  `;
  container.appendChild(row);
  // Show synthetic loading immediately so that by the time the reference loads,
  // the synthetic box already displays the loading design
  setLoading(id, true);
  const refImg = row.querySelector('img.synth-ref');
  if (refImg) {
    // When the reference image finishes loading, show it and kick off synthetic generation
    refImg.onload = () => { refImg.style.display = ''; regenerateSynthRow(id); };
    // In case the image is cached and loads instantly, force a visibility check
    if (refImg.complete) { refImg.style.display = ''; regenerateSynthRow(id); }
  } else {
    // Fallback: if no reference image element found, still trigger generation
    regenerateSynthRow(id);
  }
  }

  function removeSynthRow(id) {
    synthRows = synthRows.filter(r => r.id !== id);
    const el = document.querySelector(`[data-row-id="${id}"]`);
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  function setLoading(id, isLoading) {
    const ld = document.getElementById(`synth-loading-${id}`);
    const img = document.getElementById(`synth-prev-${id}`);
    const cnv = document.getElementById(`synth-obb-${id}`);
    const nameEl = document.getElementById(`synth-name-${id}`);
    const ldTxt = document.getElementById(`synth-loading-text-${id}`);
    if (ld) ld.style.display = isLoading ? 'block' : 'none';
    if (ldTxt && nameEl && isLoading) ldTxt.textContent = `Generating ${nameEl.textContent}`;
    if (img) img.style.display = isLoading ? 'none' : '';
    // Hide overlay while loading; when loading is done, do NOT force it visible.
    // Its visibility will be controlled by obbVisibility + drawObbsForRow.
    if (cnv) { if (isLoading) cnv.style.display = 'none'; }
  }

  function updateSynthT(id, val) {
    const v = parseFloat(val);
    const lbl = document.getElementById(`synth-t-label-${id}`);
    const nameEl = document.getElementById(`synth-name-${id}`);
    if (lbl) lbl.textContent = v.toFixed(2);
    if (nameEl) nameEl.textContent = `t=${v.toFixed(2)}.jpg`;
  }

  const obbVisibility = new Map(); // rowId -> boolean

  // Global OBB draw percentage (0-100). Default to 100.
  let obbPercent = 100;
  function setObbPercent(val) {
    const v = parseInt(val, 10);
    if (!isNaN(v)) obbPercent = Math.max(0, Math.min(100, v));
    // Redraw currently visible rows to reflect new percentage
    synthRows.forEach(r => {
      // We cannot easily re-use last obbs without refetch; trigger a redraw using last image size when available
      const img = document.getElementById(`synth-prev-${r.id}`);
      if (!img) return;
      const iw = img.naturalWidth || 260;
      const ih = img.naturalHeight || 260;
      // If obb visibility is on, we force re-generate to fetch obbs again for accurate sampling
      if (obbVisibility.get(r.id)) {
        regenerateSynthRow(r.id);
      }
    });
  }

  function drawObbsForRow(id, obbs, imgNaturalW, imgNaturalH) {
    const canvas = document.getElementById(`synth-obb-${id}`);
    const img = document.getElementById(`synth-prev-${id}`);
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!obbVisibility.get(id)) { canvas.style.display = 'none'; return; } // hidden by default
    canvas.style.display = '';
    // Compute displayed size of image within container (object-fit: contain)
    const cw = canvas.width, ch = canvas.height;
    const arImg = imgNaturalW / imgNaturalH;
    const arBox = cw / ch;
    let dispW, dispH, offX, offY;
    if (arImg > arBox) {
      dispW = cw; dispH = cw / arImg; offX = 0; offY = (ch - dispH) / 2;
    } else {
      dispH = ch; dispW = ch * arImg; offY = 0; offX = (cw - dispW) / 2;
    }
    const scaleX = dispW / imgNaturalW;
    const scaleY = dispH / imgNaturalH;
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.85)';
    // Decrease line width per user request
    ctx.lineWidth = 0.8;
    // Sample a subset of OBBs according to global percentage
    const all = Array.isArray(obbs) ? obbs : [];
    const nDraw = Math.max(0, Math.round(all.length * (obbPercent / 100.0)));
    const items = all.slice(0, nDraw);
    items.forEach(ob => {
      const cs = ob.corners || [];
      if (cs.length !== 4) return;
      ctx.beginPath();
      for (let i = 0; i < 4; i++) {
        const x = offX + cs[i][0] * scaleX;
        const y = offY + cs[i][1] * scaleY;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    });
    // Store metadata to enable high-res zoom composition
    try {
      canvas.dataset.obbs = JSON.stringify(obbs || []);
      canvas.dataset.imgNaturalW = String(imgNaturalW || 0);
      canvas.dataset.imgNaturalH = String(imgNaturalH || 0);
      canvas.dataset.obbPercent = String(obbPercent);
    } catch {}
  }

  async function regenerateSynthRow(id) {
    const row = synthRows.find(r => r.id === id);
    if (!row) return;
    const tEl = document.getElementById(`synth-t-${id}`);
    if (tEl) row.t = parseFloat(tEl.value);
    updateSynthT(id, row.t);
    const cfg = getSynthConfigFromForm();
    const imgEl = document.getElementById(`synth-prev-${id}`);
    setLoading(id, true);
    try {
      const res = await fetch('/synth_preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ t: row.t, config: cfg, quality: (document.getElementById('synJpegQ') ? parseInt(document.getElementById('synJpegQ').value, 10) : 85), return_obbs: true, mode: 'preview' }),
      });
      const data = await res.json();
      if (!data.ok) { if (typeof showAlert === 'function') showAlert('danger', data.error || 'Failed to generate preview'); return; }
      if (imgEl) {
        imgEl.src = data.image_b64;
        imgEl.onload = () => { imgEl.style.display = ''; };
        if (data.highres_token) {
          imgEl.dataset.highresToken = data.highres_token;
        } else {
          delete imgEl.dataset.highresToken;
        }
      }
      // draw obbs after image loads
      const obbs = data.obbs || [];
      const iw = data.width || 0, ih = data.height || 0;
      drawObbsForRow(id, obbs, iw, ih);
    } catch (e) {
      if (typeof showAlert === 'function') showAlert('danger', 'Preview error: ' + e.message);
    } finally { setLoading(id, false); }
  }

  async function regenerateAllSynth() {
    if (synthRows.length === 0) { if (typeof showAlert === 'function') showAlert('warning', 'Add at least one row'); return; }
    const rows = synthRows.map(r => {
      const tEl = document.getElementById(`synth-t-${r.id}`);
      return { id: r.id, t: tEl ? parseFloat(tEl.value) : r.t };
    });
    const cfg = getSynthConfigFromForm();
    const qEl = document.getElementById('synJpegQ');
    const quality = qEl ? parseInt(qEl.value, 10) : 85;
    // show loading on each row
    rows.forEach(r => setLoading(r.id, true));
    try {
      const res = await fetch('/synth_preview_bulk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rows, config: cfg, quality, return_obbs: true })
      });
      const data = await res.json();
      if (!data.ok) { if (typeof showAlert === 'function') showAlert('danger', data.error || 'Bulk preview failed'); return; }
      const images = data.images || {};
      const obbsById = data.obbs || {};
      for (const rid in images) {
        const el = document.getElementById(`synth-prev-${rid}`);
        if (el && images[rid]) el.src = images[rid];
        if (el) el.onload = () => { el.style.display = ''; };
        const obbs = obbsById[rid] || [];
        const iw = data.width || 0, ih = data.height || 0; // bulk doesn't include per-row size; we will rely on image natural size when loaded
        // We cannot get natural size from response; defer drawing slightly
        setTimeout(() => {
          const img = document.getElementById(`synth-prev-${rid}`);
          if (!img) return;
          // Try to get natural size via object, fallback to 260x260 canvas scale handling
          const iw2 = img.naturalWidth || 260;
          const ih2 = img.naturalHeight || 260;
          drawObbsForRow(rid, obbs, iw2, ih2);
        }, 50);
        setLoading(rid, false);
      }
      if (typeof showAlert === 'function') showAlert('success', 'Regenerated all previews');
    } catch (e) {
      if (typeof showAlert === 'function') showAlert('danger', 'Bulk preview error: ' + e.message);
      rows.forEach(r => setLoading(r.id, false));
    }
  }

  // Toggle OBB overlay for a single row
  function toggleObbRow(id) {
    const current = !!obbVisibility.get(id);
    obbVisibility.set(id, !current);
    // Re-generate to fetch OBBs and redraw according to visibility
    regenerateSynthRow(id);
  }

  // Toggle OBB overlay for all rows
  function toggleObbAll() {
    if (synthRows.length === 0) return;
    const firstId = synthRows[0].id;
    const target = !obbVisibility.get(firstId);
    synthRows.forEach(r => obbVisibility.set(r.id, target));
    regenerateAllSynth();
  }

  // Save the current globals into the currently selected preset.
  // If the current preset is 'standard', it uses the /synth_save_standard endpoint.
  // Otherwise, it updates the named preset via /synth_save_preset.
  async function saveSynthInCurrent() {
    const cfg = getSynthConfigFromForm();
    const name = (currentPresetName || 'standard').trim();
    try {
      if (name.toLowerCase() === 'standard') {
        const res = await fetch('/synth_save_standard', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: cfg })
        });
        const data = await res.json();
        if (!data.ok) { if (typeof showAlert === 'function') showAlert('danger', data.error || 'Failed to save standard'); return; }
        if (typeof showAlert === 'function') showAlert('success', 'Saved in current preset: standard');
      } else {
        const res = await fetch('/synth_save_preset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, config: cfg })
        });
        const data = await res.json();
        if (!data.ok) { if (typeof showAlert === 'function') showAlert('danger', data.error || `Failed to save preset: ${name}`); return; }
        // Set the current preset to this newly saved name (use sanitized name if provided)
        currentPresetName = data.name || name;
        if (typeof showAlert === 'function') showAlert('success', `Saved preset: ${currentPresetName}`);
        // If the preset modal is open, refresh its contents to reflect changes
        const modalEl = document.getElementById('presetModal');
        if (modalEl && modalEl.classList.contains('show')) {
          try { await openPresetModal(); } catch {}
        }
      }
    } catch (e) {
      if (typeof showAlert === 'function') showAlert('danger', e.message);
    }
  }

    // Open the presets modal and populate with available presets
    async function openPresetModal() {
      try {
        const res = await fetch('/synth_presets');
        const data = await res.json();
        if (!data.ok) { if (typeof showAlert === 'function') showAlert('danger', data.error || 'Failed to load presets'); return; }
        const list = document.getElementById('presetList');
        if (!list) return;
        list.innerHTML = '';
        const presets = Array.isArray(data.presets) ? data.presets : [];
        if (presets.length === 0) {
          const empty = document.createElement('div');
          empty.className = 'text-muted';
          empty.textContent = 'No presets saved yet.';
          list.appendChild(empty);
        } else {
          presets.forEach(name => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'list-group-item list-group-item-action';
            item.textContent = name;
            if (name === currentPresetName) item.classList.add('active');
            item.addEventListener('click', async () => {
              try {
                const r = await fetch(`/synth_get_preset?name=${encodeURIComponent(name)}`);
                const d = await r.json();
                if (!d.ok || !d.config) { if (typeof showAlert === 'function') showAlert('danger', d.error || 'Failed to load preset'); return; }
                
                // Apply config to form (handles nested/flat)
                applyConfigToForm(d.config);
                
                currentPresetName = name;
                if (typeof showAlert === 'function') showAlert('success', `Loaded preset: ${name}`);
              } catch (e) {
                if (typeof showAlert === 'function') showAlert('danger', e.message);
              }
            });
            list.appendChild(item);
          });
        }
        // Show the modal
        const modalEl = document.getElementById('presetModal');
        if (modalEl && typeof bootstrap !== 'undefined' && bootstrap.Modal) {
          new bootstrap.Modal(modalEl).show();
        }
      } catch (e) {
        if (typeof showAlert === 'function') showAlert('danger', e.message);
      }
    }

    // Start a synthesis batch job (local or slurm)
    async function startSynthBatch() {
      const cfg = getSynthConfigFromForm();
      const nEl = document.getElementById('synBatchN');
      const outEl = document.getElementById('synBatchOut');
      const passEl = document.getElementById('synBatchPassword');
      const tasksEl = document.getElementById('synBatchTasks');
      const partEl = document.getElementById('synBatchPartition');
      const seedEl = document.getElementById('synSeedBase');
      const idxEl = document.getElementById('synIndexOffset');
      const lamMinEl = document.getElementById('synLambdaMin');
      const lamMaxEl = document.getElementById('synLambdaMax');
      const body = {
        n_images: nEl ? parseInt(nEl.value, 10) : 100, // FIX: backend expects n_images
        out_dir: outEl ? outEl.value : '',             // FIX: backend expects out_dir
        password: passEl ? passEl.value : '',
        n_tasks: tasksEl ? parseInt(tasksEl.value, 10) : 1, // backend expects n_tasks
        partition: partEl ? partEl.value : undefined,
        seed_base: seedEl ? parseInt(seedEl.value, 10) : undefined,
        index_offset: idxEl ? parseInt(idxEl.value, 10) : 0,
        stage_lambda_range: [lamMinEl ? parseFloat(lamMinEl.value) : 0.1, lamMaxEl ? parseFloat(lamMaxEl.value) : 10.0],
        config: cfg,
      };
      try {
        const res = await fetch('/synth_batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!data.ok) { if (typeof showAlert === 'function') showAlert('danger', data.error || 'Batch submission failed'); return; }
        // Notify with job id and tasks
        const mode = data.mode || 'local';
        const jobId = data.job_id ? String(data.job_id) : 'unknown';
        const tasks = data.tasks != null ? String(data.tasks) : (body.n_tasks || 1);
        const outPath = data.out_dir || body.out_dir || '(default)';
        if (typeof showAlert === 'function') showAlert('success', `Batch submitted (${mode}). Job ID: ${jobId}. Tasks: ${tasks}. Out: ${outPath}`);
      } catch (e) {
        if (typeof showAlert === 'function') showAlert('danger', e.message);
      }
    }

    // Clear all synthesis rows
    function clearSynthRows() {
      synthRows = [];
      synthRowCounter = 0;
      const container = document.getElementById('synth-rows');
      if (container) container.innerHTML = '';
    }

    // Helper actions for saving presets
    async function saveSynthStandard() {
      try {
        currentPresetName = 'standard';
        await saveSynthInCurrent();
      } catch (e) {
        if (typeof showAlert === 'function') showAlert('danger', e.message);
      }
    }

    async function saveSynthPresetPrompt() {
      try {
        let name = prompt('Enter a name for the new preset');
        if (name == null) return; // canceled
        name = name.trim();
        if (!name) return; // empty
        currentPresetName = name;
        await saveSynthInCurrent();
      } catch (e) {
        if (typeof showAlert === 'function') showAlert('danger', e.message);
      }
    }

    // Expose functions to global scope for inline HTML event handlers
    window.addSynthRowFromFile = addSynthRowFromFile;
    window.addSynthRow = addSynthRow;
    window.removeSynthRow = removeSynthRow;
    window.updateSynthT = updateSynthT;
    window.setObbPercent = setObbPercent;
    window.regenerateSynthRow = regenerateSynthRow;
    window.regenerateAllSynth = regenerateAllSynth;
    window.toggleObbRow = toggleObbRow;
    window.toggleObbAll = toggleObbAll;
    window.saveSynthInCurrent = saveSynthInCurrent;
    window.saveSynthStandard = saveSynthStandard;
    window.saveSynthPresetPrompt = saveSynthPresetPrompt;
    window.openPresetModal = openPresetModal;
    window.startSynthBatch = startSynthBatch;
    window.clearSynthRows = clearSynthRows;

    })();

// Open high-resolution synthetic image in global zoom modal
async function openSynthHighResModal(rowId) {
  const imgEl = document.getElementById(`synth-prev-${rowId}`);
  if (!imgEl) return;
  const token = imgEl.dataset.highresToken;
  const spinner = document.getElementById('imageZoomSpinner');
  const modalTitle = document.getElementById('imageZoomModalTitle');
  if (modalTitle) modalTitle.textContent = 'Synthetic (High-Res)';
  if (spinner) spinner.style.display = 'block';

  // If backend provided a token, request high-res via URL; otherwise fall back to current src
  let highResUrl = null;
  try {
    if (token) {
      const res = await fetch(`/synth_highres?token=${encodeURIComponent(token)}`);
      const data = await res.json();
      if (data && data.ok && data.image_b64) highResUrl = data.image_b64;
    }
  } catch (e) { /* ignore, fallback below */ }
  if (!highResUrl) highResUrl = imgEl.src;

  // Try to overlay OBBs on the high-res image if the preview overlay is currently visible
  const overlayCanvas = document.getElementById(`synth-obb-${rowId}`);
  const overlayVisible = !!(overlayCanvas && overlayCanvas.style.display !== 'none');
  if (overlayVisible && overlayCanvas && typeof window.showImageInModal === 'function') {
    try {
      const obbs = JSON.parse(overlayCanvas.dataset.obbs || '[]');
      const natW = parseInt(overlayCanvas.dataset.imgNaturalW || '0', 10);
      const natH = parseInt(overlayCanvas.dataset.imgNaturalH || '0', 10);
      const pct = parseInt(overlayCanvas.dataset.obbPercent || '100', 10);
      if (Array.isArray(obbs) && obbs.length > 0 && natW > 0 && natH > 0) {
        const hiImg = new Image();
        hiImg.onload = function() {
          const hw = hiImg.naturalWidth || natW;
          const hh = hiImg.naturalHeight || natH;
          const cnv = document.createElement('canvas');
          cnv.width = hw; cnv.height = hh;
          const ctx = cnv.getContext('2d');
          // Draw base image
          ctx.drawImage(hiImg, 0, 0, hw, hh);
          // Draw OBBs scaled from original natural coords to high-res size
          const scaleX = hw / natW;
          const scaleY = hh / natH;
          const nDraw = Math.max(0, Math.round(obbs.length * (pct / 100.0)));
          const items = obbs.slice(0, nDraw);
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.85)';
          // Scale line width with image size for readability
          ctx.lineWidth = Math.max(1, Math.min(hw, hh) * 0.0012);
          items.forEach(ob => {
            const cs = ob.corners || [];
            if (cs.length !== 4) return;
            ctx.beginPath();
            for (let i = 0; i < 4; i++) {
              const x = cs[i][0] * scaleX;
              const y = cs[i][1] * scaleY;
              if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.stroke();
          });
          // Show composed canvas in modal
          window.showImageInModal(null, 'Synthetic (High-Res)', true, cnv);
        };
        hiImg.src = highResUrl;
        return; // prevent fallback from executing before canvas is ready
      }
    } catch (e) {
      // If anything fails, fall back to plain image below
    }
  }

  if (typeof window.showImageInModal === 'function') {
    window.showImageInModal(highResUrl, 'Synthetic (High-Res)');
  }
}

window.openSynthHighResModal = openSynthHighResModal;
