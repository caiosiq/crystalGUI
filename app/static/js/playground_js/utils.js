// Low-level DOM & Formatting Utilities

export function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
    updateLabelFor(id, val);
}

export function setChk(id, val) {
    const el = document.getElementById(id);
    if (el) el.checked = !!val;
}

export function getVal(id, def = 0) {
    const el = document.getElementById(id);
    if (!el) return def;
    const v = parseFloat(el.value);
    return isNaN(v) ? def : v;
}

export function getInt(id, def = 0) {
    const el = document.getElementById(id);
    if (!el) return def;
    const v = parseInt(el.value, 10);
    return isNaN(v) ? def : v;
}

export function getChk(id, def = false) {
    const el = document.getElementById(id);
    return el ? !!el.checked : def;
}

export function showToast(msg) {
    // Simple alert for now or use bootstrap toast if html exists
    // Future: implement Bootstrap toast trigger
    alert(msg);
}

export function debounce(func, wait) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

export function formatVal(val) {
    if (val === undefined) return '';
    if (typeof val === 'number') return val.toFixed(2);
    if (Array.isArray(val)) return `[${val.join(', ')}]`;
    return String(val);
}

const PARAM_DISPLAY_NAMES = {
    // Physics - Rods/Specs
    'physics.rod_specs.count_range': 'Rod Count',
    'physics.rod_specs.length_range': 'Rod Length',
    'physics.rod_specs.aspect_range': 'Rod Aspect Ratio',
    'physics.rod_specs.surf_roughness': 'Roughness',
    'physics.rod_specs.ragged_p': 'Geometry Jitter',
    'physics.rod_specs.polarity_p': 'Polarity Probability',
    'physics.rod_specs.inclusions': 'Inclusions',
    'physics.rod_specs.grain_size': 'Grain Size',
    'physics.rod_specs.anisotropy': 'Anisotropy',
    'physics.rod_specs.anisotropy_angle_deg': 'Anisotropy Angle',
    
    // Agglomeration
    'physics.fused.p1': 'Agglomeration Prob.',
    'physics.fused.sintering_strength': 'Sintering Strength',
    
    // Dynamics
    'physics.flow_shear_rate': 'Flow Shear Rate',
    'physics.sedimentation_strength': 'Sedimentation',
    
    // Optics
    'optics.polarizer_angle_deg': 'Polarizer Angle',
    'optics.shadow_gain': 'DIC Shadow Gain',
    'optics.focus_z': 'Focus Plane (Z) [Legacy]',
    'optics.aperture': 'Aperture [Legacy]',
    
    // Sensor / Renderer
    'sensor.bg_noise_std': 'Background Noise',
    'sensor.blur_sigma': 'Blur Sigma',
    'sensor.vignette_strength': 'Vignette',
    'sensor.chromatic_aberration_strength': 'Chromatic Aberration',
    'sensor.spectral_dispersion_strength': 'Spectral Dispersion',
    'sensor.diffraction_spikes_intensity': 'Diffraction Intensity',
    'sensor.diffraction_spikes_length': 'Diffraction Length',
    'sensor.diffraction_spikes_angle_deg': 'Diffraction Angle',
    'sensor.diffraction_spikes_threshold': 'Diffraction Threshold',
    
    // DoF
    'sensor.focus_z': 'DoF Focus Plane (Z)',
    'sensor.aperture': 'Aperture Size',
    
    // Fouling
    'sensor.fouling_prob': 'Fouling Probability',
    'sensor.fouling_opacity': 'Fouling Opacity',
    
    // Distractors
    'sensor.distractor_blur_sigma': 'Distractor Blur',
    'sensor.distractor_opacity': 'Distractor Opacity',
    'sensor.distractor_anisotropy': 'Distractor Stretch',
    'sensor.scalebar.prob': 'Scale Label Probability',
};

export function getDisplayName(key) {
    if (PARAM_DISPLAY_NAMES[key]) return PARAM_DISPLAY_NAMES[key];
    // Fallback: clean up key
    const parts = key.split('.');
    const last = parts[parts.length-1];
    // Convert snake_case to Title Case
    return last.split('_')
               .map(w => w.charAt(0).toUpperCase() + w.slice(1))
               .join(' ');
}

export function updateLabelFor(id, val) {
    // Map input IDs to label IDs
    const map = {
        'synRoughness': 'lblRoughness',
        'synPolarity': 'lblPolarity',
        'synInclusions': 'lblInclusions',
        'synAgglo': 'lblAgglo',
        'synSinter': 'lblSinter',
        'synFlowDir': 'lblFlowDir',
        'synFlowShear': 'lblFlowShear',
        'synSedStr': 'lblSedStr',
        'synPolarizerAngle': 'lblPolAngle',
        'synShGain': 'lblShGain',
        'synFocusZ': 'lblFocus',
        'synBgNoise': 'lblNoise',
        'synBlur': 'lblBlur',
        'synVignette': 'lblVignette',
        'synChromAb': 'lblChromAb',
        'synSpectralStr': 'lblSpectralStr',
        'synDiffractInt': 'lblDiffractInt',
        'synDiffractLen': 'lblDiffractLen',
        'synDiffractAng': 'lblDiffractAng',
        'synDiffractThresh': 'lblDiffractThresh',
        'synBubbleAttach': 'lblBubbleAttach',
        'synFoulingProb': 'lblFoulingProb',
        'synFoulingOp': 'lblFoulingOp',
        'synScalebarProb': 'lblScalebarProb',
        'synGhostFraction': 'lblGhostFraction',
        'synGhostSizeScale': 'lblGhostSizeScale',
        'synGhostBlur': 'lblGhostBlur',
        'synGhostCurv': 'lblGhostCurv',
        'synGhostSlopeGain': 'lblGhostSlopeGain',
        'synGhostDeltaAtten': 'lblGhostDeltaAtten',
        'synPolyIrreg': 'lblPolyIrreg',
        'synIncrustFrac': 'lblIncrustFrac',
        'synLightAz': 'lblLightAz',
        'synLightEl': 'lblLightEl',
        'synAnisotropy': 'lblAnisotropy',
        'synAnisoAngle': 'lblAnisoAngle',
        'synGeoJitter': 'lblGeoJitter',
        'synCornerRound': 'lblCornerRound',
        'synCornerBend': 'lblCornerBend',
        'synLabelMergeOverlap': 'lblLabelMergeOverlap',
        'synGrainSize': 'lblGrainSize',
        'synDistractorCount': 'lblDistractorCount',
        'synDistractorBlur': 'lblDistractorBlur',
        'synDistractorStretch': 'lblDistractorStretch',
        'synDistractorOp': 'lblDistractorOp',
        'synAperture': 'lblAperture'
    };
    if (map[id]) {
        const lbl = document.getElementById(map[id]);
        if (lbl) {
            // formatting
            if (id === 'synPolarizerAngle' || id === 'synFlowDir' || id === 'synLightAz' || id === 'synLightEl' || id === 'synAnisoAngle') lbl.textContent = Math.round(val) + '°';
            else if (id === 'synLabelMergeOverlap') lbl.textContent = parseFloat(val).toFixed(2);
            else if (id === 'synFocusZ') lbl.textContent = parseFloat(val).toFixed(1);
            else if (id === 'synAperture') lbl.textContent = parseFloat(val).toFixed(2);
            else if (id === 'synShGain') {
                 // Context-aware label for Gain
                 const modeEl = document.getElementById('synOpticsMode');
                 if(modeEl) {
                     // const mode = modeEl.value; // unused for now, just show val
                     lbl.textContent = val;
                 }
            }
            else lbl.textContent = val;
        }
    }
}

/** Map batch λ to stage t (matches osog.physics.stage.lambda_to_t). */
export function lambdaToT(lmbda) {
    const lam = Number(lmbda);
    if (!Number.isFinite(lam) || lam <= 0) return null;
    const t = (Math.log10(Math.max(1e-6, lam)) + 1.0) / 2.0;
    return Math.min(1, Math.max(0, t));
}

/** Update t readouts under batch λ min/max inputs. */
export function updateLambdaTLabels() {
    const pairs = [
        ['batchLambdaMin', 'batchLambdaMinT'],
        ['batchLambdaMax', 'batchLambdaMaxT'],
    ];
    for (const [inputId, labelId] of pairs) {
        const input = document.getElementById(inputId);
        const label = document.getElementById(labelId);
        if (!input || !label) continue;
        const t = lambdaToT(parseFloat(input.value));
        label.textContent = t == null ? 't = —' : `t = ${t.toFixed(3)}`;
    }
}

/** Sync all range slider value labels on load or after preset apply. */
export function syncRangeLabels() {
    const map = {
        'synGhostFraction': 'lblGhostFraction',
        'synGhostSizeScale': 'lblGhostSizeScale',
        'synGhostBlur': 'lblGhostBlur',
        'synGhostCurv': 'lblGhostCurv',
        'synGhostSlopeGain': 'lblGhostSlopeGain',
        'synGhostDeltaAtten': 'lblGhostDeltaAtten',
        'synFoulingProb': 'lblFoulingProb',
        'synFoulingOp': 'lblFoulingOp',
    };
    for (const [inputId, labelId] of Object.entries(map)) {
        const el = document.getElementById(inputId);
        if (el) updateLabelFor(inputId, el.value);
    }
    updateLambdaTLabels();
}

const TEXTURE_TYPE_HELP = {
    none: '<strong>Smooth / Generic</strong> uses the material default (usually smooth). Only faint micro-grain appears unless <strong>Surface Roughness</strong> is above ~0.1.',
    striated: '<strong>Striated</strong> adds sharp longitudinal ridges. Use <strong>Surface Roughness</strong> for intensity and <strong>Grain Size</strong> for ridge spacing.',
    pitted: '<strong>Pitted</strong> adds patchy etched regions. Use <strong>Surface Roughness</strong> for intensity and <strong>Grain Size</strong> for patch scale.',
    stepped: '<strong>Stepped</strong> adds growth terraces. Use <strong>Surface Roughness</strong> for intensity and <strong>Grain Size</strong> for terrace density.',
};

export function updateTextureTypeHelp() {
    const typeEl = document.getElementById('synTextureType');
    const helpEl = document.getElementById('textureTypeHelp');
    const partnerBox = document.getElementById('texturePartnerBox');
    if (!typeEl || !helpEl) return;

    const type = typeEl.value || 'none';
    const rough = parseFloat(document.getElementById('synRoughness')?.value || '0');

    const typeMsg = TEXTURE_TYPE_HELP[type] || TEXTURE_TYPE_HELP.none;
    helpEl.innerHTML = typeMsg;

    const emphasizePartners = type !== 'none' || rough > 0.05;
    if (partnerBox) {
        partnerBox.style.background = emphasizePartners
            ? 'rgba(var(--bs-primary-rgb), 0.08)'
            : 'rgba(var(--bs-secondary-rgb), 0.04)';
        partnerBox.style.border = emphasizePartners
            ? '1px solid rgba(var(--bs-primary-rgb), 0.35)'
            : '1px solid rgba(var(--bs-secondary-rgb), 0.2)';
    }
}
