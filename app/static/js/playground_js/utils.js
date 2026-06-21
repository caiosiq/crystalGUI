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
    'sensor.distractor_anisotropy': 'Distractor Stretch'
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
        'synGhostFraction': 'lblGhostFraction',
        'synGhostSizeScale': 'lblGhostSizeScale',
        'synGhostBlur': 'lblGhostBlur',
        'synGhostCurv': 'lblGhostCurv',
        'synGhostGainMult': 'lblGhostGain',
        'synGhostDeltaAtten': 'lblGhostDeltaAtten',
        'synPolyIrreg': 'lblPolyIrreg',
        'synIncrustFrac': 'lblIncrustFrac',
        'synLightAz': 'lblLightAz',
        'synLightEl': 'lblLightEl',
        'synAnisotropy': 'lblAnisotropy',
        'synAnisoAngle': 'lblAnisoAngle',
        'synGeoJitter': 'lblGeoJitter',
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
