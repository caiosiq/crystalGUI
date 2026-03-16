// ------------------------------------------------------------------
// Sidebar Tabs
// ------------------------------------------------------------------

function switchSidebarTab(tabName) {
    // Buttons
    document.querySelectorAll('.sidebar-nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');
    
    // Panes
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    document.getElementById(`pane-${tabName}`).classList.add('active');
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
    // Also update label if exists
    updateLabelFor(id, val);
}

function setChk(id, val) {
    const el = document.getElementById(id);
    if (el) el.checked = !!val;
}

function getVal(id, def = 0) {
    const el = document.getElementById(id);
    if (!el) return def;
    const v = parseFloat(el.value);
    return isNaN(v) ? def : v;
}

function getInt(id, def = 0) {
    const el = document.getElementById(id);
    if (!el) return def;
    const v = parseInt(el.value, 10);
    return isNaN(v) ? def : v;
}

function getChk(id, def = false) {
    const el = document.getElementById(id);
    return el ? !!el.checked : def;
}

function updateLabelFor(id, val) {
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
                 const mode = document.getElementById('synOpticsMode').value;
                 lbl.textContent = val;
            }
            else lbl.textContent = val;
        }
    }
}

// ------------------------------------------------------------------
// Validation & Metrics
// ------------------------------------------------------------------

let targetImageFilename = null;

function setupDragDrop() {
    const dz = document.getElementById('dropZone');
    const inp = document.getElementById('fileInput');
    
    if (!dz) return;
    
    dz.onclick = () => inp.click();
    
    inp.onchange = (e) => {
        if (e.target.files && e.target.files[0]) {
            loadRefImage(e.target.files[0]);
        }
    };
    
    dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('bg-secondary'); };
    dz.ondragleave = (e) => { e.preventDefault(); dz.classList.remove('bg-secondary'); };
    dz.ondrop = (e) => {
        e.preventDefault();
        dz.classList.remove('bg-secondary');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            loadRefImage(e.dataTransfer.files[0]);
        }
    };
}

function loadRefImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('refImage').src = e.target.result;
        document.getElementById('compareImageDisplay').src = e.target.result; // Update comparison image
        
        document.getElementById('dropZone').style.display = 'none';
        document.getElementById('refImageContainer').style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Upload for backend use
    uploadRefImage(file);
}

async function uploadRefImage(file) {
    const fd = new FormData();
    fd.append('file', file);
    try {
        const res = await fetch('/upload_target', {method: 'POST', body: fd});
        const data = await res.json();
        if(data.ok) {
            targetImageFilename = data.filename;
            const badge = document.getElementById('targetImgName');
            if(badge) {
                badge.textContent = data.filename;
                badge.classList.remove('bg-secondary');
                badge.classList.add('bg-success');
            }
        } else {
            console.error("Upload failed", data.error);
        }
    } catch(e) {
        console.error("Upload error", e);
    }
}

function clearRefImage() {
    document.getElementById('refImage').src = '';
    document.getElementById('compareImageDisplay').src = '';
    document.getElementById('compareContainer').style.display = 'none';
    
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('refImageContainer').style.display = 'none';
    
    targetImageFilename = null;
    document.getElementById('targetImgName').textContent = 'None';
    document.getElementById('targetImgName').classList.remove('bg-success');
    document.getElementById('targetImgName').classList.add('bg-secondary');
    
    // Reset comparison mode if active
    isCompareMode = false;
    document.getElementById('btnCompareMode').classList.remove('active');
}

let isCompareMode = false;
function toggleComparisonMode() {
    isCompareMode = !isCompareMode;
    const btn = document.getElementById('btnCompareMode');
    const cmpContainer = document.getElementById('compareContainer');
    const mainContainer = document.getElementById('canvasContainer');
    
    if (isCompareMode) {
        if (!targetImageFilename && !document.getElementById('refImage').src) {
             showToast("Load a reference image first!");
             isCompareMode = false;
             return;
        }
        btn.classList.add('active');
        cmpContainer.style.display = 'flex'; // Show side by side
        // Flexbox handles width automatically (50/50 if flex-grow)
    } else {
        btn.classList.remove('active');
        cmpContainer.style.display = 'none';
    }
    // Resize 3D if active
    if (is3DInit) onWindowResize();
}

function updateMetrics(data) {
    if (!data || !data.meta) return;
    const m = data.meta;
    
    // Count (mock logic if meta not fully populated, or read actuals)
    let count = 0;
    if (m.rods) count += m.rods.count || 0;
    
    const metricEl = document.getElementById('metricCount');
    if(metricEl) metricEl.textContent = count > 0 ? count : '-';
}


// ------------------------------------------------------------------
// Config & Logic
// ------------------------------------------------------------------

function getConfig() {
    const hexToRgb = (hex) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)] : [255, 255, 255];
    };
    
    // Light Direction Calc
    const az = getVal('synLightAz', 45) * (Math.PI / 180);
    const el = getVal('synLightEl', 45) * (Math.PI / 180);
    const lx = Math.cos(el) * Math.cos(az);
    const ly = Math.cos(el) * Math.sin(az);
    const lz = Math.sin(el);

    return {
        canvas: { width: 1024, height: 1024, use_gpu: true },
        physics: {
            // Global flags
            use_specific_specs: true, // Prefer specific specs for better control
            
            // Rods (Legacy/Main) - Populating both for compatibility
            rods: {
                enable: getChk('synRodEnable'),
                enable_3d: getChk('synEnable3d'),
                n_rods_rng_lo_hi: [getInt('synRodCountLo', 50), getInt('synRodCountHi', 200), getInt('synRodCountHi', 200)],
                rod_len_px_lo_hi: [getVal('synRodLenLo', 30), getVal('synRodLenHi', 150), getVal('synRodLenHi', 150)],
                rod_aspect_lo_hi: [getVal('synRodAspLo', 0.02), getVal('synRodAspHi', 0.1), getVal('synRodAspHi', 0.1)]
                // material: Not supported in ParticlesConfig (Legacy)
            },

            // Specific Specs
            rod_specs: {
                enable: getChk('synRodEnable'),
                count_range: [getInt('synRodCountLo', 50), getInt('synRodCountHi', 200)],
                length_range: [getVal('synRodLenLo', 30), getVal('synRodLenHi', 150)],
                aspect_range: [getVal('synRodAspLo', 0.02), getVal('synRodAspHi', 0.1)],
                material: document.getElementById('synRodMaterial').value,
                
                // Morphology
                ragged_p: getVal('synGeoJitter'), // Decoupled from Roughness
                polarity_p: getVal('synPolarity'),
                inclusions: getVal('synInclusions'),
                shape_mode: document.getElementById('synShapeMode').value,
                
                // Add texture-specific fields for TextureShader
                texture_type: document.getElementById('synTextureType').value,
                surf_roughness: getVal('synRoughness'),
                grain_size: getVal('synGrainSize'),
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity'),
                
                // Phase 4.4.2.3.1: Anisotropy
                anisotropy: getVal('synAnisotropy'),
                anisotropy_angle_deg: getVal('synAnisoAngle')
            },

            sphere_specs: {
                enable: getChk('synSphereEnable'),
                count_range: [getInt('synSphereCountLo', 10), getInt('synSphereCountHi', 50)],
                diameter_range: [getInt('synSphereDiamLo', 20), getInt('synSphereDiamHi', 100)],
                material: document.getElementById('synSphereMaterial').value,
                
                // Propagate texture params to spheres too!
                texture_type: document.getElementById('synTextureType').value,
                surf_roughness: getVal('synRoughness'),
                grain_size: getVal('synGrainSize'),
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity'),
                
                anisotropy: getVal('synAnisotropy'),
                anisotropy_angle_deg: getVal('synAnisoAngle')
            },
            cube_specs: {
                enable: getChk('synCubeEnable'),
                count_range: [getInt('synCubeCountLo', 10), getInt('synCubeCountHi', 50)],
                size_range: [getInt('synCubeSizeLo', 20), getInt('synCubeSizeHi', 100)],
                material: document.getElementById('synCubeMaterial').value,
                
                // Propagate texture params
                texture_type: document.getElementById('synTextureType').value,
                surf_roughness: getVal('synRoughness'),
                grain_size: getVal('synGrainSize'),
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity'),
                
                anisotropy: getVal('synAnisotropy'),
                anisotropy_angle_deg: getVal('synAnisoAngle')
            },
            plate_specs: {
                enable: getChk('synPlateEnable'),
                count_range: [getInt('synPlateCountLo', 10), getInt('synPlateCountHi', 50)],
                size_range: [getInt('synPlateSizeLo', 30), getInt('synPlateSizeHi', 150)],
                aspect_range: [getVal('synPlateAspLo', 0.1), getVal('synPlateAspHi', 0.8)],
                thickness_range: [getVal('synPlateThickLo', 0.05), getVal('synPlateThickHi', 0.2)],
                material: document.getElementById('synPlateMaterial').value,
                
                // Propagate texture params
                texture_type: document.getElementById('synTextureType').value,
                surf_roughness: getVal('synRoughness'),
                grain_size: getVal('synGrainSize'),
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity'),
                
                ragged_p: getVal('synGeoJitter'),
                anisotropy: getVal('synAnisotropy'),
                anisotropy_angle_deg: getVal('synAnisoAngle')
            },
            bubble_specs: {
                enable: getChk('synBubbleEnable'),
                count_range: [getInt('synBubbleCountLo', 5), getInt('synBubbleCountHi', 20)],
                diameter_range: [getInt('synBubbleDiamLo', 10), getInt('synBubbleDiamHi', 50)],
                attach_prob: getVal('synBubbleAttach'),
                material: document.getElementById('synBubbleMaterial').value
            },
            droplet_specs: {
                enable: getChk('synDropletEnable'),
                count_range: [getInt('synDropletCountLo', 5), getInt('synDropletCountHi', 20)],
                diameter_range: [getInt('synDropletDiamLo', 10), getInt('synDropletDiamHi', 50)],
                material: document.getElementById('synDropletMaterial').value
            },
            polyhedra_specs: {
                enable: getChk('synPolyEnable'),
                count_range: [getInt('synPolyCountLo', 5), getInt('synPolyCountHi', 20)],
                size_range: [getInt('synPolySizeLo', 30), getInt('synPolySizeHi', 100)],
                num_planes_range: [getInt('synPolyPlanesLo', 6), getInt('synPolyPlanesHi', 12)],
                irregularity: getVal('synPolyIrreg'),
                material: document.getElementById('synPolyMaterial').value,
                
                // Propagate texture params
                texture_type: document.getElementById('synTextureType').value,
                surf_roughness: getVal('synRoughness'),
                grain_size: getVal('synGrainSize'),
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity'),
                
                anisotropy: getVal('synAnisotropy'),
                anisotropy_angle_deg: getVal('synAnisoAngle')
            },
            
            incrustation_specs: {
                enable: getChk('synIncrustEnable'),
                fraction: getVal('synIncrustFrac'),
                count_range: [getInt('synIncrustCountLo', 50), getInt('synIncrustCountHi', 200)],
                size_range: [getVal('synIncrustSizeLo', 1.0), getVal('synIncrustSizeHi', 3.0)],
                material: document.getElementById('synIncrustMaterial').value
            },

            // Fused / Agglomeration
            fused: {
                enable: (getVal('synAgglo') > 0.001),
                p1: getVal('synAgglo'),
                sintering_strength: getVal('synSinter'),
                dlca_enable: getChk('synDLCA'),
                cluster_weights: [
                    getChk('chkAggRandom') ? 1.0 : 0.0,
                    getChk('chkAggStack') ? 1.0 : 0.0,
                    getChk('chkAggChain') ? 1.0 : 0.0,
                    getChk('chkAggCross') ? 1.0 : 0.0,
                    getChk('chkAggSnow') ? 1.0 : 0.0,
                    getChk('chkAggSphere') ? 1.0 : 0.0
                ]
            },

            // Dynamics
            flow_enable: getChk('synFlowEnable'),
            flow_direction: getVal('synFlowDir'),
            flow_shear_rate: getVal('synFlowShear'),
            sedimentation_enable: getChk('synSedEnable'),
            sedimentation_strength: getVal('synSedStr'),
            size_segregation_enable: getChk('synSizeSeg'),

            ghosts: {
                enable: getChk('synGhostEnable'),
                fraction: getVal('synGhostFraction')
            },
            debris: {
                rate: getVal('synDebrisRate')
            }
        },
        optics: {
            mode: document.getElementById('synOpticsMode').value,
            polarizer_angle_deg: getVal('synPolarizerAngle'),
            shadow_gain: [getVal('synShGain'), getVal('synShGain') * 2.0], // Tuple
            focus_z: getVal('synFocusZ'),
            medium_refractive_index: getVal('synSolventRi', 1.333), // Phase 4.4.2.1
            light_direction: [lx, ly, lz]
        },
        sensor: {
            bg_noise_std: getVal('synBgNoise'),
            blur_sigma: getVal('synBlur'),
            vignette_strength: getVal('synVignette'),
            chromatic_aberration_strength: getVal('synChromAb'),
            
            // Phase 4.4.2.3.3: Optical Realism
            spectral_dispersion_enable: getChk('synSpectralEnable'),
            spectral_dispersion_strength: getVal('synSpectralStr'),
            
            diffraction_spikes_enable: getChk('synDiffractEnable'),
            diffraction_spikes_intensity: getVal('synDiffractInt'),
            diffraction_spikes_length: getVal('synDiffractLen'),
            diffraction_spikes_count: parseInt(document.getElementById('synDiffractCount').value) || 4,
            diffraction_spikes_angle_deg: getVal('synDiffractAng'),
            diffraction_spikes_threshold: getVal('synDiffractThresh'),
            
            // Phase 4.4.2.4: Shallow Depth of Field
            dof_enable: getChk('synDofEnable'),
            focus_z: getVal('synFocusZ'),
            aperture: getVal('synAperture'),
            
            solvent_color: hexToRgb(document.getElementById('synSolventColor').value || '#ffffff'), // Phase 4.4.2.1
            
            tilt_enable: getChk('synTiltEnable'),
            relief_field_enable: getChk('synReliefEnable'),
            
            // Fouling
            fouling_enable: getChk('synFoulingEnable'),
            fouling_prob: getVal('synFoulingProb'),
            fouling_count_range: [getInt('synFoulingCountLo', 1), getInt('synFoulingCountHi', 5)],
            fouling_opacity: getVal('synFoulingOp'),

            // Phase 4.4.2.4: Distractors
            distractor_enable: getChk('synDistractorEnable'),
            distractor_count_range: [getInt('synDistractorCount', 200), getInt('synDistractorCount', 200)],
            distractor_blur_sigma: getVal('synDistractorBlur', 2.0),
            distractor_opacity: getVal('synDistractorOp', 0.5),
            distractor_anisotropy: getVal('synDistractorStretch', 0.0)
        }
    };
}

function applyConfigToUI(p) {
    // Map nested config back to UI
    if (!p) return;
    
    // Physics
    if (p.physics) {
        // enable_3d is inside rods in ParticlesConfig
        const e3d = (p.physics.rods && p.physics.rods.enable_3d !== undefined) ? p.physics.rods.enable_3d : p.physics.enable_3d;
        setChk('synEnable3d', e3d);
        
        // Rods (Legacy/Main)
        if (p.physics.rods) {
            setChk('synRodEnable', p.physics.rods.enable);
            if (p.physics.rods.n_rods_rng_lo_hi) {
                setVal('synRodCountLo', p.physics.rods.n_rods_rng_lo_hi[0]);
                setVal('synRodCountHi', p.physics.rods.n_rods_rng_lo_hi[1]);
            }
            // Length/Aspect might be in rod_specs if specific specs used
        }
        
        // Specific Specs (Preferred)
        if (p.physics.rod_specs) {
             const rs = p.physics.rod_specs;
             // Overwrite if specific specs are populated
             if (rs.count_range) {
                 setVal('synRodCountLo', rs.count_range[0]);
                 setVal('synRodCountHi', rs.count_range[1]);
             }
             if (rs.length_range) {
                 setVal('synRodLenLo', rs.length_range[0]);
                 setVal('synRodLenHi', rs.length_range[1]);
             }
             if (rs.aspect_range) {
                 setVal('synRodAspLo', rs.aspect_range[0]);
                 setVal('synRodAspHi', rs.aspect_range[1]);
             }
             if (rs.ragged_p !== undefined) setVal('synGeoJitter', rs.ragged_p);
             if (rs.surf_roughness !== undefined) setVal('synRoughness', rs.surf_roughness);
             if (rs.grain_size !== undefined) setVal('synGrainSize', rs.grain_size);
             if (rs.polarity_p !== undefined) setVal('synPolarity', rs.polarity_p);
             if (rs.inclusions !== undefined) setVal('synInclusions', rs.inclusions);
             if (rs.shape_mode) document.getElementById('synShapeMode').value = rs.shape_mode;
             if (rs.texture_type) document.getElementById('synTextureType').value = rs.texture_type;
             if (rs.material) document.getElementById('synRodMaterial').value = rs.material;
             
             if (rs.anisotropy !== undefined) setVal('synAnisotropy', rs.anisotropy);
             if (rs.anisotropy_angle_deg !== undefined) setVal('synAnisoAngle', rs.anisotropy_angle_deg);
        }

        if (p.physics.sphere_specs) {
            const s = p.physics.sphere_specs;
            setChk('synSphereEnable', s.enable);
            if (s.count_range) { setVal('synSphereCountLo', s.count_range[0]); setVal('synSphereCountHi', s.count_range[1]); }
            if (s.diameter_range) { setVal('synSphereDiamLo', s.diameter_range[0]); setVal('synSphereDiamHi', s.diameter_range[1]); }
            if (s.material) document.getElementById('synSphereMaterial').value = s.material;
        }

        if (p.physics.cube_specs) {
            const c = p.physics.cube_specs;
            setChk('synCubeEnable', c.enable);
            if (c.count_range) { setVal('synCubeCountLo', c.count_range[0]); setVal('synCubeCountHi', c.count_range[1]); }
            if (c.size_range) { setVal('synCubeSizeLo', c.size_range[0]); setVal('synCubeSizeHi', c.size_range[1]); }
            if (c.material) document.getElementById('synCubeMaterial').value = c.material;
        }

        if (p.physics.plate_specs) {
            const pl = p.physics.plate_specs;
            setChk('synPlateEnable', pl.enable);
            if (pl.count_range) { setVal('synPlateCountLo', pl.count_range[0]); setVal('synPlateCountHi', pl.count_range[1]); }
            if (pl.size_range) { setVal('synPlateSizeLo', pl.size_range[0]); setVal('synPlateSizeHi', pl.size_range[1]); }
            if (pl.aspect_range) { setVal('synPlateAspLo', pl.aspect_range[0]); setVal('synPlateAspHi', pl.aspect_range[1]); }
            if (pl.thickness_range) { setVal('synPlateThickLo', pl.thickness_range[0]); setVal('synPlateThickHi', pl.thickness_range[1]); }
            if (pl.material) document.getElementById('synPlateMaterial').value = pl.material;
        }

        if (p.physics.bubble_specs) {
            const b = p.physics.bubble_specs;
            setChk('synBubbleEnable', b.enable);
            if (b.count_range) { setVal('synBubbleCountLo', b.count_range[0]); setVal('synBubbleCountHi', b.count_range[1]); }
            if (b.diameter_range) { setVal('synBubbleDiamLo', b.diameter_range[0]); setVal('synBubbleDiamHi', b.diameter_range[1]); }
            if (b.attach_prob !== undefined) setVal('synBubbleAttach', b.attach_prob);
            if (b.material) document.getElementById('synBubbleMaterial').value = b.material;
        }

        if (p.physics.droplet_specs) {
            const d = p.physics.droplet_specs;
            setChk('synDropletEnable', d.enable);
            if (d.count_range) { setVal('synDropletCountLo', d.count_range[0]); setVal('synDropletCountHi', d.count_range[1]); }
            if (d.diameter_range) { setVal('synDropletDiamLo', d.diameter_range[0]); setVal('synDropletDiamHi', d.diameter_range[1]); }
            if (d.material) document.getElementById('synDropletMaterial').value = d.material;
        }

        if (p.physics.polyhedra_specs) {
            const py = p.physics.polyhedra_specs;
            setChk('synPolyEnable', py.enable);
            if (py.count_range) { setVal('synPolyCountLo', py.count_range[0]); setVal('synPolyCountHi', py.count_range[1]); }
            if (py.size_range) { setVal('synPolySizeLo', py.size_range[0]); setVal('synPolySizeHi', py.size_range[1]); }
            if (py.num_planes_range) { setVal('synPolyPlanesLo', py.num_planes_range[0]); setVal('synPolyPlanesHi', py.num_planes_range[1]); }
            if (py.irregularity !== undefined) setVal('synPolyIrreg', py.irregularity);
            if (py.material) document.getElementById('synPolyMaterial').value = py.material;
        }

        if (p.physics.incrustation_specs) {
            const inc = p.physics.incrustation_specs;
            setChk('synIncrustEnable', inc.enable);
            if (inc.fraction !== undefined) setVal('synIncrustFrac', inc.fraction);
            if (inc.count_range) { setVal('synIncrustCountLo', inc.count_range[0]); setVal('synIncrustCountHi', inc.count_range[1]); }
            if (inc.size_range) { setVal('synIncrustSizeLo', inc.size_range[0]); setVal('synIncrustSizeHi', inc.size_range[1]); }
            if (inc.material) document.getElementById('synIncrustMaterial').value = inc.material;
        }

        // Dynamics (Flattened in PhysicsConfig)
        if (p.physics.flow_enable !== undefined) setChk('synFlowEnable', p.physics.flow_enable);
        if (p.physics.flow_direction !== undefined) setVal('synFlowDir', p.physics.flow_direction);
        if (p.physics.flow_shear_rate !== undefined) setVal('synFlowShear', p.physics.flow_shear_rate);
        
        if (p.physics.sedimentation_enable !== undefined) setChk('synSedEnable', p.physics.sedimentation_enable);
        if (p.physics.sedimentation_strength !== undefined) setVal('synSedStr', p.physics.sedimentation_strength);
        if (p.physics.size_segregation_enable !== undefined) setChk('synSizeSeg', p.physics.size_segregation_enable);
        
        // Fused
        if (p.physics.fused) {
            setVal('synAgglo', p.physics.fused.p1 || 0);
            setVal('synSinter', p.physics.fused.sintering_strength || 0);
            setChk('synDLCA', p.physics.fused.dlca_enable);
        }
        
        // Ghosts
        if (p.physics.ghosts) {
            setChk('synGhostEnable', p.physics.ghosts.enable);
            setVal('synGhostFraction', p.physics.ghosts.fraction);
        }
        
        // Debris
        if (p.physics.debris) {
            setVal('synDebrisRate', p.physics.debris.rate);
        }
    }
    
    // Optics
    if (p.optics) {
        setVal('synOpticsMode', p.optics.mode);
        updateOpticsControls(p.optics.mode); // Trigger UI update
        
        // Handle name change
        const ang = p.optics.polarizer_angle_deg !== undefined ? p.optics.polarizer_angle_deg : p.optics.polarizer_angle;
        setVal('synPolarizerAngle', ang || 0);
        
        // Handle Tuple vs Float for shadow_gain
        let sg = p.optics.shadow_gain;
        if (Array.isArray(sg)) sg = sg[0];
        setVal('synShGain', sg);
        
        setVal('synFocusZ', p.optics.focus_z || 0.0);
        setVal('synSolventRi', p.optics.medium_refractive_index || 1.333);
        
        if (p.optics.light_direction && p.optics.light_direction.length >= 3) {
            const l = p.optics.light_direction;
            // Vector to Az/El
            const lx = l[0], ly = l[1], lz = l[2];
            const r = Math.sqrt(lx*lx + ly*ly + lz*lz);
            const el = Math.asin(lz/r) * (180/Math.PI);
            const az = Math.atan2(ly, lx) * (180/Math.PI);
            setVal('synLightEl', el);
            setVal('synLightAz', az < 0 ? az + 360 : az);
        }
    }

    // Sensor
    if (p.sensor) {
        setVal('synBgNoise', p.sensor.bg_noise_std !== undefined ? p.sensor.bg_noise_std : p.sensor.noise);
        setVal('synBlur', p.sensor.blur_sigma !== undefined ? p.sensor.blur_sigma : p.sensor.blur);
        setVal('synVignette', p.sensor.vignette_strength !== undefined ? p.sensor.vignette_strength : p.sensor.vignette);
        setVal('synChromAb', p.sensor.chromatic_aberration_strength !== undefined ? p.sensor.chromatic_aberration_strength : p.sensor.chromatic_aberration);
        
        // Phase 4.4.2.3.3
        setChk('synSpectralEnable', p.sensor.spectral_dispersion_enable);
        if (p.sensor.spectral_dispersion_strength !== undefined) setVal('synSpectralStr', p.sensor.spectral_dispersion_strength);
        
        setChk('synDiffractEnable', p.sensor.diffraction_spikes_enable);
        if (p.sensor.diffraction_spikes_intensity !== undefined) setVal('synDiffractInt', p.sensor.diffraction_spikes_intensity);
        if (p.sensor.diffraction_spikes_length !== undefined) setVal('synDiffractLen', p.sensor.diffraction_spikes_length);
        if (p.sensor.diffraction_spikes_count !== undefined) document.getElementById('synDiffractCount').value = p.sensor.diffraction_spikes_count;
        if (p.sensor.diffraction_spikes_angle_deg !== undefined) setVal('synDiffractAng', p.sensor.diffraction_spikes_angle_deg);
        if (p.sensor.diffraction_spikes_threshold !== undefined) setVal('synDiffractThresh', p.sensor.diffraction_spikes_threshold);
        
        // Phase 4.4.2.4: Shallow Depth of Field
        setChk('synDofEnable', p.sensor.dof_enable);
        if (p.sensor.aperture !== undefined) setVal('synAperture', p.sensor.aperture);
        if (p.sensor.focus_z !== undefined) setVal('synFocusZ', p.sensor.focus_z);
        
        // Solvent Color
        if (p.sensor.solvent_color) {
            const c = p.sensor.solvent_color;
            if (Array.isArray(c) && c.length === 3) {
                 const toHex = (x) => ('0' + x.toString(16)).slice(-2);
                 setVal('synSolventColor', '#' + toHex(c[0]) + toHex(c[1]) + toHex(c[2]));
            }
        }
        
        setChk('synTiltEnable', p.sensor.tilt_enable);
        setChk('synReliefEnable', p.sensor.relief_field_enable !== undefined ? p.sensor.relief_field_enable : p.sensor.relief_enable);
        
        // Fouling (Flattened in SensorConfig)
        if (p.sensor.fouling_enable !== undefined) setChk('synFoulingEnable', p.sensor.fouling_enable);
        if (p.sensor.fouling_prob !== undefined) setVal('synFoulingProb', p.sensor.fouling_prob);
        if (p.sensor.fouling_opacity !== undefined) setVal('synFoulingOp', p.sensor.fouling_opacity);
        
        // Phase 4.4.2.4: Distractors
        setChk('synDistractorEnable', p.sensor.distractor_enable);
        if (p.sensor.distractor_count_range) {
             setVal('synDistractorCount', p.sensor.distractor_count_range[0]);
        }
        if (p.sensor.distractor_blur_sigma !== undefined) setVal('synDistractorBlur', p.sensor.distractor_blur_sigma);
        if (p.sensor.distractor_opacity !== undefined) setVal('synDistractorOp', p.sensor.distractor_opacity);
        if (p.sensor.distractor_anisotropy !== undefined) setVal('synDistractorStretch', p.sensor.distractor_anisotropy);
    }
}

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

function getDisplayName(key) {
    if (PARAM_DISPLAY_NAMES[key]) return PARAM_DISPLAY_NAMES[key];
    // Fallback: clean up key
    const parts = key.split('.');
    const last = parts[parts.length-1];
    // Convert snake_case to Title Case
    return last.split('_')
               .map(w => w.charAt(0).toUpperCase() + w.slice(1))
               .join(' ');
}
let isOptimizing = false; // Lock flag during optimization runs
let optimizationRules = {};
let paramLinks = {}; // Tracks which params are linked (true) or overridden (false)

// Initialize links when rules are loaded
function initParamLinks() {
    Object.keys(optimizationRules).forEach(key => {
        if (paramLinks[key] === undefined) {
            paramLinks[key] = true; // Default to linked
        }
    });
}

function applyOverrides(config, overrides) {
    const clone = JSON.parse(JSON.stringify(config));
    
    // Only apply overrides where link is broken
    Object.keys(overrides).forEach(key => {
        if (paramLinks[key] === false) { // Only apply if unlinked
            let val = overrides[key];
            let path = key;
            let index = null;
            
            // Check rules for mapping
            if (optimizationRules[key] && optimizationRules[key].target_attr) {
                 path = optimizationRules[key].target_attr[0];
                 index = optimizationRules[key].target_attr[1];
            }
            
            const parts = path.split('.');
            let current = clone;
            
            // Navigate to parent
            for(let i=0; i<parts.length-1; i++) {
                if(current[parts[i]] === undefined) current[parts[i]] = {};
                current = current[parts[i]];
            }
            
            const lastPart = parts[parts.length-1];
            
            if (index !== null) {
                // Target is an array element
                if (Array.isArray(current[lastPart])) {
                    current[lastPart][index] = val;
                }
            } else {
                current[lastPart] = val;
                
                // Sync legacy/dual parameters
                if (path === 'optics.focus_z' && clone.sensor) clone.sensor.focus_z = val;
                if (path === 'optics.aperture' && clone.sensor) clone.sensor.aperture = val;
            }
        }
    });
    return clone;
}

// ------------------------------------------------------------------
// Generation & Preview
// ------------------------------------------------------------------

let debounceTimer = null;
let currentSeed = null;

function scheduleRegenerate() {
    // If optimizing, don't auto-regenerate (AI is driving)
    if (isOptimizing) return; 
    if (isValidateMode && optimizationJobId) return; 

    const status = document.getElementById('statusText');
    status.textContent = 'Changed...';
    
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        regenerate();
        // Also refresh optimization param list values if in validate mode
        if(isValidateMode) {
             // Lightweight UI update
             updateOptimizationListValues();
             updateGTTunerValues();
        }
    }, 50); // 50ms debounce for responsiveness
}

function showError(msg, traceback) {
    const status = document.getElementById('statusText');
    status.textContent = 'Error!';
    status.classList.add('text-danger');
    
    document.getElementById('errorMsg').textContent = msg || "Unknown Error";
    document.getElementById('errorTrace').textContent = traceback || "No traceback available.";
    
    const modal = new bootstrap.Modal(document.getElementById('errorModal'));
    modal.show();
    console.error(msg, traceback);
}

async function fetchPreview(config, seed, targetImgId, isGt = false) {
    const payload = { 
        t: 0.5, 
        config: config, 
        return_heads: true,
        return_obbs: !isGt // Only main view needs OBBs for now
    };
    
    if (seed !== null) {
        payload.seed = seed;
    }

    const res = await fetch('/synth_preview', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    if (data.ok) {
        // Determine image source
        let imgSrc = data.image_b64;
        
        // If it's the main view, we support head switching
        if (!isGt) {
            // Update hidden head images
             if (data.heads) {
                if (data.heads.optical) document.getElementById('img-optical').src = data.heads.optical;
                else if (data.image_b64) document.getElementById('img-optical').src = data.image_b64;
                
                if (data.heads.height) document.getElementById('img-height').src = data.heads.height;
                if (data.heads.depth) document.getElementById('img-depth').src = data.heads.depth;
                if (data.heads.mask) document.getElementById('img-mask').src = data.heads.mask;
                
                if (data.heads.brightfield) {
                    document.getElementById('img-brightfield').src = data.heads.brightfield;
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
            img.src = imgSrc;
            img.style.display = 'block';
            
            // Seed
            if (data.seed_used !== undefined) currentSeed = data.seed_used;
            
            // OBBs
            if (data.obbs) {
                drawObbs(data.obbs, data.width, data.height);
                update3DScene(data.obbs, data.width, data.height);
            }
        } else {
            // GT View - Just show optical/default
            const img = document.getElementById(targetImgId);
            img.src = imgSrc;
            img.style.display = 'block';
        }

        return data;
    } else {
        throw new Error(data.error);
    }
}

async function regenerate() {
    const status = document.getElementById('statusText');
    status.textContent = 'Generating...';
    
    try {
        const synthConfig = getConfig();
        
        // 1. Generate Synthetic (Left)
        const p1 = fetchPreview(synthConfig, currentSeed, 'mainImage', false);
        
        // 2. Generate GT (Right) if in Validate Mode
        let p2 = Promise.resolve(null);
        if (isValidateMode) {
             const gtConfig = applyOverrides(synthConfig, gtOverrides);
             // We use the same seed for GT to ensure fair comparison
             // unless we want to test robustness against seed variance?
             // Usually for parameter validation, we want same seed.
             p2 = fetchPreview(gtConfig, currentSeed, 'compareImageDisplay', true);
        }

        const [data1, data2] = await Promise.all([p1, p2]);
        
        status.textContent = `Ready (${data1.width}x${data1.height})`;
        status.classList.remove('text-danger');
        status.classList.add('text-success');
        updateMetrics(data1);

    } catch (e) {
        showError(e.message, e.stack);
    }
}

async function regenerateOld() {
    const status = document.getElementById('statusText');
    const img = document.getElementById('mainImage');
    
    status.textContent = 'Generating...';
    
    try {
        const config = getConfig();
        const payload = { 
            t: 0.5, 
            config: config, 
            return_heads: true,
            return_obbs: true
        };
        
        if (currentSeed !== null) {
            payload.seed = currentSeed;
        }

        const res = await fetch('/synth_preview', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        if (data.ok) {
            // 1. Update all head sources first (hidden thumbnails)
            if (data.heads) {
                if (data.heads.optical) document.getElementById('img-optical').src = data.heads.optical;
                else if (data.image_b64) document.getElementById('img-optical').src = data.image_b64; // Fallback
                
                if (data.heads.height) document.getElementById('img-height').src = data.heads.height;
                if (data.heads.depth) document.getElementById('img-depth').src = data.heads.depth;
                if (data.heads.mask) document.getElementById('img-mask').src = data.heads.mask;
                
                // Aux heads
                if (data.heads.brightfield) {
                    document.getElementById('img-brightfield').src = data.heads.brightfield;
                    document.getElementById('thumb-brightfield').style.display = 'block';
                } else {
                    document.getElementById('thumb-brightfield').style.display = 'none';
                }
            } else {
                // Legacy fallback if no heads dict
                document.getElementById('img-optical').src = data.image_b64;
            }

            // 2. Determine currently active head
            const activeThumb = document.querySelector('.head-thumb.active');
            let activeType = 'optical'; // Default
            if (activeThumb) {
                // ID is "thumb-optical", "thumb-height", etc.
                activeType = activeThumb.id.replace('thumb-', '');
            }

            // 3. Update Main Image based on ACTIVE head
            // If active head is '3d', we don't update mainImage src (it's hidden)
            if (activeType !== '3d') {
                const activeSrcEl = document.getElementById(`img-${activeType}`);
                if (activeSrcEl && activeSrcEl.src) {
                    img.src = activeSrcEl.src;
                    img.style.display = 'block';
                } else {
                    // Fallback if active head not found (e.g. switched modes and head gone)
                    img.src = data.image_b64;
                    img.style.display = 'block';
                    // Reset selection to optical
                    document.querySelectorAll('.head-thumb').forEach(t => t.classList.remove('active'));
                    document.getElementById('thumb-optical').classList.add('active');
                }
            }
            
            // Capture seed used by backend if we didn't have one
            if (data.seed_used !== undefined) {
                currentSeed = data.seed_used;
            }

            // Draw OBBs
            if (data.obbs) {
                drawObbs(data.obbs, data.width, data.height);
                update3DScene(data.obbs, data.width, data.height);
            }
            
            status.textContent = `Ready (${data.width}x${data.height})`;
            status.classList.remove('text-danger');
            status.classList.add('text-success');
            updateMetrics(data); // if backend sends meta
        } else {
            showError(data.error, data.traceback);
        }
    } catch (e) {
        showError(e.message, e.stack);
    }
}

function drawObbs(obbs, w, h) {
    const canvas = document.getElementById('obbCanvas');
    const container = document.getElementById('canvasContainer');
    
    // Canvas should match image display size? No, canvas is absolute overlay
    // We need to sync canvas size to displayed image size
    const img = document.getElementById('mainImage');
    
    // Wait for image to layout
    requestAnimationFrame(() => {
        const rect = img.getBoundingClientRect();
        const parentRect = container.getBoundingClientRect();
        
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.left = (rect.left - parentRect.left) + 'px';
        canvas.style.top = (rect.top - parentRect.top) + 'px';
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const scaleX = rect.width / w;
        const scaleY = rect.height / h;
        
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.7)';
        ctx.lineWidth = 1;
        
        obbs.forEach(ob => {
            const cs = ob.corners;
            ctx.beginPath();
            ctx.moveTo(cs[0][0] * scaleX, cs[0][1] * scaleY);
            for(let i=1; i<4; i++) ctx.lineTo(cs[i][0] * scaleX, cs[i][1] * scaleY);
            ctx.closePath();
            ctx.stroke();
        });
    });
}

function toggleObb() {
    const cvs = document.getElementById('obbCanvas');
    cvs.style.display = cvs.style.display === 'none' ? 'block' : 'none';
}

function switchHead(type) {
    document.querySelectorAll('.head-thumb').forEach(t => t.classList.remove('active'));
    document.getElementById(`thumb-${type}`).classList.add('active');
    
    const main = document.getElementById('mainImage');
    const obbCvs = document.getElementById('obbCanvas');
    
    if (type === '3d') {
        if (!is3DInit) init3DViewer();
        if (renderer3d) renderer3d.domElement.style.display = 'block';
        main.style.display = 'none';
        obbCvs.style.display = 'none';
        
        // Trigger resize just in case
        onWindowResize();
        return;
    }
    
    // Hide 3D
    if (renderer3d) renderer3d.domElement.style.display = 'none';
    
    // If switching back to optical, ensure main image is visible if source is available
    if (type === 'optical') {
        const imgOptical = document.getElementById('img-optical');
        if (imgOptical && imgOptical.src && imgOptical.src.startsWith('data:')) {
            main.src = imgOptical.src;
            main.style.display = 'block';
            
            // Restore OBB canvas if it was enabled before?
            // For now, let's keep it hidden unless toggled, or maybe check user preference.
            // But main image MUST be block.
        }
        return;
    }
    
    const src = document.getElementById(`img-${type}`).src;
    if (src && src.startsWith('data:')) {
        main.src = src;
        main.style.display = 'block';
    }
}

// ------------------------------------------------------------------
// Presets Management
// ------------------------------------------------------------------

async function loadPresets() {
    try {
        const res = await fetch('/synth_presets');
        const data = await res.json();
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
    } catch (e) {
        console.error("Failed to load presets", e);
    }
}

async function loadSelectedPreset() {
    const sel = document.getElementById('presetSelector');
    const name = sel.value;
    if (!name) return;
    
    try {
        const res = await fetch(`/synth_get_preset?name=${encodeURIComponent(name)}`);
        const data = await res.json();
        if (data.ok) {
            applyConfigToUI(data.config);
            showToast(`Loaded preset: ${name}`);
            regenerate();
        } else {
            showToast(`Error: ${data.error}`);
        }
    } catch (e) {
        showToast(`Error: ${e.message}`);
    }
}

async function deleteSelectedPreset() {
    const sel = document.getElementById('presetSelector');
    const name = sel.value;
    if (!name) return;
    
    if (!confirm(`Delete preset "${name}"?`)) return;
    
    try {
        const res = await fetch(`/synth_delete_preset/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.ok) {
            showToast(`Deleted preset: ${name}`);
            loadPresets();
        } else {
            showToast(`Error: ${data.error}`);
        }
    } catch (e) {
        showToast(`Error: ${e.message}`);
    }
}

function savePresetPrompt() {
    let name = prompt("Enter preset name:");
    if (name) {
        const config = getConfig();
        fetch('/synth_save_preset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name, config })
        }).then(r => r.json()).then(data => {
            if(data.ok) {
                showToast('Saved!');
                loadPresets();
            } else {
                showToast('Error: ' + data.error);
            }
        });
    }
}

// ------------------------------------------------------------------
// Batch Job Queue
// ------------------------------------------------------------------

async function submitBatchJob() {
    const count = parseInt(document.getElementById('batchCount').value) || 100;
    const tasks = parseInt(document.getElementById('batchTasks').value) || 4;
    const outDir = document.getElementById('batchOutDir').value.trim();
    
    const config = getConfig();
    
    const payload = {
        config: config,
        n_images: count,
        n_tasks: tasks
    };
    if (outDir) payload.out_dir = outDir;
    
    try {
        const res = await fetch('/synth_batch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.ok) {
            showToast(`Job Submitted! ID: ${data.job_id}`);
            refreshJobs();
        } else {
            showToast(`Error: ${data.error}`);
        }
    } catch (e) {
        showToast(`Error: ${e.message}`);
    }
}

async function refreshJobs() {
    try {
        const res = await fetch('/synth_jobs');
        const data = await res.json();
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
                    <td><span title="${job.job_id}">${shortId}</span></td>
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
    } catch (e) {
        console.error("Failed to refresh jobs", e);
    }
}

async function deleteJob(jobId) {
    if (!confirm("Delete this job?")) return;
    try {
        await fetch(`/synth_delete_job/${jobId}`, { method: 'DELETE' });
        refreshJobs();
    } catch (e) {
        showToast(`Error: ${e.message}`);
    }
}

function showToast(msg) {
    // Simple alert for now or use bootstrap toast if html exists
    alert(msg);
}

// ------------------------------------------------------------------
// 3D Viewer
// ------------------------------------------------------------------

let scene3d, camera3d, renderer3d, controls3d, particlesGroup, focalPlaneMesh;
let is3DInit = false;

function init3DViewer() {
    if (is3DInit) return;
    
    const container = document.getElementById('canvasContainer');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    scene3d = new THREE.Scene();
    scene3d.background = new THREE.Color(0x1e1e1e); // Dark gray match

    // Camera
    camera3d = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
    camera3d.position.set(512, 500, 1500); // Back up a bit
    camera3d.lookAt(512, -512, 0);

    // Renderer
    renderer3d = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer3d.setSize(width, height);
    renderer3d.domElement.id = 'canvas3d';
    renderer3d.domElement.style.display = 'none'; // Hidden by default
    renderer3d.domElement.style.position = 'absolute';
    renderer3d.domElement.style.top = '0';
    renderer3d.domElement.style.left = '0';
    container.appendChild(renderer3d.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene3d.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(200, 500, 500);
    scene3d.add(dirLight);

    // Helpers
    // Grid on Z=0 plane (Focal Plane)
    // Three.js GridHelper is XZ. Rotate to XY.
    // We map Image Y -> Three -Y.
    const gridHelper = new THREE.GridHelper(2000, 20, 0x444444, 0x222222);
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.set(512, -512, 0); 
    scene3d.add(gridHelper);
    
    // Focal Plane Visualizer
    const planeGeo = new THREE.PlaneGeometry(1024, 1024);
    const planeMat = new THREE.MeshBasicMaterial({ 
        color: 0x00ff00, 
        transparent: true, 
        opacity: 0.1, 
        side: THREE.DoubleSide,
        depthWrite: false
    });
    focalPlaneMesh = new THREE.Mesh(planeGeo, planeMat);
    focalPlaneMesh.position.set(512, -512, 0);
    scene3d.add(focalPlaneMesh);

    const axesHelper = new THREE.AxesHelper(100);
    scene3d.add(axesHelper);

    // Particles Container
    particlesGroup = new THREE.Group();
    scene3d.add(particlesGroup);

    // Controls
    if (typeof THREE.OrbitControls !== 'undefined') {
        controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
        controls3d.enableDamping = true;
        controls3d.dampingFactor = 0.05;
        controls3d.target.set(512, -512, 0); // Look at center of 1024x1024 area
    }

    // Resize listener
    window.addEventListener('resize', onWindowResize, false);

    // Animation Loop
    animate3D();
    
    is3DInit = true;
}

function onWindowResize() {
    if (!camera3d || !renderer3d) return;
    const container = document.getElementById('canvasContainer');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    camera3d.aspect = width / height;
    camera3d.updateProjectionMatrix();
    renderer3d.setSize(width, height);
}

function animate3D() {
    requestAnimationFrame(animate3D);
    if (controls3d) controls3d.update();
    if (renderer3d && scene3d && camera3d) {
        // Only render if visible
        if (renderer3d.domElement.style.display !== 'none') {
            renderer3d.render(scene3d, camera3d);
        }
    }
}

function update3DScene(obbs, imgW, imgH) {
    if (!is3DInit) init3DViewer();
    
    // Clear old particles
    while(particlesGroup.children.length > 0){ 
        particlesGroup.remove(particlesGroup.children[0]); 
    }

    if (!obbs || obbs.length === 0) return;
    
    // Update Focal Plane Z
    const focusZ = getVal('synFocusZ', 0.0);
    if (focalPlaneMesh) {
        focalPlaneMesh.position.z = focusZ;
    }

    obbs.forEach(ob => {
        // OSOG OBB has: cx, cy, z, L, W, H, angle_deg, beta, gamma
        // OSOG Coords: X right, Y down. Z depth?
        // Three.js: X right, Y up, Z depth (towards camera).
        
        // Map:
        // x -> x
        // y -> -y (invert Y)
        // z -> z
        
        const w = ob.L; // Length is along local X
        const h = ob.W; // Width is along local Y
        const d = ob.H || (ob.W * 0.1); // Thickness
        
        const geometry = new THREE.BoxGeometry(w, h, d);
        
        // Color based on shape or random?
        // Let's use a nice crystal color
        const material = new THREE.MeshStandardMaterial({ 
            color: 0x00aaff,
            roughness: 0.3,
            metalness: 0.1,
            transparent: true,
            opacity: 0.8
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        
        // Position
        mesh.position.set(ob.cx, -ob.cy, ob.z);
        
        // Color based on Z distance from focus
        // Green = In Focus, Red = Far
        const dist = Math.abs(ob.z - focusZ);
        const maxDist = 100.0; // approximate z-range
        const t = Math.min(dist / maxDist, 1.0);
        
        // Lerp Color: 0x00ff00 -> 0xff0000
        const r = Math.floor(t * 255);
        const g = Math.floor((1 - t) * 255);
        mesh.material.color.setRGB(r/255, g/255, 0.2);
        
        const deg2rad = Math.PI / 180.0;
        
        // Simple approx:
        // Z rotation = -angle (since Y flipped)
        mesh.rotation.z = -ob.angle_deg * deg2rad; 
        mesh.rotation.x = ob.beta * deg2rad; 
        mesh.rotation.y = ob.gamma * deg2rad; 

        particlesGroup.add(mesh);
    });
}

// ------------------------------------------------------------------
// Optics UI Logic
// ------------------------------------------------------------------

function updateOpticsControls(mode) {
    // Helper to show/hide by ID
    const toggle = (id, show) => {
        const el = document.getElementById(id);
        if (el) el.style.display = show ? 'block' : 'none';
    };

    // 1. Shadow Gain (DIC Only)
    toggle('groupShGain', mode === 'dic');

    // 2. Polarizer Angle
    toggle('groupPolAngle', false); 

    // 3. Light Direction (Brightfield, Blaze, PVM)
    const useLightDir = ['brightfield', 'blaze'].includes(mode);
    toggle('groupLightDir', useLightDir);

    // 4. Focus Z (Only if DoF is enabled)
    const dof = document.getElementById('synDofEnable');
    toggle('groupFocusZ', dof && dof.checked);
}

// ------------------------------------------------------------------
// Optimization Logic (Validate Mode)
// ------------------------------------------------------------------

let isValidateMode = false;
let optimizationJobId = null;
let optimizationTimer = null;

function toggleValidateMode() {
    isValidateMode = !isValidateMode;
    
    const btn = document.getElementById('btnValidateMode');
    const rightSidebar = document.getElementById('gtSidebar');
    const compareContainer = document.getElementById('compareContainer');
    
    if(btn) {
        if(isValidateMode) {
            btn.classList.add('active', 'btn-info');
            btn.classList.remove('btn-outline-info');
            document.body.classList.add('validate-mode');
            
            // Show GT Sidebar
            if(rightSidebar) rightSidebar.style.display = 'flex';
            if(compareContainer) compareContainer.style.display = 'flex'; // Show side by side
            
            // Reset GT Overrides
            gtOverrides = {};
            
            // Auto-load params and Build GT Tuner
            buildGTTuner();
            
            // Trigger Dual Regen
            regenerate();
            
        } else {
            btn.classList.remove('active', 'btn-info');
            btn.classList.add('btn-outline-info');
            document.body.classList.remove('validate-mode');
            
            if(rightSidebar) rightSidebar.style.display = 'none';
            if(compareContainer) compareContainer.style.display = 'none';
            
            // Revert to single view regen
            regenerate();
        }
    }
}

async function buildGTTuner() {
    const container = document.getElementById('gtControlsContent');
    
    // Ensure params are loaded
    if(Object.keys(optimizationRules).length === 0) {
        await loadOptimizationParams();
    }
    
    const params = optimizationRules;
    
    if(Object.keys(params).length > 0) {
        // Initial Build (Run Once)
        if (container.children.length > 0) { // Naive check: if content exists, update
             // But if we want to re-order groups, we might need to rebuild if structure changed?
             // Since groups are static constant, structure is stable.
             // However, if we didn't use groups before, we need to rebuild NOW to apply groups.
             // Let's assume we need to rebuild at least once to apply the new grouping logic.
             // To be safe, let's clear and rebuild.
             // Wait, if we rebuild, we lose scroll position and maybe focus?
             // But this function is called on toggleValidateMode, so it's fine.
             // Does regenerate call this? No.
             // So safe to rebuild.
        }
        
        container.innerHTML = '';
        
        // Group parameters
        const groups = {};
        Object.keys(params).forEach(key => {
            const g = getParamGroup(key);
            if (!groups[g]) groups[g] = [];
            groups[g].push(key);
        });
        
        // Render Groups
        // Order keys based on PARAM_GROUPS definition order
        const orderedGroups = Object.keys(PARAM_GROUPS).filter(g => groups[g]);
        // Add 'Other' if exists
        if (groups['Other']) orderedGroups.push('Other');
        
        orderedGroups.forEach(groupName => {
            // Group Header
            const groupHeader = document.createElement('div');
            groupHeader.className = 'text-uppercase text-secondary fw-bold small mt-3 mb-2 px-1 border-bottom border-dark';
            groupHeader.textContent = groupName;
            container.appendChild(groupHeader);
            
            // Render Items in this group
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
    // Wrapper
    const div = document.createElement('div');
    div.className = 'mb-2 pb-2 border-bottom border-secondary gt-tuner-item';
    div.id = `gt_item_${key}`;
    div.style.display = 'none'; // Hidden by default until value is valid
    
    // Header: Label + Link Button + Value Display
    const header = document.createElement('div');
    header.className = 'd-flex justify-content-between align-items-center mb-1';
    
    const labelGroup = document.createElement('div');
    labelGroup.className = 'd-flex align-items-center gap-2';
    
    // Link Button
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
    
    // Controls
    let inputContainer = document.createElement('div');
    
    // Determine type from meta or guess
    // We don't have value yet, so rely on meta.type
    // If meta.bounds is array of length 2 -> scalar range
    // If meta.type is 'bool' -> checkbox
    // What if it is array value (range tuple)? Meta usually doesn't say "tuple".
    // We might need to guess from initial config or just support scalar for now 
    // and rebuild if array detected later?
    // Let's assume scalar unless we know otherwise.
    // Actually, updateGTTunerValues will handle value updates. 
    // We just need structure. 
    // Problem: we don't know if we need 1 slider or 2 inputs until we see the value.
    // Solution: Create a generic container and let updateGTTunerValues populate inputs if missing?
    // Or just check initial config now.
    
    // We'll leave inputContainer empty and let updateGTTunerValues build inputs if needed?
    // That mixes concerns. 
    // Let's try to get initial value even if undefined to guess type.
    const currentConfig = getConfig();
    let val = getParamValue(currentConfig, key, meta);
    
    // If undefined, assume scalar float based on bounds
    if (val === undefined) {
         if (meta.bounds) val = meta.bounds[0];
         else val = 0;
    }

    if (Array.isArray(val)) {
        // Range
        inputContainer.className = 'd-flex gap-1 align-items-center';
        
        // Min Input
        const minInput = document.createElement('input');
        minInput.type = 'number';
        minInput.className = 'form-control form-control-sm gt-input';
        minInput.dataset.key = key;
        minInput.dataset.index = 0;
        minInput.step = (meta.type === 'int') ? 1 : 0.1;
        minInput.placeholder = 'Min';
        minInput.onchange = (e) => handleGTChange(key, parseFloat(e.target.value), 0);
        
        // Separator
        const sep = document.createElement('span');
        sep.className = 'text-muted small';
        sep.textContent = '-';
        
        // Max Input
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
        // Scalar
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

function updateGTTunerValues() {
    // Read config, apply overrides
    const synthConfig = getConfig();
    const gtConfig = applyOverrides(synthConfig, gtOverrides); // This uses paramLinks logic
    
    Object.keys(optimizationRules).forEach(key => {
        const meta = optimizationRules[key];
        const val = getParamValue(gtConfig, key, meta);
        
        const itemDiv = document.getElementById(`gt_item_${key}`);
        
        // Show/Hide based on availability
        if (val === undefined) {
             // Inactive
             if(itemDiv) {
                 itemDiv.style.display = 'none'; // Or move to bottom and dim?
                 // User asked: "not activated are at the bottom clearly stated as not activated"
                 // To move to bottom, we need flex order or physical move.
                 itemDiv.style.order = '9999';
                 itemDiv.style.display = 'block';
                 itemDiv.classList.add('opacity-50');
                 // Disable inputs
                 itemDiv.querySelectorAll('input').forEach(i => i.disabled = true);
                 
                 // Update badge to "N/A"
                 const badge = document.getElementById(`gt_val_${key}`);
                 if(badge) badge.textContent = "Inactive";
             }
             return;
        }
        
        // Active
        if(itemDiv) {
            itemDiv.style.display = 'block';
            itemDiv.style.order = '0';
            itemDiv.classList.remove('opacity-50');
            itemDiv.querySelectorAll('input').forEach(i => i.disabled = false);
        }

        // Update Badge
        const badge = document.getElementById(`gt_val_${key}`);
        if(badge) badge.textContent = formatVal(val);
        
        // Update Inputs
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
        
        // Update Link Icon
        const linkBtn = document.getElementById(`btn_link_${key}`);
        if(linkBtn) {
            linkBtn.innerHTML = paramLinks[key] ? '<i class="bi bi-link text-success"></i>' : '<i class="bi bi-link-45deg text-warning"></i>';
        }
    });
}

function toggleParamLink(key) {
    if (paramLinks[key]) {
        // Break Link
        paramLinks[key] = false;
        // Capture current synthetic value as override base
        const currentConfig = getConfig();
        const meta = optimizationRules[key];
        const val = getParamValue(currentConfig, key, meta);
        
        // Store deeply copy
        if(Array.isArray(val)) gtOverrides[key] = [...val];
        else gtOverrides[key] = val;
        
    } else {
        // Restore Link
        paramLinks[key] = true;
        delete gtOverrides[key];
    }
    
    // Refresh UI
    updateGTTunerValues();
    updateOptimizationListValues();
    regenerate();
}

function handleGTChange(key, value, index = null) {
    // User manually changed a GT input -> Break Link automatically
    if (paramLinks[key]) {
        paramLinks[key] = false;
        // The value passed is the NEW value, so we just set override
    }
    
    if (index !== null) {
        if (!gtOverrides[key]) {
             // Need to init array from current config if not exists
             const currentConfig = getConfig();
             const meta = optimizationRules[key];
             const val = getParamValue(currentConfig, key, meta);
             gtOverrides[key] = [...val];
        }
        gtOverrides[key][index] = value;
    } else {
        gtOverrides[key] = value;
    }
    
    // Update Badge
    const badge = document.getElementById(`gt_val_${key}`);
    if(badge) {
        if (index !== null) badge.textContent = formatVal(gtOverrides[key]);
        else badge.textContent = formatVal(value);
    }
    
    // Refresh Link Icon
    const linkBtn = document.getElementById(`btn_link_${key}`);
    if(linkBtn) linkBtn.innerHTML = '<i class="bi bi-link-45deg text-warning"></i>';
    
    // Auto-update Optimization List
    updateOptimizationListValues();
    
    // Regenerate
    regenerate();
}

async function loadOptimizationParams(skipFetch = false) {
    const list = document.getElementById('optParamsList');
    
    // Only fetch if rules not loaded
    if(Object.keys(optimizationRules).length === 0 && !skipFetch) {
        list.innerHTML = '<div class="text-center text-muted small mt-2">Loading params...</div>';
        try {
            const res = await fetch('/calibration/params');
            const data = await res.json();
            if(data.ok && data.params) {
                optimizationRules = data.params;
                initParamLinks(); // Init links once rules are loaded
            }
        } catch(e) {
            console.error(e);
            return;
        }
    }
    
    if(Object.keys(optimizationRules).length > 0) {
        // If list is already built, just update values (Lightweight update)
        if (list.children.length > 0) {
            // Need to check if structure matches. 
            // If we introduced grouping to Optimization List too, we need rebuild.
            // Let's implement grouping for the Optimization List as well to match GT Tuner.
            // Since structure changes, we rebuild once.
            // But this function is called frequently? No, only once on load.
            // The lightweight update is 'updateOptimizationListValues'.
            // So we can rebuild here safely if it's the initial load or refresh.
            // Wait, loadOptimizationParams is called by 'scheduleRegenerate' in original code?
            // NO, we replaced that with updateOptimizationListValues.
            // So loadOptimizationParams is only called on startup or refresh.
        }
        
        list.innerHTML = '';
        const currentConfig = getConfig();
        
        // Group parameters
        const groups = {};
        Object.keys(optimizationRules).forEach(key => {
            const g = getParamGroup(key);
            if (!groups[g]) groups[g] = [];
            groups[g].push(key);
        });
        
        // Order keys based on PARAM_GROUPS definition order
        const orderedGroups = Object.keys(PARAM_GROUPS).filter(g => groups[g]);
        if (groups['Other']) orderedGroups.push('Other');
        
        orderedGroups.forEach(groupName => {
            // Group Header
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
                
                // Initial Value
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

// Helper to update values without rebuilding DOM
// Now takes an optional config parameter to allow forcing an update from backend state
function updateOptimizationListValues(forcedConfig = null) {
    const currentConfig = forcedConfig || getConfig();
    
    Object.keys(optimizationRules).forEach(key => {
        const meta = optimizationRules[key];
        const val = getParamValue(currentConfig, key, meta);
        const badge = document.getElementById(`badge_opt_${key}`);
        if(badge) {
            badge.textContent = formatVal(val);
        }
        
        // Auto-check if unlinked (diverged)
        const chk = document.getElementById(`chk_opt_${key}`);
        if (chk) {
            if (paramLinks[key] === false) {
                if (!chk.checked) chk.checked = true;
                chk.disabled = true; // Lock it if forced by override? Or just let user uncheck?
                // Actually, if it's overridden in GT, it MUST be optimized or the divergence makes no sense.
                // But user might want to temporarily disable optimization for it.
                // Let's just auto-check but allow uncheck.
                chk.disabled = false;
            }
        }
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

function formatVal(val) {
    if (val === undefined) return '';
    if (typeof val === 'number') return val.toFixed(2);
    if (Array.isArray(val)) return `[${val.join(', ')}]`;
    return String(val);
}

async function startOptimization() {
    // 1. Handle Validate Mode Auto-Generation
    if (isValidateMode) {
        const status = document.getElementById('optStatus');
        status.textContent = 'Generating Target...';
        
        try {
            const synthConfig = getConfig();
            const gtConfig = applyOverrides(synthConfig, gtOverrides);
            
            // Generate GT
            // Use same seed as current view for consistency
            const res = await fetchPreview(gtConfig, currentSeed, 'compareImageDisplay', true);
            
            // Convert Base64 to Blob
            const b64 = res.image_b64;
            // Check if prefix exists
            const base64Data = b64.includes(',') ? b64.split(',')[1] : b64;
            const byteString = atob(base64Data);
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            const blob = new Blob([ab], { type: 'image/jpeg' });
            const file = new File([blob], "validate_target.jpg", { type: "image/jpeg" });
            
            // Upload
            await uploadRefImage(file);
            
        } catch (e) {
            alert("Failed to generate/upload validation target: " + e.message);
            status.textContent = 'Error';
            return;
        }
    }

    if(!targetImageFilename) {
        alert("Please upload a target reference image first!");
        return;
    }
    
    // Gather selected params
    const selected = [];
    document.querySelectorAll('.opt-param-chk:checked').forEach(c => selected.push(c.value));
    
    if(selected.length === 0) {
        alert("Select at least one parameter to optimize.");
        return;
    }
    
    // UI Update
    document.getElementById('btnStartOpt').disabled = true;
    document.getElementById('btnStartOpt').classList.add('d-none');
    document.getElementById('btnStopOpt').classList.remove('d-none');
    document.getElementById('optStatus').textContent = 'Starting...';
    document.getElementById('optProgressBar').style.width = '0%';
    resetLossChart(); // Reset chart
    isOptimizing = true; // Lock UI from manual regens
    
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
        const res = await fetch('/calibration/start', {method: 'POST', body: fd});
        const data = await res.json();
        
        if(data.ok) {
            optimizationJobId = data.job_id;
            document.getElementById('optStatus').textContent = 'Running...';
            // Start polling
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

function stopOptimization() {
    if(optimizationJobId) {
        fetch(`/calibration/stop/${optimizationJobId}`, {method: 'POST'});
        // We will let the polling loop catch the "stopped" state naturally
        // to ensure we get the final config back from the backend.
        document.getElementById('optStatus').textContent = 'Stopping...';
    }
}

async function checkOptimizationStatus() {
    if(!optimizationJobId) return;
    
    try {
        const res = await fetch(`/calibration/status/${optimizationJobId}`);
        const data = await res.json();
        
        if(data.ok && data.status) {
            const s = data.status;
            // Map backend keys to frontend expected keys
            const state = s.status; // "initializing", "running", "finished", "stopped", "error"
            const step = s.step || 0;
            const max_steps = s.max_steps || 200;
            const current_loss = s.loss;
            const current_config = s.current_params; // backend returns physical values
            
            // Update UI
            const pct = max_steps > 0 ? (step / max_steps) * 100 : 0;
            document.getElementById('optProgressBar').style.width = `${pct}%`;
            document.getElementById('optStatus').textContent = `${state} (${step}/${max_steps})`;
            
            if(current_loss !== undefined && current_loss !== 0 && current_loss !== null) {
                document.getElementById('optLoss').textContent = `Loss: ${current_loss.toFixed(4)}`;
                
                // Show Floating Chart Dashboard
                const dashboard = document.getElementById('optFloatingDashboard');
                if (dashboard.style.display === 'none') {
                    dashboard.style.display = 'block';
                    if (lossChart) lossChart.resize();
                }
                
                // Init chart if not exists
                if(!lossChart) {
                    initLossChart();
                }
                
                // Only add point if it's new
                const lastStep = lossData.labels[lossData.labels.length - 1];
                if (lastStep !== step) {
                    lossData.labels.push(step);
                    lossData.datasets[0].data.push(current_loss);
                    lossChart.update();
                }
            }
            
            // Live Feedback: Update Synthetic Config
            // backend 'current_params' is a flat dict of {'physics.rod_specs.count_range': val, ...}
            // We need to convert it back to a nested config structure to use applyConfigToUI
            // Wait, applyConfigToUI expects a full nested config object.
            // Let's create a helper to merge flat params into current config.
            if (current_config && Object.keys(current_config).length > 0) {
                  const fullConfig = getConfig(); // get current base
                  // We need to build an override object that matches applyOverrides expectations.
                  // applyOverrides expects { 'physics.rod_specs.count_range': val }
                  // backend returns { 'physics.rod_specs.count_range': val }
                  // BUT, applyOverrides checks paramLinks. We WANT to force this update regardless of links!
                  // Let's write a bypass to update the config directly.
                  const mergedConfig = JSON.parse(JSON.stringify(fullConfig));
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
                  updateUIFromConfig(mergedConfig);
                  
                  // Update text badges and synthetic image every 10 steps (or on step 1)
                  if (step % 5 === 0 || step === 1) {
                      if (isValidateMode) {
                          // Update badges manually since we unhooked them from updateUIFromConfig
                          updateOptimizationListValues(mergedConfig);
                      }
                      // Fetch new preview using updated config without resetting target
                      // We don't use scheduleRegenerate() here to avoid debouncing loops, 
                      // just fire it directly for the main image.
                      fetchPreview(mergedConfig, currentSeed, 'mainImage', false);
                  }
             }
            
            // If finished or stopped
            if(['finished', 'stopped', 'error', 'failed'].includes(state)) {
                clearInterval(optimizationTimer);
                optimizationTimer = null;
                optimizationJobId = null;
                isOptimizing = false; // Unlock UI
                
                document.getElementById('btnStopOpt').classList.add('d-none');
                document.getElementById('btnStartOpt').disabled = false;
                document.getElementById('btnStartOpt').classList.remove('d-none');
                
                if(state === 'finished' || state === 'stopped') {
                     document.getElementById('btnApplyOpt').classList.remove('d-none');
                     
                     // If we have final params, apply them
                     if(current_config && Object.keys(current_config).length > 0) {
                         const fullConfig = getConfig();
                         const finalConfig = applyOverrides(fullConfig, current_config);
                         applyConfigToUI(finalConfig);
                         regenerate();
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

// Helper to update UI sliders from a config object (reverse of applyConfigToUI but lighter)
function updateUIFromConfig(config) {
    // We assume applyConfigToUI exists and works.
    // It updates the Designer tab.
    // Since GT is linked to Designer (unless overridden), GT sliders will also update 
    // if we call updateGTTunerValues() after.
    
    applyConfigToUI(config);
    
    // Auto-update visual numbers
    updateMetrics(config); // if needed, though this expects backend response
    
    // Update GT Tuner to reflect new synthetic values (for linked params)
    if(isValidateMode) {
        updateGTTunerValues();
        // Don't update optimization list values here to avoid breaking checkboxes
        // updateOptimizationListValues(); 
    }
}

function applyOptimizationResults() {
    // This is just a manual trigger if we didn't auto-apply, 
    // but we auto-applied in the checkOptimizationStatus.
    // We can hide the button now.
    document.getElementById('btnApplyOpt').classList.add('d-none');
}

function resetOptUI() {
    document.getElementById('btnStartOpt').disabled = false;
    document.getElementById('btnStartOpt').classList.remove('d-none');
    document.getElementById('btnStopOpt').classList.add('d-none');
    document.getElementById('optStatus').textContent = 'Ready';
    document.getElementById('optProgressBar').style.width = '0%';
}

async function evaluateLoss() {
     if(!targetImageFilename) {
        alert("Please upload a target reference image first!");
        return;
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
        const res = await fetch('/calibration/compute_loss', {method: 'POST', body: fd});
        const data = await res.json();
        
        const resDiv = document.getElementById('lossResults');
        if(data.ok) { // calibration_manager returns dict directly usually? No, check main.py
            // main.py returns result of compute_loss
            // It typically returns {mean_loss: ..., std_loss: ...}
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

function confirmExitValidateMode() {
    // Logic to clear validation state if needed
    toggleValidateMode(); // Toggle back off
}


// ------------------------------------------------------------------
// Initialization
// ------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // Attach listeners to all inputs for live update
    document.querySelectorAll('input, select').forEach(el => {
        if (el.type === 'range' || el.type === 'number' || el.type === 'checkbox' || el.tagName === 'SELECT') {
            el.addEventListener('input', (e) => {
                // Don't trigger if it's an optimization parameter checkbox
                if(e.target.classList.contains('opt-param-chk')) return;

                // Special handling for Optics Mode change
                if (e.target.id === 'synOpticsMode' || e.target.id === 'synDofEnable') {
                    updateOpticsControls(document.getElementById('synOpticsMode').value);
                }
                
                // Update label immediately
                updateLabelFor(e.target.id, e.target.value);
                // Schedule regenerate
                scheduleRegenerate();
            });
        }
    });
    
    // Regenerate button
    document.getElementById('btnRegenerate').onclick = () => {
        currentSeed = null; // Force new seed
        regenerate();
    };
    
    // Spacebar to regenerate
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
            e.preventDefault();
            currentSeed = null; // Force new seed
            regenerate();
        }
    });

    loadPresets();
    refreshJobs();
    setupDragDrop();
    loadOptimizationParams(); // Load optimization params on startup
    
    // Initial generation
    regenerate();
    
    // Auto-refresh jobs
    setInterval(refreshJobs, 5000);
});
