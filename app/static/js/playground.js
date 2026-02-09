
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
        'synBubbleAttach': 'lblBubbleAttach',
        'synFoulingProb': 'lblFoulingProb',
        'synFoulingOp': 'lblFoulingOp',
        'synPolyIrreg': 'lblPolyIrreg'
    };
    if (map[id]) {
        const lbl = document.getElementById(map[id]);
        if (lbl) {
            // formatting
            if (id === 'synPolarizerAngle' || id === 'synFlowDir') lbl.textContent = Math.round(val) + '°';
            else if (id === 'synFocusZ') lbl.textContent = parseFloat(val).toFixed(1);
            else if (id === 'synShGain') {
                 // Context-aware label for Gain
                 const mode = document.getElementById('synOpticsMode').value;
                 if (mode === 'pvm') lbl.textContent = val + ' (Laser Power)';
                 else lbl.textContent = val;
            }
            else lbl.textContent = val;
        }
    }
}

// ------------------------------------------------------------------
// Validation & Metrics
// ------------------------------------------------------------------

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
        document.getElementById('compareImage').src = e.target.result;
        
        document.getElementById('dropZone').style.display = 'none';
        document.getElementById('refImageContainer').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearRefImage() {
    document.getElementById('refImage').src = '';
    document.getElementById('compareImage').src = '';
    document.getElementById('compareImage').style.display = 'none';
    
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('refImageContainer').style.display = 'none';
    
    // Reset comparison mode if active
    isCompareMode = false;
    document.getElementById('btnCompareMode').classList.remove('active');
}

let isCompareMode = false;
function toggleComparisonMode() {
    isCompareMode = !isCompareMode;
    const btn = document.getElementById('btnCompareMode');
    const cmpImg = document.getElementById('compareImage');
    const mainImg = document.getElementById('mainImage');
    
    if (isCompareMode) {
        btn.classList.add('active');
        if (cmpImg.src && cmpImg.src !== window.location.href) {
            cmpImg.style.display = 'block';
            cmpImg.style.width = '50%';
            cmpImg.style.left = '0';
            
            mainImg.style.position = 'absolute';
            mainImg.style.width = '50%';
            mainImg.style.left = '50%';
            mainImg.style.height = '100%';
            mainImg.style.objectFit = 'contain';
        } else {
            showToast("Load a reference image first!");
            isCompareMode = false;
            btn.classList.remove('active');
        }
    } else {
        btn.classList.remove('active');
        cmpImg.style.display = 'none';
        
        // Reset Main Image
        mainImg.style.position = 'static';
        mainImg.style.width = '';
        mainImg.style.left = '';
        mainImg.style.height = '';
        mainImg.style.objectFit = 'contain';
        mainImg.style.maxWidth = '95%';
        mainImg.style.maxHeight = '95%';
    }
}

function updateMetrics(data) {
    if (!data || !data.meta) return;
    const m = data.meta;
    
    // Count (mock logic if meta not fully populated, or read actuals)
    // The backend generate_image doesn't always return counts in meta, but let's assume it might
    let count = 0;
    if (m.rods) count += m.rods.count || 0;
    
    document.getElementById('metricCount').textContent = count > 0 ? count : '-';
}


// ------------------------------------------------------------------
// Config & Logic
// ------------------------------------------------------------------

function getConfig() {
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
                ragged_p: getVal('synRoughness'),
                polarity_p: getVal('synPolarity'),
                inclusions: getVal('synInclusions'),
                shape_mode: document.getElementById('synShapeMode').value,
                
                // Add texture-specific fields for TextureShader
                texture_type: document.getElementById('synTextureType').value,
                surf_roughness: getVal('synRoughness'),
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity')
            },

            sphere_specs: {
                enable: getChk('synSphereEnable'),
                count_range: [getInt('synSphereCountLo', 10), getInt('synSphereCountHi', 50)],
                diameter_range: [getInt('synSphereDiamLo', 20), getInt('synSphereDiamHi', 100)],
                material: document.getElementById('synSphereMaterial').value,
                
                // Propagate texture params to spheres too!
                texture_type: document.getElementById('synTextureType').value,
                surf_roughness: getVal('synRoughness'),
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity')
            },
            cube_specs: {
                enable: getChk('synCubeEnable'),
                count_range: [getInt('synCubeCountLo', 10), getInt('synCubeCountHi', 50)],
                size_range: [getInt('synCubeSizeLo', 20), getInt('synCubeSizeHi', 100)],
                material: document.getElementById('synCubeMaterial').value,
                
                // Propagate texture params
                texture_type: document.getElementById('synTextureType').value,
                surf_roughness: getVal('synRoughness'),
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity')
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
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity')
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
                internal_inclusions: getVal('synInclusions'),
                polarity_flip_p: getVal('synPolarity')
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
            focus_z: getVal('synFocusZ')
        },
        sensor: {
            bg_noise_std: getVal('synBgNoise'),
            blur_sigma: getVal('synBlur'),
            vignette_strength: getVal('synVignette'),
            chromatic_aberration_strength: getVal('synChromAb'),
            
            tilt_enable: getChk('synTiltEnable'),
            relief_field_enable: getChk('synReliefEnable'),
            
            // Fouling
            fouling_enable: getChk('synFoulingEnable'),
            fouling_prob: getVal('synFoulingProb'),
            fouling_count_range: [getInt('synFoulingCountLo', 1), getInt('synFoulingCountHi', 5)],
            fouling_opacity: getVal('synFoulingOp')
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
             if (rs.ragged_p !== undefined) setVal('synRoughness', rs.ragged_p);
             if (rs.polarity_p !== undefined) setVal('synPolarity', rs.polarity_p);
             if (rs.inclusions !== undefined) setVal('synInclusions', rs.inclusions);
             if (rs.shape_mode) document.getElementById('synShapeMode').value = rs.shape_mode;
             if (rs.texture_type) document.getElementById('synTextureType').value = rs.texture_type;
             if (rs.material) document.getElementById('synRodMaterial').value = rs.material;
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
        // Handle name change
        const ang = p.optics.polarizer_angle_deg !== undefined ? p.optics.polarizer_angle_deg : p.optics.polarizer_angle;
        setVal('synPolarizerAngle', ang || 0);
        
        // Handle Tuple vs Float for shadow_gain
        let sg = p.optics.shadow_gain;
        if (Array.isArray(sg)) sg = sg[0];
        setVal('synShGain', sg);
        
        setVal('synFocusZ', p.optics.focus_z || 0.0);
    }

    // Sensor
    if (p.sensor) {
        setVal('synBgNoise', p.sensor.bg_noise_std !== undefined ? p.sensor.bg_noise_std : p.sensor.noise);
        setVal('synBlur', p.sensor.blur_sigma !== undefined ? p.sensor.blur_sigma : p.sensor.blur);
        setVal('synVignette', p.sensor.vignette_strength !== undefined ? p.sensor.vignette_strength : p.sensor.vignette);
        setVal('synChromAb', p.sensor.chromatic_aberration_strength !== undefined ? p.sensor.chromatic_aberration_strength : p.sensor.chromatic_aberration);
        
        setChk('synTiltEnable', p.sensor.tilt_enable);
        setChk('synReliefEnable', p.sensor.relief_field_enable !== undefined ? p.sensor.relief_field_enable : p.sensor.relief_enable);
        
        // Fouling (Flattened in SensorConfig)
        if (p.sensor.fouling_enable !== undefined) setChk('synFoulingEnable', p.sensor.fouling_enable);
        if (p.sensor.fouling_prob !== undefined) setVal('synFoulingProb', p.sensor.fouling_prob);
        if (p.sensor.fouling_opacity !== undefined) setVal('synFoulingOp', p.sensor.fouling_opacity);
    }
}

// ------------------------------------------------------------------
// Generation & Preview
// ------------------------------------------------------------------

let debounceTimer = null;
let currentSeed = null;

function scheduleRegenerate() {
    const status = document.getElementById('statusText');
    status.textContent = 'Changed...';
    
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        regenerate();
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

async function regenerate() {
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
                if (data.heads.pvm) {
                    document.getElementById('img-pvm').src = data.heads.pvm;
                    document.getElementById('thumb-pvm').style.display = 'block';
                } else {
                    document.getElementById('thumb-pvm').style.display = 'none';
                }
                
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

let scene3d, camera3d, renderer3d, controls3d, particlesGroup;
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
    camera3d = new THREE.PerspectiveCamera(45, width / height, 0.1, 5000);
    camera3d.position.set(500, 500, 1000);
    camera3d.lookAt(512, 512, 0);

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
    const gridHelper = new THREE.GridHelper(2000, 20, 0x444444, 0x222222);
    // Grid usually on XZ plane, but our image is XY. Let's rotate grid to match XY?
    // Actually, let's keep standard 3D orientation: Y is up.
    // We map Image (x, y) -> 3D (x, -y, z).
    // So grid should be on XY plane? No, usually grid is "ground".
    // Let's just put grid on Z=0 plane (XY plane in standard math, XZ in Three.js default).
    // Three.js GridHelper is on XZ plane.
    // If we map Image Y to Three.js -Y, then we are looking at XY plane.
    // Let's rotate grid to be on XY plane.
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.set(512, -512, 0); // Center roughly
    scene3d.add(gridHelper);

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

    // Center camera if first load? Maybe not, keep user view.
    
    // Scale factor? We assume 1 unit = 1 pixel
    
    const geometryCache = {}; // Reuse geometries if possible? BoxGeometry is cheap.

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
        
        // Rotation
        // OSOG rotations are likely Intrinsic Z-Y'-X'' or similar.
        // angle_deg (alpha) is around Z (in image plane).
        // beta is tilt.
        // gamma is roll.
        
        // Let's try standard Euler ZYX order?
        // Note: Inverted Y axis might affect rotation direction.
        // If Y is inverted, rotation around Z (CW vs CCW) flips.
        
        const deg2rad = Math.PI / 180.0;
        // In OSOG: angle is CCW from X axis? Or CW?
        // Usually image coords angle is CW.
        
        // Let's apply rotations.
        // Three.js Euler default is XYZ.
        // We might need to construct a quaternion.
        
        // Simple approx:
        // Z rotation = -angle (since Y flipped)
        mesh.rotation.z = -ob.angle_deg * deg2rad; 
        mesh.rotation.x = ob.beta * deg2rad; 
        mesh.rotation.y = ob.gamma * deg2rad; 

        particlesGroup.add(mesh);
    });
}

// ------------------------------------------------------------------
// Initialization
// ------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // Attach listeners to all inputs for live update
    document.querySelectorAll('input, select').forEach(el => {
        if (el.type === 'range' || el.type === 'number' || el.type === 'checkbox' || el.tagName === 'SELECT') {
            el.addEventListener('input', (e) => {
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
    
    // Initial generation
    regenerate();
    
    // Auto-refresh jobs
    setInterval(refreshJobs, 5000);
});
