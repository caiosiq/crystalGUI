import { getVal, getInt, getChk, setVal, setChk } from './utils.js';
import { updateOpticsControls } from './ui.js';

export function getConfig() {
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

export function applyConfigToUI(p) {
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
