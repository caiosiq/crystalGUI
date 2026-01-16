// OSOG Lab - Playground Logic

let currentHeads = {};
let currentObbs = [];
let activeHead = 'optical';
let showObbs = false;
let isPending = false;
let needsUpdate = false;
let debounceTimer = null;
let presetsConstraints = {};

// ID Mapping for Validation
const idToConstraint = {
  // Rods
  'synRodLenLo': { comp: 'rod', param: 'length_range' },
  'synRodLenHi': { comp: 'rod', param: 'length_range' },
  'synRodAspLo': { comp: 'rod', param: 'aspect_range' },
  'synRodAspHi': { comp: 'rod', param: 'aspect_range' },
  'synRodCountLo': { comp: 'rod', param: 'count_range' },
  'synRodCountHi': { comp: 'rod', param: 'count_range' },
  
  // Spheres
  'synSphereDiamLo': { comp: 'sphere', param: 'diameter_range' },
  'synSphereDiamHi': { comp: 'sphere', param: 'diameter_range' },
  'synSphereCountLo': { comp: 'sphere', param: 'count_range' },
  'synSphereCountHi': { comp: 'sphere', param: 'count_range' },

  // Bubbles
  'synBubbleDiamLo': { comp: 'bubble', param: 'diameter_range' },
  'synBubbleDiamHi': { comp: 'bubble', param: 'diameter_range' },
  
  // Optics
  'synShGain': { comp: 'dic', param: 'shadow_gain' }
};

// Initial Load
document.addEventListener('DOMContentLoaded', async () => {
  // Load Constraints first
  await loadConstraints();
  // Load Defaults next
  await loadDefaults();

  // Attach listeners to all inputs
  document.querySelectorAll('input, select').forEach(el => {
    // Validate on change (commit)
    el.addEventListener('change', (e) => {
        validateInput(e.target);
        scheduleUpdate();
    });
    
    // Live update (no validation to allow typing)
    el.addEventListener('input', () => {
      updateLabels();
      scheduleUpdate();
    });
  });
  
  // Specific buttons
  document.getElementById('btnRegenerate').addEventListener('click', () => {
    currentSeed = Math.floor(Math.random() * 2000000000);
    scheduleUpdate();
  });

  // Initial update
  updateLabels();
  scheduleUpdate();
});

async function loadDefaults() {
    try {
        const res = await fetch('/synth_default_config');
        const data = await res.json();
        if (data.ok && data.config) {
            applyConfigToUI(data.config);
            console.log("Loaded Defaults:", data.source);
        }
    } catch (e) {
        console.error("Failed to load defaults", e);
    }
}

function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) {
        el.value = val;
    }
}

function setChk(id, val) {
    const el = document.getElementById(id);
    if (el) {
        el.checked = val;
    }
}

function applyConfigToUI(cfg) {
    const p = cfg.physics || {};
    
    // Rods
    if (p.rod_specs) {
        setChk('synRodEnable', p.rod_specs.enable);
        if (p.rod_specs.count_range) {
            setVal('synRodCountLo', p.rod_specs.count_range[0]);
            setVal('synRodCountHi', p.rod_specs.count_range[1]);
        }
        if (p.rod_specs.length_range) {
            setVal('synRodLenLo', p.rod_specs.length_range[0]);
            setVal('synRodLenHi', p.rod_specs.length_range[1]);
        }
        if (p.rod_specs.aspect_range) {
            setVal('synRodAspLo', p.rod_specs.aspect_range[0]);
            setVal('synRodAspHi', p.rod_specs.aspect_range[1]);
        }
        if (p.rod_specs.material) {
            setVal('synRodMaterial', p.rod_specs.material);
        }
        setVal('synRoughness', p.rod_specs.ragged_p);
        setVal('synPolarity', p.rod_specs.polarity_p);
        setVal('synShapeMode', p.rod_specs.shape_mode);
    }

    // Spheres
    if (p.sphere_specs) {
        setChk('synSphereEnable', p.sphere_specs.enable);
        if (p.sphere_specs.count_range) {
            setVal('synSphereCountLo', p.sphere_specs.count_range[0]);
            setVal('synSphereCountHi', p.sphere_specs.count_range[1]);
        }
        if (p.sphere_specs.diameter_range) {
            setVal('synSphereDiamLo', p.sphere_specs.diameter_range[0]);
            setVal('synSphereDiamHi', p.sphere_specs.diameter_range[1]);
        }
        if (p.sphere_specs.material) {
            setVal('synSphereMaterial', p.sphere_specs.material);
        }
    }

    // Cubes
    if (p.cube_specs) {
        setChk('synCubeEnable', p.cube_specs.enable);
        if (p.cube_specs.count_range) {
            setVal('synCubeCountLo', p.cube_specs.count_range[0]);
            setVal('synCubeCountHi', p.cube_specs.count_range[1]);
        }
        if (p.cube_specs.size_range) {
            setVal('synCubeSizeLo', p.cube_specs.size_range[0]);
            setVal('synCubeSizeHi', p.cube_specs.size_range[1]);
        }
        if (p.cube_specs.material) {
            setVal('synCubeMaterial', p.cube_specs.material);
        }
    }

    // Plates
    if (p.plate_specs) {
        setChk('synPlateEnable', p.plate_specs.enable);
        if (p.plate_specs.count_range) {
            setVal('synPlateCountLo', p.plate_specs.count_range[0]);
            setVal('synPlateCountHi', p.plate_specs.count_range[1]);
        }
        if (p.plate_specs.size_range) {
            setVal('synPlateSizeLo', p.plate_specs.size_range[0]);
            setVal('synPlateSizeHi', p.plate_specs.size_range[1]);
        }
        if (p.plate_specs.aspect_range) {
            setVal('synPlateAspLo', p.plate_specs.aspect_range[0]);
            setVal('synPlateAspHi', p.plate_specs.aspect_range[1]);
        }
        if (p.plate_specs.thickness_range) {
            setVal('synPlateThickLo', p.plate_specs.thickness_range[0]);
            setVal('synPlateThickHi', p.plate_specs.thickness_range[1]);
        }
        if (p.plate_specs.material) {
            setVal('synPlateMaterial', p.plate_specs.material);
        }
        
        // Physics 2.0
    }

    // Bubbles
    if (p.bubble_specs) {
        setChk('synBubbleEnable', p.bubble_specs.enable);
        if (p.bubble_specs.count_range) {
            setVal('synBubbleCountLo', p.bubble_specs.count_range[0]);
            setVal('synBubbleCountHi', p.bubble_specs.count_range[1]);
        }
        if (p.bubble_specs.diameter_range) {
            setVal('synBubbleDiamLo', p.bubble_specs.diameter_range[0]);
            setVal('synBubbleDiamHi', p.bubble_specs.diameter_range[1]);
        }
        if (p.bubble_specs.material) {
            setVal('synBubbleMaterial', p.bubble_specs.material);
        }
    }

    // Droplets
    if (p.droplet_specs) {
        setChk('synDropletEnable', p.droplet_specs.enable);
        if (p.droplet_specs.count_range) {
            setVal('synDropletCountLo', p.droplet_specs.count_range[0]);
            setVal('synDropletCountHi', p.droplet_specs.count_range[1]);
        }
        if (p.droplet_specs.diameter_range) {
            setVal('synDropletDiamLo', p.droplet_specs.diameter_range[0]);
            setVal('synDropletDiamHi', p.droplet_specs.diameter_range[1]);
        }
        if (p.droplet_specs.material) {
            setVal('synDropletMaterial', p.droplet_specs.material);
        }
    }

    // Ghosts
    if (p.ghosts) {
        setChk('synGhostEnable', p.ghosts.enable);
        setVal('synGhostFraction', p.ghosts.fraction);
    }

    // Debris
    if (p.debris) {
        setVal('synDebrisRate', p.debris.rate);
    }

    // Fused
    if (p.fused) {
        setVal('synAgglo', p.fused.p1);
    }
    
    // Optics
    const o = cfg.optics || {};
    setVal('synOpticsMode', o.mode);
    setVal('synPolarizerAngle', o.polarizer_angle_deg);
    if (o.shadow_gain) {
        setVal('synShGain', o.shadow_gain[0]); // Approx
    }

    // Sensor
    const s = cfg.sensor || {};
    setVal('synBgNoise', s.bg_noise_std);
    setVal('synBlur', s.blur_sigma);
    setVal('synVignette', s.vignette_strength);
    setChk('synTiltEnable', s.tilt_enable);
    setChk('synReliefEnable', s.relief_field_enable);
    setChk('synFoulingEnable', s.fouling_enable);
    setVal('synFoulingProb', s.fouling_prob);
    setVal('synFoulingOp', s.fouling_opacity);
    if (s.fouling_count_range) {
        setVal('synFoulingCountLo', s.fouling_count_range[0]);
        setVal('synFoulingCountHi', s.fouling_count_range[1]);
    }
}

async function loadConstraints() {
    try {
        const res = await fetch('/synth_constraints');
        const data = await res.json();
        if (data.ok) {
            presetsConstraints = data.constraints;
            console.log("Loaded Constraints:", presetsConstraints);
        }
    } catch (e) {
        console.error("Failed to load constraints", e);
    }
}

function showToast(message) {
    let toast = document.getElementById('error-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'error-toast';
        toast.style.position = 'fixed';
        toast.style.top = '20px';
        toast.style.left = '50%';
        toast.style.transform = 'translateX(-50%)';
        toast.style.backgroundColor = '#dc3545';
        toast.style.color = 'white';
        toast.style.padding = '10px 20px';
        toast.style.borderRadius = '5px';
        toast.style.zIndex = '10000';
        toast.style.boxShadow = '0 4px 6px rgba(0,0,0,0.3)';
        toast.style.transition = 'opacity 0.5s';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.opacity = '1';
    toast.style.display = 'block';
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => { toast.style.display = 'none'; }, 500);
    }, 3000);
}

function validateInput(el) {
    const map = idToConstraint[el.id];
    if (!map) return;
    
    const comp = presetsConstraints[map.comp];
    if (!comp) return;
    
    const param = comp[map.param];
    if (!param) return;
    
    let val = parseFloat(el.value);
    if (isNaN(val)) return;
    
    let clamped = val;
    let msg = "";
    
    if (param.hard_min !== null && val < param.hard_min) {
        clamped = param.hard_min;
        msg = `${map.comp} ${map.param} cannot be less than ${param.hard_min}`;
    } else if (param.hard_max !== null && val > param.hard_max) {
        clamped = param.hard_max;
        msg = `${map.comp} ${map.param} cannot be greater than ${param.hard_max}`;
    }
    
    if (clamped !== val) {
        el.value = clamped;
        showToast(`⚠️ Constraint Reached: ${msg}. Resetting to limit.`);
        updateLabels(); // Update label if it's a slider
    }
}

let currentSeed = Math.floor(Math.random() * 2000000000);

function updateLabels() {
  const labelMap = {
    'synPolarizerAngle': 'lblPolAngle',
    'synShGain': 'lblShGain',
    'synBgNoise': 'lblNoise',
    'synBlur': 'lblBlur',
    'synVignette': 'lblVignette',
    'synFoulingProb': 'lblFoulingProb',
    'synFoulingOp': 'lblFoulingOp',
    'synRoughness': 'lblRoughness',
    'synPolarity': 'lblPolarity',
    'synAgglo': 'lblAgglo'
  };
  
  Object.keys(labelMap).forEach(id => {
    const lblId = labelMap[id];
    const lbl = document.getElementById(lblId);
    const el = document.getElementById(id);
    if (lbl && el) {
      lbl.textContent = el.value + (id.includes('Angle') ? '°' : '');
    }
  });
}

function scheduleUpdate() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(performUpdate, 150); // 150ms debounce
}

async function performUpdate() {
  if (isPending) {
    needsUpdate = true;
    return;
  }
  
  isPending = true;
  document.getElementById('statusText').textContent = 'Rendering...';
  // document.getElementById('loadingOverlay').style.display = 'block'; // Too intrusive?
  // Just status text is better for "realtime" feel.

  const config = getConfig();
  
  try {
    const res = await fetch('/synth_preview', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        t: 0.5, // Center of time
        seed: currentSeed,
        config: config,
        return_obbs: true,
        return_heads: true,
        quality: 85
      })
    });
    
    const data = await res.json();
    if (data.ok) {
      // Store results
      currentHeads['optical'] = data.image_b64;
      if (data.heads) {
        Object.assign(currentHeads, data.heads);
      }
      currentObbs = data.obbs || [];
      
      // Update Thumbnails
      for (const k in currentHeads) {
        const thumb = document.getElementById(`img-${k}`);
        if (thumb) thumb.src = currentHeads[k];
      }
      
      // Update Main View
      updateMainView();
      
      document.getElementById('statusText').textContent = `Ready (${data.timings.total_s.toFixed(3)}s)`;
    } else {
      document.getElementById('statusText').textContent = 'Error: ' + data.error;
    }
  } catch (e) {
    document.getElementById('statusText').textContent = 'Error: ' + e.message;
  } finally {
    isPending = false;
    // document.getElementById('loadingOverlay').style.display = 'none';
    if (needsUpdate) {
      needsUpdate = false;
      performUpdate();
    }
  }
}

function updateMainView() {
  const img = document.getElementById('mainImage');
  const src = currentHeads[activeHead];
  if (src) {
    // Update canvas size to match image
    img.onload = () => {
      drawObbs();
    };
    img.src = src;
    img.style.display = 'block';
  }
  
  // Highlight active thumb
  document.querySelectorAll('.head-thumb').forEach(el => el.classList.remove('active'));
  const activeThumb = document.getElementById(`thumb-${activeHead}`);
  if (activeThumb) activeThumb.classList.add('active');
}

function switchHead(headName) {
  activeHead = headName;
  document.getElementById('viewTitle').textContent = headName.charAt(0).toUpperCase() + headName.slice(1) + " Output";
  updateMainView();
}

function toggleObb() {
  showObbs = !showObbs;
  document.getElementById('btnObb').classList.toggle('active', showObbs);
  drawObbs();
}

function drawObbs() {
  const canvas = document.getElementById('obbCanvas');
  const img = document.getElementById('mainImage');
  if (!canvas || !img) return;
  
  if (!showObbs || activeHead !== 'optical') {
    canvas.style.display = 'none';
    return;
  }
  
  // Match canvas to displayed image size
  // Note: mainImage is object-fit: contain.
  // We need to calculate the actual displayed rectangle of the image.
  // This is tricky.
  // Simplification: Set canvas size to image natural size and scale via CSS?
  // No, we want to draw on top.
  // Let's assume the canvas covers the container, and we compute transform.
  // Actually, easier: Set canvas to natural size, and apply same styles?
  // But object-fit: contain makes it hard.
  
  // Robust approach for object-fit: contain
  // 1. Get natural dimensions
  const nw = img.naturalWidth;
  const nh = img.naturalHeight;
  if (!nw || !nh) return;
  
  // 2. Get element dimensions
  const rect = img.getBoundingClientRect();
  const ew = rect.width;
  const eh = rect.height;
  
  // 3. Calculate rendered dimensions
  const ar_n = nw / nh;
  const ar_e = ew / eh;
  
  let rw, rh, ox, oy;
  
  if (ar_n > ar_e) {
    // Image is wider than element (constrained by width)
    rw = ew;
    rh = ew / ar_n;
    ox = 0;
    oy = (eh - rh) / 2;
  } else {
    // Image is taller than element (constrained by height)
    rh = eh;
    rw = eh * ar_n;
    ox = (ew - rw) / 2;
    oy = 0;
  }
  
  // 4. Set canvas to match RENDERED rect
  // We need to position it relative to the container.
  // The img rect is relative to viewport.
  // The canvas is absolute inside container.
  const container = document.getElementById('canvasContainer');
  const cRect = container.getBoundingClientRect();
  
  // Offset of the img element relative to container
  const imgRelLeft = rect.left - cRect.left;
  const imgRelTop = rect.top - cRect.top;
  
  // Final canvas position = img element pos + internal offset (ox, oy)
  canvas.width = rw;
  canvas.height = rh;
  canvas.style.left = (imgRelLeft + ox) + 'px';
  canvas.style.top = (imgRelTop + oy) + 'px';
  canvas.style.display = 'block';
  
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Scale is now simply Rendered / Natural
  const scaleX = rw / nw;
  const scaleY = rh / nh;
  
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 1.5;
  
  currentObbs.forEach(ob => {
    const cs = ob.corners; // [[x,y], ...]
    if (!cs || cs.length !== 4) return;
    
    ctx.beginPath();
    ctx.moveTo(cs[0][0] * scaleX, cs[0][1] * scaleY);
    for (let i=1; i<4; i++) {
      ctx.lineTo(cs[i][0] * scaleX, cs[i][1] * scaleY);
    }
    ctx.closePath();
    ctx.stroke();
  });
}

// Window resize handling for OBB overlay
window.addEventListener('resize', () => {
  drawObbs();
});

// Config Helper (Simplified from synth.js)
function getConfig() {
  const val = (id, def) => {
    const el = document.getElementById(id);
    return el ? (parseFloat(el.value) || def) : def;
  };
  const chk = (id) => {
    const el = document.getElementById(id);
    return el ? el.checked : false;
  };
  const txt = (id, def) => {
    const el = document.getElementById(id);
    return el ? el.value : def;
  };

  return {
    canvas: { width: 1024, height: 768, use_gpu: true }, 
    physics: {
      // Global / Legacy flags
      rods: {
        enable: false, // Force disable legacy path to avoid ghost objects
        enable_3d: chk('synEnable3d'),
        // Dummy values for legacy fields
        n_rods_rng_lo_hi: [50, 200, 200],
        rod_len_px_lo_hi: [30, 380, 380],
        rod_aspect_lo_hi: [0.02, 0.3, 0.3],
        rod_delta_rng: [-12, 0, 0]
      },
      
      // New Specific Specs (Playground Mode)
      use_specific_specs: true,
      
      rod_specs: {
        enable: chk('synRodEnable'),
        count_range: [val('synRodCountLo', 50), val('synRodCountHi', 200)],
        length_range: [val('synRodLenLo', 30), val('synRodLenHi', 150)],
        aspect_range: [val('synRodAspLo', 0.02), val('synRodAspHi', 0.10)],
        material: txt('synRodMaterial', 'standard'),
        
        // Physics 2.0
        ragged_p: val('synRoughness', 0.0),
        polarity_p: val('synPolarity', 0.0),
        shape_mode: txt('synShapeMode', 'straight')
      },
      sphere_specs: {
        enable: chk('synSphereEnable'),
        count_range: [val('synSphereCountLo', 10), val('synSphereCountHi', 50)],
        diameter_range: [val('synSphereDiamLo', 20), val('synSphereDiamHi', 100)],
        material: txt('synSphereMaterial', 'standard')
      },
      cube_specs: {
        enable: chk('synCubeEnable'),
        count_range: [val('synCubeCountLo', 10), val('synCubeCountHi', 50)],
        size_range: [val('synCubeSizeLo', 20), val('synCubeSizeHi', 100)],
        material: txt('synCubeMaterial', 'standard')
      },
      plate_specs: {
        enable: chk('synPlateEnable'),
        count_range: [val('synPlateCountLo', 10), val('synPlateCountHi', 50)],
        size_range: [val('synPlateSizeLo', 30), val('synPlateSizeHi', 150)],
        aspect_range: [val('synPlateAspLo', 0.1), val('synPlateAspHi', 0.8)],
        thickness_range: [val('synPlateThickLo', 0.05), val('synPlateThickHi', 0.2)],
        material: txt('synPlateMaterial', 'standard'),
        
        // Physics 2.0
        ragged_p: val('synRoughness', 0.0),
        polarity_p: val('synPolarity', 0.0),
        shape_mode: txt('synShapeMode', 'straight')
      },
      bubble_specs: {
        enable: chk('synBubbleEnable'),
        count_range: [val('synBubbleCountLo', 5), val('synBubbleCountHi', 20)],
        diameter_range: [val('synBubbleDiamLo', 10), val('synBubbleDiamHi', 50)],
        material: txt('synBubbleMaterial', 'air')
      },
      droplet_specs: {
        enable: chk('synDropletEnable'),
        count_range: [val('synDropletCountLo', 5), val('synDropletCountHi', 20)],
        diameter_range: [val('synDropletDiamLo', 10), val('synDropletDiamHi', 50)],
        material: txt('synDropletMaterial', 'oil')
      },

      ghosts: {
        enable: chk('synGhostEnable'),
        fraction: val('synGhostFraction', 0.2),
        gain_mult: 0.5,
        blur_sigma: 0.0
      },
      debris: {
        rate: val('synDebrisRate', 0.0),
        int_delta: [-6, 6],
        size_px: [1, 3]
      },
      fused: {
        enable: true,
        p0: 0.0001,
        p1: val('synAgglo', 0.003)
      }
    },
    optics: {
      mode: txt('synOpticsMode', 'dic'),
      polarizer_angle_deg: val('synPolarizerAngle', 0),
      shadow_gain: [val('synShGain', 10), val('synShGain', 10)*2], // Range?
      taper_strength: 0.45,
      rod_halo_sigma: 3.2
    },
    sensor: {
      bg_gray_range: [90, 97],
      bg_noise_std: val('synBgNoise', 0.0),
      blur_sigma: val('synBlur', 0.0),
      vignette_strength: val('synVignette', 0.0),
      tilt_enable: chk('synTiltEnable'),
      relief_field_enable: chk('synReliefEnable'),
      
      // Fouling
      fouling_enable: chk('synFoulingEnable'),
      fouling_prob: val('synFoulingProb', 0.3),
      fouling_count_range: [val('synFoulingCountLo', 1), val('synFoulingCountHi', 5)],
      fouling_opacity: val('synFoulingOp', 0.5),

      scalebar: { enable: true, prob: 1.0 }
    }
  };
}

async function savePresetPrompt() {
  let name = prompt("Enter preset name:");
  if (name) {
    const config = getConfig();
    await fetch('/synth_save_preset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ name, config })
    });
    alert('Saved!');
  }
}
