// OSOG Lab - Playground Logic

let currentHeads = {};
let currentObbs = [];
let activeHead = 'optical';
let showObbs = false;
let isPending = false;
let needsUpdate = false;
let debounceTimer = null;
let presetsConstraints = {};

// Three.js Globals
let scene, camera, renderer, controls;
let is3DInit = false;

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

  // Window Resize
  window.addEventListener('resize', () => {
      drawObbs();
      if (activeHead === '3d' && is3DInit) {
          resize3D();
      }
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
        setVal('synInclusions', p.rod_specs.inclusions);
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
        setVal('synBubbleAttach', p.bubble_specs.attach_prob || 0.0);
    }
    
    // Phase 4
    setChk('synFlowEnable', p.flow_enable);
    setVal('synFlowDir', p.flow_direction || 0);
    setVal('synFlowShear', p.flow_shear_rate || 0);
    setChk('synSedEnable', p.sedimentation_enable);
    setVal('synSedStr', p.sedimentation_strength || 0);
    setChk('synSizeSeg', p.size_segregation_enable);

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
        setChk('synCoalesceEnable', p.droplet_specs.coalesce_enable);
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
        // Map weights back to checkboxes if possible
        // Weights: [Random, Stack, Chain, Cross]
        const w = p.fused.cluster_weights || [1, 1, 1, 1, 0, 0];
        setChk('chkAggRandom', w[0] > 0);
        setChk('chkAggStack', w[1] > 0);
        setChk('chkAggChain', w[2] > 0);
        setChk('chkAggCross', w[3] > 0);
        setChk('chkAggSnow', (w[4] || 0) > 0);
        setChk('chkAggSphere', (w[5] || 0) > 0);

        setChk('synDLCA', p.fused.dlca_enable);
        setVal('synSinter', p.fused.sintering_strength || 0);
    }
    
    // Optics
    const o = cfg.optics || {};
    setVal('synOpticsMode', o.mode || 'dic');
    setVal('synPolarizerAngle', o.polarizer_angle_deg);
    setVal('synLightAngle', o.lighting_angle_deg || 45);
    setVal('synFocusZ', o.focus_z || 0.0);
    setVal('synAperture', o.aperture || 0.0);
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
    'synLightAngle': 'lblLightAngle',
    'synShGain': 'lblShGain',
    'synBgNoise': 'lblNoise',
    'synBlur': 'lblBlur',
    'synVignette': 'lblVignette',
    'synFoulingProb': 'lblFoulingProb',
    'synFoulingOp': 'lblFoulingOp',
    'synRoughness': 'lblRoughness',
    'synPolarity': 'lblPolarity',
    'synAgglo': 'lblAgglo',
    'synSinter': 'lblSinter',
    'synBubbleAttach': 'lblBubbleAttach',
    'synFlowDir': 'lblFlowDir',
    'synFlowShear': 'lblFlowShear',
    'synSedStr': 'lblSedStr',
    'synFocusZ': 'lblFocusZ',
    'synAperture': 'lblAperture'
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
    if (needsUpdate) {
      needsUpdate = false;
      performUpdate();
    }
  }
}

function updateMainView() {
  const container = document.getElementById('canvasContainer');
  const img = document.getElementById('mainImage');
  const canvas3d = document.getElementById('canvas3d'); // We will create this dynamically
  
  if (activeHead === '3d') {
      // Hide Image
      img.style.display = 'none';
      document.getElementById('obbCanvas').style.display = 'none';
      
      // Show 3D Canvas
      if (!canvas3d) {
          init3D();
      } else {
          canvas3d.style.display = 'block';
      }
      draw3D();
      
  } else {
      // Hide 3D
      if (canvas3d) canvas3d.style.display = 'none';
      
      // Show Image
      const src = currentHeads[activeHead];
      if (src) {
        img.onload = () => {
          drawObbs();
        };
        img.src = src;
        img.style.display = 'block';
      }
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
  
  if (!showObbs || activeHead !== 'optical' || activeHead === '3d') {
    canvas.style.display = 'none';
    return;
  }
  
  // (Same OBB drawing logic as before...)
  const nw = img.naturalWidth;
  const nh = img.naturalHeight;
  if (!nw || !nh) return;
  
  const rect = img.getBoundingClientRect();
  const ew = rect.width;
  const eh = rect.height;
  
  const ar_n = nw / nh;
  const ar_e = ew / eh;
  
  let rw, rh, ox, oy;
  
  if (ar_n > ar_e) {
    rw = ew;
    rh = ew / ar_n;
    ox = 0;
    oy = (eh - rh) / 2;
  } else {
    rh = eh;
    rw = eh * ar_n;
    ox = (ew - rw) / 2;
    oy = 0;
  }
  
  const container = document.getElementById('canvasContainer');
  const cRect = container.getBoundingClientRect();
  
  const imgRelLeft = rect.left - cRect.left;
  const imgRelTop = rect.top - cRect.top;
  
  canvas.width = rw;
  canvas.height = rh;
  canvas.style.left = (imgRelLeft + ox) + 'px';
  canvas.style.top = (imgRelTop + oy) + 'px';
  canvas.style.display = 'block';
  
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  const scaleX = rw / nw;
  const scaleY = rh / nh;
  
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 1.5;
  
  currentObbs.forEach(ob => {
    const cs = ob.corners;
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

// ------------------------------------------------------------------
// 3D Visualization Logic (Three.js)
// ------------------------------------------------------------------

function init3D() {
    if (is3DInit) return;
    
    const container = document.getElementById('canvasContainer');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);
    
    // Camera (Orthographic to match microscope)
    // View size = image size (approx 1024)
    const viewSize = 1024;
    const aspect = width / height;
    camera = new THREE.OrthographicCamera(
        viewSize * aspect / -2, viewSize * aspect / 2,
        viewSize / 2, viewSize / -2,
        1, 2000
    );
    camera.position.z = 1000;
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.domElement.id = 'canvas3d';
    renderer.domElement.style.position = 'absolute';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    container.appendChild(renderer.domElement);
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(1, 1, 2);
    scene.add(dirLight);
    
    // Grid Helper (1024x1024)
    const gridHelper = new THREE.GridHelper(1024, 20, 0x444444, 0x333333);
    gridHelper.rotation.x = Math.PI / 2; // Flat on XY plane
    scene.add(gridHelper);
    
    is3DInit = true;
    
    // Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        if (activeHead === '3d') {
            renderer.render(scene, camera);
        }
    }
    animate();
    
    // Add Mouse Controls (Simple Rotation)
    let isDragging = false;
    let prevX = 0, prevY = 0;
    
    renderer.domElement.addEventListener('mousedown', e => {
        isDragging = true;
        prevX = e.clientX;
        prevY = e.clientY;
    });
    
    window.addEventListener('mouseup', () => isDragging = false);
    
    window.addEventListener('mousemove', e => {
        if (!isDragging) return;
        const dx = e.clientX - prevX;
        const dy = e.clientY - prevY;
        
        // Rotate scene or camera? Let's rotate the camera container
        // Actually, easiest to just rotate the root object containing particles
        if (scene.getObjectByName("root")) {
            scene.getObjectByName("root").rotation.y += dx * 0.01;
            scene.getObjectByName("root").rotation.x += dy * 0.01;
        }
        
        prevX = e.clientX;
        prevY = e.clientY;
    });
}

function resize3D() {
    const container = document.getElementById('canvasContainer');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    const viewSize = 1024;
    const aspect = width / height;
    
    camera.left = viewSize * aspect / -2;
    camera.right = viewSize * aspect / 2;
    camera.top = viewSize / 2;
    camera.bottom = viewSize / -2;
    camera.updateProjectionMatrix();
    
    renderer.setSize(width, height);
}

function draw3D() {
    if (!scene) return;
    
    // Clear old objects
    const oldRoot = scene.getObjectByName("root");
    if (oldRoot) scene.remove(oldRoot);
    
    const root = new THREE.Group();
    root.name = "root";
    scene.add(root);
    
    // Add Particles
    currentObbs.forEach(obj => {
        let geom, mat;
        
        // Map shape_id to geometry
        // 0: Rod, 1: Plate, 2: Cube, 3: Sphere, 4: Bubble, 5: Droplet
        const sid = obj.shape_id || 0;
        
        // Color based on shape
        let color = 0x00ff00;
        if (sid === 0) color = 0x00ff00; // Rod (Green)
        else if (sid === 1) color = 0x00ffff; // Plate (Cyan)
        else if (sid === 2) color = 0xff00ff; // Cube (Magenta)
        else if (sid === 3) color = 0xffff00; // Sphere (Yellow)
        else if (sid === 4) color = 0xffffff; // Bubble (White)
        else if (sid === 5) color = 0xff8800; // Droplet (Orange)
        
        // Geometry
        if (sid === 3 || sid === 4 || sid === 5) {
            // Sphere/Bubble/Droplet
            // Use L as diameter
            const diam = obj.L;
            geom = new THREE.SphereGeometry(diam / 2, 16, 16);
            
            // Fix: Spheres/Bubbles shouldn't rotate visually in a way that looks like a box
            // But we still apply position
        } else if (sid === 0) {
             // Rod (Cylinder-ish or Box)
             // Use CylinderGeometry for better look? Or Capsule?
             // Box is fine for performance, but let's try Cylinder for Rods if possible
             // Box: L, W, H. Rods are long in L.
             geom = new THREE.BoxGeometry(obj.L, obj.W, obj.H);
        } else {
            // Box (Plate/Cube)
            // L, W, H
            geom = new THREE.BoxGeometry(obj.L, obj.W, obj.H);
        }
        
        mat = new THREE.MeshPhongMaterial({ 
            color: color, 
            transparent: true, 
            opacity: 0.8,
            specular: 0x555555,
            shininess: 30
        });
        
        const mesh = new THREE.Mesh(geom, mat);
        
        // Position
        // cx, cy are image coords (0,0 top-left).
        // 3D world: 0,0 center. Y up.
        // Image Width/Height assumed 1024x768 (or whatever config says)
        // Let's assume 1024x1024 for simplicity or center it
        const imgW = 1024;
        const imgH = 768; // Should come from config, but hardcoded in playground js default
        
        mesh.position.x = obj.cx - imgW/2;
        mesh.position.y = -(obj.cy - imgH/2); // Flip Y
        mesh.position.z = obj.z * 100; // Z is depth (-1 to 1). Scale it up for visibility
        
        // Rotation
        // obj.angle_deg is Z rotation (in image plane)
        // obj.beta is X rotation (tumble)
        // obj.gamma is Y rotation (roll)
        // Order matters. Usually we rotate geometry or use Euler
        
        // In 2D engine:
        // Alpha (Z) is applied first?
        // Let's try standard ZYX
        mesh.rotation.order = 'ZYX'; 
        
        if (sid === 3 || sid === 4 || sid === 5) {
             // Spheres: Only Z rotation matters if they are slightly non-spherical?
             // Actually spheres don't show rotation well.
             // But let's apply it anyway.
        }
        
        mesh.rotation.z = -THREE.Math.degToRad(obj.angle_deg); // Negate for correct direction?
        mesh.rotation.x = THREE.Math.degToRad(obj.beta);
        mesh.rotation.y = THREE.Math.degToRad(obj.gamma);
        
        root.add(mesh);
    });
}


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
        shape_mode: txt('synShapeMode', 'straight'),
        inclusions: val('synInclusions', 0.0)
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
        material: txt('synBubbleMaterial', 'air'),
        attach_prob: val('synBubbleAttach', 0.0)
      },
      droplet_specs: {
        enable: chk('synDropletEnable'),
        count_range: [val('synDropletCountLo', 5), val('synDropletCountHi', 20)],
        diameter_range: [val('synDropletDiamLo', 10), val('synDropletDiamHi', 50)],
        material: txt('synDropletMaterial', 'oil'),
        coalesce_enable: chk('synCoalesceEnable')
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
        p1: val('synAgglo', 0.003),
        dlca_enable: chk('synDLCA'),
        sintering_strength: val('synSinter', 0.0),
        cluster_weights: [
            chk('chkAggRandom') ? 1.0 : 0.0,
            chk('chkAggStack') ? 1.0 : 0.0,
            chk('chkAggChain') ? 1.0 : 0.0,
            chk('chkAggCross') ? 1.0 : 0.0,
            chk('chkAggSnow') ? 1.0 : 0.0,
            chk('chkAggSphere') ? 1.0 : 0.0
        ]
      },
      
      // Phase 4
      flow_enable: chk('synFlowEnable'),
      flow_direction: val('synFlowDir', 0.0),
      flow_shear_rate: val('synFlowShear', 0.0),
      sedimentation_enable: chk('synSedEnable'),
      sedimentation_strength: val('synSedStr', 0.0),
      size_segregation_enable: chk('synSizeSeg')
    },
    optics: {
      mode: txt('synOpticsMode', 'dic'),
      polarizer_angle_deg: val('synPolarizerAngle', 0),
      lighting_angle_deg: val('synLightAngle', 45),
      focus_z: val('synFocusZ', 0.0),
      aperture: val('synAperture', 0.0),
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
