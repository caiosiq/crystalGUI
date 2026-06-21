import { showToast, getVal } from './utils.js?v=4';

// ------------------------------------------------------------------
// Sidebar Tabs
// ------------------------------------------------------------------

export function switchSidebarTab(tabName) {
    // Buttons
    document.querySelectorAll('.sidebar-nav-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById(`tab-${tabName}`);
    if(btn) btn.classList.add('active');
    
    // Panes
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    const pane = document.getElementById(`pane-${tabName}`);
    if(pane) pane.classList.add('active');
}

// ------------------------------------------------------------------
// Heads / Preview
// ------------------------------------------------------------------

export function switchHead(type, callbacks = {}) {
    const { on3DInit, onResize } = callbacks;
    
    document.querySelectorAll('.head-thumb').forEach(t => t.classList.remove('active'));
    const thumb = document.getElementById(`thumb-${type}`);
    if(thumb) thumb.classList.add('active');
    
    const main = document.getElementById('mainImage');
    const obbCvs = document.getElementById('obbCanvas');
    const renderer3dEl = document.getElementById('canvas3d'); // Assuming ID set in render.js
    
    if (type === '3d') {
        if (on3DInit) on3DInit();
        if (renderer3dEl) renderer3dEl.style.display = 'block';
        if (main) main.style.display = 'none';
        if (obbCvs) obbCvs.style.display = 'none';
        
        // Trigger resize just in case
        if (onResize) onResize();
        return;
    }
    
    // Hide 3D
    if (renderer3dEl) renderer3dEl.style.display = 'none';
    
    // If switching back to optical, ensure main image is visible if source is available
    if (type === 'optical') {
        const imgOptical = document.getElementById('img-optical');
        if (imgOptical && imgOptical.src && imgOptical.src.startsWith('data:')) {
            main.src = imgOptical.src;
            main.style.display = 'block';
        }
        return;
    }
    
    const srcEl = document.getElementById(`img-${type}`);
    if (srcEl && srcEl.src && srcEl.src.startsWith('data:')) {
        main.src = srcEl.src;
        main.style.display = 'block';
    }
}

export function toggleObb() {
    const cvs = document.getElementById('obbCanvas');
    if(cvs) cvs.style.display = cvs.style.display === 'none' ? 'block' : 'none';
}

// ------------------------------------------------------------------
// Optics UI Logic
// ------------------------------------------------------------------

export function updateOpticsControls(mode) {
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
// Validation / Comparison UI
// ------------------------------------------------------------------

// State for comparison mode
let isCompareMode = false;
let hasTargetImage = false;

export function setTargetImageState(hasImage) {
    hasTargetImage = hasImage;
}

export function toggleComparisonMode(onResize) {
    isCompareMode = !isCompareMode;
    const btn = document.getElementById('btnCompareMode');
    const cmpContainer = document.getElementById('compareContainer');
    const mainContainer = document.getElementById('canvasContainer');
    
    if (isCompareMode) {
        if (!hasTargetImage && !document.getElementById('refImage').src) {
             showToast("Load a reference image first!");
             isCompareMode = false;
             return;
        }
        btn.classList.add('active');
        cmpContainer.style.display = 'flex'; // Show side by side
    } else {
        btn.classList.remove('active');
        cmpContainer.style.display = 'none';
    }
    // Resize 3D if active
    if (onResize) onResize();
}

export function updateMetrics(data) {
    if (!data || !data.meta) return;
    const m = data.meta;
    
    // Count (mock logic if meta not fully populated, or read actuals)
    let count = 0;
    if (m.rods) count += m.rods.count || 0;
    
    const metricEl = document.getElementById('metricCount');
    if(metricEl) metricEl.textContent = count > 0 ? count : '-';
}

// ------------------------------------------------------------------
// Drag & Drop / File Input
// ------------------------------------------------------------------

export function setupDragDrop(onFileLoaded) {
    const dz = document.getElementById('dropZone');
    const inp = document.getElementById('fileInput');
    
    if (!dz) return;
    
    dz.onclick = () => inp.click();
    
    inp.onchange = (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0], onFileLoaded);
        }
    };
    
    dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('bg-secondary'); };
    dz.ondragleave = (e) => { e.preventDefault(); dz.classList.remove('bg-secondary'); };
    dz.ondrop = (e) => {
        e.preventDefault();
        dz.classList.remove('bg-secondary');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0], onFileLoaded);
        }
    };
}

function handleFile(file, onFileLoaded) {
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('refImage').src = e.target.result;
        document.getElementById('compareImageDisplay').src = e.target.result; // Update comparison image
        
        document.getElementById('dropZone').style.display = 'none';
        document.getElementById('refImageContainer').style.display = 'block';
        
        hasTargetImage = true;
        
        if (onFileLoaded) onFileLoaded(file);
    };
    reader.readAsDataURL(file);
}

export function clearRefImage() {
    document.getElementById('refImage').src = '';
    document.getElementById('compareImageDisplay').src = '';
    document.getElementById('compareContainer').style.display = 'none';
    
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('refImageContainer').style.display = 'none';
    
    document.getElementById('targetImgName').textContent = 'None';
    document.getElementById('targetImgName').classList.remove('bg-success');
    document.getElementById('targetImgName').classList.add('bg-secondary');
    
    // Reset comparison mode if active
    isCompareMode = false;
    hasTargetImage = false;
    document.getElementById('btnCompareMode').classList.remove('active');
}
