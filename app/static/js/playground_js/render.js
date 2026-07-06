// Rendering & 3D Visualization
import { getVal } from './utils.js?v=4';

let scene3d, camera3d, renderer3d, controls3d, particlesGroup, focalPlaneMesh;
let is3DInit = false;

// ------------------------------------------------------------------
// 2D Canvas / OBBs
// ------------------------------------------------------------------

export function drawObbs(obbs, w, h, options = {}) {
    const { rawObbs = null, showRaw = false } = options;
    const canvas = document.getElementById('obbCanvas');
    const container = document.getElementById('canvasContainer');
    const img = document.getElementById('mainImage');
    
    if (!canvas || !container || !img) return;

    const drawOne = (ctx, list, scaleX, scaleY, style) => {
        if (!list || !list.length) return;
        ctx.save();
        ctx.strokeStyle = style.color;
        ctx.lineWidth = style.width || 1;
        if (style.dash) ctx.setLineDash(style.dash);
        list.forEach(ob => {
            const cs = ob.corners;
            if (!cs || cs.length < 4) return;
            ctx.beginPath();
            ctx.moveTo(cs[0][0] * scaleX, cs[0][1] * scaleY);
            for (let i = 1; i < 4; i++) ctx.lineTo(cs[i][0] * scaleX, cs[i][1] * scaleY);
            ctx.closePath();
            ctx.stroke();
        });
        ctx.restore();
    };

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

        if (showRaw && rawObbs && rawObbs.length) {
            drawOne(ctx, rawObbs, scaleX, scaleY, {
                color: 'rgba(255, 220, 80, 0.85)',
                width: 1,
                dash: [4, 3],
            });
        }
        drawOne(ctx, obbs, scaleX, scaleY, {
            color: 'rgba(0, 255, 0, 0.85)',
            width: 1.5,
        });
    });
}

// ------------------------------------------------------------------
// 3D Viewer (Three.js)
// ------------------------------------------------------------------

export function init3DViewer() {
    if (is3DInit) return;
    
    const container = document.getElementById('canvasContainer');
    if (!container) return;
    
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
        controls3d.target.set(512, -512, 0);
    }

    // Resize listener
    window.addEventListener('resize', onWindowResize, false);

    // Animation Loop
    animate3D();
    
    is3DInit = true;
}

export function onWindowResize() {
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

export function update3DScene(obbs, imgW, imgH) {
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
        const w = ob.L; 
        const h = ob.W; 
        const d = ob.H || (ob.W * 0.1); 
        
        const geometry = new THREE.BoxGeometry(w, h, d);
        
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
        const dist = Math.abs(ob.z - focusZ);
        const maxDist = 100.0;
        const t = Math.min(dist / maxDist, 1.0);
        
        const r = Math.floor(t * 255);
        const g = Math.floor((1 - t) * 255);
        mesh.material.color.setRGB(r/255, g/255, 0.2);
        
        const deg2rad = Math.PI / 180.0;
        
        // Rotation
        mesh.rotation.z = -ob.angle_deg * deg2rad; 
        mesh.rotation.x = ob.beta * deg2rad; 
        mesh.rotation.y = ob.gamma * deg2rad; 

        particlesGroup.add(mesh);
    });
}
