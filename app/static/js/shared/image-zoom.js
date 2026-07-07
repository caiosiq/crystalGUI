/* shared/image-zoom.js */

function waitForModalShown(modalEl) {
  return new Promise((resolve) => {
    if (!modalEl) { resolve(); return; }
    if (modalEl.classList.contains('show')) {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
      return;
    }
    modalEl.addEventListener('shown.bs.modal', () => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    }, { once: true });
  });
}

function getZoomViewportLimits(viewport) {
  const maxW = Math.min(Math.floor(window.innerWidth * 0.88), 1400);
  const maxH = Math.min(Math.floor(window.innerHeight * 0.72), 900);
  const vpW = Math.max(200, viewport?.clientWidth || maxW);
  const vpH = Math.max(200, viewport?.clientHeight || maxH);
  return {
    vpW: Math.min(vpW, maxW),
    vpH: Math.min(vpH, maxH),
    maxW,
    maxH,
  };
}

function computeFitCanvasSize(natW, natH, viewport) {
  const { maxW, maxH } = getZoomViewportLimits(viewport);
  const scale = Math.min(maxW / natW, maxH / natH, 1);
  return {
    drawW: Math.max(1, Math.round(natW * scale)),
    drawH: Math.max(1, Math.round(natH * scale)),
    fitScale: scale,
  };
}

function fitZoomContentToViewport(viewportEl, contentEl) {
  if (!viewportEl || !contentEl) return 1;
  const natW = contentEl.width || contentEl.naturalWidth || contentEl.offsetWidth;
  const natH = contentEl.height || contentEl.naturalHeight || contentEl.offsetHeight;
  if (!natW || !natH) return 1;

  const { vpW, vpH } = getZoomViewportLimits(viewportEl);
  const fitScale = Math.min(vpW / natW, vpH / natH, 1);
  const state = viewportEl._zoomState || { scale: 1, x: 0, y: 0 };
  state.fitScale = fitScale;
  state.minScale = fitScale;
  state.maxScale = Math.max(fitScale * 8, 4);
  state.scale = fitScale;
  state.x = Math.max(0, (vpW - natW * fitScale) / 2);
  state.y = Math.max(0, (vpH - natH * fitScale) / 2);
  viewportEl._zoomState = state;

  contentEl.style.width = `${natW}px`;
  contentEl.style.height = `${natH}px`;
  contentEl.style.maxWidth = 'none';
  contentEl.style.maxHeight = 'none';
  contentEl.style.transformOrigin = '0 0';
  contentEl.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
  return fitScale;
}

function updateImageZoomObbButton() {
  const btn = document.getElementById('imageZoomToggleObbsBtn');
  const label = document.getElementById('imageZoomToggleObbsLabel');
  if (!btn) return;
  if (!imageZoomPreprocContext) {
    btn.style.display = 'none';
    return;
  }
  btn.style.display = 'inline-block';
  if (label) label.textContent = imageZoomPreprocContext.showObbs ? 'Hide OBBs' : 'Show OBBs';
}

async function redrawPreprocZoomModal() {
  if (!imageZoomPreprocContext || !preprocImg || preprocImg.naturalWidth <= 0) return;
  const modalCanvas = document.getElementById('imageZoomCanvas');
  const viewport = document.getElementById('imageZoomViewport');
  const spinner = document.getElementById('imageZoomSpinner');
  if (!modalCanvas || !viewport) return;

  if (spinner) spinner.style.display = 'block';
  modalCanvas.style.display = 'none';

  const { drawW, drawH } = computeFitCanvasSize(
    preprocImg.naturalWidth,
    preprocImg.naturalHeight,
    viewport
  );
  modalCanvas.width = drawW;
  modalCanvas.height = drawH;

  await renderPreprocToCanvas(
    modalCanvas,
    imageZoomPreprocContext.kind,
    imageZoomPreprocContext.showObbs,
    drawW,
    drawH
  );

  if (spinner) spinner.style.display = 'none';
  modalCanvas.style.display = 'block';
  resetZoomState(viewport);
  enableInteractiveZoom(viewport, modalCanvas);
}

async function showPreprocImageInModal(kind, title = 'Image Preview') {
  const modalEl = document.getElementById('imageZoomModal');
  if (!modalEl || typeof bootstrap === 'undefined' || !bootstrap.Modal) {
    showImageInModal(null, title, true, kind === 'original' ? preprocCanvasOrig : preprocCanvasProc);
    return;
  }

  imageZoomPreprocContext = { kind, showObbs: true };
  updateImageZoomObbButton();

  const modal = bootstrap.Modal.getOrCreateInstance(modalEl, { keyboard: true });
  const modalTitle = document.getElementById('imageZoomModalTitle');
  const modalImg = document.getElementById('imageZoomImg');
  const modalCanvas = document.getElementById('imageZoomCanvas');
  const spinner = document.getElementById('imageZoomSpinner');
  const viewport = document.getElementById('imageZoomViewport');

  resetZoomState(viewport);
  if (modalTitle) modalTitle.textContent = title;
  if (modalImg) modalImg.style.display = 'none';
  if (modalCanvas) modalCanvas.style.display = 'none';
  if (spinner) spinner.style.display = 'block';

  modal.show();
  await waitForModalShown(modalEl);
  await redrawPreprocZoomModal();
}

function toggleImageZoomObbs() {
  if (!imageZoomPreprocContext) return;
  imageZoomPreprocContext.showObbs = !imageZoomPreprocContext.showObbs;
  updateImageZoomObbButton();
  redrawPreprocZoomModal();
}

function showImageInModal(imageSrc, title = 'Image Preview', isCanvas = false, canvasElement = null) {
  imageZoomPreprocContext = null;
  updateImageZoomObbButton();
  const modalEl = document.getElementById('imageZoomModal');
  if (!modalEl) {
    console.warn('imageZoomModal element not found in DOM');
    if (typeof showAlert === 'function') showAlert('danger', 'Zoom modal not found. Please reload the page.');
    return;
  }
  if (typeof bootstrap === 'undefined' || !bootstrap.Modal) {
    console.warn('Bootstrap Modal not available');
    if (typeof showAlert === 'function') showAlert('danger', 'Bootstrap JS not loaded. Check base.html includes.');
    return;
  }
  const modal = new bootstrap.Modal(modalEl, { keyboard: true });
  const modalTitle = document.getElementById('imageZoomModalTitle');
  const modalImg = document.getElementById('imageZoomImg');
  const modalCanvas = document.getElementById('imageZoomCanvas');
  const spinner = document.getElementById('imageZoomSpinner');
  const viewport = document.getElementById('imageZoomViewport');
  
  // Reset zoom state
  resetZoomState(viewport);
  
  // Set title
  if (modalTitle) modalTitle.textContent = title;
  
  // Hide both img and canvas initially
  if (modalImg) modalImg.style.display = 'none';
  if (modalCanvas) modalCanvas.style.display = 'none';
  if (spinner) spinner.style.display = 'block';
  
  if (isCanvas && canvasElement && modalCanvas) {
    const ctx = modalCanvas.getContext('2d');
    modalCanvas.width = canvasElement.width;
    modalCanvas.height = canvasElement.height;
    ctx.drawImage(canvasElement, 0, 0);

    if (spinner) spinner.style.display = 'none';
    modalCanvas.style.display = 'block';
    modal.show();
    waitForModalShown(modalEl).then(() => {
      resetZoomState(viewport);
      enableInteractiveZoom(viewport, modalCanvas);
    });
    return;
  } else if (modalImg) {
    // Load image
    modalImg.onload = function() {
      if (spinner) spinner.style.display = 'none';
      modalImg.style.display = 'block';
      modalImg.style.maxWidth = 'none';
      modalImg.style.maxHeight = 'none';
      waitForModalShown(modalEl).then(() => {
        resetZoomState(viewport);
        enableInteractiveZoom(viewport, modalImg);
      });
    };
    modalImg.onerror = function() {
      if (spinner) spinner.style.display = 'none';
      modalImg.style.display = 'block';
      modalImg.alt = 'Failed to load image';
    };
    modalImg.src = imageSrc;
  }
  
  modal.show();
}

function enableInteractiveZoom(viewportEl, contentEl) {
  if (!viewportEl || !contentEl) return;
  viewportEl.style.position = 'relative';
  viewportEl.style.overflow = 'hidden';
  viewportEl.style.maxHeight = '72vh';
  viewportEl.style.maxWidth = '88vw';
  viewportEl.style.width = '88vw';
  viewportEl.style.height = '72vh';
  viewportEl.style.touchAction = 'none';
  contentEl.style.willChange = 'transform';
  contentEl.style.userSelect = 'none';

  if (!viewportEl._zoomBound) {
    viewportEl._zoomBound = true;
    bindInteractiveZoomHandlers(viewportEl);
  }

  fitZoomContentToViewport(viewportEl, contentEl);
}

function bindInteractiveZoomHandlers(viewportEl) {
  viewportEl.addEventListener('wheel', (e) => {
    const contentEl = viewportEl.querySelector('.zoom-content');
    const state = viewportEl._zoomState;
    if (!contentEl || !state) return;
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1.1 : 0.9;
    const prev = state.scale;
    const minScale = state.minScale || 1;
    const maxScale = state.maxScale || 10;
    state.scale = Math.min(Math.max(state.scale * delta, minScale), maxScale);
    const vp = viewportEl.getBoundingClientRect();
    const cx = e.clientX - vp.left - state.x;
    const cy = e.clientY - vp.top - state.y;
    state.x -= (cx / prev) * (state.scale - prev);
    state.y -= (cy / prev) * (state.scale - prev);
    clampZoomPosition(viewportEl, contentEl, state);
    applyZoomTransform(contentEl, state);
  }, { passive: false });

  let dragging = false, lastX = 0, lastY = 0;
  viewportEl.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    viewportEl.setPointerCapture(e.pointerId);
  });
  viewportEl.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const contentEl = viewportEl.querySelector('.zoom-content');
    const state = viewportEl._zoomState;
    if (!contentEl || !state) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    state.x += dx;
    state.y += dy;
    clampZoomPosition(viewportEl, contentEl, state);
    applyZoomTransform(contentEl, state);
  });
  viewportEl.addEventListener('pointerup', () => { dragging = false; });
  viewportEl.addEventListener('pointercancel', () => { dragging = false; });

  let pinchDist = 0;
  viewportEl.addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) {
      pinchDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
    }
  }, { passive: true });
  viewportEl.addEventListener('touchmove', (e) => {
    if (e.touches.length === 2 && pinchDist) {
      e.preventDefault();
      const contentEl = viewportEl.querySelector('.zoom-content');
      const state = viewportEl._zoomState;
      if (!contentEl || !state) return;
      const newDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      const delta = newDist / pinchDist;
      pinchDist = newDist;
      const minScale = state.minScale || 1;
      const maxScale = state.maxScale || 10;
      state.scale = Math.min(Math.max(state.scale * delta, minScale), maxScale);
      clampZoomPosition(viewportEl, contentEl, state);
      applyZoomTransform(contentEl, state);
    }
  }, { passive: false });

  viewportEl.addEventListener('dblclick', () => {
    const contentEl = viewportEl.querySelector('.zoom-content');
    if (!contentEl) return;
    fitZoomContentToViewport(viewportEl, contentEl);
  });
}

function applyZoomTransform(contentEl, state) {
  contentEl.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
}

function clampZoomPosition(viewportEl, contentEl, state) {
  const natW = contentEl.width || contentEl.naturalWidth || contentEl.offsetWidth;
  const natH = contentEl.height || contentEl.naturalHeight || contentEl.offsetHeight;
  const vpW = viewportEl.clientWidth;
  const vpH = viewportEl.clientHeight;
  const contentW = natW * state.scale;
  const contentH = natH * state.scale;
  const minX = Math.min(0, vpW - contentW);
  const minY = Math.min(0, vpH - contentH);
  state.x = Math.min(Math.max(state.x, minX), 0);
  state.y = Math.min(Math.max(state.y, minY), 0);
}

function resetZoomState(viewportEl) {
  if (!viewportEl) return;
  viewportEl._zoomState = { scale: 1, x: 0, y: 0 };
}

function composeImageWithOverlay(imageEl, overlayCanvas) {
  if (!imageEl || !overlayCanvas) return null;
  // If overlay is hidden, we don't compose
  const isHidden = overlayCanvas.style && overlayCanvas.style.display === 'none';
  if (isHidden) return null;
  const w = overlayCanvas.width || imageEl.clientWidth || 512;
  const h = overlayCanvas.height || imageEl.clientHeight || 512;
  const cnv = document.createElement('canvas');
  cnv.width = w; cnv.height = h;
  const ctx = cnv.getContext('2d');
  // Compute object-fit contain placement for the image so it matches the overlay drawing
  const natW = imageEl.naturalWidth || w;
  const natH = imageEl.naturalHeight || h;
  const arImg = natW / natH;
  const arBox = w / h;
  let dispW, dispH, offX, offY;
  if (arImg > arBox) { dispW = w; dispH = w / arImg; offX = 0; offY = (h - dispH) / 2; }
  else { dispH = h; dispW = h * arImg; offY = 0; offX = (w - dispW) / 2; }
  ctx.clearRect(0,0,w,h);
  ctx.drawImage(imageEl, 0, 0, natW, natH, offX, offY, dispW, dispH);
  // Draw the overlay canvas on top (it already has proper scaling/offset)
  try { ctx.drawImage(overlayCanvas, 0, 0); } catch {}
  return cnv;
}

function setupImageClickHandlers() {
  // Handle clickable images
  document.addEventListener('click', function(e) {
    // Special handling for synthetic previews: request high-res image before opening
    if (e.target.id && e.target.id.startsWith('synth-prev-')) {
      const rid = e.target.id.replace('synth-prev-', '');
      if (window.openSynthHighResModal) {
        e.preventDefault();
        window.openSynthHighResModal(rid);
        return;
      }
    }

    if (e.target.classList.contains('clickable-image')) {
      const title = e.target.alt || 'Image Preview';
      // Try composing with any visible overlay canvas in the same container
      let overlayCanvas = null;
      // Prefer an overlay canvas inside the same positioned container
      const parent = e.target.parentElement;
      if (parent) {
        overlayCanvas = parent.querySelector('canvas.image-overlay-canvas');
      }
      // If none found, search a bit wider within the same row/card
      if (!overlayCanvas) {
        let p = e.target.closest('.position-relative, .card, .row, .col');
        if (p) overlayCanvas = p.querySelector('canvas.image-overlay-canvas');
      }
      // For synth rows, skip composing overlay to preserve high-res clarity (low-res overlay would downscale)
      if (e.target.id && e.target.id.startsWith('synth-prev-')) {
        overlayCanvas = null;
      }
      const composite = overlayCanvas ? composeImageWithOverlay(e.target, overlayCanvas) : null;
      if (composite) {
        showImageInModal(null, title, true, composite);
      } else {
        showImageInModal(e.target.src, title);
      }
    }
    
    if (e.target.classList.contains('clickable-canvas')) {
      const isPreprocOrig = e.target.id === 'preprocOriginalCanvas';
      const isPreprocProc = e.target.id === 'preprocProcessedCanvas';
      if (preprocInferenceLocked && (isPreprocOrig || isPreprocProc)) {
        const kind = isPreprocOrig ? 'original' : 'processed';
        const title = isPreprocOrig ? 'Original Image' : 'Processed Image';
        showPreprocImageInModal(kind, title);
        return;
      }
      const title = e.target.id.includes('Original') ? 'Original Image' : 
                   e.target.id.includes('Processed') ? 'Processed Image' : 'Canvas Preview';
      showImageInModal(null, title, true, e.target);
    }
  });
}
