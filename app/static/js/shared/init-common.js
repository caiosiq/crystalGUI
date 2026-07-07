/* Shared DOM init (all pages) */
document.addEventListener('DOMContentLoaded', function () {
  try { setupImageClickHandlers(); } catch (e) { console.warn('Failed to setup click handlers', e); }

  const zoomToggleObbsBtn = document.getElementById('imageZoomToggleObbsBtn');
  if (zoomToggleObbsBtn) zoomToggleObbsBtn.addEventListener('click', toggleImageZoomObbs);
  const zoomModalEl = document.getElementById('imageZoomModal');
  if (zoomModalEl) {
    zoomModalEl.addEventListener('hidden.bs.modal', () => {
      imageZoomPreprocContext = null;
      updateImageZoomObbButton();
    });
  }
});
