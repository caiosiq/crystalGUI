/* Inference page init */
document.addEventListener('DOMContentLoaded', function () {
  restoreSelectedImage();
  setTimeout(() => {
    try { checkGpuAvailability(); } catch (e) { /* optional */ }
    try { loadAvailableModels(); } catch (e) { console.warn('loadAvailableModels missing'); }
  }, 100);
});
