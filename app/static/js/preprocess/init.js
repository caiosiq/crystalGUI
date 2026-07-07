/* Preprocess page init */
document.addEventListener('DOMContentLoaded', function () {
  restoreSelectedImage();
  setTimeout(() => {
    try { loadPreprocModels(); } catch (e) { console.warn('loadPreprocModels missing'); }
    try { preprocLoadPresetsList(); } catch (e) { console.warn('preprocLoadPresetsList missing'); }
  }, 100);
});
