/* Restore cross-page selection and run page setup */
function restoreSelectedImage() {
  try {
    const saved = sessionStorage.getItem('selectedImage');
    if (saved) selectImage(saved);
  } catch (e) { /* ignore */ }
}
