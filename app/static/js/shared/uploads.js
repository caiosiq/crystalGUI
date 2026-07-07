/* shared/uploads.js */

async function deleteUploadedImage(imageName, btnEl) {
  if (!imageName) return;
  const msg = `Delete "${imageName}"?\n\nThis permanently removes the uploaded image and any cached inference/preprocess results for it.`;
  if (!window.confirm(msg)) return;

  try {
    const formData = new FormData();
    formData.append('image_name', imageName);
    const response = await fetch('/delete_upload', { method: 'POST', body: formData });
    const data = await response.json();
    if (!data.ok) {
      showAlert('danger', data.error || 'Failed to delete image');
      return;
    }

    document.querySelectorAll(`[data-upload-image="${CSS.escape(imageName)}"]`).forEach(el => el.remove());

    const imageList = document.getElementById('image-list');
    const preprocList = document.getElementById('preproc-image-list-top');
    if (imageList && !imageList.querySelector('[data-upload-image]')) {
      imageList.innerHTML = '<p class="text-muted mb-0">No images uploaded yet.</p>';
    }
    if (preprocList && !preprocList.querySelector('[data-upload-image]')) {
      preprocList.innerHTML = '<p class="text-muted mb-0">No images uploaded yet.</p>';
    }

    if (selectedImage === imageName) {
      selectedImage = null;
      const selBadge = document.getElementById('selected-image');
      if (selBadge) selBadge.textContent = 'None selected';
      const preprocBadge = document.getElementById('preproc-selected-image');
      if (preprocBadge) preprocBadge.textContent = 'None selected';
      hideLoadingInterface();
      const resultsDisplay = document.getElementById('results-display');
      if (resultsDisplay) resultsDisplay.style.display = 'none';
      const empty = document.getElementById('preproc-empty');
      const preview = document.getElementById('preproc-preview');
      if (empty) empty.style.display = 'block';
      if (preview) preview.style.display = 'none';
      preprocImg = null;
      preprocBaseImg = null;
    }

    showAlert('success', `Deleted ${imageName}`);
  } catch (error) {
    console.error('Delete upload error:', error);
    showAlert('danger', `Delete failed: ${error.message}`);
  }
}
