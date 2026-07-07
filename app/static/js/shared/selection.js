/* shared/selection.js */

function selectImage(name) {
  selectedImage = name;
  try { sessionStorage.setItem('selectedImage', name); } catch (e) { /* ignore */ }
  const badge = document.getElementById('selected-image');
  if (badge) badge.innerText = name;
  const preBadge = document.getElementById('preproc-selected-image');
  if (preBadge) preBadge.textContent = name;
  if (typeof setupPreprocPreview === 'function') setupPreprocPreview(name);
}
