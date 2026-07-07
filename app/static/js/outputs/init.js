/* Outputs page init */
document.addEventListener('DOMContentLoaded', function () {
  try {
    loadOutputsModels();
    fetchOutputsDatasets();
    outputsLoadPresetsList();
  } catch (e) { console.warn('Failed initial Outputs preload', e); }

  const outputsSearch = document.getElementById('outputsDatasetSearch');
  if (outputsSearch) {
    outputsSearch.addEventListener('input', () => renderOutputsDatasetsList(outputsUploadedDatasets));
  }
});
