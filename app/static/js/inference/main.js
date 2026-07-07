/* inference/main.js */

async function preprocessSelected(operation) {
  if (!selectedImage) {
    alert('Select an image first.');
    return;
  }
  const form = new FormData();
  form.append('image_name', selectedImage);
  form.append('operation', operation);
  const res = await fetch('/preprocess', { method: 'POST', body: form });
  const data = await res.json();
  if (data.ok) {
    document.getElementById('overlay').src = data.processed_path;
  }
}

function displayInferenceResults(data) {
  console.log('Inference results:', data);
  
  // Get UI elements
  const resultsDisplay = document.getElementById('results-display');
  const emptyState = document.getElementById('empty-state');
  const overlayImg = document.getElementById('overlay');
  const modelNameSpan = document.getElementById('results-model-name');
  const statsText = document.getElementById('statsText');
  
  // Show results and hide empty state
  if (resultsDisplay) resultsDisplay.style.display = 'block';
  if (emptyState) emptyState.style.display = 'none';
  
  // Display overlay image (cache-bust: same path is overwritten each run)
  if (overlayImg && data.overlay_url) {
    overlayImg.src = cacheBustStaticUrl(data.overlay_url);
    overlayImg.alt = `Inference results for ${data.image}`;
  }
  
  // Display model name
  if (modelNameSpan && data.model_info) {
    modelNameSpan.textContent = data.model_info.name || 'Unknown Model';
  }
  
  // Display statistics
  if (statsText && data.stats) {
    const fmt = (v) => (v!=null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
    let statsHtml = '<h6>Detection Statistics:</h6>';
    if (data.stats.count !== undefined) {
      statsHtml += `<p><strong>Detections:</strong> ${data.stats.count}</p>`;
    }
    if (data.stats.mean_length !== undefined) {
      statsHtml += `<p><strong>Mean Length:</strong> ${fmt(data.stats.mean_length)} px</p>`;
    }
    if (data.stats.mean_width !== undefined) {
      statsHtml += `<p><strong>Mean Width:</strong> ${fmt(data.stats.mean_width)} px</p>`;
    }
    if (data.stats.mean_aspect_ratio !== undefined) {
      statsHtml += `<p><strong>Mean Aspect Ratio:</strong> ${fmt(data.stats.mean_aspect_ratio)}</p>`;
    }
    if (data.stats.mean_confidence !== undefined) {
      statsHtml += `<p><strong>Mean Confidence:</strong> ${(Number(data.stats.mean_confidence) * 100).toFixed(1)}%</p>`;
    }
    statsText.innerHTML = statsHtml;
  }
  
  // Update charts (length, width, aspect ratio)
  const s = data.stats || {};
  renderBarChart('statsChartLen', s.lengths || [], 'Length (px)', '#5b9cff');
  renderBarChart('statsChartWid', s.widths || [], 'Width (px)', '#9cf');
  renderBarChart('statsChartAR', s.aspect_ratios || [], 'Aspect Ratio', '#f59e0b');
}

async function runInference(imageName) {
  if (!imageName) {
    showAlert('warning', 'No image selected for inference.');
    return;
  }

  // Show loading interface
  showLoadingInterface(imageName);

  try {
    const formData = new FormData();
    formData.append('image_name', imageName);

    const response = await fetch('/inference', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.ok) {
      displayInferenceResults(data);
      showAlert('success', `Inference completed for ${imageName}`);
    } else {
      showAlert('danger', `Inference failed: ${data.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.error('Inference error:', error);
    showAlert('danger', `Inference failed: ${error.message}`);
  } finally {
    hideLoadingInterface();
  }
}

function showLoadingInterface(imageName) {
  const loadingInterface = document.getElementById('loading-interface');
  const resultsDisplay = document.getElementById('results-display');
  const emptyState = document.getElementById('empty-state');
  const loadingImageName = document.getElementById('loading-image-name');
  const loadingModelName = document.getElementById('loading-model-name');

  if (loadingInterface) loadingInterface.style.display = 'block';
  if (resultsDisplay) resultsDisplay.style.display = 'none';
  if (emptyState) emptyState.style.display = 'none';
  if (loadingImageName) loadingImageName.textContent = imageName;
  if (loadingModelName) {
    const currentModel = document.getElementById('current-model-name');
    loadingModelName.textContent = currentModel ? currentModel.textContent : 'Unknown';
  }
}

function hideLoadingInterface() {
  const loadingInterface = document.getElementById('loading-interface');
  if (loadingInterface) loadingInterface.style.display = 'none';
}

function updateStatsDisplay(stats) {
  const statsText = document.getElementById('statsText');
  if (statsText) {
    const fmt = (v) => (v != null && !isNaN(v)) ? Number(v).toFixed(2) : '0.00';
    statsText.innerHTML = `
      <div>Count: <strong>${stats.count || 0}</strong></div>
      <div>Mean length: <strong>${fmt(stats.mean_length)}</strong></div>
      <div>Mean width: <strong>${fmt(stats.mean_width)}</strong></div>
      <div>Mean aspect ratio: <strong>${fmt(stats.mean_aspect_ratio)}</strong></div>
    `;
  }
}

function updateStatsCharts(stats) {
  const lengths = stats.lengths || [];
  const widths = stats.widths || [];
  const aspectRatios = stats.aspect_ratios || [];

  // Render binned histograms instead of per-crystal bars
  renderHistogramChart('statsChartLen', lengths, 'Length (px)', '#5b9cff');
  renderHistogramChart('statsChartWid', widths, 'Width (px)', '#9cf');
  renderHistogramChart('statsChartAR', aspectRatios, 'Aspect Ratio', '#f59e0b');
}
