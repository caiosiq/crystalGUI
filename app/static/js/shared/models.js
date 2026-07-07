/* shared/models.js */

async function checkGpuAvailability() {
  try {
    const res = await fetch('/system_info');
    const data = await res.json();
    const select = document.getElementById('model-device-select');
    if (data.ok && data.gpu_available && select) {
      select.value = "cuda:0";
    }
  } catch (e) {
    console.warn('Failed to check GPU status', e);
  }
}

async function loadAvailableModels() {
  try {
    // Check GPU first
    checkGpuAvailability();

    const res = await fetch('/model_statuses');
    const data = await res.json();
    if (data.ok && data.statuses && data.statuses.length > 0) {
      createModelButtons(data.statuses);
    } else {
      showNoModelsMessage();
    }
  } catch (error) {
    console.error('Failed to load available models:', error);
    showNoModelsMessage();
  }
}

function showNoModelsMessage() {
  const container = document.getElementById('model-buttons-container');
  container.innerHTML = '<div class="text-warning"><i class="bi bi-exclamation-triangle"></i> No models found. Please check the models folder.</div>';
}

function selectModel(model) {
  // Update UI to show selected model
  const buttons = document.querySelectorAll('#model-buttons-container button');
  buttons.forEach(btn => {
    // Reset all buttons to outline style
    btn.classList.remove('active', 'btn-primary', 'btn-outline-primary');
    btn.classList.add('btn-outline-primary');
    
    // Highlight the selected button
    if (btn.textContent.trim() === model.name) {
      btn.classList.remove('btn-outline-primary');
      btn.classList.add('btn-primary', 'active');
    }
  });
  
  // Load the model from the models folder
  loadModel(model.id);
}

function loadModel(modelId) {
  const formData = new FormData();
  formData.append('folder_path', modelId);
  
  const deviceSelect = document.getElementById('model-device-select');
  if (deviceSelect) {
    formData.append('device', deviceSelect.value);
  }
  
  fetch('/select_model_folder', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.ok) {
      currentModel = data.model;
      updateCurrentModelIndicator(data.model.name);
      showAlert('success', `Model "${data.model.name}" loaded successfully`);
    } else {
      showAlert('danger', `Failed to load model: ${data.error || 'Unknown error'}`);
    }
  })
  .catch(error => {
    console.error('Error loading model:', error);
    showAlert('danger', `Error loading model: ${error.message}`);
  });
}

function updateCurrentModelIndicator(modelName) {
  const indicator = document.getElementById('current-model-name');
  if (indicator) {
    indicator.textContent = modelName;
    indicator.classList.remove('text-muted');
    indicator.classList.add('text-success');
  }
}

function createModelButtons(models) {
  const container = document.getElementById('model-buttons-container');
  if (!container) {
    console.error('Model buttons container not found');
    return;
  }
  
  // Clear existing content
  container.innerHTML = '';
  
  if (!models || models.length === 0) {
    showNoModelsMessage();
    return;
  }
  
  models.forEach(model => {
    const button = document.createElement('button');
    // Format name if missing
    const displayName = model.name || model.id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    model.name = displayName; // Ensure name property exists for selectModel
    
    if (model.status && model.status !== 'ok') {
        button.className = 'btn btn-outline-danger me-2 mb-2';
        button.disabled = true;
        button.title = model.error || 'Unknown error';
        button.innerHTML = `<i class="bi bi-exclamation-triangle"></i> ${displayName}`;
    } else {
        button.className = 'btn btn-outline-primary me-2 mb-2';
        button.textContent = displayName;
        button.addEventListener('click', () => selectModel(model));
    }
    
    container.appendChild(button);
  });
}
