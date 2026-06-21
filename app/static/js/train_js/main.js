
const API_BASE = "";

// State
let selectedDataset = null;
let selectedJob = null;
let allDatasets = [];

document.addEventListener('DOMContentLoaded', () => {
    fetchDatasets();
    fetchJobs();

    const searchEl = document.getElementById('datasetSearch');
    if (searchEl) {
        searchEl.addEventListener('input', () => renderDatasetsList(allDatasets));
    }
    
    // Auto-refresh jobs
    setInterval(fetchJobs, 5000);
});

function renderDatasetsList(datasets) {
    const list = document.getElementById('datasetsList');
    if (!list) return;
    list.innerHTML = '';

    const query = (document.getElementById('datasetSearch')?.value || '').trim().toLowerCase();
    const filtered = query
        ? datasets.filter(ds =>
            ds.name.toLowerCase().includes(query) ||
            ds.path.toLowerCase().includes(query)
        )
        : datasets;

    if (!filtered.length) {
        list.innerHTML = '<div class="text-center text-muted py-2">No synthetic datasets found</div>';
        return;
    }

    filtered.forEach(ds => {
        const item = document.createElement('a');
        item.className = `list-group-item list-group-item-action bg-transparent text-light border-secondary py-2${selectedDataset && selectedDataset.path === ds.path ? ' active' : ''}`;
        item.href = '#';
        item.onclick = (e) => {
            e.preventDefault();
            selectDataset(ds);
        };

        let badges = '';
        if (ds.has_dota) badges += '<span class="badge bg-info me-1">DOTA</span>';
        if (ds.has_yolo) badges += '<span class="badge bg-primary me-1">YOLO</span>';
        if (ds.is_split) badges += '<span class="badge bg-success me-1">SPLIT</span>';

        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span class="fw-bold text-truncate" title="${ds.name}">${ds.name}</span>
                <span class="badge bg-dark border border-secondary">${ds.image_count}</span>
            </div>
            <div class="mt-1">${badges}</div>
            <div class="small text-muted text-truncate" style="font-size:0.7rem;">${ds.path}</div>
        `;
        list.appendChild(item);
    });
}

async function fetchDatasets() {
    try {
        const res = await fetch(`${API_BASE}/training/datasets`);
        const data = await res.json();

        if (data.ok && data.datasets) {
            allDatasets = data.datasets;
            renderDatasetsList(allDatasets);

            // Re-sync sidebar selection and prep controls after refresh
            if (selectedDataset) {
                const refreshed = data.datasets.find(d => d.path === selectedDataset.path);
                if (refreshed) selectDataset(refreshed);
            }
        }
    } catch (e) {
        console.error("Failed to fetch datasets", e);
    }
}

function selectDataset(ds) {
    selectedDataset = ds;
    document.getElementById('selectedDatasetLabel').textContent = ds.name;
    document.getElementById('selectedDatasetLabel').className = 'text-success small fw-bold';
    
    // Update status panel
    const statusList = document.getElementById('datasetStatusList');
    statusList.innerHTML = `
        <li class="list-group-item bg-transparent text-light">Path: <span class="text-muted">${ds.path}</span></li>
        <li class="list-group-item bg-transparent text-light">Images: <span class="text-muted">${ds.image_count}</span></li>
        <li class="list-group-item bg-transparent text-light">DOTA Labels: <span class="${ds.has_dota ? 'text-success' : 'text-danger'}">${ds.has_dota ? 'Found' : 'Missing'}</span></li>
        <li class="list-group-item bg-transparent text-light">YOLO Labels: <span class="${ds.has_yolo ? 'text-success' : 'text-danger'}">${ds.has_yolo ? 'Found' : 'Missing'}</span></li>
        <li class="list-group-item bg-transparent text-light">Split (Train/Val/Test): <span class="${ds.is_split ? 'text-success' : 'text-danger'}">${ds.is_split ? 'Done' : 'Not Split'}</span></li>
    `;

    updatePrepControls(ds);
}

function datasetReadyToSplit(ds) {
    if (!ds) return false;
    // Support older API responses that only expose has_yolo
    return !!(ds.has_yolo_for_split ?? ds.has_yolo);
}

function updatePrepControls(ds) {
    const splitBtn = document.getElementById('splitBtn');
    const splitHint = document.getElementById('splitHint');
    if (!splitBtn || !splitHint) return;

    if (!ds) {
        splitBtn.disabled = true;
        splitHint.textContent = 'Select a dataset from the sidebar to enable splitting.';
        splitHint.className = 'small text-muted mt-2';
        return;
    }

    if (ds.is_split) {
        splitBtn.disabled = true;
        splitHint.textContent = 'Dataset is already split.';
        splitHint.className = 'small text-warning mt-2';
    } else if (!datasetReadyToSplit(ds)) {
        splitBtn.disabled = true;
        splitHint.textContent = 'Step 1 required: convert DOTA labels to YOLO in labels/ before splitting.';
        splitHint.className = 'small text-warning mt-2';
    } else {
        splitBtn.disabled = false;
        splitHint.textContent = 'Ready: will split images/ and labels/ together into train, val, and test.';
        splitHint.className = 'small text-success mt-2';
    }
}

async function runConvertLabels() {
    if (!selectedDataset) return alert("Select a dataset first!");
    
    const width = document.getElementById('imgWidth').value;
    const height = document.getElementById('imgHeight').value;
    const btn = event.target;
    
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Converting...';
    
    try {
        const fd = new FormData();
        fd.append('dataset_path', selectedDataset.path);
        fd.append('width', width);
        fd.append('height', height);
        
        const res = await fetch(`${API_BASE}/training/convert_labels`, { method: 'POST', body: fd });
        const data = await res.json();
        
        if (data.ok) {
            document.getElementById('convertResult').innerHTML = `<span class="text-success">Converted ${data.converted} files.</span>`;
            await fetchDatasets();
            if (selectedDataset) {
                const res = await fetch(`${API_BASE}/training/datasets`);
                const dsData = await res.json();
                const refreshed = dsData.datasets?.find(d => d.path === selectedDataset.path);
                if (refreshed) selectDataset(refreshed);
            }
        } else {
            document.getElementById('convertResult').innerHTML = `<span class="text-danger">Error: ${data.error}</span>`;
        }
    } catch (e) {
        document.getElementById('convertResult').innerHTML = `<span class="text-danger">Error: ${e.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Convert';
    }
}

async function runSplitData() {
    if (!selectedDataset) return alert("Select a dataset first!");
    if (!datasetReadyToSplit(selectedDataset)) {
        return alert("Convert DOTA labels to YOLO first (step 1).");
    }
    if (selectedDataset.is_split) {
        return alert("Dataset is already split.");
    }
    
    const train = document.getElementById('splitTrain').value;
    const val = document.getElementById('splitVal').value;
    const test = document.getElementById('splitTest').value;
    const btn = event.target;
    
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Splitting...';
    
    try {
        const fd = new FormData();
        fd.append('dataset_path', selectedDataset.path);
        fd.append('train_ratio', train);
        fd.append('val_ratio', val);
        fd.append('test_ratio', test);
        
        const res = await fetch(`${API_BASE}/training/split_data`, { method: 'POST', body: fd });
        const data = await res.json();
        
        if (data.ok) {
            document.getElementById('splitResult').innerHTML = `<span class="text-success">Split done. Train: ${data.counts.train}, Val: ${data.counts.val}, Test: ${data.counts.test}</span>`;
            await fetchDatasets();
            if (selectedDataset) {
                const res = await fetch(`${API_BASE}/training/datasets`);
                const dsData = await res.json();
                const refreshed = dsData.datasets?.find(d => d.path === selectedDataset.path);
                if (refreshed) selectDataset(refreshed);
            }
        } else {
            document.getElementById('splitResult').innerHTML = `<span class="text-danger">Error: ${data.error}</span>`;
        }
    } catch (e) {
        document.getElementById('splitResult').innerHTML = `<span class="text-danger">Error: ${e.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Split Dataset';
    }
}

async function startTrainingJob() {
    if (!selectedDataset) return alert("Select a dataset first!");
    
    const model = document.getElementById('trainModel').value;
    const epochs = document.getElementById('trainEpochs').value;
    const batch = document.getElementById('trainBatch').value;
    const imgSize = document.getElementById('trainImgSize').value;
    const partition = document.getElementById('slurmPartition').value;
    const gpu = document.getElementById('slurmGpu').value;
    const time = document.getElementById('slurmTime').value;
    
    if(!confirm(`Start training ${model} on ${selectedDataset.name}?`)) return;
    
    const btn = event.target;
    btn.disabled = true;
    
    try {
        const fd = new FormData();
        fd.append('dataset_path', selectedDataset.path);
        fd.append('model_name', model);
        fd.append('epochs', epochs);
        fd.append('batch_size', batch);
        fd.append('img_size', imgSize);
        fd.append('partition', partition);
        fd.append('gpu', gpu);
        fd.append('time_limit', time);
        
        const res = await fetch(`${API_BASE}/training/start`, { method: 'POST', body: fd });
        const data = await res.json();
        
        if (data.ok) {
            alert(`Job Submitted! ID: ${data.job_id}`);
            fetchJobs();
            // Switch to monitor tab
            const tab = new bootstrap.Tab(document.querySelector('#monitor-tab'));
            tab.show();
            selectJob(data.job_id);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
    }
}

async function fetchJobs() {
    try {
        const res = await fetch(`${API_BASE}/training/jobs`);
        const data = await res.json();
        
        const list = document.getElementById('activeJobsList');
        list.innerHTML = '';
        
        if (data.ok && data.jobs) {
            if (data.jobs.length === 0) {
                list.innerHTML = '<div class="text-center text-muted">No active jobs</div>';
                return;
            }
            
            // Sort by submitted_at desc
            data.jobs.sort((a, b) => new Date(b.submitted_at) - new Date(a.submitted_at));
            
            data.jobs.forEach(job => {
                const item = document.createElement('a');
                item.className = `list-group-item list-group-item-action bg-transparent text-light border-secondary py-2 ${selectedJob === job.job_id ? 'active' : ''}`;
                item.href = '#';
                item.onclick = (e) => {
                    e.preventDefault();
                    selectJob(job.job_id);
                };
                
                let statusColor = 'secondary';
                if (job.status === 'running') statusColor = 'primary';
                if (job.status === 'completed') statusColor = 'success';
                if (job.status === 'error') statusColor = 'danger';
                
                item.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="fw-bold text-truncate" title="${job.job_id}">${job.job_id}</span>
                        <span class="badge bg-${statusColor}">${job.status}</span>
                    </div>
                    <div class="small text-muted mt-1">
                        <div>Model: ${job.model_name}</div>
                        <div>Slurm ID: ${job.slurm_id || 'N/A'}</div>
                    </div>
                `;
                list.appendChild(item);
            });
        }
    } catch (e) {
        console.error("Failed to fetch jobs", e);
    }
}

function selectJob(jobId) {
    selectedJob = jobId;
    document.getElementById('monitorJobTitle').textContent = `Job: ${jobId}`;
    fetchJobs(); // Update UI active state
    refreshLog();
}

async function refreshLog() {
    if (!selectedJob) return;
    
    const pre = document.getElementById('jobLogContent');
    pre.textContent = "Loading log...";
    
    try {
        const res = await fetch(`${API_BASE}/training/logs/${selectedJob}`);
        const data = await res.json();
        
        if (data.ok) {
            pre.textContent = data.log || "Log is empty or not found yet.";
            // Auto scroll to bottom
            pre.scrollTop = pre.scrollHeight;
        } else {
            pre.textContent = "Failed to load log.";
        }
    } catch (e) {
        pre.textContent = "Error loading log: " + e.message;
    }
}

async function refreshTrainedModels() {
    const table = document.getElementById('trainedModelsTable');
    table.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Loading...</td></tr>';
    
    try {
        const res = await fetch(`${API_BASE}/training/trained_models`);
        const data = await res.json();
        
        table.innerHTML = '';
        if (data.ok && data.models && data.models.length > 0) {
            data.models.forEach(m => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${m.name}</td>
                    <td>${m.date}</td>
                    <td>${m.size_mb} MB</td>
                    <td>
                        <button class="btn btn-sm btn-outline-success" onclick="prepDeploy('${m.name}', '${m.path.replace(/\\/g, '\\\\')}')">
                            <i class="bi bi-cloud-upload"></i> Select
                        </button>
                    </td>
                `;
                table.appendChild(tr);
            });
        } else {
            table.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No trained models found.</td></tr>';
        }
    } catch (e) {
        table.innerHTML = `<tr><td colspan="4" class="text-center text-danger">Error: ${e.message}</td></tr>`;
    }
}

function prepDeploy(name, path) {
    document.getElementById('deployConfigCard').classList.remove('d-none');
    document.getElementById('deploySrcName').textContent = name;
    document.getElementById('deploySrcPath').value = path;
    document.getElementById('deployDestName').value = name;
}

async function runDeployModel() {
    const srcPath = document.getElementById('deploySrcPath').value;
    const destName = document.getElementById('deployDestName').value;
    
    if (!destName) return alert("Enter a model name");
    
    const btn = event.target;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Deploying...';
    
    try {
        const fd = new FormData();
        fd.append('weights_path', srcPath);
        fd.append('model_name', destName);
        
        const res = await fetch(`${API_BASE}/training/deploy_model`, { method: 'POST', body: fd });
        const data = await res.json();
        
        if (data.ok) {
            alert(`Model deployed successfully to models/${destName}`);
            document.getElementById('deployConfigCard').classList.add('d-none');
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Deploy Model';
    }
}

// Add event listener for tab change to refresh models
document.getElementById('deploy-tab').addEventListener('shown.bs.tab', () => {
    refreshTrainedModels();
});

// Make functions global
window.runConvertLabels = runConvertLabels;
window.runSplitData = runSplitData;
window.startTrainingJob = startTrainingJob;
window.refreshLog = refreshLog;
window.selectDataset = selectDataset;
window.selectJob = selectJob;
window.refreshTrainedModels = refreshTrainedModels;
window.prepDeploy = prepDeploy;
window.runDeployModel = runDeployModel;
