# UI Integration Roadmap for DiffOSOG

This roadmap outlines the plan to integrate the **DiffOSOG Calibration Engine** into the **OSOG Playground** UI. The goal is to enable users to run parameter optimization tasks directly from the web interface, either against real images ("Optimize") or synthetic ground truth ("Validate").

## 1. Architecture Overview

### Backend (FastAPI)
- **Async Job Management**: Optimization is a long-running process (seconds to minutes). We will use a job-based system similar to the existing `OUTPUTS_JOBS` or `SYNTH_JOBS`.
- **Background Worker**: A dedicated thread/process will run the `CalibrationEngine.calibrate()` loop.
- **State Tracking**: A global dictionary `CALIBRATION_JOBS` will store live progress (step, loss, current parameters, intermediate images).
- **Endpoints**:
  - `POST /api/calibrate/start`: Initialize and start a job.
  - `GET /api/calibrate/status/{job_id}`: Poll for progress updates (metrics + intermediate image URL).
  - `POST /api/calibrate/stop/{job_id}`: Signal the engine to abort.
  - `POST /api/compare`: Static comparison of Real vs. Synthetic (Current Params).

### Frontend (Playground)
- **New Tab/Section**: "Optimize & Compare" (replacing or augmenting the existing "Validate" section).
- **Three Modes**:
  1.  **Compare**: Static side-by-side of Real Image vs. Synthetic (generated with current UI params).
  2.  **Optimize (Real)**: Optimize parameters to match an uploaded Real Image.
  3.  **Validate (Synthetic)**: Optimize parameters to match a Synthetic Image generated from hidden Ground Truth (GT) parameters.

---

## 2. Implementation Phases

### Phase 1: Backend Infrastructure
**Goal**: Enable the Python backend to run the calibration engine asynchronously and report status.

1.  **Job Manager**:
    - Create `CALIBRATION_JOBS` store in `app/main.py`.
    - Implement `_calibration_worker` function:
        - Instantiates `CalibrationEngine`.
        - Defines a `progress_callback` that updates the job state.
        - Handles image generation for previews (e.g., every 5 or 10 steps).
2.  **API Endpoints**:
    - `POST /api/calibrate/start`:
        - Inputs: `target_image_path` (or `gt_config`), `initial_config`, `selected_params` (list), `settings` (max_steps, etc.).
        - Returns: `job_id`.
    - `GET /api/calibrate/status`:
        - Returns: `step`, `total_steps`, `current_loss`, `stage`, `param_values`, `preview_url` (optional).
    - `POST /api/calibrate/stop`: Terminates the job.

### Phase 2: Frontend "Compare" Mode
**Goal**: visual confirmation of the starting point before optimization.

1.  **UI Layout**:
    - **Left**: Target Image Uploader (Dropzone).
    - **Right**: Current Synthetic Output (Standard Preview).
    - **Action**: "Compare" button.
2.  **Logic**:
    - Send Real Image + Current Config to `POST /api/compare`.
    - Backend computes losses (VGG, Gram, Histogram) without optimizing.
    - Display results:
        - **Visual**: Overlay slider (Real <-> Synthetic).
        - **Metrics**: Display the calculated loss values.

### Phase 3: Frontend "Optimize" Mode
**Goal**: The core feature—running the calibration loop.

1.  **Parameter Selection UI**:
    - Add checkboxes next to existing sliders in the Playground? **Or** a dedicated "Optimization Config" panel.
    - **Recommendation**: A dedicated list in the Optimize tab.
        - Group by **Stage**: "Geometry" (Camera, Object) and "Texture" (Noise, Blur).
        - Checkboxes to "Enable Optimization" for each parameter.
2.  **Controls**:
    - "Start Optimization" button.
    - "Stop" button.
    - Settings: `Max Steps`, `Learning Rate` (simple/advanced toggle).
3.  **Live Feedback**:
    - **Progress Bar**: Show overall progress and current Stage (Geometry vs Texture).
    - **Live Preview**: Update the synthetic image view every N steps.
    - **Loss Graph**: Simple line chart (using Chart.js or Plotly) showing Total Loss over time.

### Phase 4: "Validate" Mode (Synthetic GT)
**Goal**: Educational/Debugging tool to verify engine convergence.

1.  **Workflow**:
    - User sets "Target" parameters (GT) in a separate config panel (or loads a preset).
    - User scrambles "Current" parameters (Starting Point).
    - System generates the "Target Image" internally using GT params.
    - Optimization runs trying to move "Current" -> "Target".
2.  **Visualization**:
    - Show **Target Value** vs **Current Value** for each optimized parameter.
    - Success metric: Did it converge to the GT value?

### Phase 5: Refinement & Polish
1.  **Result Application**:
    - When optimization finishes, provide an "Apply to UI" button to update the main Playground sliders with the new optimized values.
2.  **History/Logs**:
    - Allow downloading the `param_history.csv` for detailed analysis.
3.  **Error Handling**:
    - Handle divergence (NaN losses).
    - Handle memory issues (if running on GPU).

---

## 3. Technical Specifications

### Data Structures

**Job Status Object**:
```json
{
  "job_id": "uuid",
  "status": "running", // running, finished, error, stopped
  "step": 45,
  "max_steps": 200,
  "stage": "Geometry",
  "loss": 0.1234,
  "losses_breakdown": {"vgg": 0.1, "gram": 0.02},
  "current_params": {"optics.focus_z": 1.2, ...},
  "preview_url": "/static/results/calibration/job_id/step_45.jpg",
  "history": {
      "loss": [0.5, 0.4, ...],
      "steps": [0, 1, ...]
  }
}
```

### File Organization
- **`app/routers/calibration.py`**: (Optional) Separate router if `main.py` gets too large.
- **`app/static/js/calibration.js`**: Frontend logic for the optimize tab.

## 4. Next Steps
1.  Implement **Phase 1** (Backend Endpoints).
2.  Implement **Phase 2** (Compare UI).
3.  Implement **Phase 3** (Optimize UI & Loop).
