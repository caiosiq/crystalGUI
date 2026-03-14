# Optimization & Validation Mode Roadmap

## 1. Overview
The goal is to implement a "Synthetic vs. Synthetic" validation workflow. This allows users to create a "Ground Truth" (GT) configuration by deviating specific parameters from the current "Synthetic" configuration, and then test if the optimization engine can recover those parameters.

## 2. User Experience (UX) Flow
1.  **Enter Validate Mode**:
    *   **Action**: User clicks "Validate Mode".
    *   **System**:
        *   Initializes `gtConfig` as a clone of the current `synthConfig`.
        *   Displays two viewports side-by-side: Left (Synthetic/Optimizing), Right (Ground Truth).
        *   Opens the "GT Tuner" panel on the right.
        *   Populates "GT Tuner" with controls for all *optimizable* parameters.
2.  **Modify Ground Truth**:
    *   **Action**: User changes a parameter (e.g., `rod_length`) in the "GT Tuner".
    *   **System**:
        *   Updates `gtConfig`.
        *   Marks this parameter as **Diverged**.
        *   Automatically checks this parameter in the "Parameters to Optimize" list (indicating this is a target for optimization).
        *   Regenerates the GT image.
3.  **Modify Designer (Synthetic)**:
    *   **Action**: User changes a parameter in the main Designer sidebar.
    *   **System**:
        *   Updates `synthConfig`.
        *   **Sync Check**:
            *   If the parameter is **NOT** marked as Diverged (i.e., user hasn't manually changed it in GT Tuner), the system *also* updates `gtConfig` to match. (e.g., changing "Optics Mode" updates both).
            *   If the parameter **IS** Diverged (checked for optimization), `gtConfig` remains unchanged (preserving the "target" value).
        *   Regenerates the Synthetic image (and GT image if it was synced).

## 3. Technical Implementation Plan

### 3.1. State Management
We need to introduce new global state variables in `playground.js`:
*   `gtConfig`: Stores the configuration for the Ground Truth image.
*   `divergedParams`: A `Set<string>` containing the IDs/paths of parameters that have been manually modified in the GT Tuner.

### 3.2. GT Tuner Panel (Dynamic Generation)
Unlike the manually crafted Designer sidebar, the GT Tuner should likely be generated dynamically based on the list of optimizable parameters returned by `/calibration/params`.
*   **Challenge**: Mapping the flat parameter paths (e.g., `physics.rod_specs.count_range`) back to friendly UI controls (Sliders/Inputs).
*   **Solution**: We will create a mapping or metadata structure that links the internal config path to:
    *   Label (e.g., "Rod Count")
    *   Type (Range, Select, Checkbox)
    *   Min/Max/Step values (can be derived from `calibration_constraints` or existing HTML attributes).

### 3.3. Synchronization Logic
We will modify the event listeners in `playground.js`.
*   **Current**: `input` event -> `updateLabel` -> `scheduleRegenerate`.
*   **New**: `input` event -> `updateLabel` -> `syncToGT(paramId, value)` -> `scheduleRegenerate`.

**`syncToGT(paramId, value)`**:
```javascript
function syncToGT(paramId, value) {
    const configPath = mapIdToConfigPath(paramId); // Need a helper for this
    if (!divergedParams.has(configPath)) {
        // Update GT Config Value
        setDeepValue(gtConfig, configPath, value);
        // Update GT Tuner UI Control (to reflect the sync)
        updateGTTunerControl(configPath, value);
    }
}
```

### 3.4. Dual Generation
Refactor `regenerate()` to handle the dual-view logic:
*   If `!isValidateMode`: Fetch `/synth_preview` for `synthConfig` only.
*   If `isValidateMode`:
    *   Fetch `/synth_preview` for `synthConfig` (Left View).
    *   Fetch `/synth_preview` for `gtConfig` (Right View).
    *   *Optimization*: Use `Promise.all` to run these in parallel.

### 3.5. Backend Adjustments
*   No major backend changes needed for the generation itself (the existing `/synth_preview` endpoint works for both).
*   The Optimization Job (`/calibration/start`) currently expects a `target_image_name`. We need to support starting an optimization job where the target is a **configuration** (synthetic GT) rather than an uploaded image file.
    *   **Option A**: Generate the GT image, save it temporarily on the server, and pass that filename to the optimizer. (Easiest integration with existing `CalibrationManager`).
    *   **Option B**: Pass `gtConfig` directly to the optimizer and have it generate the reference on the fly.

## 4. Feasibility & Risks

### 4.1. Risks
1.  **Parameter Mapping Complexity**: `playground.js` currently constructs the config object manually in `getConfig()`. There is no centralized "map" of ID -> ConfigPath.
    *   *Mitigation*: We must create a `paramMap` object that explicitly links DOM IDs (e.g., `synRodCountLo`) to Config Paths (e.g., `physics.rod_specs.count_range[0]`). This is the most labor-intensive part but necessary for robust syncing.
2.  **Performance**: Generating two images on every slider drag (debounce 50ms) might be sluggish.
    *   *Mitigation*: Increase debounce time in Validate Mode or optimize the backend preview generation (reduce resolution/quality for previews).
3.  **UI Clutter**: The GT Tuner could become very long if we list *all* optimizable parameters.
    *   *Mitigation*: Only show parameters relevant to the currently enabled features (e.g., don't show Rod params if Rods are disabled), or use a collapsible accordion structure similar to the Designer.

### 4.2. Conclusion
The proposed workflow is feasible. The primary effort will be in the JavaScript layer: refactoring `getConfig` to be bidirectional (or creating the mapping layer) and managing the synchronization state. The backend is largely ready to support this via the existing synthetic generation endpoints.
