# OSOG Lab UI Roadmap: The Cockpit for Synthetic Data

This roadmap outlines the evolution of the OSOG Lab (GUI) to match the capabilities of the underlying OSOG engine. The goal is to provide a professional, intuitive "Flight Simulator" interface for data synthesis.

## Current Status (v1.0 - "The Playground")
*Last Updated: 2026-01-15*

**Existing Features:**
*   Basic Parameter Controls (Rods, Plates, Spheres, Cubes).
*   Optical Mode Selection (DIC, Brightfield, etc.).
*   Live Preview with debounced rendering.
*   Multi-head visualization (Optical, Height, Depth, Mask).
*   Basic OBB (Oriented Bounding Box) overlay.

---

## Phase 1: The "Artifact Designer" (Negative Classes)
*Goal: Granular control over the "Enemies of the Probe" to fine-tune False Positive robustness.*

- [x] **Advanced Artifact Controls**
    - [x] **Bubble & Droplet Panel**: Sliders for density, size distribution, and "stickiness" (tendency to attach to crystals).
    - [x] **Fouling Mixer**: A visual mixer to blend "Lens Dirt", "Biofilm", and "Smudges" independently.
    - [x] **Defect Injector**: Controls to introduce cracks, solvent inclusions, and broken crystals. (Implemented as Shape Modes: Wavy, Kink, Noisy)

## Phase 2: 3D Interaction & Habits (Completed)
*Goal: Bridge the gap between 2D images and 3D reality.*

- [x] **Live 3D Preview Widget**
    - [x] **WebGL/Three.js Viewer**: Interactive 3D scene showing the true geometry of the simulation.
    - [x] **Sync**: Updates in real-time as parameters change.
- [x] **Morphology Designer: Habit Editor**
    - [x] **Visual Sliders**: Controls for aspect ratio, thickness, and shape mode (Tumble/Straight).

## Phase 3: Advanced Optics Laboratory (Completed)
*Goal: Simulate the microscope hardware itself.*

- [x] **Polarization Studio**
    - [x] **Birefringence Control**: Material-specific birefringence slider/presets.
    - [x] **Polarizer Angle**: Rotate the virtual polarizer (0-180°).
- [x] **Depth of Field & Focus**
    - [x] **Focus Slider**: Manually sweep through the Z-stack.
    - [x] **Aperture Control**: Adjust Numerical Aperture to control depth-of-field blur intensity.
- [x] **Fluorescence & Lighting**
    - [x] **Lighting Director**: Rotate the shear angle for DIC (Lighting Angle).

## Phase 4: Workflow & Engineering
*Goal: Make the tool usable for production datasets.*

- [x] **Preset Management System**
    - [x] **Load/Save/Delete**: JSON-based configuration storage.
    - [x] **Preset Gallery**: Dropdown list of available presets (User + Built-in).
- [x] **Dataset Generation Queue**
    - [x] **Batch Job Config**: UI to set up a bulk generation run (Count, Tasks, Directory).
    - [x] **Progress Monitor**: Table showing active jobs, status, and completion %.
    - [x] **Job Control**: Ability to cancel/delete jobs.
- [x] **Validation Dashboard**
    - [x] **Real vs. Synth Side-by-Side**: Drag-and-drop a real image to compare with the current synthetic output.
    - [x] **Metric Display**: Show calculated stats (e.g., coverage fraction, particle count) in real-time.

## Phase 4.5: Interactive Annotation & "Human-in-the-Loop" (Immediate)
*Goal: Bridge the gap between the synthetic engine and the real-world validation workflow.*

- [ ] **Interactive Annotation Tool**
    - [ ] **Bounding Box Editor**: Allow users to draw/edit bounding boxes directly on uploaded real images in the "Validate" tab.
    - [ ] **Save Annotations**: Export user-drawn boxes to YOLO format (txt) to start building a "Gold Standard" validation set.
- [ ] **Failure Case Highlighter**
    - [ ] **"Mark as Failure" Button**: A one-click action in the Validation tab to flag a real image where the current synthetic model performs poorly.
    - [ ] **Visual Diff**: Highlight regions where the synthetic distribution (histograms) diverges significantly from the real image stats.
- [ ] **Playground "Quick-Sim" Integration**
    - [ ] **"Simulate Like This"**: A button in the Inference tab that jumps to the Playground and attempts to pre-load parameters that match the detected particle count and size of the current real image.

## Phase 5: Advanced Visualization (Future)
*Goal: Seamless immersion.*

- [ ] **"Smart" Auto-Tune**
    - [ ] Button to "Optimize parameters to match uploaded image" (Frontend for the Inverse Rendering feature).
- [ ] **Neural Style Transfer Toggle**
    - [ ] On/Off switch for the GAN-based texture refiner.
- [ ] **Export Studio**
    - [ ] Download current view as PNG/TIFF/JSON (labels).
    - [ ] One-click "Send to Training Folder".

## Phase 6: Advanced Interactivity & Tools
*Goal: Boost productivity and experimentation speed.*

- [ ] **Comparison Mode (A/B Testing)**
    - [ ] **Split-Screen View**: Compare two different parameter sets side-by-side.
    - [ ] **Real vs. Synth Overlay**: Onion-skinning (transparency) overlay of a real reference image on top of the synthetic view.
- [ ] **History & Undo Stack**
    - [ ] **Parameter History**: "Ctrl+Z" support for slider changes.
    - [ ] **Snapshot Timeline**: Visual timeline of generated previews to quickly jump back to a "good" state.
- [ ] **Scenario Wizard**
    - [ ] **Guided Setup**: A step-by-step wizard to configure complex scenarios (e.g., "I want to simulate a filter clogging event").
    - [ ] **Difficulty Presets**: One-click setup for "Easy", "Medium", and "Hard" detection challenges.

## Phase 7: Differentiable & AI Tools
*Goal: Interfaces for the "Smart" Engine.*

- [ ] **Inverse Rendering Dashboard**
    - [ ] **"Fit to Image" Wizard**: Upload a real image, select a region, and watch OSOG iteratively adjust parameters (via Gradient Descent) to match the appearance.
    - [ ] **Loss Landscape Visualization**: See how sensitive the image is to specific parameters (e.g., Focus vs. Thickness).
- [ ] **Learned Component Designer**
    - [ ] **Aperture Editor**: Draw or evolve custom condenser apertures to test novel optical setups.
- [ ] **Wave Visualization**
    - [ ] **Phase View**: Visualize the raw Optical Path Difference (OPD) map before it hits the sensor.
    - [ ] **Interference Pattern Inspector**: Zoom in to see the constructive/destructive interference fringes.
