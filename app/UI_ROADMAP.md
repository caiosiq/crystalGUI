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

## Phase 2: 3D Interaction & Habits
*Goal: Visualize and manipulate the 3D nature of the particles.*

- [ ] **Live 3D Preview Widget**
    - [ ] A WebGL/Three.js viewport showing a wireframe or simplified representation of the particle distribution in 3D space.
    - [ ] Visual feedback for rotation (tumble/roll) distributions.
- [ ] **Morphology Designer**
    - [ ] **Habit Editor**: Visual curve editor for defining crystal aspect ratio distributions (not just min/max sliders).
    - [x] **Agglomeration Controls**: Settings for cluster tightness, stacking probability, and overlap density.
    - [x] **Surface Roughness**: Slider for smooth vs. jagged edges.
    - [x] **Polarity Switch**: Toggle between light/dark crystals (birefringence simulation).

## Phase 3: Advanced Optics Laboratory
*Goal: Expose the full power of the Multi-Modal Optics engine.*

- [ ] **Polarization Studio**
    - [ ] **Michel-Levy Chart**: Visual reference for interference colors.
    - [ ] **Birefringence Sliders**: Control the optical properties of the material itself.
- [ ] **Depth of Field & Focus**
    - [ ] **Focus Slider**: Manually move the focal plane through the Z-stack.
    - [ ] **Aperture Control**: Adjust the simulated numerical aperture (NA) to change depth of field.
- [ ] **Fluorescence & Lighting**
    - [ ] **Channel Mixer**: Pick "dyes" and assign colors to different channels.
    - [ ] **Lighting Director**: Adjust the angle and intensity of the virtual light source (for DIC/Shadowgraphy).

## Phase 4: Workflow & Engineering
*Goal: Make the tool usable for production datasets.*

- [ ] **Preset Management System**
    - [ ] **Load/Save/Delete**: Full CRUD for configuration presets.
    - [ ] **Preset Gallery**: Visual browser of saved presets with thumbnails.
- [ ] **Dataset Generation Queue**
    - [ ] **Batch Job Config**: UI to set up a bulk generation run (e.g., "Generate 10k images with these settings").
    - [ ] **Progress Monitor**: Real-time progress bar and estimated time remaining.
- [ ] **Validation Dashboard**
    - [ ] **Real vs. Synth Side-by-Side**: Drag-and-drop a real image to compare with the current synthetic output.
    - [ ] **Metric Display**: Show calculated stats (e.g., coverage fraction, particle count) in real-time.

## Phase 5: The "Digital Twin" Experience
*Goal: Seamless immersion.*

- [ ] **"Smart" Auto-Tune**
    - [ ] Button to "Optimize parameters to match uploaded image" (Frontend for the Inverse Rendering feature).
- [ ] **Neural Style Transfer Toggle**
    - [ ] On/Off switch for the GAN-based texture refiner.
- [ ] **Export Studio**
    - [ ] Download current view as PNG/TIFF/JSON (labels).
    - [ ] One-click "Send to Training Folder".
