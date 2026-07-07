/* outputs/state.js */

let outputsModel = null; // selected model for Outputs tab

let outputsCurrentPresetName = null; // track current preset name in Outputs

let outputsBatchSummary = null;

let outputsTimeUnit = 'min';

let outputsBatchPerImage = [];

let outputsDrillSelectedName = '';

let outputsUploadedDatasets = [];

let outputsSelectedDatasetPath = '';

let outputsSelectedDataset = null;

let outputsScaleByDataset = {};

let outputsScaleSampleIndex = 0;

let outputsScaleLinePoints = []; // image-space [{x,y}, {x,y}]

const OUTPUTS_EVOLUTION_CHART_SPECS = [
