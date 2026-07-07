/* preprocess/state.js */

let preprocCurrentPresetName = null; // track current preset name in Preprocess

let preprocImg = null; // original image element

let preprocCanvasOrig = null;

let preprocCanvasProc = null;

let compareChart = null;

let preprocParams = { desaturate: 0, invert: false, gradient_strength: 0, clahe: false, equalize: false, clahe_clip_limit: 2.0, clahe_tile_grid: 8 };

let preprocModel = null; // independent model selection for Preprocess tab

let preprocLoadingEl = null; // loading overlay element for preprocess actions

let preprocBaseImg = null; // server-processed preview image (full pipeline)

let preprocPreviewCache = new Map(); // cache of server-preprocessed images keyed by image+pipeline

let preprocWaitingForBase = false; // guards applying client ops until server base is ready

let preprocInferenceLocked = false; // when true, processed canvas shows inference overlay until params change

let preprocInferenceOverlays = { originalDetections: [], processedBase: null, processedDetections: [] };

let imageZoomPreprocContext = null; // { kind: 'original'|'processed', showObbs: boolean }

let preprocPreviewRequestSeq = 0; // ignore stale preview responses after newer requests

let previewDebounce = null;

let previewAbortController = null;

let previewOverlayGuard = null;
