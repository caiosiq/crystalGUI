const API_BASE = "";

let browserState = {
    datasetPath: null,
    isSplit: false,
    split: "train",
    splitCounts: {},
    total: 0,
    offset: 0,
    width: 1024,
    height: 1024,
    obbs: [],
    obbVisible: false,
    loading: false,
};

function qs(id) {
    return document.getElementById(id);
}

function imageUrl(datasetPath, relPath) {
    const params = new URLSearchParams({
        dataset_path: datasetPath,
        rel_path: relPath,
    });
    return `${API_BASE}/training/dataset_image?${params.toString()}`;
}

function samplesUrl(offset, limit, includeLabels) {
    const params = new URLSearchParams({
        dataset_path: browserState.datasetPath,
        split: browserState.split,
        offset: String(offset),
        limit: String(limit),
        include_labels: includeLabels ? "true" : "false",
    });
    return `${API_BASE}/training/dataset_samples?${params.toString()}`;
}

export function resetDatasetBrowser() {
    browserState = {
        datasetPath: null,
        isSplit: false,
        split: "train",
        splitCounts: {},
        total: 0,
        offset: 0,
        width: 1024,
        height: 1024,
        obbs: [],
        obbVisible: false,
        loading: false,
    };
    const browser = qs("datasetBrowser");
    const obbBtn = qs("btnToggleObb");
    if (browser) browser.classList.add("d-none");
    if (obbBtn) {
        obbBtn.classList.add("d-none");
        obbBtn.classList.remove("active");
    }
    hideObbCanvas();
}

export async function initDatasetBrowser(ds) {
    if (!ds || !ds.path) {
        resetDatasetBrowser();
        return;
    }

    browserState.datasetPath = ds.path;
    browserState.isSplit = !!ds.is_split;
    browserState.split = ds.is_split ? "train" : "all";
    browserState.splitCounts = ds.split_counts || {};
    browserState.offset = 0;
    browserState.obbVisible = false;

    const browser = qs("datasetBrowser");
    const obbBtn = qs("btnToggleObb");
    if (browser) browser.classList.remove("d-none");
    if (obbBtn) {
        obbBtn.classList.remove("d-none");
        obbBtn.classList.remove("active");
    }
    hideObbCanvas();

    renderSplitTabs();
    await loadSample(0);
}

function renderSplitTabs() {
    const tabs = qs("datasetSplitTabs");
    if (!tabs) return;

    if (!browserState.isSplit) {
        tabs.classList.add("d-none");
        tabs.innerHTML = "";
        return;
    }

    tabs.classList.remove("d-none");
    tabs.innerHTML = "";

    for (const name of ["train", "val", "test"]) {
        const count = browserState.splitCounts[name] ?? 0;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `btn btn-outline-secondary${browserState.split === name ? " active" : ""}`;
        btn.textContent = `${name.charAt(0).toUpperCase() + name.slice(1)} (${count})`;
        btn.onclick = () => switchSplit(name);
        tabs.appendChild(btn);
    }
}

async function switchSplit(name) {
    if (browserState.split === name) return;
    browserState.split = name;
    browserState.offset = 0;
    renderSplitTabs();
    await loadSample(0);
}

async function loadSample(offset) {
    if (!browserState.datasetPath || browserState.loading) return;
    browserState.loading = true;

    const placeholder = qs("datasetPreviewPlaceholder");
    const img = qs("datasetPreviewImg");
    if (placeholder) {
        placeholder.style.display = "block";
        placeholder.textContent = "Loading preview...";
    }
    if (img) img.style.display = "none";

    try {
        const res = await fetch(samplesUrl(offset, 1, true));
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "Failed to load sample");

        browserState.total = data.total || 0;
        browserState.offset = data.offset || 0;
        browserState.splitCounts = data.split_counts || browserState.splitCounts;
        if (data.is_split !== undefined) browserState.isSplit = data.is_split;

        if (!data.items || data.items.length === 0) {
            if (placeholder) {
                placeholder.textContent = "No images in this split.";
                placeholder.style.display = "block";
            }
            updateControls();
            return;
        }

        const item = data.items[0];
        browserState.width = item.width || 1024;
        browserState.height = item.height || 1024;
        browserState.obbs = item.obbs || [];

        if (img) {
            img.onload = () => {
                if (placeholder) placeholder.style.display = "none";
                img.style.display = "block";
                if (browserState.obbVisible) drawObbs();
            };
            img.onerror = () => {
                if (placeholder) {
                    placeholder.textContent = "Failed to load image.";
                    placeholder.style.display = "block";
                }
            };
            img.src = imageUrl(browserState.datasetPath, item.image_rel);
        }

        const stemEl = qs("datasetPreviewStem");
        if (stemEl) stemEl.textContent = item.stem || "";

        updateControls();
        if (browserState.obbVisible) drawObbs();
    } catch (e) {
        console.error(e);
        if (placeholder) {
            placeholder.textContent = `Error: ${e.message}`;
            placeholder.style.display = "block";
        }
    } finally {
        browserState.loading = false;
    }
}

function updateControls() {
    const total = browserState.total;
    const offset = browserState.offset;
    const idxLabel = qs("datasetIndexLabel");
    const slider = qs("datasetSeekSlider");
    const goTo = qs("datasetGoToIndex");
    const prevBtn = qs("datasetPrevBtn");
    const nextBtn = qs("datasetNextBtn");

    if (idxLabel) {
        idxLabel.textContent = total > 0 ? `${offset + 1} / ${total}` : "0 / 0";
    }
    if (slider) {
        slider.min = "0";
        slider.max = String(Math.max(0, total - 1));
        slider.value = String(Math.max(0, offset));
        slider.disabled = total <= 1;
    }
    if (goTo) {
        goTo.max = String(Math.max(0, total - 1));
        goTo.placeholder = total > 0 ? `0–${total - 1}` : "Index";
    }
    if (prevBtn) prevBtn.disabled = offset <= 0;
    if (nextBtn) nextBtn.disabled = total === 0 || offset >= total - 1;

    renderSplitTabs();
}

function hideObbCanvas() {
    const canvas = qs("datasetObbCanvas");
    if (canvas) canvas.style.display = "none";
}

function drawObbs() {
    const canvas = qs("datasetObbCanvas");
    const wrap = qs("datasetPreviewWrap");
    const img = qs("datasetPreviewImg");
    if (!canvas || !wrap || !img || img.style.display === "none") return;

    requestAnimationFrame(() => {
        const rect = img.getBoundingClientRect();
        const parentRect = wrap.getBoundingClientRect();

        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.left = `${rect.left - parentRect.left}px`;
        canvas.style.top = `${rect.top - parentRect.top}px`;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
        canvas.style.display = browserState.obbVisible ? "block" : "none";

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!browserState.obbVisible || !browserState.obbs.length) return;

        const scaleX = rect.width / browserState.width;
        const scaleY = rect.height / browserState.height;

        ctx.strokeStyle = "rgba(0, 255, 0, 0.7)";
        ctx.lineWidth = 1;

        browserState.obbs.forEach(ob => {
            const cs = ob.corners;
            if (!cs || cs.length < 4) return;
            ctx.beginPath();
            ctx.moveTo(cs[0][0] * scaleX, cs[0][1] * scaleY);
            for (let i = 1; i < 4; i++) {
                ctx.lineTo(cs[i][0] * scaleX, cs[i][1] * scaleY);
            }
            ctx.closePath();
            ctx.stroke();
        });
    });
}

export function toggleDatasetObb() {
    browserState.obbVisible = !browserState.obbVisible;
    const btn = qs("btnToggleObb");
    if (btn) btn.classList.toggle("active", browserState.obbVisible);
    if (browserState.obbVisible) {
        drawObbs();
    } else {
        hideObbCanvas();
    }
}

export async function datasetBrowsePrev() {
    if (browserState.offset <= 0) return;
    await loadSample(browserState.offset - 1);
}

export async function datasetBrowseNext() {
    if (browserState.offset >= browserState.total - 1) return;
    await loadSample(browserState.offset + 1);
}

export async function datasetBrowseRandom() {
    if (browserState.total <= 0) return;
    const idx = Math.floor(Math.random() * browserState.total);
    await loadSample(idx);
}

export async function datasetBrowseGoTo() {
    const input = qs("datasetGoToIndex");
    if (!input || browserState.total <= 0) return;
    let idx = parseInt(input.value, 10);
    if (Number.isNaN(idx)) return;
    idx = Math.max(0, Math.min(browserState.total - 1, idx));
    await loadSample(idx);
}

export function bindDatasetBrowserEvents() {
    const slider = qs("datasetSeekSlider");
    if (slider) {
        slider.addEventListener("input", () => {
            const idxLabel = qs("datasetIndexLabel");
            const total = browserState.total;
            const val = parseInt(slider.value, 10);
            if (idxLabel && total > 0) {
                idxLabel.textContent = `${val + 1} / ${total}`;
            }
        });
        slider.addEventListener("change", () => {
            const val = parseInt(slider.value, 10);
            if (!Number.isNaN(val)) loadSample(val);
        });
    }

    window.addEventListener("resize", () => {
        if (browserState.obbVisible) drawObbs();
    });

    document.addEventListener("keydown", (e) => {
        if (!browserState.datasetPath) return;
        const tag = (e.target && e.target.tagName) || "";
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (e.key === "ArrowLeft") {
            e.preventDefault();
            datasetBrowsePrev();
        } else if (e.key === "ArrowRight") {
            e.preventDefault();
            datasetBrowseNext();
        }
    });
}
