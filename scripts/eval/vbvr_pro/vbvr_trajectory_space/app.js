"use strict";

const state = {
  index: null,
  sample: null,
  domain: "all",
  left: { model: "baseline", sampler: "cps-0.7" },
  right: { model: "checkpoint-2200", sampler: "cps-0.7" },
  view: "step",
  stepIndex: 0,
  autoAdvance: false,
  matrixRenderedKey: null,
};

const elements = {
  archiveStatus: document.querySelector("#archive-status"),
  loadingOverlay: document.querySelector("#loading-overlay"),
  toast: document.querySelector("#toast"),
  heroSamples: document.querySelector("#hero-samples"),
  heroCells: document.querySelector("#hero-cells"),
  heroVideos: document.querySelector("#hero-videos"),
  domain: document.querySelector("#domain-select"),
  task: document.querySelector("#task-select"),
  sampleSelect: document.querySelector("#sample-select"),
  previousSample: document.querySelector("#previous-sample"),
  randomSample: document.querySelector("#random-sample"),
  nextSample: document.querySelector("#next-sample"),
  samplePosition: document.querySelector("#sample-position"),
  sampleDomain: document.querySelector("#sample-domain"),
  sampleSeed: document.querySelector("#sample-seed"),
  sampleName: document.querySelector("#sample-name"),
  sampleRawName: document.querySelector("#sample-raw-name"),
  samplePrompt: document.querySelector("#sample-prompt"),
  compareSection: document.querySelector("#compare-section"),
  viewSwitcher: document.querySelector("#view-switcher"),
  viewDescription: document.querySelector("#view-description"),
  trajectoryNote: document.querySelector("#trajectory-note"),
  playPair: document.querySelector("#play-pair"),
  pausePair: document.querySelector("#pause-pair"),
  restartPair: document.querySelector("#restart-pair"),
  swapSides: document.querySelector("#swap-sides"),
  copyLink: document.querySelector("#copy-link"),
  stepNavigator: document.querySelector("#step-navigator"),
  previousStep: document.querySelector("#previous-step"),
  nextStep: document.querySelector("#next-step"),
  stepNumber: document.querySelector("#step-number"),
  stepSigma: document.querySelector("#step-sigma"),
  stepSlider: document.querySelector("#step-slider"),
  stepStrip: document.querySelector("#step-strip"),
  playPath: document.querySelector("#play-path"),
  matrixDetails: document.querySelector("#matrix-details"),
  matrixTitle: document.querySelector("#matrix-title"),
  matrixDescription: document.querySelector("#matrix-description"),
  matrix: document.querySelector("#trajectory-matrix"),
  playMatrix: document.querySelector("#play-matrix"),
  pauseMatrix: document.querySelector("#pause-matrix"),
};

for (const side of ["left", "right"]) {
  elements[`${side}Model`] = document.querySelector(`#${side}-model`);
  elements[`${side}Sampler`] = document.querySelector(`#${side}-sampler`);
  elements[`${side}Video`] = document.querySelector(`#${side}-video`);
  elements[`${side}Loading`] = document.querySelector(`#${side}-loading`);
  elements[`${side}Label`] = document.querySelector(`#${side}-label`);
  elements[`${side}Meta`] = document.querySelector(`#${side}-meta`);
  elements[`${side}Score`] = document.querySelector(`#${side}-score`);
  elements[`${side}ScoreMean`] = document.querySelector(`#${side}-score-mean`);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function prettyTaskName(name) {
  const withoutSuffix = name.replace(/_data-generator$/, "");
  const [prefix, ...words] = withoutSuffix.split("_");
  const phrase = words.join(" ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  return `${prefix} · ${phrase}`;
}

function domainLabel(domain) {
  return domain === "in-domain" ? "In-Domain" : "Out-of-Domain";
}

function modelById(id) {
  return state.index.models.find((model) => model.id === id);
}

function samplerById(id) {
  return state.index.samplers.find((sampler) => sampler.id === id);
}

function cellFor(selection) {
  return state.index.cells.find(
    (cell) => cell.model === selection.model && cell.sampler === selection.sampler,
  );
}

function scoreFor(selection, sample = state.sample) {
  const cell = cellFor(selection);
  const sampleIndex = state.index.samples.findIndex((candidate) => candidate.id === sample.id);
  return state.index.scores[cell.id][sampleIndex];
}

function formatScore(score) {
  return Number(score).toFixed(state.index.scoreContract.displayPrecision);
}

function scheduleEntry(selection, stepIndex = state.stepIndex) {
  return samplerById(selection.sampler).schedule[stepIndex];
}

function encodePath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

function originalStepFilename(stepIndex) {
  const media = state.index.media;
  return `${media.stepFilenamePrefix}${String(stepIndex).padStart(2, "0")}${media.stepFilenameExtension}`;
}

function mediaUrl(selection, sample, view = state.view, stepIndex = state.stepIndex) {
  const cell = cellFor(selection);
  let prefix = state.index.mediaUrlPrefix;
  let filename;
  if (view === "step") {
    prefix = state.index.stepMediaUrlPrefixes[selection.model];
    filename = originalStepFilename(stepIndex);
  } else {
    filename = view === "grid" ? state.index.media.gridFilename : state.index.media.finalFilename;
  }
  const suffix = `${cell.id}/${encodePath(sample.id)}/${filename}`;
  return `${prefix.replace(/\/$/, "")}/${suffix}`;
}

async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  return response.json();
}

function validCellId(id) {
  return state.index.cells.some((cell) => cell.id === id);
}

function selectionFromCellId(id, fallback) {
  if (!validCellId(id)) return fallback;
  const cell = state.index.cells.find((candidate) => candidate.id === id);
  return { model: cell.model, sampler: cell.sampler };
}

function loadUrlState() {
  const params = new URLSearchParams(window.location.search);
  const requestedSample = state.index.samples.find((sample) => sample.id === params.get("sample"));
  if (requestedSample) {
    state.sample = requestedSample;
    state.domain = requestedSample.domain;
  }
  state.left = selectionFromCellId(params.get("left"), state.left);
  state.right = selectionFromCellId(params.get("right"), state.right);
  if (["step", "grid", "final"].includes(params.get("view"))) state.view = params.get("view");
  const requestedStep = Number.parseInt(params.get("step"), 10);
  if (requestedStep >= 1 && requestedStep <= state.index.stepCount) {
    state.stepIndex = requestedStep - 1;
  }
}

function updateUrl() {
  if (!state.sample) return;
  const params = new URLSearchParams();
  params.set("sample", state.sample.id);
  params.set("left", cellFor(state.left).id);
  params.set("right", cellFor(state.right).id);
  params.set("view", state.view);
  params.set("step", String(state.stepIndex + 1));
  history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => elements.toast.classList.remove("visible"), 1800);
}

function populateStaticControls() {
  const modelOptions = state.index.models
    .map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.label)}</option>`)
    .join("");
  const samplerOptions = state.index.samplers
    .map((sampler) => `<option value="${escapeHtml(sampler.id)}">${escapeHtml(sampler.label)}</option>`)
    .join("");
  for (const side of ["left", "right"]) {
    elements[`${side}Model`].innerHTML = modelOptions;
    elements[`${side}Sampler`].innerHTML = samplerOptions;
    elements[`${side}Model`].value = state[side].model;
    elements[`${side}Sampler`].value = state[side].sampler;
  }
  elements.stepStrip.innerHTML = Array.from(
    { length: state.index.stepCount },
    (_, stepIndex) =>
      `<button type="button" data-step="${stepIndex}" aria-label="Show original step ${stepIndex + 1}">${String(stepIndex + 1).padStart(2, "0")}</button>`,
  ).join("");
  elements.stepSlider.max = String(state.index.stepCount);
  elements.domain.value = state.domain;
  renderStepControls();
  renderViewControls();
}

function filteredTasks() {
  const samples = state.index.samples.filter(
    (sample) => state.domain === "all" || sample.domain === state.domain,
  );
  return [...new Set(samples.map((sample) => sample.task))].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
}

function populateTaskControl(preferredTask = state.sample?.task) {
  const tasks = filteredTasks();
  elements.task.innerHTML = tasks
    .map((task) => `<option value="${escapeHtml(task)}">${escapeHtml(prettyTaskName(task))}</option>`)
    .join("");
  const selectedTask = tasks.includes(preferredTask) ? preferredTask : tasks[0];
  elements.task.value = selectedTask;
  populateSampleControl(selectedTask, state.sample?.id);
}

function populateSampleControl(task, preferredSampleId = null) {
  const samples = state.index.samples.filter((sample) => sample.task === task);
  elements.sampleSelect.innerHTML = samples
    .map(
      (sample, index) =>
        `<option value="${escapeHtml(sample.id)}">Sample ${index + 1} · ${escapeHtml(sample.sample)}</option>`,
    )
    .join("");
  const selected = samples.find((sample) => sample.id === preferredSampleId) ?? samples[0];
  elements.sampleSelect.value = selected.id;
  selectSample(selected, { refreshControls: false });
}

function selectSample(sample, { refreshControls = true } = {}) {
  if (!sample) return;
  stopPathPlayback();
  state.sample = sample;
  state.domain = elements.domain.value;
  if (refreshControls) {
    if (elements.task.value !== sample.task) populateTaskControl(sample.task);
    elements.sampleSelect.value = sample.id;
  }
  renderSample();
  renderCompare();
  renderMatrix();
  updateUrl();
}

function renderSample() {
  const sample = state.sample;
  const globalIndex = state.index.samples.findIndex((candidate) => candidate.id === sample.id);
  elements.samplePosition.textContent = `${globalIndex + 1} / ${state.index.sampleCount}`;
  elements.sampleDomain.textContent = domainLabel(sample.domain);
  elements.sampleSeed.textContent = `seed ${sample.seed}`;
  elements.sampleName.textContent = prettyTaskName(sample.task);
  elements.sampleRawName.textContent = `${sample.task} / ${sample.sample}`;
  elements.samplePrompt.textContent = sample.prompt;
}

function formatSigma(value) {
  return Number(value).toFixed(6);
}

function stepMeta(selection) {
  const preview = scheduleEntry(selection);
  if (preview.kind === "final_latent") {
    return `Step ${String(preview.step).padStart(2, "0")} / ${state.index.stepCount} · final latent · σ 0`;
  }
  return `Step ${String(preview.step).padStart(2, "0")} / ${state.index.stepCount} · x₀ from σ ${formatSigma(preview.sourceSigma)}`;
}

function renderSide(side) {
  const selection = state[side];
  const model = modelById(selection.model);
  const sampler = samplerById(selection.sampler);
  const cell = cellFor(selection);
  const video = elements[`${side}Video`];
  const loading = elements[`${side}Loading`];
  elements[`${side}Model`].value = selection.model;
  elements[`${side}Sampler`].value = selection.sampler;
  elements[`${side}Label`].textContent = `${model.shortLabel} · ${sampler.shortLabel}`;
  elements[`${side}Score`].textContent = formatScore(scoreFor(selection));
  elements[`${side}ScoreMean`].textContent = `cell mean ${formatScore(cell.scoreSummary.overall)}`;
  if (state.view === "step") {
    elements[`${side}Meta`].textContent = `${stepMeta(selection)} · native 512×512`;
  } else if (state.view === "grid") {
    elements[`${side}Meta`].textContent = "Compressed 30-step overview · 6×5 grid";
  } else {
    elements[`${side}Meta`].textContent = "512×512 · 81 frames · sigma 0";
  }
  loading.textContent = `Loading ${side === "left" ? "A" : "B"}…`;
  loading.classList.remove("hidden");
  video.dataset.loaded = "false";
  video.pause();
  video.src = mediaUrl(selection, state.sample);
  video.load();
}

function renderStepControls() {
  const displayStep = state.stepIndex + 1;
  elements.stepNumber.textContent = `${String(displayStep).padStart(2, "0")} / ${state.index.stepCount}`;
  elements.stepSlider.value = String(displayStep);
  const leftPreview = scheduleEntry(state.left);
  const rightPreview = scheduleEntry(state.right);
  const sameSigma = leftPreview.sourceSigma === rightPreview.sourceSigma;
  elements.stepSigma.textContent = sameSigma
    ? `source σ ${formatSigma(leftPreview.sourceSigma)}`
    : `A σ ${formatSigma(leftPreview.sourceSigma)} · B σ ${formatSigma(rightPreview.sourceSigma)}`;
  elements.stepStrip.querySelectorAll("[data-step]").forEach((button) => {
    const active = Number(button.dataset.step) === state.stepIndex;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  elements.previousStep.disabled = state.stepIndex === 0;
  elements.nextStep.disabled = state.stepIndex === state.index.stepCount - 1;
}

function renderViewControls() {
  elements.compareSection.dataset.view = state.view;
  elements.stepNavigator.hidden = state.view !== "step";
  elements.viewSwitcher.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === state.view);
  });
  if (state.view === "step") {
    elements.viewDescription.textContent = state.index.media.stepDescription;
    elements.matrixTitle.textContent = `Show all 12 samplers at step ${state.stepIndex + 1}`;
    elements.matrixDescription.textContent =
      `Two models × six matched samplers using their native step_${String(state.stepIndex).padStart(2, "0")}.mp4 files.`;
  } else if (state.view === "grid") {
    elements.viewDescription.textContent = state.index.media.gridDescription;
    elements.matrixTitle.textContent = "Show all 12 compressed trajectory overviews";
    elements.matrixDescription.textContent = "Two models × six matched samplers in the optional 6×5 overview format.";
  } else {
    elements.viewDescription.textContent = state.index.media.finalDescription;
    elements.matrixTitle.textContent = "Show all 12 native final outputs";
    elements.matrixDescription.textContent = "Two models × six matched samplers at sigma zero.";
  }
}

function renderCompare() {
  renderStepControls();
  renderViewControls();
  renderSide("left");
  renderSide("right");
  elements.trajectoryNote.textContent =
    `${state.index.trajectorySemantics} ` +
    "Displayed EvalKit scores are final-output scores only (step 30 / final); intermediate previews are not rescored.";
}

function changeSelection(side) {
  stopPathPlayback({ pause: true });
  state[side] = {
    model: elements[`${side}Model`].value,
    sampler: elements[`${side}Sampler`].value,
  };
  renderStepControls();
  renderSide(side);
  updateUrl();
}

function filteredSamplesForNavigation() {
  return state.index.samples.filter(
    (sample) => state.domain === "all" || sample.domain === state.domain,
  );
}

function navigateSample(offset) {
  const samples = filteredSamplesForNavigation();
  const index = samples.findIndex((sample) => sample.id === state.sample.id);
  const next = samples[(index + offset + samples.length) % samples.length];
  if (elements.task.value !== next.task) populateTaskControl(next.task);
  selectSample(next);
}

function randomSample() {
  const samples = filteredSamplesForNavigation();
  if (samples.length < 2) return;
  let next = state.sample;
  while (next.id === state.sample.id) next = samples[Math.floor(Math.random() * samples.length)];
  if (elements.task.value !== next.task) populateTaskControl(next.task);
  selectSample(next);
}

function pairVideos() {
  return [elements.leftVideo, elements.rightVideo];
}

async function playPair({ restart = false } = {}) {
  const videos = pairVideos();
  const target = restart ? 0 : Math.min(...videos.map((video) => video.currentTime || 0));
  videos.forEach((video) => {
    video.currentTime = target;
  });
  await Promise.allSettled(videos.map((video) => video.play()));
}

function pausePair() {
  pairVideos().forEach((video) => video.pause());
}

function stopPathPlayback({ pause = false } = {}) {
  state.autoAdvance = false;
  elements.playPath.textContent = "▶ Play path from here";
  elements.playPath.classList.remove("active");
  if (pause) pausePair();
}

function maybeStartPathStep() {
  if (!state.autoAdvance) return;
  if (pairVideos().every((video) => video.dataset.loaded === "true")) {
    playPair({ restart: true });
  }
}

function maybeAdvancePath() {
  if (!state.autoAdvance || !pairVideos().every((video) => video.ended)) return;
  if (state.stepIndex === state.index.stepCount - 1) {
    stopPathPlayback();
    showToast("Reached step 30");
    return;
  }
  setStep(state.stepIndex + 1, { keepAutoAdvance: true });
}

function startPathPlayback() {
  if (state.view !== "step") {
    state.view = "step";
    renderCompare();
    renderMatrix();
  }
  state.autoAdvance = true;
  elements.playPath.textContent = "■ Stop path playback";
  elements.playPath.classList.add("active");
  maybeStartPathStep();
  updateUrl();
}

function setStep(stepIndex, { keepAutoAdvance = false } = {}) {
  const clamped = Math.max(0, Math.min(state.index.stepCount - 1, stepIndex));
  if (clamped === state.stepIndex && state.view === "step") return;
  if (!keepAutoAdvance) stopPathPlayback();
  state.stepIndex = clamped;
  renderStepControls();
  renderViewControls();
  if (state.view === "step") {
    renderSide("left");
    renderSide("right");
  }
  renderMatrix();
  updateUrl();
}

function matrixRenderKey() {
  return `${state.sample.id}|${state.view}|${state.stepIndex}`;
}

function releaseMatrixVideos() {
  elements.matrix.querySelectorAll("video").forEach((video) => {
    video.pause();
    video.removeAttribute("src");
    video.load();
  });
}

function renderMatrix() {
  renderViewControls();
  if (!elements.matrixDetails.open) {
    state.matrixRenderedKey = null;
    return;
  }
  const key = matrixRenderKey();
  if (state.matrixRenderedKey === key) return;
  releaseMatrixVideos();
  elements.matrix.dataset.view = state.view;
  elements.matrix.innerHTML = state.index.models
    .flatMap((model) =>
      state.index.samplers.map((sampler) => {
        const selection = { model: model.id, sampler: sampler.id };
        const cell = cellFor(selection);
        const score = scoreFor(selection);
        const suffix = state.view === "step" ? ` · step ${state.stepIndex + 1}` : "";
        return `
          <article class="matrix-card">
            <div class="matrix-card-heading">
              <div class="matrix-card-label">
                <span>${escapeHtml(model.shortLabel)}</span>
                <strong>${escapeHtml(sampler.shortLabel)}${escapeHtml(suffix)}</strong>
              </div>
              <div class="matrix-score" title="Final-output EvalKit score; cell mean ${formatScore(cell.scoreSummary.overall)}">
                <span>FINAL SCORE</span>
                <strong>${formatScore(score)}</strong>
              </div>
            </div>
            <video controls playsinline muted preload="metadata" src="${escapeHtml(mediaUrl(selection, state.sample))}"></video>
            <div class="matrix-card-actions">
              <button type="button" data-send-side="left" data-cell="${escapeHtml(cell.id)}">Send to A</button>
              <button type="button" data-send-side="right" data-cell="${escapeHtml(cell.id)}">Send to B</button>
            </div>
          </article>
        `;
      }),
    )
    .join("");
  state.matrixRenderedKey = key;
}

function sendMatrixCell(side, cellId) {
  stopPathPlayback({ pause: true });
  state[side] = selectionFromCellId(cellId, state[side]);
  renderStepControls();
  renderSide(side);
  updateUrl();
  document.querySelector(`.side-${side}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function bindEvents() {
  elements.domain.addEventListener("change", () => {
    state.domain = elements.domain.value;
    populateTaskControl();
  });
  elements.task.addEventListener("change", () => populateSampleControl(elements.task.value));
  elements.sampleSelect.addEventListener("change", () => {
    selectSample(state.index.samples.find((sample) => sample.id === elements.sampleSelect.value));
  });
  elements.previousSample.addEventListener("click", () => navigateSample(-1));
  elements.nextSample.addEventListener("click", () => navigateSample(1));
  elements.randomSample.addEventListener("click", randomSample);

  for (const side of ["left", "right"]) {
    elements[`${side}Model`].addEventListener("change", () => changeSelection(side));
    elements[`${side}Sampler`].addEventListener("change", () => changeSelection(side));
    elements[`${side}Video`].addEventListener("loadeddata", () => {
      elements[`${side}Video`].dataset.loaded = "true";
      elements[`${side}Loading`].classList.add("hidden");
      maybeStartPathStep();
    });
    elements[`${side}Video`].addEventListener("ended", maybeAdvancePath);
    elements[`${side}Video`].addEventListener("error", () => {
      elements[`${side}Loading`].textContent = `Could not load ${side.toUpperCase()}`;
      stopPathPlayback({ pause: true });
    });
  }

  elements.viewSwitcher.addEventListener("click", (event) => {
    const button = event.target.closest("[data-view]");
    if (!button || button.dataset.view === state.view) return;
    stopPathPlayback({ pause: true });
    state.view = button.dataset.view;
    renderCompare();
    renderMatrix();
    updateUrl();
  });
  elements.previousStep.addEventListener("click", () => setStep(state.stepIndex - 1));
  elements.nextStep.addEventListener("click", () => setStep(state.stepIndex + 1));
  elements.stepSlider.addEventListener("change", () => setStep(Number(elements.stepSlider.value) - 1));
  elements.stepStrip.addEventListener("click", (event) => {
    const button = event.target.closest("[data-step]");
    if (button) setStep(Number(button.dataset.step));
  });
  elements.playPath.addEventListener("click", () => {
    if (state.autoAdvance) stopPathPlayback({ pause: true });
    else startPathPlayback();
  });
  elements.playPair.addEventListener("click", () => {
    stopPathPlayback();
    playPair();
  });
  elements.pausePair.addEventListener("click", () => stopPathPlayback({ pause: true }));
  elements.restartPair.addEventListener("click", () => {
    stopPathPlayback();
    playPair({ restart: true });
  });
  elements.swapSides.addEventListener("click", () => {
    stopPathPlayback();
    [state.left, state.right] = [state.right, state.left];
    renderCompare();
    updateUrl();
  });
  elements.copyLink.addEventListener("click", async () => {
    updateUrl();
    try {
      await navigator.clipboard.writeText(window.location.href);
      showToast("Comparison link copied");
    } catch {
      showToast("Copy the URL from your browser");
    }
  });

  elements.matrixDetails.addEventListener("toggle", renderMatrix);
  elements.matrix.addEventListener("click", (event) => {
    const button = event.target.closest("[data-send-side]");
    if (button) sendMatrixCell(button.dataset.sendSide, button.dataset.cell);
  });
  elements.playMatrix.addEventListener("click", () => {
    elements.matrix.querySelectorAll("video").forEach((video) => {
      video.currentTime = 0;
      video.play().catch(() => {});
    });
  });
  elements.pauseMatrix.addEventListener("click", () => {
    elements.matrix.querySelectorAll("video").forEach((video) => video.pause());
  });
  document.addEventListener("keydown", (event) => {
    const tag = event.target.tagName;
    if (state.view !== "step" || ["INPUT", "SELECT", "VIDEO", "BUTTON"].includes(tag)) return;
    if (event.key === "ArrowLeft") setStep(state.stepIndex - 1);
    if (event.key === "ArrowRight") setStep(state.stepIndex + 1);
  });
}

async function initialize() {
  try {
    state.index = await loadJson("./data/index.json");
    if (
      state.index.schemaVersion < 3 ||
      !state.index.stepMediaUrlPrefixes ||
      !state.index.scores ||
      !state.index.scoreContract
    ) {
      throw new Error("This Space index does not contain native steps and aligned final scores");
    }
    const scoreArraysAreValid = state.index.cells.every(
      (cell) =>
        Array.isArray(state.index.scores[cell.id]) &&
        state.index.scores[cell.id].length === state.index.sampleCount,
    );
    if (!scoreArraysAreValid) {
      throw new Error("This Space index contains incomplete per-test-case scores");
    }
    state.sample = state.index.samples[0];
    loadUrlState();
    populateStaticControls();
    populateTaskControl(state.sample.task);
    bindEvents();
    renderSample();
    renderCompare();
    updateUrl();

    elements.heroSamples.textContent = state.index.sampleCount.toLocaleString("en-US");
    elements.heroCells.textContent = state.index.cellCount.toLocaleString("en-US");
    elements.heroVideos.textContent = state.index.videoCount.toLocaleString("en-US");
    elements.archiveStatus.textContent =
      `${state.index.sampleCount} samples · ${state.index.originalStepVideoCount.toLocaleString("en-US")} native step videos · ready`;
    document.querySelector(".status-dot")?.classList.add("ready");
    elements.loadingOverlay.classList.add("hidden");
  } catch (error) {
    console.error(error);
    elements.archiveStatus.textContent = "Archive failed to load";
    elements.loadingOverlay.querySelector("p").textContent = error.message;
    elements.loadingOverlay.querySelector(".loading-bar").hidden = true;
  }
}

initialize();
