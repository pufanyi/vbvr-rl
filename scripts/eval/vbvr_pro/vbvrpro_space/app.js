"use strict";

const MODE_META = {
  unipc: { label: "UniPC ODE", short: "UniPC", color: "#3659a7" },
  euler: { label: "Euler ODE", short: "Euler", color: "#7351a6" },
  cps03: { label: "CPS 0.3", short: "CPS 0.3", color: "#c37b21" },
  cps07: { label: "CPS 0.7", short: "CPS 0.7", color: "#0a7457" },
};

const state = {
  index: null,
  run: null,
  runData: null,
  baselineData: null,
  domain: "all",
  query: "",
  sort: "delta-desc",
  dialogTask: null,
  loadToken: 0,
};

const elements = {
  checkpoint: document.querySelector("#checkpoint-select"),
  modeSwitcher: document.querySelector("#mode-switcher"),
  scoreCards: document.querySelector("#score-cards"),
  trendLegend: document.querySelector("#trend-legend"),
  trendChart: document.querySelector("#trend-chart"),
  taskSearch: document.querySelector("#task-search"),
  taskSort: document.querySelector("#task-sort"),
  domainFilter: document.querySelector("#domain-filter"),
  taskTableBody: document.querySelector("#task-table-body"),
  taskSummary: document.querySelector("#task-summary"),
  emptyState: document.querySelector("#empty-state"),
  dialog: document.querySelector("#task-dialog"),
  dialogClose: document.querySelector("#dialog-close"),
  dialogKicker: document.querySelector("#dialog-kicker"),
  dialogTitle: document.querySelector("#dialog-title"),
  dialogRawName: document.querySelector("#dialog-raw-name"),
  dialogMetrics: document.querySelector("#dialog-metrics"),
  dialogPrompt: document.querySelector("#dialog-prompt"),
  sampleGrid: document.querySelector("#sample-grid"),
  datasetStatus: document.querySelector("#dataset-status"),
  loadingOverlay: document.querySelector("#loading-overlay"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtScore(value) {
  return Number(value).toFixed(6);
}

function fmtDelta(value, precision = 6) {
  const numeric = Number(value);
  if (Math.abs(numeric) < 0.5 * 10 ** -precision) {
    return (0).toFixed(precision);
  }
  return `${numeric > 0 ? "+" : "−"}${Math.abs(numeric).toFixed(precision)}`;
}

function deltaClass(value) {
  if (value > 1e-10) return "positive";
  if (value < -1e-10) return "negative";
  return "neutral";
}

function deltaArrow(value) {
  if (value > 1e-10) return "↑";
  if (value < -1e-10) return "↓";
  return "→";
}

function prettyTaskName(name) {
  const withoutSuffix = name.replace(/_data-generator$/, "");
  const [prefix, ...rest] = withoutSuffix.split("_");
  const phrase = rest.join(" ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  return `${prefix} · ${phrase}`;
}

function domainLabel(domain) {
  return domain === "In_Domain" ? "In-Domain" : "Out-of-Domain";
}

function bestDefaultRun(index) {
  return [...index.runs]
    .filter((run) => !run.isBaseline)
    .sort((a, b) => b.scores.overall - a.scores.overall)[0];
}

function runFromUrl(index) {
  const params = new URLSearchParams(window.location.search);
  const checkpoint = Number(params.get("checkpoint"));
  const mode = params.get("mode");
  return index.runs.find(
    (run) => !run.isBaseline && run.checkpoint === checkpoint && run.mode === mode,
  );
}

function updateUrl(taskName = null) {
  if (!state.run) return;
  const params = new URLSearchParams();
  params.set("checkpoint", state.run.checkpoint);
  params.set("mode", state.run.mode);
  const selectedTask = taskName ?? state.dialogTask;
  if (selectedTask) params.set("task", selectedTask);
  history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
}

async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${url} (${response.status})`);
  }
  return response.json();
}

async function initialize() {
  try {
    state.index = await loadJson("./data/index.json");
    state.baselineData = await loadJson(state.index.baseline.dataUrl);
    state.run = runFromUrl(state.index) ?? bestDefaultRun(state.index);

    populateHeader();
    populateRunControls();
    bindEvents();
    renderTrend();
    await selectRun(state.run);

    const requestedTask = new URLSearchParams(window.location.search).get("task");
    if (requestedTask && state.index.tasks.some((task) => task.name === requestedTask)) {
      await openTask(requestedTask);
    }

    document.querySelector(".status-dot")?.classList.add("ready");
    elements.datasetStatus.textContent = `${state.index.taskCount} tasks · ${state.index.sampleCountPerRun} matched samples`;
    elements.loadingOverlay.classList.add("hidden");
  } catch (error) {
    console.error(error);
    elements.datasetStatus.textContent = "Archive failed to load";
    elements.loadingOverlay.querySelector("p").textContent = error.message;
    elements.loadingOverlay.querySelector(".loading-line").hidden = true;
  }
}

function populateHeader() {
  document.querySelector("#hero-run-count").textContent = state.index.runCount;
  document.querySelector("#hero-video-count").textContent =
    state.index.totalVideoCount.toLocaleString("en-US");
  document.querySelector("#hero-task-count").textContent = state.index.taskCount;
}

function populateRunControls() {
  const checkpoints = [...new Set(state.index.runs.filter((run) => !run.isBaseline).map((run) => run.checkpoint))].sort(
    (a, b) => a - b,
  );
  elements.checkpoint.innerHTML = checkpoints
    .map((checkpoint) => `<option value="${checkpoint}">Step ${checkpoint}</option>`)
    .join("");
  elements.checkpoint.value = state.run.checkpoint;

  elements.modeSwitcher.innerHTML = state.index.modeOrder
    .map((mode) => {
      const meta = MODE_META[mode];
      return `<button type="button" data-mode="${mode}">${escapeHtml(meta.label)}</button>`;
    })
    .join("");
  updateModeSelection();
}

function bindEvents() {
  elements.checkpoint.addEventListener("change", () => {
    chooseRun(Number(elements.checkpoint.value), state.run.mode);
  });

  elements.modeSwitcher.addEventListener("click", (event) => {
    const button = event.target.closest("[data-mode]");
    if (!button) return;
    chooseRun(state.run.checkpoint, button.dataset.mode);
  });

  elements.taskSearch.addEventListener("input", () => {
    state.query = elements.taskSearch.value.trim().toLowerCase();
    renderTasks();
  });

  elements.taskSort.addEventListener("change", () => {
    state.sort = elements.taskSort.value;
    renderTasks();
  });

  elements.domainFilter.addEventListener("click", (event) => {
    const button = event.target.closest("[data-domain]");
    if (!button) return;
    state.domain = button.dataset.domain;
    elements.domainFilter.querySelectorAll("button").forEach((candidate) => {
      candidate.classList.toggle("active", candidate === button);
    });
    renderTasks();
  });

  elements.taskTableBody.addEventListener("click", (event) => {
    const button = event.target.closest("[data-task]");
    if (button) openTask(button.dataset.task);
  });

  elements.dialogClose.addEventListener("click", closeDialog);
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) closeDialog();
  });
  elements.dialog.addEventListener("close", () => {
    state.dialogTask = null;
    updateUrl();
    stopDialogVideos();
  });

  elements.sampleGrid.addEventListener("click", (event) => {
    const button = event.target.closest("[data-sync-pair]");
    if (!button) return;
    const card = button.closest(".sample-card");
    const videos = [...card.querySelectorAll("video")];
    videos.forEach((video) => {
      video.currentTime = 0;
      video.play().catch(() => {});
    });
  });
}

function updateModeSelection() {
  elements.modeSwitcher.querySelectorAll("[data-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.run.mode);
  });
}

function chooseRun(checkpoint, mode) {
  const run = state.index.runs.find(
    (candidate) =>
      !candidate.isBaseline && candidate.checkpoint === checkpoint && candidate.mode === mode,
  );
  if (!run || run.id === state.run.id) return;
  selectRun(run);
}

async function selectRun(run) {
  const loadToken = ++state.loadToken;
  state.run = run;
  elements.checkpoint.value = run.checkpoint;
  updateModeSelection();
  updateUrl();
  renderScoreCards();
  renderTrend();

  elements.taskTableBody.style.opacity = "0.45";
  const runData = await loadJson(run.dataUrl);
  if (loadToken !== state.loadToken) return;
  state.runData = runData;
  elements.taskTableBody.style.opacity = "";
  renderTasks();

  if (elements.dialog.open && state.dialogTask) {
    await renderTaskDialog(state.dialogTask);
  }
}

function renderScoreCards() {
  const baseline = state.index.baseline.scores;
  const scoreItems = [
    ["Overall", state.run.scores.overall, baseline.overall, "500 samples"],
    ["In-Domain", state.run.scores.inDomain, baseline.inDomain, "250 samples"],
    ["Out-of-Domain", state.run.scores.outOfDomain, baseline.outOfDomain, "250 samples"],
    [
      "Task wins",
      state.run.taskStats.wins,
      state.index.taskCount,
      `${state.run.taskStats.losses} drops · ${state.run.taskStats.ties} ties`,
    ],
  ];

  elements.scoreCards.innerHTML = scoreItems
    .map(([label, score, base, note], index) => {
      const delta = index === 3 ? null : score - base;
      const value = index === 3 ? `${score}/${base}` : fmtScore(score);
      const baselineText = index === 3 ? "vs task baseline" : `baseline ${fmtScore(base)}`;
      return `
        <article class="score-card">
          <div class="score-card-label">
            <span>${escapeHtml(label)}</span>
            ${delta === null ? "" : `<span class="delta ${deltaClass(delta)}">${deltaArrow(delta)} ${fmtDelta(delta, 4)}</span>`}
          </div>
          <div class="score-value">${value}</div>
          <div class="score-footer">
            <span>${escapeHtml(baselineText)}</span>
            <span>${escapeHtml(note)}</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderTrend() {
  if (!state.index || !state.run) return;
  const modes = state.index.modeOrder;
  const checkpoints = [...new Set(state.index.runs.filter((run) => !run.isBaseline).map((run) => run.checkpoint))].sort(
    (a, b) => a - b,
  );
  const runsByMode = Object.fromEntries(
    modes.map((mode) => [
      mode,
      checkpoints.map((checkpoint) =>
        state.index.runs.find((run) => run.mode === mode && run.checkpoint === checkpoint),
      ),
    ]),
  );

  elements.trendLegend.innerHTML = `
    <span class="legend-item">
      <span class="legend-line shared-origin"></span>
      Step 0 · SFT ODE
    </span>
    ${modes
      .map(
        (mode) => `
        <span class="legend-item ${mode === state.run.mode ? "selected" : ""}">
          <span class="legend-line" style="--legend-color:${MODE_META[mode].color}"></span>
          ${escapeHtml(MODE_META[mode].label)}
        </span>
      `,
      )
      .join("")}
  `;

  const values = [
    state.index.baseline.scores.overall,
    ...state.index.runs.filter((run) => !run.isBaseline).map((run) => run.scores.overall),
  ];
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const min = Math.floor((rawMin - 0.015) * 100) / 100;
  const max = Math.ceil((rawMax + 0.015) * 100) / 100;
  const width = 900;
  const height = 330;
  const pad = { top: 38, right: 35, bottom: 48, left: 58 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const trajectorySteps = [0, ...checkpoints];
  const x = (index) => pad.left + (index / (trajectorySteps.length - 1)) * plotWidth;
  const y = (value) => pad.top + ((max - value) / (max - min)) * plotHeight;
  const ticks = 5;

  const horizontalGrid = Array.from({ length: ticks + 1 }, (_, index) => {
    const value = min + ((max - min) * index) / ticks;
    const py = y(value);
    return `
      <line class="chart-grid" x1="${pad.left}" x2="${width - pad.right}" y1="${py}" y2="${py}" />
      <text class="chart-axis-label" x="${pad.left - 12}" y="${py + 3}" text-anchor="end">${value.toFixed(2)}</text>
    `;
  }).join("");

  const xLabels = trajectorySteps
    .map(
      (step, index) => `
        <text class="chart-axis-label" x="${x(index)}" y="${height - 18}" text-anchor="middle">${step}</text>
      `,
    )
    .join("");

  const baselineY = y(state.index.baseline.scores.overall);
  const baseline = `
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${baselineY}" y2="${baselineY}"
      stroke="#17201d" stroke-width="1.2" stroke-dasharray="5 6" opacity="0.58" />
    <text class="chart-axis-label" x="${width - pad.right}" y="${baselineY - 8}" text-anchor="end">
      SFT ODE ${fmtScore(state.index.baseline.scores.overall)}
    </text>
  `;
  const sharedOrigin = `
    <circle class="chart-shared-origin" cx="${x(0)}" cy="${baselineY}" r="5.5"
      fill="#17201d" stroke="#fffdf8" stroke-width="2.5" />
  `;

  const lines = modes
    .map((mode) => {
      const selected = mode === state.run.mode;
      const points = runsByMode[mode];
      const polyline = [
        `${x(0)},${baselineY}`,
        ...points.map((run, index) => `${x(index + 1)},${y(run.scores.overall)}`),
      ].join(" ");
      const circles = points
        .map((run, index) => {
          const active = run.id === state.run.id;
          return `
            <circle cx="${x(index + 1)}" cy="${y(run.scores.overall)}" r="${active ? 7 : selected ? 4.5 : 3.5}"
              fill="${MODE_META[mode].color}" stroke="#fffdf8" stroke-width="${active ? 3 : 2}" />
            ${
              active
                ? `<text class="chart-point-label" x="${x(index + 1)}" y="${y(run.scores.overall) - 15}" text-anchor="middle">${run.scores.overall.toFixed(4)}</text>`
                : ""
            }
          `;
        })
        .join("");
      return `
        <polyline points="${polyline}" fill="none" stroke="${MODE_META[mode].color}"
          stroke-width="${selected ? 3.8 : 2}" stroke-linecap="round" stroke-linejoin="round"
          opacity="${selected ? 1 : 0.43}" />
        ${circles}
      `;
    })
    .join("");

  elements.trendChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
      ${horizontalGrid}
      ${baseline}
      ${lines}
      ${sharedOrigin}
      ${xLabels}
      <text class="chart-axis-label" x="${pad.left}" y="${height - 3}">CHECKPOINT STEP</text>
    </svg>
  `;
}

function taskRows() {
  const scoreMap = state.run.taskScores;
  const baselineMap = state.index.baseline.taskScores;
  const tasks = state.index.tasks
    .filter((task) => state.domain === "all" || task.domain === state.domain)
    .filter((task) => {
      if (!state.query) return true;
      return (
        task.name.toLowerCase().includes(state.query) ||
        prettyTaskName(task.name).toLowerCase().includes(state.query) ||
        task.category.toLowerCase().includes(state.query)
      );
    })
    .map((task) => ({
      ...task,
      score: scoreMap[task.name],
      baseline: baselineMap[task.name],
      delta: scoreMap[task.name] - baselineMap[task.name],
    }));

  const comparators = {
    "delta-desc": (a, b) => b.delta - a.delta || a.name.localeCompare(b.name),
    "delta-asc": (a, b) => a.delta - b.delta || a.name.localeCompare(b.name),
    "score-desc": (a, b) => b.score - a.score || a.name.localeCompare(b.name),
    "score-asc": (a, b) => a.score - b.score || a.name.localeCompare(b.name),
    "name-asc": (a, b) => a.name.localeCompare(b.name),
  };
  return tasks.sort(comparators[state.sort]);
}

function renderTasks() {
  if (!state.runData) return;
  const tasks = taskRows();
  const gains = tasks.filter((task) => task.delta > 1e-10).length;
  const drops = tasks.filter((task) => task.delta < -1e-10).length;
  elements.taskSummary.textContent = `${tasks.length} shown · ${gains} gains · ${drops} drops`;
  elements.emptyState.hidden = tasks.length > 0;

  elements.taskTableBody.innerHTML = tasks
    .map(
      (task) => `
        <tr class="task-row">
          <td>
            <div class="task-pretty-name">${escapeHtml(prettyTaskName(task.name))}</div>
            <div class="task-raw-name" title="${escapeHtml(task.name)}">${escapeHtml(task.name)}</div>
          </td>
          <td>
            <div class="tag-stack">
              <span class="tag">${escapeHtml(domainLabel(task.domain))}</span>
              <span class="tag">${escapeHtml(task.category)}</span>
            </div>
          </td>
          <td><span class="numeric-score">${fmtScore(task.score)}</span></td>
          <td><span class="numeric-score">${fmtScore(task.baseline)}</span></td>
          <td>
            <span class="delta ${deltaClass(task.delta)}">${deltaArrow(task.delta)} ${fmtDelta(task.delta)}</span>
          </td>
          <td>
            <button class="inspect-button" type="button" data-task="${escapeHtml(task.name)}">Videos</button>
          </td>
        </tr>
      `,
    )
    .join("");
}

async function openTask(taskName) {
  state.dialogTask = taskName;
  await renderTaskDialog(taskName);
  if (!elements.dialog.open) elements.dialog.showModal();
  updateUrl(taskName);
}

async function renderTaskDialog(taskName) {
  const task = state.index.tasks.find((candidate) => candidate.name === taskName);
  if (!task || !state.runData) return;
  stopDialogVideos();

  const score = state.run.taskScores[task.name];
  const baseline = state.index.baseline.taskScores[task.name];
  const delta = score - baseline;
  const samples = state.runData.samples.filter((sample) => sample.taskName === task.name);
  const baselineById = Object.fromEntries(state.baselineData.samples.map((sample) => [sample.id, sample]));

  elements.dialogKicker.textContent =
    `${domainLabel(task.domain)} · ${task.category} · ${state.run.label}`;
  elements.dialogTitle.textContent = prettyTaskName(task.name);
  elements.dialogRawName.textContent = task.name;
  elements.dialogPrompt.textContent =
    "Prompts are sample-specific for this task. The exact prompt used by each matched sample appears beneath its video pair.";
  elements.dialogMetrics.innerHTML = [
    ["Current task score", fmtScore(score)],
    ["SFT baseline", fmtScore(baseline)],
    ["Δ vs baseline", fmtDelta(delta)],
    ["Matched samples", samples.length],
  ]
    .map(
      ([label, value], index) => `
        <div class="dialog-metric">
          <span>${escapeHtml(label)}</span>
          <strong class="${index === 2 ? `delta ${deltaClass(delta)}` : ""}">${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");

  elements.sampleGrid.innerHTML = samples
    .map((sample) => {
      const baselineSample = baselineById[sample.id];
      const sampleDelta = sample.score - baselineSample.score;
      return `
        <article class="sample-card">
          <div class="sample-card-header">
            <span class="sample-id">SAMPLE ${escapeHtml(sample.videoFile.replace(".mp4", ""))}</span>
            <div class="sample-actions">
              <span class="delta ${deltaClass(sampleDelta)}">
                ${deltaArrow(sampleDelta)} ${fmtDelta(sampleDelta)} sample Δ
              </span>
              <button class="sync-button" type="button" data-sync-pair>Play pair</button>
            </div>
          </div>
          <div class="video-pair">
            ${videoPanel(sample.videoUrl, state.run.shortLabel, sample.score)}
            ${videoPanel(baselineSample.videoUrl, "SFT baseline", baselineSample.score)}
          </div>
          <details class="sample-prompt">
            <summary>Exact prompt for sample ${escapeHtml(sample.videoFile.replace(".mp4", ""))}</summary>
            <p>${escapeHtml(baselineSample.prompt)}</p>
          </details>
        </article>
      `;
    })
    .join("");
}

function videoPanel(url, label, score) {
  return `
    <div class="video-panel">
      <video controls playsinline muted preload="auto" src="${escapeHtml(url)}"></video>
      <div class="video-meta">
        <span class="video-run-label">${escapeHtml(label)}</span>
        <span class="video-score">score ${fmtScore(score)}</span>
      </div>
    </div>
  `;
}

function stopDialogVideos() {
  elements.sampleGrid.querySelectorAll("video").forEach((video) => {
    video.pause();
    video.removeAttribute("src");
    video.load();
  });
}

function closeDialog() {
  if (elements.dialog.open) elements.dialog.close();
}

initialize();
