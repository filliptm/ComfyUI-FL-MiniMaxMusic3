import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import {
  MINIMAX_MUSIC3_ARTIFACTS,
  applyInventory,
  applyLoaderEvent,
  createLoaderState,
  formatBytes,
  formatEta,
  markLoaderCached,
  markLoaderFailed,
  progressRatio,
} from "./minimax_music3_loader_state.js";

const MIN_NODE_WIDTH = 330;
const MIN_NODE_HEIGHT = 290;
const MIN_PANEL_HEIGHT = 200;
const STATUS_ENDPOINT = "/fl/minimax-music3/status";
const STATUS_EVENT = "fl_minimax_music3_loader_status";
const panels = new Map();

const STYLES = `
  .flmm3-container {
    container-type: size;
    height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
    width: 100%;
  }
  .flmm3-panel {
    --flmm3-accent: #8b5cf6;
    --flmm3-border: var(--border-color, #343741);
    --flmm3-muted: var(--descrip-text, #979cab);
    --flmm3-surface: color-mix(in srgb, var(--comfy-menu-bg, #17181d) 88%, black);
    background: var(--comfy-menu-bg, #17181d);
    border: 1px solid var(--flmm3-border);
    border-radius: 7px;
    box-sizing: border-box;
    color: var(--input-text, #f4f4f5);
    display: grid;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 10px;
    gap: 5px;
    grid-template-rows: auto minmax(0, 1fr) auto;
    height: 100%;
    min-height: 0;
    overflow: hidden;
    padding: 6px;
    width: 100%;
  }
  .flmm3-panel * { box-sizing: border-box; }
  .flmm3-header {
    align-items: center;
    display: grid;
    gap: 1px 5px;
    grid-template-columns: minmax(0, 1fr) auto auto;
  }
  .flmm3-title {
    font-size: 11.5px;
    font-weight: 750;
    line-height: 1.2;
  }
  .flmm3-subtitle {
    color: var(--flmm3-muted);
    font-size: 8.5px;
    grid-column: 1 / -1;
  }
  .flmm3-badge {
    background: #3f3f46;
    border-radius: 999px;
    color: #e4e4e7;
    font-size: 8px;
    font-variant-numeric: tabular-nums;
    font-weight: 750;
    line-height: 1;
    max-width: 90px;
    overflow: hidden;
    padding: 4px 6px;
    text-overflow: ellipsis;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .flmm3-panel[data-state="installed"] .flmm3-badge,
  .flmm3-panel[data-state="verified"] .flmm3-badge { background: #155e75; color: #cffafe; }
  .flmm3-panel[data-state="downloading"] .flmm3-badge,
  .flmm3-panel[data-state="loading"] .flmm3-badge,
  .flmm3-panel[data-state="checking"] .flmm3-badge,
  .flmm3-panel[data-state="verifying"] .flmm3-badge { background: #5b21b6; color: #ede9fe; }
  .flmm3-panel[data-state="ready"] .flmm3-badge,
  .flmm3-panel[data-state="cached"] .flmm3-badge { background: #166534; color: #dcfce7; }
  .flmm3-panel[data-state="partial"] .flmm3-badge,
  .flmm3-panel[data-state="attention"] .flmm3-badge,
  .flmm3-panel[data-state="missing"] .flmm3-badge { background: #854d0e; color: #fef3c7; }
  .flmm3-panel[data-state="error"] .flmm3-badge,
  .flmm3-panel[data-state="interrupted"] .flmm3-badge { background: #991b1b; color: #fee2e2; }
  .flmm3-refresh {
    align-items: center;
    background: var(--comfy-input-bg, #24262d);
    border: 1px solid var(--flmm3-border);
    border-radius: 5px;
    color: inherit;
    cursor: pointer;
    display: inline-flex;
    font: inherit;
    height: 22px;
    justify-content: center;
    padding: 0;
    width: 24px;
  }
  .flmm3-refresh:hover:not(:disabled) { border-color: var(--flmm3-accent); color: #c4b5fd; }
  .flmm3-refresh:focus-visible { outline: 2px solid var(--flmm3-accent); outline-offset: 1px; }
  .flmm3-refresh:disabled { cursor: default; opacity: .45; }
  .flmm3-artifacts {
    align-content: start;
    display: grid;
    gap: 4px;
    min-height: 0;
    overflow: hidden;
  }
  .flmm3-artifact {
    background: var(--flmm3-surface);
    border: 1px solid var(--flmm3-border);
    border-radius: 5px;
    column-gap: 5px;
    display: grid;
    grid-template-columns: 17px minmax(0, 1fr) auto;
    padding: 4px 6px;
    row-gap: 2px;
  }
  .flmm3-icon {
    align-items: center;
    background: #27272a;
    border-radius: 999px;
    color: #a1a1aa;
    display: flex;
    font-size: 9.5px;
    font-weight: 800;
    grid-row: 1 / 3;
    height: 16px;
    justify-content: center;
    width: 16px;
  }
  .flmm3-artifact[data-state="downloading"] .flmm3-icon { background: #5b21b6; color: #ede9fe; }
  .flmm3-artifact[data-state="verifying"] .flmm3-icon,
  .flmm3-artifact[data-state="loading"] .flmm3-icon { background: #155e75; color: #cffafe; }
  .flmm3-artifact[data-state="present"] .flmm3-icon,
  .flmm3-artifact[data-state="verified"] .flmm3-icon,
  .flmm3-artifact[data-state="ready"] .flmm3-icon { background: #166534; color: #dcfce7; }
  .flmm3-artifact[data-state="partial"] .flmm3-icon,
  .flmm3-artifact[data-state="missing"] .flmm3-icon,
  .flmm3-artifact[data-state="invalid_size"] .flmm3-icon,
  .flmm3-artifact[data-state="invalid"] .flmm3-icon { background: #854d0e; color: #fef3c7; }
  .flmm3-artifact[data-state="error"] .flmm3-icon,
  .flmm3-artifact[data-state="interrupted"] .flmm3-icon { background: #991b1b; color: #fee2e2; }
  .flmm3-name {
    font-size: 10px;
    font-weight: 700;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flmm3-state {
    color: #d4d4d8;
    font-size: 8.5px;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    text-align: right;
    white-space: nowrap;
  }
  .flmm3-meta {
    color: var(--flmm3-muted);
    font-size: 8.25px;
    font-variant-numeric: tabular-nums;
    grid-column: 2 / 4;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flmm3-track {
    background: #27272a;
    border-radius: 999px;
    grid-column: 1 / -1;
    height: 3px;
    overflow: hidden;
    position: relative;
  }
  .flmm3-fill {
    background: linear-gradient(90deg, #7c3aed, #22c55e);
    height: 100%;
    transition: width 150ms linear;
    width: 0%;
  }
  .flmm3-track[data-indeterminate="true"] .flmm3-fill {
    animation: flmm3-indeterminate 1s ease-in-out infinite;
    background: linear-gradient(90deg, transparent, #8b5cf6, #22d3ee, transparent);
    width: 45% !important;
  }
  @keyframes flmm3-indeterminate {
    from { transform: translateX(-115%); }
    to { transform: translateX(255%); }
  }
  .flmm3-message {
    background: color-mix(in srgb, var(--comfy-input-bg, #24262d) 75%, transparent);
    border: 1px solid var(--flmm3-border);
    border-radius: 5px;
    color: #d4d4d8;
    font-size: 8.75px;
    line-height: 1.2;
    min-height: 24px;
    overflow-wrap: anywhere;
    padding: 5px 6px;
  }
  .flmm3-panel[data-state="error"] .flmm3-message,
  .flmm3-panel[data-state="interrupted"] .flmm3-message { color: #fecaca; border-color: #7f1d1d; }
  @container (max-width: 350px) {
    .flmm3-panel { padding: 5px; }
    .flmm3-badge { max-width: 76px; }
    .flmm3-meta { font-size: 8px; }
  }
  @container (max-height: 215px) {
    .flmm3-panel { gap: 4px; padding: 5px; }
    .flmm3-subtitle { display: none; }
    .flmm3-artifacts { gap: 3px; }
    .flmm3-artifact { padding: 3px 5px; }
    .flmm3-message { min-height: 20px; padding: 3px 5px; }
  }
`;

function injectStyles() {
  if (document.getElementById("flmm3-loader-styles")) return;
  const style = document.createElement("style");
  style.id = "flmm3-loader-styles";
  style.textContent = STYLES;
  document.head.appendChild(style);
}

function nodeKey(value) {
  return value === null || value === undefined ? "" : String(value);
}

function eventNode(detail) {
  if (detail && typeof detail === "object") return detail.node ?? detail.node_id;
  return detail;
}

function registerPanel(panel) {
  for (const [key, value] of panels.entries()) {
    if (value === panel) panels.delete(key);
  }
  panels.set(nodeKey(panel.node.id), panel);
}

function enforceMinimumNodeSize(node) {
  node.min_size = [
    Math.max(node.min_size?.[0] || 0, MIN_NODE_WIDTH),
    Math.max(node.min_size?.[1] || 0, MIN_NODE_HEIGHT),
  ];
  node.setSize([
    Math.max(node.size[0], MIN_NODE_WIDTH),
    Math.max(node.size[1], MIN_NODE_HEIGHT),
  ]);
}

function stateLabel(artifact) {
  const percent = Math.round(progressRatio(artifact) * 100);
  const labels = {
    unknown: "Not checked",
    missing: "Missing",
    partial: `${percent}% saved`,
    invalid_size: "Needs repair",
    invalid: "Needs repair",
    present: "Installed",
    verifying: "Verifying",
    downloading: `${percent}%`,
    verified: "Verified",
    loading: "Loading",
    ready: "Ready",
    error: "Error",
    interrupted: "Interrupted",
  };
  return labels[artifact.state] || artifact.state;
}

function stateIcon(state) {
  if (["present", "verified", "ready"].includes(state)) return "✓";
  if (state === "downloading") return "↓";
  if (["verifying", "loading"].includes(state)) return "↻";
  if (["error", "interrupted"].includes(state)) return "×";
  if (["partial", "missing", "invalid", "invalid_size"].includes(state)) return "!";
  return "•";
}

function artifactMeta(artifact) {
  const size = formatBytes(artifact.max || artifact.expectedSize);
  if (artifact.state === "present") return `${size} installed · verification pending`;
  if (artifact.state === "missing") return `${size} · downloads on queue`;
  if (["invalid", "invalid_size"].includes(artifact.state)) return `${formatBytes(artifact.value)} found · repairs on queue`;
  if (artifact.state === "partial") return `${formatBytes(artifact.value)} / ${size} · resumes on queue`;
  if (artifact.state === "verifying") return `Checksum verification · ${size}`;
  if (artifact.state === "downloading") {
    const parts = [`${formatBytes(artifact.value)} / ${size}`];
    if (artifact.speed > 0) {
      parts.push(`${formatBytes(artifact.speed)}/s`);
      const eta = formatEta((artifact.max - artifact.value) / artifact.speed);
      if (eta) parts.push(eta);
    }
    if (artifact.resumed) parts.push("resumed");
    return parts.join(" · ");
  }
  if (artifact.state === "verified") return `Checksum verified · ${size}`;
  if (artifact.state === "loading") return `Creating ComfyUI model · ${size}`;
  if (artifact.state === "ready") return `Loaded and ready · ${size}`;
  return `${size} expected`;
}

function badgeLabel(state) {
  const labels = {
    idle: "Inspecting",
    installed: "Installed",
    missing: "Missing files",
    partial: "Resume ready",
    attention: "Needs repair",
    checking: "Checking",
    verifying: "Verifying",
    downloading: "Downloading",
    verified: "Verified",
    files_ready: "Verified",
    loading: "Loading",
    ready: "Ready",
    complete: "Ready",
    cached: "Cached",
    error: "Error",
    interrupted: "Interrupted",
  };
  return labels[state] || state;
}

class MiniMaxMusic3LoaderPanel {
  constructor(node, container) {
    this.node = node;
    this.container = container;
    this.state = createLoaderState();
    this.active = false;
    this.disposed = false;
    this.inventoryRequest = 0;
    this.rows = new Map();
    this.build();
    this.render();
    void this.refreshInventory();
  }

  build() {
    injectStyles();
    this.container.innerHTML = `
      <section class="flmm3-panel" data-state="idle" aria-label="MiniMax Music 3 model status">
        <header class="flmm3-header">
          <div class="flmm3-title">MiniMax Music 3</div>
          <span class="flmm3-badge" data-role="badge">Inspecting</span>
          <button class="flmm3-refresh" data-action="refresh" type="button" title="Refresh installed-file status" aria-label="Refresh installed-file status">↻</button>
          <div class="flmm3-subtitle">Official Comfy-Org bundle · 13.34 GiB</div>
        </header>
        <div class="flmm3-artifacts" data-role="artifacts"></div>
        <div class="flmm3-message" data-role="message" role="status" aria-live="polite"></div>
      </section>
    `;
    this.panel = this.container.querySelector(".flmm3-panel");
    this.badge = this.container.querySelector('[data-role="badge"]');
    this.message = this.container.querySelector('[data-role="message"]');
    this.refreshButton = this.container.querySelector('[data-action="refresh"]');
    const artifacts = this.container.querySelector('[data-role="artifacts"]');

    for (const artifact of MINIMAX_MUSIC3_ARTIFACTS) {
      const row = document.createElement("div");
      row.className = "flmm3-artifact";
      row.dataset.artifact = artifact.key;
      row.innerHTML = `
        <span class="flmm3-icon" data-role="icon">•</span>
        <span class="flmm3-name" data-role="name"></span>
        <span class="flmm3-state" data-role="state"></span>
        <span class="flmm3-meta" data-role="meta"></span>
        <span class="flmm3-track" data-role="track" role="progressbar">
          <span class="flmm3-fill" data-role="fill"></span>
        </span>
      `;
      row.title = artifact.filename;
      artifacts.appendChild(row);
      this.rows.set(artifact.key, {
        row,
        icon: row.querySelector('[data-role="icon"]'),
        name: row.querySelector('[data-role="name"]'),
        state: row.querySelector('[data-role="state"]'),
        meta: row.querySelector('[data-role="meta"]'),
        track: row.querySelector('[data-role="track"]'),
        fill: row.querySelector('[data-role="fill"]'),
      });
    }
    this.refreshButton.addEventListener("click", () => void this.refreshInventory());
  }

  async refreshInventory() {
    if (this.active || this.disposed) return;
    const request = ++this.inventoryRequest;
    this.state = createLoaderState();
    this.render();
    try {
      const response = await api.fetchApi(STATUS_ENDPOINT);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not inspect MiniMax Music 3 files.");
      if (this.disposed || request !== this.inventoryRequest) return;
      this.state = applyInventory(this.state, payload);
      this.render();
    } catch (error) {
      if (this.disposed || request !== this.inventoryRequest) return;
      this.state = markLoaderFailed(this.state, `Status unavailable: ${error.message}`);
      this.render();
    }
  }

  beginExecution() {
    this.inventoryRequest += 1;
    this.active = true;
    this.state = applyLoaderEvent(createLoaderState(), {
      state: "checking",
      message: "Waiting for backend model checks…",
    });
    this.render();
  }

  update(detail) {
    this.active = !["complete", "error", "interrupted"].includes(detail.state);
    this.state = applyLoaderEvent(this.state, detail);
    this.render();
  }

  markCached() {
    this.active = false;
    this.state = markLoaderCached(this.state);
    this.render();
  }

  fail(message) {
    this.active = false;
    if (this.state.state !== "error") this.state = markLoaderFailed(this.state, message);
    this.render();
  }

  interrupt() {
    this.active = false;
    this.state = markLoaderFailed(this.state, "MiniMax Music 3 loading interrupted", true);
    this.render();
  }

  configure() {
    registerPanel(this);
    if (!this.active) void this.refreshInventory();
  }

  render() {
    this.panel.dataset.state = this.state.state;
    this.badge.textContent = badgeLabel(this.state.state);
    this.message.textContent = this.state.message;
    this.refreshButton.disabled = this.active;

    for (const [key, elements] of this.rows.entries()) {
      const artifact = this.state.artifacts[key];
      if (!artifact) continue;
      const ratio = progressRatio(artifact);
      const indeterminate = ["verifying", "loading"].includes(artifact.state);
      elements.row.dataset.state = artifact.state;
      elements.row.title = artifact.filename;
      elements.icon.textContent = stateIcon(artifact.state);
      elements.name.textContent = artifact.label;
      elements.state.textContent = stateLabel(artifact);
      elements.meta.textContent = artifactMeta(artifact);
      elements.track.dataset.indeterminate = String(indeterminate);
      elements.track.setAttribute("aria-valuemin", "0");
      elements.track.setAttribute("aria-valuemax", "100");
      elements.track.setAttribute("aria-valuetext", `${artifact.label}: ${stateLabel(artifact)}`);
      if (indeterminate) elements.track.removeAttribute("aria-valuenow");
      else elements.track.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));
      elements.fill.style.width = `${ratio * 100}%`;
    }
    this.node.setDirtyCanvas?.(true, true);
  }

  dispose() {
    this.disposed = true;
    this.inventoryRequest += 1;
    for (const [key, value] of panels.entries()) {
      if (value === this) panels.delete(key);
    }
    this.container.replaceChildren();
  }
}

app.registerExtension({
  name: "ComfyUI.FL_MiniMaxMusic3.Loader",
  nodeCreated(node) {
    if (node.comfyClass !== "FL_MiniMaxMusic3Loader") return;

    const container = document.createElement("div");
    container.className = "flmm3-container";
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = `${MIN_PANEL_HEIGHT}px`;
    container.style.overflow = "hidden";

    const widget = node.addDOMWidget("fl_minimax_music3_status", "fl-minimax-music3-status", container, {
      getMinHeight: () => MIN_PANEL_HEIGHT,
      hideOnZoom: false,
      serialize: false,
    });
    enforceMinimumNodeSize(node);
    requestAnimationFrame(() => enforceMinimumNodeSize(node));

    const panel = new MiniMaxMusic3LoaderPanel(node, container);
    registerPanel(panel);

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      panel.configure();
      requestAnimationFrame(() => enforceMinimumNodeSize(this));
      return result;
    };

    widget.onRemove = () => panel.dispose();
  },
});

api.addEventListener("executing", (event) => {
  panels.get(nodeKey(eventNode(event.detail)))?.beginExecution();
});

api.addEventListener(STATUS_EVENT, (event) => {
  const detail = event.detail || {};
  panels.get(nodeKey(detail.node))?.update(detail);
});

api.addEventListener("execution_cached", (event) => {
  const nodes = Array.isArray(event.detail?.nodes) ? event.detail.nodes : [];
  for (const nodeId of nodes) panels.get(nodeKey(nodeId))?.markCached();
});

api.addEventListener("execution_error", (event) => {
  const detail = event.detail || {};
  panels.get(nodeKey(eventNode(detail)))?.fail(detail.exception_message || "MiniMax Music 3 loading failed");
});

api.addEventListener("execution_interrupted", () => {
  for (const panel of panels.values()) {
    if (panel.active) panel.interrupt();
  }
});
