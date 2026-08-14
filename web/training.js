import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { applyTrainingEvent, createTrainingState, hydrateTrainingState, progressPercent } from "./training_state.js";

const TRAINING_WIDGET_STYLES = `
  .fl-mm3-training-widget {
    --primary: #06b6d4;
    --primary-glow: rgba(6, 182, 212, 0.4);
    --secondary: #8b5cf6;
    --success: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
    --bg-dark: #0f0f12;
    --bg-card: #18181b;
    --bg-elevated: #1f1f23;
    --border: #27272a;
    --border-hover: #3f3f46;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;

    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    box-sizing: border-box;
    height: 100%;
    min-height: 300px;
    display: flex;
    flex-direction: column;
  }

  .fl-mm3-training-widget * {
    box-sizing: border-box;
  }

  .fl-mm3-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .fl-mm3-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .fl-mm3-badge {
    padding: 2px 8px;
    background: var(--primary);
    color: white;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
  }

  .fl-mm3-badge.idle {
    background: var(--text-muted);
  }

  .fl-mm3-badge.training {
    background: var(--success);
    animation: fl-mm3-pulse 2s infinite;
  }

  .fl-mm3-badge.completed {
    background: var(--success);
  }

  .fl-mm3-badge.failed {
    background: var(--danger);
  }

  .fl-mm3-badge.interrupted {
    background: var(--warning);
  }

  @keyframes fl-mm3-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  .fl-mm3-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 10px;
    gap: 8px;
    overflow: hidden;
  }

  .fl-mm3-stats {
    display: flex;
    gap: 12px;
    justify-content: center;
    align-items: baseline;
    flex-shrink: 0;
  }

  .fl-mm3-stat {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .fl-mm3-stat-label {
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .fl-mm3-stat-value {
    font-size: 11px;
    font-weight: 600;
    color: var(--primary);
    font-variant-numeric: tabular-nums;
  }

  .fl-mm3-progress-section {
    background: var(--bg-elevated);
    border-radius: 6px;
    padding: 8px 10px;
    flex-shrink: 0;
  }

  .fl-mm3-progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
  }

  .fl-mm3-progress-label {
    font-size: 9px;
    color: var(--text-secondary);
  }

  .fl-mm3-progress-value {
    font-size: 9px;
    color: var(--text-primary);
    font-weight: 500;
  }

  .fl-mm3-progress-bar {
    height: 4px;
    background: var(--bg-dark);
    border-radius: 2px;
    overflow: hidden;
  }

  .fl-mm3-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: 2px;
    transition: width 0.3s ease;
    width: 0%;
  }

  .fl-mm3-chart-section {
    background: var(--bg-elevated);
    border-radius: 6px;
    padding: 8px 10px;
    flex: 1;
    min-height: 80px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .fl-mm3-chart-header {
    font-size: 9px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .fl-mm3-chart-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }

  .fl-mm3-training-status {
    background: var(--bg-elevated);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10px;
    color: var(--text-secondary);
    text-align: center;
    border-left: 3px solid var(--primary);
    flex-shrink: 0;
  }

  .fl-mm3-training-status.error {
    border-left-color: var(--danger);
    color: var(--danger);
  }

  .fl-mm3-training-status.warning {
    border-left-color: var(--warning);
    color: var(--warning);
  }

  .fl-mm3-training-status.success {
    border-left-color: var(--success);
    color: var(--success);
  }

  .fl-mm3-preview-section {
    background: var(--bg-elevated);
    border-radius: 6px;
    padding: 8px 10px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .fl-mm3-preview-header {
    font-size: 9px;
    color: var(--text-secondary);
    margin-bottom: 6px;
    flex-shrink: 0;
  }

  .fl-mm3-preview-carousel {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
    align-items: center;
    padding-bottom: 4px;
    scrollbar-width: thin;
    scrollbar-color: var(--border-hover) transparent;
  }

  .fl-mm3-preview-carousel::-webkit-scrollbar {
    height: 4px;
  }

  .fl-mm3-preview-carousel::-webkit-scrollbar-track {
    background: transparent;
  }

  .fl-mm3-preview-carousel::-webkit-scrollbar-thumb {
    background: var(--border-hover);
    border-radius: 2px;
  }

  .fl-mm3-preview-empty {
    width: 100%;
    text-align: center;
    font-size: 10px;
    color: var(--text-muted);
  }

  .fl-mm3-preview-tile {
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
  }

  .fl-mm3-play-btn {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 2px solid var(--border);
    background: var(--bg-dark);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    color: var(--text-secondary);
    font-size: 14px;
  }

  .fl-mm3-play-btn:hover {
    border-color: var(--primary);
    color: var(--primary);
  }

  .fl-mm3-play-btn.playing {
    border-color: var(--success);
    color: var(--success);
    animation: fl-mm3-pulse 1.5s infinite;
  }

  .fl-mm3-preview-tile .tile-label {
    font-size: 9px;
    color: var(--text-muted);
    text-align: center;
    white-space: nowrap;
  }
`;

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function statusType(status) {
  if (status === "completed") return "success";
  if (status === "interrupted" || status === "stop_requested") return "warning";
  if (status === "failed") return "error";
  return "normal";
}

class TrainingWidget {
  constructor(node, container) {
    this.node = node;
    this.container = container;
    this.element = document.createElement("div");
    this.element.className = "fl-mm3-training-widget";
    this.state = createTrainingState();
    this.validationPaths = new Set();
    this.pollTimer = null;
    this.resizeObserver = null;
    this.resizeTimeout = null;

    this.injectStyles();
    this.createUI();
    this.container.appendChild(this.element);
  }

  injectStyles() {
    const styleId = "fl-mm3-training-styles";
    if (document.getElementById(styleId)) return;
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = TRAINING_WIDGET_STYLES;
    document.head.appendChild(style);
  }

  createUI() {
    this.element.innerHTML = `
      <div class="fl-mm3-header">
        <div class="fl-mm3-title">
          <span>MiniMax Music 3 Training</span>
          <span class="fl-mm3-badge idle">Idle</span>
        </div>
      </div>

      <div class="fl-mm3-content">
        <div class="fl-mm3-stats">
          <div class="fl-mm3-stat">
            <span class="fl-mm3-stat-label">Step</span>
            <span class="fl-mm3-stat-value" data-stat="step">0/0</span>
          </div>
          <div class="fl-mm3-stat">
            <span class="fl-mm3-stat-label">Loss</span>
            <span class="fl-mm3-stat-value" data-stat="loss">-</span>
          </div>
          <div class="fl-mm3-stat">
            <span class="fl-mm3-stat-label">LR</span>
            <span class="fl-mm3-stat-value" data-stat="lr">-</span>
          </div>
        </div>

        <div class="fl-mm3-progress-section">
          <div class="fl-mm3-progress-header">
            <span class="fl-mm3-progress-label">Training Progress</span>
            <span class="fl-mm3-progress-value" data-progress-label>0%</span>
          </div>
          <div class="fl-mm3-progress-bar">
            <div class="fl-mm3-progress-fill" data-progress-fill></div>
          </div>
        </div>

        <div class="fl-mm3-chart-section">
          <div class="fl-mm3-chart-header">Loss History</div>
          <canvas class="fl-mm3-chart-canvas"></canvas>
        </div>

        <div class="fl-mm3-training-status">Ready to train</div>

        <div class="fl-mm3-preview-section">
          <div class="fl-mm3-preview-header">Validation Samples</div>
          <div class="fl-mm3-preview-carousel">
            <div class="fl-mm3-preview-empty">Audio samples will appear at each checkpoint</div>
          </div>
        </div>
      </div>
    `;

    this.badgeEl = this.element.querySelector(".fl-mm3-badge");
    this.stepValueEl = this.element.querySelector('[data-stat="step"]');
    this.lossValueEl = this.element.querySelector('[data-stat="loss"]');
    this.lrValueEl = this.element.querySelector('[data-stat="lr"]');
    this.progressFillEl = this.element.querySelector("[data-progress-fill]");
    this.progressLabelEl = this.element.querySelector("[data-progress-label]");
    this.statusEl = this.element.querySelector(".fl-mm3-training-status");
    this.canvasEl = this.element.querySelector(".fl-mm3-chart-canvas");

    const chartSection = this.element.querySelector(".fl-mm3-chart-section");
    if (this.canvasEl && chartSection) {
      this.resizeObserver = new ResizeObserver(() => {
        if (this.resizeTimeout) clearTimeout(this.resizeTimeout);
        this.resizeTimeout = window.setTimeout(() => this.drawChart(), 16);
      });
      this.resizeObserver.observe(chartSection);
    }

    this.drawChart();
  }

  update(event) {
    this.state = applyTrainingEvent(this.state, event);
    this.render();
    if (!this.state.runId) return;
    if (["completed", "failed", "interrupted"].includes(this.state.status)) {
      this.stopPolling();
      void this.loadRun(this.state.runId);
    } else {
      this.startPolling();
    }
  }

  async loadRun(runId) {
    const response = await api.fetchApi(`/fl/minimax-music3/training/runs/${encodeURIComponent(runId)}`);
    if (!response.ok) return;
    const payload = await response.json();
    if (this.state.runId && this.state.runId !== runId) return;
    const hydrated = hydrateTrainingState(payload);
    this.state = hydrated.sequence >= this.state.sequence
      ? hydrated
      : { ...this.state, history: hydrated.history, validation: hydrated.validation };
    this.render();
    if (["completed", "failed", "interrupted"].includes(this.state.status)) this.stopPolling();
  }

  startPolling() {
    if (this.pollTimer) return;
    this.pollTimer = window.setInterval(() => {
      if (this.state.runId) void this.loadRun(this.state.runId);
    }, 1500);
  }

  stopPolling() {
    if (this.pollTimer) window.clearInterval(this.pollTimer);
    this.pollTimer = null;
  }

  render() {
    const loss = finiteNumber(this.state.metrics?.loss);
    const learningRate = finiteNumber(this.state.metrics?.learning_rate);
    const percent = progressPercent(this.state);
    const status = this.state.status || "idle";
    const isTraining = ["created", "running", "stop_requested"].includes(status);

    this.badgeEl.textContent = status === "completed" ? "Complete" : status === "failed" ? "Failed" : status === "interrupted" ? "Interrupted" : isTraining ? "Training" : "Idle";
    this.badgeEl.className = `fl-mm3-badge ${isTraining ? "training" : status}`;
    this.stepValueEl.textContent = `${this.state.current}/${this.state.total}`;
    this.lossValueEl.textContent = loss === null ? "-" : loss.toFixed(6);
    this.lrValueEl.textContent = learningRate === null ? "-" : learningRate.toExponential(2);
    this.progressFillEl.style.width = `${percent}%`;
    this.progressLabelEl.textContent = `${percent.toFixed(1)}%`;
    this.updateStatus(this.state.error || this.state.message, statusType(status));
    this.syncValidation(this.state.validation);
    this.drawChart();
  }

  updateStatus(message, type = "normal") {
    this.statusEl.textContent = message || "Ready to train";
    this.statusEl.className = "fl-mm3-training-status";
    if (type !== "normal") this.statusEl.classList.add(type);
  }

  syncValidation(validation) {
    const items = Array.isArray(validation) ? [...validation].reverse() : [];
    for (const item of items) {
      if (!item?.path || this.validationPaths.has(item.path)) continue;
      this.addValidationAudio(item);
    }
  }

  addValidationAudio(item) {
    const carousel = this.element.querySelector(".fl-mm3-preview-carousel");
    if (!carousel || !this.state.runId) return;
    this.validationPaths.add(item.path);

    const empty = carousel.querySelector(".fl-mm3-preview-empty");
    if (empty) empty.remove();

    const filename = item.name || item.path.replace(/\\/g, "/").split("/").at(-1) || "sample";
    const stepMatch = filename.match(/(?:step|checkpoint)[_-]?(\d+)/i);
    const shortLabel = stepMatch ? `S${stepMatch[1]}` : filename.replace(/\.wav$/i, "").slice(0, 12);
    const tile = document.createElement("div");
    tile.className = "fl-mm3-preview-tile";

    const audio = document.createElement("audio");
    audio.src = api.apiURL(`/fl/minimax-music3/training/runs/${encodeURIComponent(this.state.runId)}/artifact?path=${encodeURIComponent(item.path)}`);
    audio.preload = "none";

    const playButton = document.createElement("button");
    playButton.className = "fl-mm3-play-btn";
    playButton.type = "button";
    playButton.innerHTML = "&#9654;";
    playButton.title = filename;
    playButton.addEventListener("click", () => {
      carousel.querySelectorAll("audio").forEach((otherAudio) => {
        if (otherAudio !== audio) {
          otherAudio.pause();
          otherAudio.currentTime = 0;
        }
      });
      carousel.querySelectorAll(".fl-mm3-play-btn").forEach((otherButton) => {
        if (otherButton !== playButton) {
          otherButton.classList.remove("playing");
          otherButton.innerHTML = "&#9654;";
        }
      });

      if (audio.paused) {
        void audio.play();
        playButton.classList.add("playing");
        playButton.innerHTML = "&#9646;&#9646;";
      } else {
        audio.pause();
        playButton.classList.remove("playing");
        playButton.innerHTML = "&#9654;";
      }
    });
    audio.addEventListener("ended", () => {
      playButton.classList.remove("playing");
      playButton.innerHTML = "&#9654;";
    });

    const label = document.createElement("div");
    label.className = "tile-label";
    label.textContent = shortLabel;
    tile.append(playButton, audio, label);
    carousel.appendChild(tile);
    carousel.scrollLeft = carousel.scrollWidth;
  }

  reset() {
    this.stopPolling();
    this.state = createTrainingState();
    this.validationPaths.clear();
    this.badgeEl.textContent = "Training";
    this.badgeEl.className = "fl-mm3-badge training";
    this.stepValueEl.textContent = "0/0";
    this.lossValueEl.textContent = "-";
    this.lrValueEl.textContent = "-";
    this.progressFillEl.style.width = "0%";
    this.progressLabelEl.textContent = "0%";
    this.updateStatus("Starting training...");
    const carousel = this.element.querySelector(".fl-mm3-preview-carousel");
    carousel.innerHTML = '<div class="fl-mm3-preview-empty">Audio samples will appear at each checkpoint</div>';
    this.drawChart();
  }

  drawChart() {
    if (!this.canvasEl) return;
    const canvas = this.canvasEl;
    const context = canvas.getContext("2d");
    if (!context) return;

    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = rect.width * scale;
    canvas.height = rect.height * scale;
    context.scale(scale, scale);

    const width = rect.width;
    const height = rect.height;
    context.fillStyle = "#0f0f12";
    context.fillRect(0, 0, width, height);

    const history = this.state.history.filter((point) => finiteNumber(point.step) !== null && finiteNumber(point.loss) !== null);
    if (history.length < 2) {
      context.fillStyle = "#71717a";
      context.font = "11px Inter, sans-serif";
      context.textAlign = "center";
      context.fillText("Waiting for training data...", width / 2, height / 2);
      return;
    }

    const padding = { top: 20, right: 20, bottom: 25, left: 50 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const maxStep = Math.max(...history.map((point) => Number(point.step)));
    const minStep = Math.min(...history.map((point) => Number(point.step)));
    const maxLoss = Math.max(...history.map((point) => Number(point.loss)));
    const minLoss = Math.min(...history.map((point) => Number(point.loss)));
    const lossRange = maxLoss - minLoss || 1;

    context.strokeStyle = "#27272a";
    context.lineWidth = 0.5;
    for (let index = 0; index <= 4; index += 1) {
      const y = padding.top + (chartHeight / 4) * index;
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(width - padding.right, y);
      context.stroke();
      const lossValue = maxLoss - (lossRange / 4) * index;
      context.fillStyle = "#71717a";
      context.font = "9px Inter, sans-serif";
      context.textAlign = "right";
      context.fillText(lossValue.toFixed(4), padding.left - 5, y + 3);
    }

    const plot = () => {
      context.beginPath();
      history.forEach((point, index) => {
        const x = padding.left + ((Number(point.step) - minStep) / (maxStep - minStep || 1)) * chartWidth;
        const y = padding.top + chartHeight - ((Number(point.loss) - minLoss) / lossRange) * chartHeight;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
    };

    context.strokeStyle = "#06b6d4";
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.lineCap = "round";
    plot();
    context.stroke();

    const gradient = context.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, "rgba(6, 182, 212, 0.3)");
    gradient.addColorStop(1, "rgba(6, 182, 212, 0)");
    context.fillStyle = gradient;
    plot();
    const finalPoint = history.at(-1);
    const finalX = padding.left + ((Number(finalPoint.step) - minStep) / (maxStep - minStep || 1)) * chartWidth;
    context.lineTo(finalX, padding.top + chartHeight);
    context.lineTo(padding.left, padding.top + chartHeight);
    context.closePath();
    context.fill();

    context.fillStyle = "#fafafa";
    context.font = "bold 10px Inter, sans-serif";
    context.textAlign = "left";
    context.fillText(`Loss: ${Number(finalPoint.loss).toFixed(6)}`, padding.left + 5, padding.top + 12);
  }

  dispose() {
    this.stopPolling();
    if (this.resizeObserver) this.resizeObserver.disconnect();
    if (this.resizeTimeout) window.clearTimeout(this.resizeTimeout);
    this.element.querySelectorAll("audio").forEach((audio) => audio.pause());
    this.container.replaceChildren();
  }
}

const widgetInstances = new Map();

function createTrainingWidget(node) {
  const container = document.createElement("div");
  container.id = `fl-mm3-training-widget-${node.id}`;
  container.style.width = "100%";
  container.style.height = "100%";
  container.style.minHeight = "280px";

  const widget = node.addDOMWidget("training_ui", "training-widget", container, {
    getMinHeight: () => 400,
    hideOnZoom: false,
    serialize: false,
  });
  const trainingWidget = new TrainingWidget(node, container);
  widgetInstances.set(String(node.id), trainingWidget);
  widget.onRemove = () => {
    const instance = widgetInstances.get(String(node.id));
    if (!instance) return;
    instance.dispose();
    widgetInstances.delete(String(node.id));
  };
}

app.registerExtension({
  name: "ComfyUI.FL_MiniMaxMusic3.Training",
  nodeCreated(node) {
    if (node.comfyClass !== "FL_MiniMaxMusic3LoRATrainer") return;
    node.setSize([Math.max(node.size[0], 400), Math.max(node.size[1], 500)]);
    createTrainingWidget(node);
  },
});

api.addEventListener("fl_minimax_music3_training_status", (event) => {
  const detail = event.detail;
  if (!detail?.node) return;
  widgetInstances.get(String(detail.node))?.update(detail);
});

api.addEventListener("executing", (event) => {
  const detail = event.detail;
  if (!detail?.node) return;
  widgetInstances.get(String(detail.node))?.reset();
});
