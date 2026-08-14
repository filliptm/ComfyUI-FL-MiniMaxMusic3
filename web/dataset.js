import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "fl-mm3-dataset-style";

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .fl-mm3-dataset { box-sizing:border-box; width:100%; min-height:82px; padding:9px; border-radius:8px; background:#11151d; color:#e8edf7; font:12px/1.35 system-ui,sans-serif; display:grid; gap:7px; }
    .fl-mm3-dataset-row { display:flex; align-items:center; gap:8px; min-width:0; }
    .fl-mm3-dataset-title { font-weight:700; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .fl-mm3-dataset-badge { border-radius:999px; padding:2px 7px; background:#293248; color:#b8c8e8; font-size:10px; }
    .fl-mm3-dataset-summary { color:#9ba8bd; min-height:16px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .fl-mm3-button { border:1px solid #3b4962; background:#20293a; color:#e8edf7; border-radius:6px; padding:4px 9px; cursor:pointer; }
    .fl-mm3-button:hover { background:#2a3650; }
    .fl-mm3-button:disabled { opacity:.55; cursor:default; }
  `;
  document.head.appendChild(style);
}

function widgetValue(node, name, fallback) {
  return node.widgets?.find((widget) => widget.name === name)?.value ?? fallback;
}

class DatasetPanel {
  constructor(node, container) {
    this.node = node;
    this.container = container;
    injectStyles();
    container.innerHTML = `
      <section class="fl-mm3-dataset">
        <div class="fl-mm3-dataset-row"><span class="fl-mm3-dataset-title">Dataset preflight</span><span class="fl-mm3-dataset-badge">Not scanned</span></div>
        <div class="fl-mm3-dataset-row"><span class="fl-mm3-dataset-summary">Validate captions, lyrics, durations, and audio metadata.</span><button class="fl-mm3-button" type="button">Scan</button></div>
      </section>`;
    this.badge = container.querySelector(".fl-mm3-dataset-badge");
    this.summary = container.querySelector(".fl-mm3-dataset-summary");
    this.button = container.querySelector("button");
    this.button.addEventListener("click", () => void this.scan());
  }

  async scan() {
    this.button.disabled = true;
    this.badge.textContent = "Scanning";
    this.summary.textContent = "Inspecting audio and sidecar files…";
    const payload = {
      dataset_folder: widgetValue(this.node, "dataset_folder", ""),
      recursive: widgetValue(this.node, "recursive", true),
      caption_extension: widgetValue(this.node, "caption_extension", ".txt"),
      lyrics_extension: widgetValue(this.node, "lyrics_extension", ".lyrics"),
      missing_lyrics: widgetValue(this.node, "missing_lyrics", "instrumental"),
      min_duration: widgetValue(this.node, "min_duration", 1),
      max_duration: widgetValue(this.node, "max_duration", 60),
      duration_interval: widgetValue(this.node, "duration_interval", 3),
      audio_analysis: widgetValue(this.node, "audio_analysis", "metadata"),
      include_invalid: widgetValue(this.node, "include_invalid", false),
    };
    try {
      const response = await api.fetchApi("/fl/minimax-music3/datasets/scan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const report = await response.json();
      if (!response.ok) throw new Error(report.error || "Dataset scan failed");
      const invalid = Number(report.invalid_tracks || 0);
      this.badge.textContent = invalid ? `${invalid} invalid` : "Ready";
      this.summary.textContent = `${report.valid_tracks} tracks · ${(report.total_seconds / 60).toFixed(1)} min · ${report.warnings.length} warnings`;
      this.summary.title = [...report.errors, ...report.warnings].join("\n") || "Dataset passed preflight";
    } catch (error) {
      this.badge.textContent = "Error";
      this.summary.textContent = error.message;
    } finally {
      this.button.disabled = false;
    }
  }
}

app.registerExtension({
  name: "ComfyUI.FL_MiniMaxMusic3.Dataset",
  nodeCreated(node) {
    if (node.comfyClass !== "FL_MiniMaxMusic3Dataset") return;
    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "100%";
    const panel = new DatasetPanel(node, container);
    const widget = node.addDOMWidget("fl_minimax_music3_dataset", "fl-minimax-music3-dataset", container, { getMinHeight: () => 82, hideOnZoom: false, serialize: false });
    widget.onRemove = () => container.replaceChildren();
    node.setSize([Math.max(node.size[0], 330), Math.max(node.size[1], 350)]);
    return panel;
  },
});
