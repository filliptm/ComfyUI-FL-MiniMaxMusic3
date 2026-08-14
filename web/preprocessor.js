import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { formatBytes, parsePreprocessorSettings, preprocessPercent } from "./preprocessor_state.js";


const STYLE_ID = "fl-mm3-preprocessor-style";
const panels = new Set();


function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .fl-moss-compact { box-sizing:border-box; width:100%; height:100%; min-height:112px; padding:9px; border-radius:8px; background:#10151e; color:#eaf0fa; font:12px/1.35 system-ui,sans-serif; display:grid; grid-template-rows:auto auto auto auto; gap:6px; overflow:hidden; }
    .fl-moss-row { display:flex; align-items:center; gap:8px; min-width:0; }
    .fl-moss-title { font-weight:700; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .fl-moss-badge { border-radius:999px; padding:2px 7px; background:#293349; color:#b9c9e7; font-size:10px; text-transform:capitalize; white-space:nowrap; }
    .fl-moss-badge[data-state="completed"],.fl-moss-badge[data-state="ready"] { background:#173d35; color:#8ee3c3; }
    .fl-moss-badge[data-state="failed"],.fl-moss-badge[data-state="error"] { background:#512832; color:#ffb5c1; }
    .fl-moss-progress { height:6px; border-radius:999px; background:#283043; overflow:hidden; }
    .fl-moss-progress>span { display:block; height:100%; width:0; background:linear-gradient(90deg,#7b72ff,#55c7e9); transition:width .2s; }
    .fl-moss-message { flex:1; color:#aeb9cc; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .fl-moss-stat { color:#dce5f5; white-space:nowrap; font-variant-numeric:tabular-nums; }
    .fl-moss-button { border:1px solid #3d4a63; background:#202a3c; color:#eef3fc; border-radius:6px; padding:4px 9px; cursor:pointer; }
    .fl-moss-button:hover { background:#2a3750; }
    .fl-moss-button:disabled { opacity:.5; cursor:default; }
    .fl-moss-button.danger { border-color:#6d3944; color:#ffc0c9; }
    .fl-moss-dialog { width:min(1180px,94vw); height:min(820px,91vh); padding:0; border:1px solid #35425a; border-radius:12px; background:#0f141d; color:#edf3fd; box-shadow:0 24px 80px #000b; }
    .fl-moss-dialog::backdrop { background:#04060acf; }
    .fl-moss-modal { height:100%; display:grid; grid-template-rows:auto auto 1fr; font:13px/1.4 system-ui,sans-serif; }
    .fl-moss-modal header { display:flex; align-items:center; gap:10px; padding:13px 16px; border-bottom:1px solid #293349; }
    .fl-moss-modal h2 { margin:0; font-size:17px; flex:1; }
    .fl-moss-toolbar { display:flex; align-items:center; gap:8px; padding:9px 16px; border-bottom:1px solid #273145; flex-wrap:wrap; }
    .fl-moss-body { min-height:0; padding:14px 16px 20px; overflow:auto; display:grid; grid-template-columns:minmax(300px,.8fr) minmax(420px,1.4fr); gap:12px; align-content:start; }
    .fl-moss-card { border:1px solid #2a354a; border-radius:9px; background:#151c28; padding:11px; min-width:0; }
    .fl-moss-card h3 { margin:0 0 9px; color:#aebbd0; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
    .fl-moss-system { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .fl-moss-system article { border-radius:7px; background:#101620; padding:9px; }
    .fl-moss-system strong { display:block; margin-bottom:3px; }
    .fl-moss-muted { color:#94a2b8; font-size:11px; word-break:break-word; }
    .fl-moss-settings { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 10px; }
    .fl-moss-settings label { display:grid; gap:3px; color:#aebbd0; font-size:11px; }
    .fl-moss-settings input,.fl-moss-settings select,.fl-moss-editor textarea,.fl-moss-tracks { box-sizing:border-box; width:100%; border:1px solid #38465f; border-radius:6px; background:#101722; color:#edf3fd; padding:5px 7px; }
    .fl-moss-check { display:flex!important; grid-column:span 1; align-items:center; gap:7px!important; }
    .fl-moss-check input { width:auto; }
    .fl-moss-tracks { min-height:190px; padding:3px; }
    .fl-moss-tracks option { padding:5px; }
    .fl-moss-editor { display:grid; grid-template-rows:auto auto 1fr auto 1fr auto; gap:7px; min-height:470px; }
    .fl-moss-editor audio { width:100%; height:34px; }
    .fl-moss-editor textarea { min-height:112px; resize:vertical; font:12px/1.45 system-ui,sans-serif; }
    .fl-moss-meta { max-height:120px; overflow:auto; white-space:pre-wrap; font:10px/1.35 ui-monospace,monospace; color:#8e9db5; background:#0e141e; border-radius:6px; padding:7px; }
    .fl-moss-wide { grid-column:1/-1; }
    @media(max-width:860px) { .fl-moss-body { grid-template-columns:1fr; } .fl-moss-settings { grid-template-columns:1fr; } }
  `;
  document.head.appendChild(style);
}


function widget(node, name) {
  return node.widgets?.find((item) => item.name === name);
}


function value(node, name, fallback = "") {
  return widget(node, name)?.value ?? fallback;
}


function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}


async function payload(response) {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}


class PreprocessorPanel {
  constructor(node, container) {
    this.node = node;
    this.container = container;
    this.state = { status: "idle", phase: "idle", message: "Ready to analyze source audio", current: 0, total: 0 };
    this.system = null;
    this.tracks = [];
    this.selectedTrack = null;
    this.runId = null;
    this.pollTimer = null;
    this.settingsWidget = widget(node, "settings_json");
    this.settings = parsePreprocessorSettings(this.settingsWidget?.value);
    if (this.settingsWidget) {
      this.settingsWidget.type = "hidden";
      this.settingsWidget.computeSize = () => [0, -4];
    }
    injectStyles();
    this.buildCompact();
    this.buildDialog();
    this.renderCompact();
    panels.add(this);
  }

  buildCompact() {
    this.container.innerHTML = `
      <section class="fl-moss-compact">
        <div class="fl-moss-row"><span class="fl-moss-title">MOSS dataset preprocessor</span><span class="fl-moss-badge" data-role="status">Idle</span></div>
        <div class="fl-moss-progress"><span data-role="fill"></span></div>
        <div class="fl-moss-row"><span class="fl-moss-message" data-role="message">Ready to analyze source audio</span><span class="fl-moss-stat" data-role="count">0 / 0</span></div>
        <div class="fl-moss-row"><span class="fl-moss-stat" data-role="phase">Idle</span><span style="flex:1"></span><button class="fl-moss-button" data-action="open" type="button">Open preprocessor</button></div>
      </section>`;
    this.container.querySelector('[data-action="open"]').addEventListener("click", () => void this.open());
  }

  buildDialog() {
    this.dialog = document.createElement("dialog");
    this.dialog.className = "fl-moss-dialog";
    this.dialog.innerHTML = `
      <section class="fl-moss-modal">
        <header><h2>MiniMax Music 3 Dataset Preprocessor</h2><span class="fl-moss-badge" data-role="modal-status">Idle</span><button class="fl-moss-button" data-action="close" type="button">Close</button></header>
        <div class="fl-moss-toolbar">
          <button class="fl-moss-button" data-action="refresh" type="button">Refresh</button>
          <button class="fl-moss-button" data-action="install" type="button">Install backend</button>
          <button class="fl-moss-button" data-action="download" type="button">Download MOSS</button>
          <button class="fl-moss-button danger" data-action="cancel-download" type="button">Stop download</button>
          <button class="fl-moss-button danger" data-action="stop" type="button">Stop run</button>
          <span class="fl-moss-muted" data-role="paths"></span>
        </div>
        <main class="fl-moss-body">
          <section class="fl-moss-card fl-moss-wide"><h3>Runtime and model</h3><div class="fl-moss-system" data-role="system"></div></section>
          <div>
            <section class="fl-moss-card"><h3>Analysis settings</h3><div class="fl-moss-settings" data-role="settings"></div></section>
            <section class="fl-moss-card" style="margin-top:12px"><h3>Generated tracks</h3><select class="fl-moss-tracks" data-role="tracks" size="9"></select><p class="fl-moss-muted" data-role="dataset-message">Queue the node to create a dataset.</p></section>
          </div>
          <section class="fl-moss-card fl-moss-editor">
            <h3>Caption and lyrics review</h3>
            <audio controls preload="none" data-role="audio"></audio>
            <textarea data-role="caption" placeholder="Music caption"></textarea>
            <div class="fl-moss-muted">Lyrics and structure tags</div>
            <textarea data-role="lyrics" placeholder="[Verse]&#10;Lyrics"></textarea>
            <div class="fl-moss-row"><button class="fl-moss-button" data-action="save" type="button">Save sidecars</button><button class="fl-moss-button" data-action="approve" type="button">Approve</button><span class="fl-moss-muted" data-role="save-status"></span></div>
            <pre class="fl-moss-meta" data-role="metadata">Select a generated track to inspect its analysis and provenance.</pre>
          </section>
        </main>
      </section>`;
    document.body.appendChild(this.dialog);
    this.dialog.querySelector('[data-action="close"]').addEventListener("click", () => this.dialog.close());
    this.dialog.querySelector('[data-action="refresh"]').addEventListener("click", () => void this.refresh());
    this.dialog.querySelector('[data-action="install"]').addEventListener("click", () => void this.installBackend());
    this.dialog.querySelector('[data-action="download"]').addEventListener("click", () => void this.downloadModel());
    this.dialog.querySelector('[data-action="cancel-download"]').addEventListener("click", () => void this.stopDownload());
    this.dialog.querySelector('[data-action="stop"]').addEventListener("click", () => void this.stopRun());
    this.dialog.querySelector('[data-action="save"]').addEventListener("click", () => void this.saveTrack());
    this.dialog.querySelector('[data-action="approve"]').addEventListener("click", () => void this.approveTrack());
    this.dialog.querySelector('[data-role="tracks"]').addEventListener("change", (event) => this.selectTrack(Number(event.target.value)));
    this.dialog.addEventListener("close", () => this.stopPolling());
    this.buildSettings();
  }

  buildSettings() {
    const fields = [
      ["analysis_profile", "Analysis", "select", [["caption_only", "Caption only"], ["caption_and_lyrics", "Caption + lyrics"], ["full_analysis", "Full analysis"]]],
      ["write_policy", "Write policy", "select", [["fill_missing", "Fill missing"], ["replace_generated", "Replace generated"], ["replace_all", "Replace matching files"]]],
      ["execution_mode", "Execution", "select", [["auto_process_and_write", "Auto process + write"], ["require_review", "Require review"]]],
      ["model_policy", "Model", "select", [["download_if_missing", "Download if missing"], ["require_installed", "Require installed"]]],
      ["backend_policy", "Backend", "select", [["install_if_missing", "Install if missing"], ["require_installed", "Require installed"]]],
      ["min_segment_seconds", "Minimum seconds", "number", { min: 1, max: 1800, step: 1 }],
      ["target_segment_seconds", "Target seconds", "number", { min: 1, max: 1800, step: 1 }],
      ["max_segment_seconds", "Maximum seconds", "number", { min: 1, max: 1800, step: 1 }],
      ["output_sample_rate", "Sample rate", "select", [[32000, "32 kHz"], [44100, "44.1 kHz"], [48000, "48 kHz"]]],
      ["temperature", "Temperature", "number", { min: 0, max: 2, step: 0.05 }],
      ["max_new_tokens", "Maximum tokens", "number", { min: 128, max: 4096, step: 128 }],
      ["segment_long_tracks", "Segment long tracks", "checkbox"],
      ["preserve_channels", "Preserve channels", "checkbox"],
    ];
    const root = this.dialog.querySelector('[data-role="settings"]');
    for (const [key, labelText, type, options] of fields) {
      const label = document.createElement("label");
      if (type === "checkbox") label.className = "fl-moss-check";
      const input = document.createElement(type === "select" ? "select" : "input");
      input.dataset.setting = key;
      if (type === "select") {
        for (const [entryValue, text] of options) {
          const option = document.createElement("option");
          option.value = String(entryValue);
          option.textContent = text;
          input.appendChild(option);
        }
      } else {
        input.type = type;
        if (options) Object.assign(input, options);
      }
      if (type === "checkbox") label.append(input, document.createTextNode(labelText));
      else {
        const caption = document.createElement("span");
        caption.textContent = labelText;
        label.append(caption, input);
      }
      input.addEventListener("change", () => this.persistSettings());
      root.appendChild(label);
    }
    this.renderSettings();
  }

  renderSettings() {
    for (const input of this.dialog.querySelectorAll("[data-setting]")) {
      const current = this.settings[input.dataset.setting];
      if (input.type === "checkbox") input.checked = Boolean(current);
      else input.value = String(current);
    }
  }

  persistSettings() {
    for (const input of this.dialog.querySelectorAll("[data-setting]")) {
      let current = input.type === "checkbox" ? input.checked : input.value;
      if (input.type === "number" || input.dataset.setting === "output_sample_rate") current = Number(current);
      this.settings[input.dataset.setting] = current;
    }
    if (this.settingsWidget) {
      this.settingsWidget.value = JSON.stringify(this.settings);
      this.settingsWidget.callback?.(this.settingsWidget.value);
    }
    this.node.graph?.setDirtyCanvas(true, true);
  }

  async open() {
    if (!this.dialog.open) this.dialog.showModal();
    await this.refresh();
    this.startPolling();
  }

  async refresh() {
    await Promise.allSettled([this.refreshSystem(), this.refreshRuns(), this.refreshDataset()]);
  }

  async refreshSystem() {
    this.system = await payload(await api.fetchApi("/fl/minimax-music3/preprocess/status"));
    const model = this.system.model || {};
    const modelJob = this.system.model_job || {};
    const backend = this.system.backend || {};
    const backendJob = this.system.backend_job || {};
    const root = this.dialog.querySelector('[data-role="system"]');
    const completed = (model.files || []).filter((item) => item.state === "present").length;
    root.innerHTML = `
      <article><strong>MOSS-Music · ${model.verified ? "Verified" : modelJob.running ? "Downloading" : "Not ready"}</strong><div class="fl-moss-muted">${completed}/${(model.files || []).length} files · ${formatBytes(model.total_size)}</div><div class="fl-moss-progress" style="margin-top:7px"><span style="width:${modelJob.max ? Math.min(100, Number(modelJob.value || 0) / Number(modelJob.max) * 100) : model.verified ? 100 : 0}%"></span></div><div class="fl-moss-muted">${escapeHtml(modelJob.error || modelJob.message || model.model_path || "")}</div></article>
      <article><strong>Transformers backend · ${backend.verified ? "Verified" : backendJob.running ? "Installing" : "Not ready"}</strong><div class="fl-moss-muted">${escapeHtml(backend.backend_id || "Pinned CUDA backend")}</div><div class="fl-moss-muted">${escapeHtml(backendJob.error || backendJob.message || backend.message || backend.path || "")}</div></article>`;
    this.dialog.querySelector('[data-action="download"]').disabled = model.verified || modelJob.running;
    this.dialog.querySelector('[data-action="cancel-download"]').disabled = !modelJob.running;
    this.dialog.querySelector('[data-action="install"]').disabled = backend.verified || backendJob.running;
    this.dialog.querySelector('[data-role="paths"]').textContent = `Source: ${value(this.node, "source_folder")} · Output: ${value(this.node, "output_dataset")}`;
  }

  async refreshRuns() {
    const data = await payload(await api.fetchApi("/fl/minimax-music3/preprocess/runs"));
    const output = String(value(this.node, "output_dataset"));
    const run = (data.runs || []).find((item) => item.output_dataset === output) || data.runs?.[0];
    if (!run) return;
    this.runId = run.run_id;
    this.state = { ...this.state, ...run };
    this.renderCompact();
    const status = this.dialog.querySelector('[data-role="modal-status"]');
    status.textContent = run.status;
    status.dataset.state = run.status;
    this.dialog.querySelector('[data-action="stop"]').disabled = !["running", "stop_requested"].includes(run.status);
  }

  async refreshDataset() {
    const dataset = String(value(this.node, "output_dataset"));
    if (!dataset) return;
    const response = await api.fetchApi(`/fl/minimax-music3/preprocess/dataset/${encodeURIComponent(dataset)}/tracks`);
    if (!response.ok) {
      this.tracks = [];
      this.dialog.querySelector('[data-role="dataset-message"]').textContent = "Queue the node to create this dataset.";
      this.renderTrackList();
      return;
    }
    const data = await response.json();
    this.tracks = data.tracks || [];
    this.dialog.querySelector('[data-role="dataset-message"]').textContent = `${this.tracks.length} generated training segments`;
    this.renderTrackList();
  }

  renderTrackList() {
    const select = this.dialog.querySelector('[data-role="tracks"]');
    select.replaceChildren();
    this.tracks.forEach((track, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = track.audio;
      select.appendChild(option);
    });
    if (this.tracks.length) this.selectTrack(Math.min(this.selectedTrack ?? 0, this.tracks.length - 1));
  }

  selectTrack(index) {
    const track = this.tracks[index];
    if (!track) return;
    this.selectedTrack = index;
    this.dialog.querySelector('[data-role="tracks"]').value = String(index);
    this.dialog.querySelector('[data-role="caption"]').value = track.caption || "";
    this.dialog.querySelector('[data-role="lyrics"]').value = track.lyrics || "";
    this.dialog.querySelector('[data-role="metadata"]').textContent = JSON.stringify(track.metadata || {}, null, 2);
    const dataset = String(value(this.node, "output_dataset"));
    this.dialog.querySelector('[data-role="audio"]').src = api.apiURL(`/fl/minimax-music3/preprocess/dataset/${encodeURIComponent(dataset)}/audio?path=${encodeURIComponent(track.audio)}`);
  }

  async saveTrack() {
    const track = this.tracks[this.selectedTrack];
    if (!track) return;
    const status = this.dialog.querySelector('[data-role="save-status"]');
    status.textContent = "Saving…";
    try {
      const dataset = String(value(this.node, "output_dataset"));
      await payload(await api.fetchApi(`/fl/minimax-music3/preprocess/dataset/${encodeURIComponent(dataset)}/track`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio: track.audio,
          caption: this.dialog.querySelector('[data-role="caption"]').value,
          lyrics: this.dialog.querySelector('[data-role="lyrics"]').value,
        }),
      }));
      status.textContent = "Saved";
      await this.refreshDataset();
    } catch (error) {
      status.textContent = error.message;
    }
  }

  async approveTrack() {
    const track = this.tracks[this.selectedTrack];
    if (!track) return;
    const status = this.dialog.querySelector('[data-role="save-status"]');
    status.textContent = "Approving…";
    try {
      const dataset = String(value(this.node, "output_dataset"));
      await payload(await api.fetchApi(`/fl/minimax-music3/preprocess/dataset/${encodeURIComponent(dataset)}/track/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: track.audio }),
      }));
      status.textContent = "Approved";
      await this.refreshDataset();
    } catch (error) {
      status.textContent = error.message;
    }
  }

  async installBackend() {
    try { await payload(await api.fetchApi("/fl/minimax-music3/preprocess/backend/install", { method: "POST" })); }
    catch (error) { this.state.message = error.message; this.renderCompact(); }
    await this.refreshSystem();
  }

  async downloadModel() {
    try { await payload(await api.fetchApi("/fl/minimax-music3/preprocess/model/download", { method: "POST" })); }
    catch (error) { this.state.message = error.message; this.renderCompact(); }
    await this.refreshSystem();
  }

  async stopDownload() {
    try { await payload(await api.fetchApi("/fl/minimax-music3/preprocess/model/stop", { method: "POST" })); }
    catch (error) { this.state.message = error.message; this.renderCompact(); }
    await this.refreshSystem();
  }

  async stopRun() {
    if (!this.runId) return;
    await api.fetchApi(`/fl/minimax-music3/preprocess/runs/${encodeURIComponent(this.runId)}/stop`, { method: "POST" });
    await this.refreshRuns();
  }

  update(event) {
    this.state = { ...this.state, ...event };
    if (event.run_id) this.runId = event.run_id;
    this.renderCompact();
    if (this.dialog.open && event.status === "completed") void this.refreshDataset();
  }

  renderCompact() {
    const status = this.container.querySelector('[data-role="status"]');
    status.textContent = this.state.status || "idle";
    status.dataset.state = this.state.status || "idle";
    this.container.querySelector('[data-role="fill"]').style.width = `${preprocessPercent(this.state)}%`;
    const message = this.container.querySelector('[data-role="message"]');
    message.textContent = this.state.error || this.state.message || "Ready to analyze source audio";
    message.title = message.textContent;
    this.container.querySelector('[data-role="count"]').textContent = this.state.bytes_total ? `${formatBytes(this.state.bytes_current)} / ${formatBytes(this.state.bytes_total)}` : `${this.state.current || 0} / ${this.state.total || 0}`;
    this.container.querySelector('[data-role="phase"]').textContent = String(this.state.phase || "idle").replaceAll("_", " ");
  }

  startPolling() {
    this.stopPolling();
    this.pollTimer = setInterval(() => void this.refresh(), 1500);
  }

  stopPolling() {
    if (this.pollTimer) clearInterval(this.pollTimer);
    this.pollTimer = null;
  }

  dispose() {
    this.stopPolling();
    panels.delete(this);
    this.dialog.remove();
    this.container.replaceChildren();
  }
}


app.registerExtension({
  name: "ComfyUI.FL_MiniMaxMusic3.Preprocessor",
  nodeCreated(node) {
    if (node.comfyClass !== "FL_MiniMaxMusic3DatasetPreprocessor") return;
    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = "112px";
    const domWidget = node.addDOMWidget("fl_minimax_music3_preprocessor", "fl-minimax-music3-preprocessor", container, { getMinHeight: () => 112, hideOnZoom: false, serialize: false });
    const panel = new PreprocessorPanel(node, container);
    domWidget.onRemove = () => panel.dispose();
    node.setSize([Math.max(node.size[0], 370), Math.max(node.size[1], 285)]);
  },
});


api.addEventListener("fl_minimax_music3_preprocess_status", (event) => {
  const detail = event.detail || {};
  for (const panel of panels) {
    if (String(panel.node.id) === String(detail.node)) panel.update(detail);
  }
});
