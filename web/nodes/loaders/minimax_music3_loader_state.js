export const MINIMAX_MUSIC3_ARTIFACTS = [
  {
    key: "model",
    label: "Diffusion model",
    filename: "minimax_music3_dit_fp16.safetensors",
    expectedSize: 4914197682,
  },
  {
    key: "clip",
    label: "Text encoder",
    filename: "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
    expectedSize: 9196611886,
  },
  {
    key: "vae",
    label: "DAV VAE",
    filename: "minimax_music3_dav.safetensors",
    expectedSize: 216696128,
  },
];

export function createLoaderState() {
  return {
    state: "idle",
    message: "Checking installed model files…",
    artifacts: Object.fromEntries(MINIMAX_MUSIC3_ARTIFACTS.map((artifact) => [
      artifact.key,
      {
        ...artifact,
        state: "unknown",
        value: 0,
        max: artifact.expectedSize,
        resumed: false,
        speed: 0,
        lastValue: 0,
        lastAt: 0,
      },
    ])),
  };
}

function copyState(state) {
  return {
    ...state,
    artifacts: Object.fromEntries(Object.entries(state.artifacts).map(([key, artifact]) => [
      key,
      { ...artifact },
    ])),
  };
}

export function applyInventory(state, inventory) {
  const next = copyState(state);
  const items = Array.isArray(inventory?.artifacts) ? inventory.artifacts : [];
  for (const item of items) {
    const artifact = next.artifacts[item.key];
    if (!artifact) continue;
    artifact.label = item.label || artifact.label;
    artifact.filename = item.filename || artifact.filename;
    artifact.max = Number(item.expected_size) || artifact.expectedSize;
    artifact.value = Math.max(0, Number(item.available_bytes) || 0);
    artifact.state = item.state || "unknown";
    artifact.resumed = artifact.state === "partial";
    artifact.speed = 0;
    artifact.lastValue = artifact.value;
    artifact.lastAt = 0;
  }

  const artifacts = Object.values(next.artifacts);
  if (artifacts.every((artifact) => artifact.state === "present")) {
    next.state = "installed";
    next.message = "All files are installed · queue to verify and load";
  } else if (artifacts.some((artifact) => artifact.state === "partial")) {
    next.state = "partial";
    next.message = "A partial download will resume on the next queue";
  } else if (artifacts.some((artifact) => artifact.state === "invalid_size")) {
    next.state = "attention";
    next.message = "One or more files will be repaired on the next queue";
  } else {
    next.state = "missing";
    next.message = "Missing files will download when this node is queued";
  }
  return next;
}

export function applyLoaderEvent(state, detail, now = Date.now()) {
  const next = copyState(state);
  const eventState = String(detail?.state || "running");
  const artifact = detail?.artifact ? next.artifacts[detail.artifact] : null;
  next.state = eventState;
  next.message = detail?.message || next.message;

  if (artifact) {
    const value = Number(detail.value);
    const maximum = Number(detail.max);
    artifact.state = eventState;
    artifact.label = detail.label || artifact.label;
    artifact.filename = detail.filename || artifact.filename;
    if (Number.isFinite(maximum) && maximum > 0) artifact.max = maximum;
    if (Number.isFinite(value) && value >= 0) artifact.value = value;
    artifact.resumed = Boolean(detail.resumed);

    if (eventState === "downloading") {
      if (artifact.lastAt && now > artifact.lastAt && artifact.value >= artifact.lastValue) {
        const instantSpeed = (artifact.value - artifact.lastValue) / ((now - artifact.lastAt) / 1000);
        if (instantSpeed > 0) artifact.speed = artifact.speed ? artifact.speed * 0.65 + instantSpeed * 0.35 : instantSpeed;
      }
      artifact.lastValue = artifact.value;
      artifact.lastAt = now;
    } else {
      artifact.speed = 0;
      artifact.lastValue = artifact.value;
      artifact.lastAt = 0;
    }

    if (["verified", "ready"].includes(eventState)) artifact.value = artifact.max;
  }

  if (eventState === "files_ready") next.state = "verified";
  if (eventState === "complete") next.state = "ready";
  return next;
}

export function markLoaderCached(state) {
  const next = copyState(state);
  next.state = "cached";
  next.message = "Loader outputs reused from ComfyUI cache";
  for (const artifact of Object.values(next.artifacts)) {
    artifact.state = "ready";
    artifact.value = artifact.max;
    artifact.speed = 0;
  }
  return next;
}

export function markLoaderFailed(state, message, interrupted = false) {
  const next = copyState(state);
  next.state = interrupted ? "interrupted" : "error";
  next.message = message || (interrupted ? "Loading interrupted" : "Loading failed");
  return next;
}

export function progressRatio(artifact) {
  if (["present", "verified", "loading", "ready"].includes(artifact.state)) return 1;
  const maximum = Math.max(1, Number(artifact.max) || 1);
  return Math.max(0, Math.min(1, (Number(artifact.value) || 0) / maximum));
}

export function formatBytes(value) {
  const bytes = Math.max(0, Number(value) || 0);
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = bytes;
  let unit = -1;
  do {
    amount /= 1024;
    unit += 1;
  } while (amount >= 1024 && unit < units.length - 1);
  const digits = amount >= 100 ? 0 : amount >= 10 ? 1 : 2;
  return `${amount.toFixed(digits)} ${units[unit]}`;
}

export function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 60) return `${Math.ceil(seconds)}s left`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)}m left`;
  return `${Math.ceil(seconds / 3600)}h left`;
}
