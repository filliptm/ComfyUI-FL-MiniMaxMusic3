export const DEFAULT_PREPROCESSOR_SETTINGS = Object.freeze({
  schema_version: 1,
  analysis_profile: "caption_and_lyrics",
  segment_long_tracks: true,
  min_segment_seconds: 8,
  target_segment_seconds: 42,
  max_segment_seconds: 60,
  output_sample_rate: 44100,
  preserve_channels: true,
  write_policy: "fill_missing",
  execution_mode: "auto_process_and_write",
  model_policy: "download_if_missing",
  backend_policy: "install_if_missing",
  temperature: 0.2,
  max_new_tokens: 1024,
});

export function parsePreprocessorSettings(value) {
  try {
    const parsed = typeof value === "string" ? JSON.parse(value) : value;
    return { ...DEFAULT_PREPROCESSOR_SETTINGS, ...(parsed && typeof parsed === "object" ? parsed : {}) };
  } catch {
    return { ...DEFAULT_PREPROCESSOR_SETTINGS };
  }
}

export function preprocessPercent(state) {
  const bytesTotal = Number(state?.bytes_total || 0);
  if (bytesTotal > 0) return Math.max(0, Math.min(100, (Number(state.bytes_current || 0) / bytesTotal) * 100));
  const total = Number(state?.total || 0);
  if (total > 0) return Math.max(0, Math.min(100, (Number(state.current || 0) / total) * 100));
  return state?.status === "completed" ? 100 : 0;
}

export function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index > 2 ? 2 : 1)} ${units[index]}`;
}
