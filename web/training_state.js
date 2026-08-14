export function createTrainingState() {
  return {
    runId: null,
    sequence: -1,
    status: "idle",
    phase: "idle",
    current: 0,
    total: 0,
    message: "Ready to train",
    metrics: {},
    history: [],
    validation: [],
    adapterPath: null,
    error: null,
  };
}

export function progressPercent(state) {
  if (!state.total) return 0;
  return Math.max(0, Math.min(100, (Number(state.current) / Number(state.total)) * 100));
}

export function boundedMetrics(metrics, limit = 400) {
  if (metrics.length <= limit) return metrics;
  const stride = Math.ceil(metrics.length / limit);
  const result = metrics.filter((_, index) => index % stride === 0);
  const final = metrics.at(-1);
  if (result.at(-1) !== final) result.push(final);
  return result.slice(-limit);
}

export function applyTrainingEvent(state, event) {
  if (!event || (state.runId && event.run_id && state.runId !== event.run_id)) return state;
  const sequence = Number(event.sequence ?? state.sequence + 1);
  if (sequence <= state.sequence) return state;
  const history = event.metrics && event.current > 0
    ? boundedMetrics([...state.history, { step: Number(event.current), ...event.metrics }])
    : state.history;
  return {
    ...state,
    runId: event.run_id ?? state.runId,
    sequence,
    status: event.state ?? event.status ?? state.status,
    phase: event.phase ?? state.phase,
    current: Number(event.current ?? state.current),
    total: Number(event.total ?? state.total),
    message: event.message ?? state.message,
    metrics: event.metrics ?? state.metrics,
    history,
    adapterPath: event.adapter_path ?? event.artifact ?? state.adapterPath,
    error: event.error ?? state.error,
  };
}

export function hydrateTrainingState(payload) {
  const state = payload?.state ?? {};
  return {
    ...createTrainingState(),
    runId: state.run_id ?? payload?.spec?.run_id ?? null,
    sequence: Number(state.sequence ?? 0),
    status: state.status ?? "idle",
    phase: state.phase ?? "idle",
    current: Number(state.current ?? 0),
    total: Number(state.total ?? 0),
    message: state.message ?? "Ready to train",
    metrics: state.metrics ?? {},
    history: boundedMetrics(payload?.metrics ?? []),
    validation: payload?.validation ?? [],
    adapterPath: state.adapter_path ?? null,
    error: state.error ?? null,
  };
}
