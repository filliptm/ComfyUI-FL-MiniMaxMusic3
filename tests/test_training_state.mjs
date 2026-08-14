import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const MODULE_URL = new URL("../web/training_state.js", import.meta.url);

async function loadModule() {
  const source = await readFile(MODULE_URL, "utf8");
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}

test("training events reject stale sequences and update progress", async () => {
  const module = await loadModule();
  let state = module.createTrainingState();
  state = module.applyTrainingEvent(state, { run_id: "run-1", sequence: 2, state: "running", phase: "training", current: 5, total: 10, metrics: { loss: 0.5 } });
  const stale = module.applyTrainingEvent(state, { run_id: "run-1", sequence: 1, current: 1 });
  assert.equal(stale, state);
  assert.equal(module.progressPercent(state), 50);
  assert.equal(state.history.length, 1);
});

test("metric history remains bounded", async () => {
  const module = await loadModule();
  const metrics = Array.from({ length: 1000 }, (_, step) => ({ step, loss: 1 / (step + 1) }));
  const bounded = module.boundedMetrics(metrics, 100);
  assert.ok(bounded.length <= 100);
  assert.equal(bounded.at(-1), metrics.at(-1));
});

test("trainer UI matches the VoxCPM on-node training dashboard", async () => {
  const source = await readFile(new URL("../web/training.js", import.meta.url), "utf8");
  assert.match(source, /addDOMWidget/);
  assert.match(source, /MiniMax Music 3 Training/);
  assert.match(source, /Training Progress/);
  assert.match(source, /Loss History/);
  assert.match(source, /Validation Samples/);
  assert.match(source, /Audio samples will appear at each checkpoint/);
  assert.match(source, /getMinHeight: \(\) => 400/);
  assert.doesNotMatch(source, /Open dashboard/);
  assert.doesNotMatch(source, /showModal/);
});
