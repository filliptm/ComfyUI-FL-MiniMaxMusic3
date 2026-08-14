import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const STATE_MODULE_URL = new URL("../web/nodes/loaders/minimax_music3_loader_state.js", import.meta.url);
const PANEL_URL = new URL("../web/nodes/loaders/FL_MiniMaxMusic3Loader.js", import.meta.url);

async function loadStateModule() {
  const source = await readFile(STATE_MODULE_URL, "utf8");
  const encoded = Buffer.from(source).toString("base64");
  return import(`data:text/javascript;base64,${encoded}`);
}

test("inventory distinguishes installed, partial, and missing artifacts", async () => {
  const module = await loadStateModule();
  const initial = module.createLoaderState();
  const state = module.applyInventory(initial, {
    artifacts: [
      { key: "model", state: "present", expected_size: 100, available_bytes: 100 },
      { key: "clip", state: "partial", expected_size: 200, available_bytes: 50 },
      { key: "vae", state: "missing", expected_size: 25, available_bytes: 0 },
    ],
  });

  assert.equal(state.state, "partial");
  assert.equal(state.artifacts.model.state, "present");
  assert.equal(state.artifacts.clip.value, 50);
  assert.equal(module.progressRatio(state.artifacts.clip), 0.25);
  assert.equal(state.artifacts.vae.state, "missing");
});

test("download events track resume progress, speed, and completion", async () => {
  const module = await loadStateModule();
  let state = module.createLoaderState();
  state = module.applyLoaderEvent(state, {
    state: "downloading",
    artifact: "clip",
    value: 100,
    max: 1000,
    resumed: true,
    message: "Resuming text encoder",
  }, 1000);
  state = module.applyLoaderEvent(state, {
    state: "downloading",
    artifact: "clip",
    value: 300,
    max: 1000,
    resumed: true,
    message: "Resuming text encoder",
  }, 2000);

  assert.equal(state.artifacts.clip.speed, 200);
  assert.equal(module.progressRatio(state.artifacts.clip), 0.3);
  assert.equal(state.artifacts.clip.resumed, true);

  state = module.applyLoaderEvent(state, {
    state: "verified",
    artifact: "clip",
    value: 1000,
    max: 1000,
    message: "Text encoder verified",
  }, 3000);
  assert.equal(state.artifacts.clip.state, "verified");
  assert.equal(module.progressRatio(state.artifacts.clip), 1);
  assert.equal(state.artifacts.clip.speed, 0);
});

test("completion, cached execution, and interruption remain distinct", async () => {
  const module = await loadStateModule();
  let state = module.applyLoaderEvent(module.createLoaderState(), {
    state: "complete",
    message: "MiniMax Music 3 is ready",
  });
  assert.equal(state.state, "ready");

  state = module.markLoaderCached(state);
  assert.equal(state.state, "cached");
  assert.ok(Object.values(state.artifacts).every((artifact) => artifact.state === "ready"));

  state = module.markLoaderFailed(state, "Loading interrupted", true);
  assert.equal(state.state, "interrupted");
  assert.equal(state.message, "Loading interrupted");
});

test("byte and ETA formatting stays compact", async () => {
  const module = await loadStateModule();
  assert.equal(module.formatBytes(4914197682), "4.58 GiB");
  assert.equal(module.formatBytes(216696128), "207 MiB");
  assert.equal(module.formatEta(59.1), "60s left");
  assert.equal(module.formatEta(61), "2m left");
});

test("loader dashboard fills the resizable DOM widget", async () => {
  const source = await readFile(PANEL_URL, "utf8");
  assert.match(source, /\.flmm3-container\s*\{[^}]*height:\s*100%/s);
  assert.match(source, /\.flmm3-panel\s*\{[^}]*grid-template-rows:\s*auto minmax\(0, 1fr\) auto/s);
  assert.match(source, /\.flmm3-panel\s*\{[^}]*height:\s*100%/s);
  assert.match(source, /container\.style\.height\s*=\s*"100%"/);
});

test("loader dashboard keeps compact responsive minimums", async () => {
  const source = await readFile(PANEL_URL, "utf8");
  assert.match(source, /const MIN_NODE_WIDTH = 330;/);
  assert.match(source, /const MIN_NODE_HEIGHT = 290;/);
  assert.match(source, /const MIN_PANEL_HEIGHT = 200;/);
  assert.match(source, /container-type:\s*size/);
  assert.match(source, /@container \(max-height: 215px\)/);
  assert.match(source, /getMinHeight:\s*\(\) => MIN_PANEL_HEIGHT,/);
});
