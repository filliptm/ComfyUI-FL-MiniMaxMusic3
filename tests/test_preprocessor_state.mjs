import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_PREPROCESSOR_SETTINGS, formatBytes, parsePreprocessorSettings, preprocessPercent } from "../web/preprocessor_state.js";


test("invalid serialized settings fall back to pinned defaults", () => {
  assert.deepEqual(parsePreprocessorSettings("not json"), { ...DEFAULT_PREPROCESSOR_SETTINGS });
});

test("serialized settings override only provided values", () => {
  const value = parsePreprocessorSettings('{"max_segment_seconds":45}');
  assert.equal(value.max_segment_seconds, 45);
  assert.equal(value.analysis_profile, "caption_and_lyrics");
});

test("progress prioritizes byte progress while downloading", () => {
  assert.equal(preprocessPercent({ bytes_current: 25, bytes_total: 100, current: 8, total: 8 }), 25);
  assert.equal(preprocessPercent({ current: 3, total: 4 }), 75);
  assert.equal(preprocessPercent({ status: "completed" }), 100);
});

test("byte formatting remains compact", () => {
  assert.equal(formatBytes(1024 ** 3), "1.00 GiB");
});
