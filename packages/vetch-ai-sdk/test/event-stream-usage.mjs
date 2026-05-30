import assert from "node:assert/strict";

import {
  createStreamObservation,
  createVetchEvent,
  observeStreamPart,
} from "../dist/event.js";

const finishUsage = {
  inputTokens: { total: 42, noCache: 30, cacheRead: 12, cacheWrite: 3 },
  outputTokens: { total: 9, text: 2, reasoning: 7 },
};

const observation = createStreamObservation();
observeStreamPart({ type: "text-delta", id: "t1", delta: "OK" }, observation);
observeStreamPart(
  {
    type: "finish",
    finishReason: "stop",
    usage: finishUsage,
  },
  observation,
);

assert.equal(observation.finished, true);
assert.deepEqual(observation.usage, finishUsage);

const event = createVetchEvent({
  operation: "stream",
  model: { provider: "openai", modelId: "gpt-4.1-mini" },
  params: { maxOutputTokens: 128 },
  streamObservation: observation,
  startTimeMs: Date.now() - 5,
  options: {},
});

assert.deepEqual(event.usage?.text, {
  input_tokens: 42,
  output_tokens: 2,
  total_tokens: 44,
});
assert.deepEqual(event.usage?.reasoning, {
  input_tokens: 0,
  output_tokens: 7,
  total_tokens: 7,
});
assert.equal(event.visible_output_chars, 2);
assert.equal(event.cache_read_tokens, 12);
assert.equal(event.cache_creation_tokens, 3);
assert.equal(event.finish_reason, "stop");
assert.equal(event.complete, true);
assert.equal(event.is_stream, true);
assert.equal(event.model_known, true);
assert.equal(event.energy_source, "registry");
assert.equal(typeof event.estimated_energy_wh, "number");

const cancelledObservation = createStreamObservation();
observeStreamPart({ type: "text-delta", id: "t1", delta: "partial" }, cancelledObservation);
cancelledObservation.cancelled = true;

const cancelledEvent = createVetchEvent({
  operation: "stream",
  model: { provider: "openai", modelId: "gpt-4.1-mini" },
  params: {},
  streamObservation: cancelledObservation,
  startTimeMs: Date.now() - 3,
  options: {},
});

assert.equal(cancelledEvent.complete, false);
assert.ok(cancelledEvent.vetch_warnings.includes("stream_cancelled_partial"));
