import assert from "node:assert/strict";

import { createVetchEvent } from "../dist/event.js";

const event = createVetchEvent({
  operation: "generate",
  model: { provider: "openai", modelId: "gpt-4.1-mini" },
  params: {
    maxOutputTokens: 128,
    providerOptions: {
      vetch: {
        tags: { feature: "smoke" },
        protocol: { expectedOutputClass: "tool" },
      },
    },
  },
  result: {
    content: [
      { type: "reasoning", text: "hidden reasoning should not be visible output" },
      { type: "text", text: "OK" },
      { type: "tool-call", toolCallId: "call_1", toolName: "done", input: "{}" },
    ],
    finishReason: "tool-calls",
    usage: {
      inputTokens: { total: 42, noCache: 30, cacheRead: 12, cacheWrite: 3 },
      outputTokens: { total: 9, text: 2, reasoning: 7 },
    },
    warnings: [],
  },
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
assert.equal(event.tool_call_count, 1);
assert.equal(event.tool_result_count, 0);
assert.equal(event.cache_read_tokens, 12);
assert.equal(event.cache_creation_tokens, 3);
assert.deepEqual(event.tags, { feature: "smoke" });
assert.equal(event.model_known, true);
assert.equal(event.energy_source, "registry");
assert.equal(typeof event.estimated_energy_wh, "number");
assert.equal(typeof event.estimated_carbon_g, "number");
assert.equal(typeof event.estimated_water_l, "number");
assert.equal(typeof event.estimated_cost_usd, "number");
assert.equal(event.tracking_degraded, false);
