import assert from "node:assert/strict";

import { createVetchEvent } from "../dist/event.js";

const eventArgs = {
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
};
const event = await createVetchEvent(eventArgs);

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
assert.equal(event.estimated_water_l, null);
assert.equal(event.water_accounting_kind, null);
assert.equal(typeof event.estimated_cost_usd, "number");
assert.equal(event.tracking_degraded, false);

const overriddenWaterEvent = await createVetchEvent({
  ...eventArgs,
  options: {
    wueOverride: 0.5,
    wueBoundary: "onsite",
    waterAccountingKind: "consumption",
    wueUncertaintyPct: 25,
  },
});
assert.equal(typeof overriddenWaterEvent.estimated_water_l, "number");
assert.equal(overriddenWaterEvent.water_boundary, "onsite");
assert.equal(overriddenWaterEvent.water_accounting_kind, "consumption");
assert.equal(overriddenWaterEvent.water_uncertainty_pct, 25);

const googleWaterEvent = await createVetchEvent({
  ...eventArgs,
  model: { provider: "google.generative-ai", modelId: "gemini-2.5-pro" },
});
assert.equal(typeof googleWaterEvent.estimated_water_l, "number");
assert.equal(googleWaterEvent.water_accounting_kind, "consumption");
