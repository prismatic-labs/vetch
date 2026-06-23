import { describe, expect, it } from "vitest";

import { resolveModelMatch } from "../dist/calculation.js";
import { createVetchEvent } from "../dist/event.js";

// Mirrors Python tests/test_resolver_precision.py
const GOLDEN_PRECISION = {
  "gpt-4o": "exact",
  "gemini-3.1-pro": "exact",
  "claude-sonnet-4-6": "exact",
  "gpt-4o-2024-05-13": "alias",
  "gemini-3-flash-preview": "alias",
  "claude-sonnet-4-6-experimental": "prefix",
  "gemini-9-ultra": "family",
  "gemma-4-31b-it": "family",
  "totally-unknown-model-xyz": "fallback",
};

function generateResult(usage) {
  return {
    content: [{ type: "text", text: "OK" }],
    finishReason: { unified: "stop", raw: "stop" },
    usage: usage ?? {
      inputTokens: { total: 1000, noCache: 1000, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 500, text: 500, reasoning: 0 },
    },
    warnings: [],
  };
}

describe("resolveModelMatch precision (parity with Python)", () => {
  for (const [model, expected] of Object.entries(GOLDEN_PRECISION)) {
    it(`classifies ${model} as ${expected}`, () => {
      expect(resolveModelMatch(model).precision).toBe(expected);
    });
  }

  it("known flag is false only for fallback", () => {
    for (const [model, expected] of Object.entries(GOLDEN_PRECISION)) {
      expect(resolveModelMatch(model).known).toBe(expected !== "fallback");
    }
  });
});

describe("case-insensitivity", () => {
  for (const model of ["GPT-4O", "Gpt-4O", "gPt-4o"]) {
    it(`${model} resolves like lowercase`, () => {
      const m = resolveModelMatch(model);
      expect(m.resolvedModel).toBe("gpt-4o");
      expect(m.precision).toBe("exact");
    });
  }

  it("'31b' does not match the '1b' small hint; mid-size biases large", () => {
    const m = resolveModelMatch("Gemma-4-31b-it");
    expect(m.precision).toBe("family");
    expect(m.resolvedModel).toBe("gemini-3.1-pro");
  });
});

describe("conservative family fallback", () => {
  it("large hint picks frontier", () => {
    expect(resolveModelMatch("gemini-9-pro").resolvedModel).toBe("gemini-3.1-pro");
  });
  it("small hint picks small", () => {
    expect(resolveModelMatch("gemini-9-flash-lite").resolvedModel).toBe("gemini-3-flash");
  });
  it("ambiguous biases large (default == frontier)", () => {
    expect(resolveModelMatch("claude-9").resolvedModel).toBe("claude-sonnet-4-6");
  });
  it("unknown family is fallback", () => {
    expect(resolveModelMatch("qwen-72b").precision).toBe("fallback");
  });
});

describe("prefix/family tier floor", () => {
  it("an exact match keeps its measured tier", async () => {
    const event = await createVetchEvent({
      operation: "generate",
      model: { provider: "openai", modelId: "gpt-4o" },
      params: {},
      result: generateResult(),
      startTimeMs: Date.now(),
      options: {},
    });
    expect(event.energy_tier).toBe(1);
    expect(event.model_match).toBe("exact");
  });

  it("a prefix proxy into a Tier-1 row is floored to Tier 3", async () => {
    const event = await createVetchEvent({
      operation: "generate",
      model: { provider: "openai", modelId: "gpt-4o-frontier-2099" },
      params: {},
      result: generateResult(),
      startTimeMs: Date.now(),
      options: {},
    });
    expect(event.model_match).toBe("prefix");
    expect(event.energy_tier).toBe(3);
    expect(event.energy_uncertainty_pct).toBe(1000);
  });

  it("a family proxy is Tier 3 and flagged family", async () => {
    const event = await createVetchEvent({
      operation: "generate",
      model: { provider: "google", modelId: "gemini-9-ultra" },
      params: {},
      result: generateResult(),
      startTimeMs: Date.now(),
      options: {},
    });
    expect(event.model_match).toBe("family");
    expect(event.energy_tier).toBe(3);
    expect(event.model_known).toBe(true);
  });
});

describe("tracking_degraded recalibration (parity with Python)", () => {
  async function eventFor(provider, modelId) {
    return createVetchEvent({
      operation: "generate",
      model: { provider, modelId },
      params: {},
      result: generateResult(),
      startTimeMs: Date.now(),
      options: {},
    });
  }

  it("a healthy exact call is not degraded", async () => {
    expect((await eventFor("openai", "gpt-4o")).tracking_degraded).toBe(false);
  });

  it("an honest Tier-3 known model is not degraded", async () => {
    expect((await eventFor("google", "gemini-2.5-pro")).tracking_degraded).toBe(false);
  });

  it("a family proxy is degraded", async () => {
    expect((await eventFor("google", "gemini-9-ultra")).tracking_degraded).toBe(true);
  });

  it("an unknown model is degraded", async () => {
    expect((await eventFor("openai", "totally-unknown-xyz")).tracking_degraded).toBe(true);
  });
});
