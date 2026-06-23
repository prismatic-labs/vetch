import { readFile } from "node:fs/promises";

import { generateText, streamText } from "ai";
import { MockLanguageModelV3, simulateReadableStream } from "ai/test";
import { describe, expect, it } from "vitest";

import { createVetchEvent } from "../dist/event.js";
import {
  createVetchMiddleware,
  createVetchSession,
  detectAdvisories,
  detectPerCallAdvisories,
  VETCH_VERSION,
  withVetch,
} from "../dist/index.js";

const usageWithReasoning = {
  inputTokens: { total: 1000, noCache: 200, cacheRead: 800, cacheWrite: 25 },
  outputTokens: { total: 150, text: 100, reasoning: 50 },
};

const basicUsage = {
  inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 4, text: 2, reasoning: 2 },
};

const finishStop = { unified: "stop", raw: "stop_sequence" };

function fixturePath(name) {
  return new URL(`./fixtures/${name}`, import.meta.url);
}

async function readFixture(name) {
  return JSON.parse(await readFile(fixturePath(name), "utf8"));
}

function expectEventFields(event, fields) {
  for (const [key, expected] of Object.entries(fields)) {
    if (typeof expected === "number") {
      expect(event[key]).toBeCloseTo(expected, 12);
    } else {
      expect(event[key]).toEqual(expected);
    }
  }
}

function generateResult({ usage = basicUsage, text = "OK", finishReason = finishStop } = {}) {
  return {
    content: [{ type: "text", text }],
    finishReason,
    usage,
    warnings: [],
  };
}

function streamResult({ usage = basicUsage, text = "OK", finishReason = finishStop } = {}) {
  return {
    stream: simulateReadableStream({
      chunks: [
        { type: "text-start", id: "text-1" },
        { type: "text-delta", id: "text-1", delta: text },
        { type: "text-end", id: "text-1" },
        { type: "finish", finishReason, usage },
      ],
      initialDelayInMs: null,
      chunkDelayInMs: null,
    }),
  };
}

async function runGenerate(middleware, { sessionId, outputTokens = 10 } = {}) {
  const vetch = { protocol: { expectedToolUse: true } };
  if (sessionId !== undefined) {
    vetch.attribution = { sessionId };
  }
  await middleware.wrapGenerate({
    doGenerate: async () =>
      generateResult({
        usage: {
          inputTokens: { total: 20, noCache: 20, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: outputTokens, text: outputTokens, reasoning: 0 },
        },
      }),
    doStream: async () => {
      throw new Error("doStream should not be called");
    },
    params: {
      providerOptions: { vetch },
    },
    model: { provider: "openai", modelId: "gpt-4.1-mini" },
  });
}

describe("schema v2 calculation parity", () => {
  it("keeps the exported VETCH_VERSION in sync with package.json", async () => {
    const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
    expect(VETCH_VERSION).toBe(pkg.version);
  });

  it("matches the Python prepare_inference_metrics fixture for cache, reasoning, cost, carbon, and water", async () => {
    const fixture = await readFixture("python-gpt-41-mini-cache.json");
    const event = await createVetchEvent({
      operation: "generate",
      model: { provider: "openai", modelId: "gpt-4.1-mini" },
      params: {
        providerOptions: {
          vetch: {
            attribution: { sessionId: "golden" },
          },
        },
      },
      result: generateResult({ usage: usageWithReasoning }),
      startTimeMs: Date.now(),
      options: { region: fixture.inputs.region },
    });

    expect(event.schema_version).toBe("2");
    expect(event.session_id).toBe("golden");
    expect(event.finish_reason).toBe("stop");
    expect(event.raw_finish_reason).toBe("stop_sequence");
    expect(event.usage.text).toEqual({
      input_tokens: 1000,
      output_tokens: 100,
      total_tokens: 1100,
    });
    expect(event.usage.reasoning).toEqual({
      input_tokens: 0,
      output_tokens: 50,
      total_tokens: 50,
    });
    expectEventFields(event, fixture.fields);
  });

  it("marks model_known from registry resolution and uses fallback estimates for unknown models", async () => {
    const gatewayEvent = await createVetchEvent({
      operation: "generate",
      model: { provider: "gateway", modelId: "openai/gpt-4.1-mini" },
      params: {},
      result: generateResult(),
      startTimeMs: Date.now(),
      options: {},
    });

    expect(gatewayEvent.provider).toBe("openai");
    expect(gatewayEvent.model).toBe("gpt-4.1-mini");
    expect(gatewayEvent.model_known).toBe(true);
    expect(gatewayEvent.energy_source).toBe("registry");

    const event = await createVetchEvent({
      operation: "generate",
      model: { provider: "unknown", modelId: "totally-unknown-model-xyz" },
      params: {},
      result: generateResult({
        usage: {
          inputTokens: { total: 100, noCache: 100, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 10, text: 10, reasoning: 0 },
        },
      }),
      startTimeMs: Date.now(),
      options: {},
    });

    expect(event.model_known).toBe(false);
    expect(event.energy_source).toBe("fallback");
    expect(event.billing_tier).toBe("none");
    expect(event.estimated_energy_wh).toBeGreaterThan(0);
    expect(event.estimated_cost_usd).toBe(0);
    expect(event.vetch_warnings.some((warning) => warning.includes("not in registry"))).toBe(true);
  });

  it("estimates usage locally from visible chars when provider usage is unavailable", async () => {
    const event = await createVetchEvent({
      operation: "generate",
      model: { provider: "openai", modelId: "gpt-4.1-mini" },
      params: {},
      result: {
        content: [{ type: "text", text: "abcdefgh" }],
        finishReason: finishStop,
        warnings: [],
      },
      startTimeMs: Date.now(),
      options: {},
    });

    expect(event.usage_estimated).toBe(true);
    expect(event.usage_estimation_method).toBe("char_ratio");
    expect(event.usage.text).toEqual({
      input_tokens: 4,
      output_tokens: 2,
      total_tokens: 6,
    });
    expect(event.estimated_energy_wh).toBeGreaterThan(0);
    expect(event.energy_uncertainty_pct).toBeGreaterThanOrEqual(50);
  });
});

describe("AI SDK v6 middleware integration", () => {
  it("records generateText events with reasoning output kept out of text output", async () => {
    const events = [];
    const model = withVetch(
      new MockLanguageModelV3({
        provider: "openai",
        modelId: "gpt-4.1-mini",
        doGenerate: generateResult(),
      }),
      {
        emitter: (event) => events.push(event),
        emissionMode: "await",
      },
    );

    const result = await generateText({
      model,
      prompt: "Say OK",
      providerOptions: {
        vetch: {
          attribution: { sessionId: "generate-session" },
        },
      },
    });

    expect(result.text).toBe("OK");
    expect(events).toHaveLength(1);
    expect(events[0].ai_sdk_operation).toBe("generate");
    expect(events[0].session_id).toBe("generate-session");
    expect(events[0].usage.text).toEqual({
      input_tokens: 10,
      output_tokens: 2,
      total_tokens: 12,
    });
    expect(events[0].usage.reasoning).toEqual({
      input_tokens: 0,
      output_tokens: 2,
      total_tokens: 2,
    });
    expect(events[0].finish_reason).toBe("stop");
  });

  it("records streamText events after the stream is consumed", async () => {
    const events = [];
    const model = withVetch(
      new MockLanguageModelV3({
        provider: "openai",
        modelId: "gpt-4.1-mini",
        doStream: streamResult(),
      }),
      {
        emitter: (event) => events.push(event),
        emissionMode: "await",
      },
    );

    const result = streamText({
      model,
      prompt: "Say OK",
      providerOptions: {
        vetch: {
          attribution: { sessionId: "stream-session" },
        },
      },
    });

    await expect(result.text).resolves.toBe("OK");
    expect(events).toHaveLength(1);
    expect(events[0].ai_sdk_operation).toBe("stream");
    expect(events[0].is_stream).toBe(true);
    expect(events[0].complete).toBe(true);
    expect(events[0].visible_output_chars).toBe(2);
    expect(events[0].usage.text.output_tokens).toBe(2);
    expect(events[0].usage.reasoning.output_tokens).toBe(2);
    expect(events[0].finish_reason).toBe("stop");
  });

  it("scopes rolling advisories by attribution.sessionId on shared middleware", async () => {
    const events = [];
    const thresholds = {
      protocolVoidWindow: 2,
      protocolVoidMinOutputTokens: 1,
    };
    const middleware = createVetchMiddleware({
      emitter: (event) => events.push(event),
      emissionMode: "await",
      thresholds,
    });

    await runGenerate(middleware, { sessionId: "session-a" });
    await runGenerate(middleware, { sessionId: "session-a" });
    await runGenerate(middleware, { sessionId: "session-b" });

    expect(events[0].advisories.map((advisory) => advisory.code)).not.toContain("PROTO-001");
    expect(events[1].advisories.map((advisory) => advisory.code)).toContain("PROTO-001");
    expect(events[2].advisories.map((advisory) => advisory.code)).not.toContain("PROTO-001");
  });

  it("uses sessionFactory and LRU eviction for explicit session IDs", async () => {
    const created = [];
    const middleware = createVetchMiddleware({
      emitter: () => undefined,
      emissionMode: "await",
      maxSessionCount: 1,
      sessionFactory: (sessionId) => {
        created.push(sessionId);
        return createVetchSession();
      },
    });

    await runGenerate(middleware, { sessionId: "session-a" });
    await runGenerate(middleware, { sessionId: "session-b" });
    await runGenerate(middleware, { sessionId: "session-a" });

    expect(created).toEqual(["session-a", "session-b", "session-a"]);
  });

  it("honors VETCH_DISABLED without emitting events", async () => {
    const previous = process.env.VETCH_DISABLED;
    process.env.VETCH_DISABLED = "true";
    try {
      const events = [];
      const model = withVetch(
        new MockLanguageModelV3({
          provider: "openai",
          modelId: "gpt-4.1-mini",
          doGenerate: generateResult(),
        }),
        {
          emitter: (event) => events.push(event),
          emissionMode: "await",
        },
      );
      await generateText({ model, prompt: "Say OK" });
      expect(events).toHaveLength(0);
    } finally {
      if (previous === undefined) {
        delete process.env.VETCH_DISABLED;
      } else {
        process.env.VETCH_DISABLED = previous;
      }
    }
  });

  it("honors the explicit disabled option without emitting events", async () => {
    const events = [];
    const model = withVetch(
      new MockLanguageModelV3({
        provider: "openai",
        modelId: "gpt-4.1-mini",
        doGenerate: generateResult(),
      }),
      {
        disabled: true,
        emitter: (event) => events.push(event),
        emissionMode: "await",
      },
    );

    const result = await generateText({ model, prompt: "Say OK" });

    expect(result.text).toBe("OK");
    expect(events).toHaveLength(0);
  });

  it("retires the 0.8.1 Naples release Easter egg on later versions", async () => {
    // The NAPLES-081 egg is gated to VETCH_VERSION === "0.8.1"; on 0.9.0+ it is
    // dormant. The event still emits, just without the release advisory.
    const events = [];
    const model = withVetch(
      new MockLanguageModelV3({
        provider: "openai",
        modelId: "gpt-4.1-mini",
        doGenerate: generateResult(),
      }),
      {
        easterEggs: true,
        emitter: (event) => events.push(event),
        emissionMode: "await",
      },
    );

    await generateText({ model, prompt: "Say OK" });

    expect(events).toHaveLength(1);
    const codes = (events[0].advisories ?? []).map((a) => a.code);
    expect(codes).not.toContain("NAPLES-081");
  });

  it("labels Ollama OpenAI-compat endpoints as provider ollama", async () => {
    const event = await createVetchEvent({
      operation: "generate",
      model: {
        provider: "openai",
        modelId: "llama3.1:8b",
        baseURL: "http://localhost:11434/v1",
      },
      params: {},
      result: generateResult({ usage: usageWithReasoning }),
      startTimeMs: Date.now(),
      options: {},
    });

    expect(event.provider).toBe("ollama");
  });

  it("honors providerOptions.vetch.providerOverride", async () => {
    const event = await createVetchEvent({
      operation: "generate",
      model: { provider: "openai", modelId: "gpt-4.1-mini" },
      params: {
        providerOptions: {
          vetch: {
            providerOverride: "custom-local",
          },
        },
      },
      result: generateResult({ usage: usageWithReasoning }),
      startTimeMs: Date.now(),
      options: {},
    });

    expect(event.provider).toBe("custom-local");
  });

  it("records budget metadata and budget_exceeded", async () => {
    const event = await createVetchEvent({
      operation: "generate",
      model: { provider: "openai", modelId: "gpt-4.1-mini" },
      params: {
        providerOptions: {
          vetch: {
            budget: { cost_usd: 0.000001 },
            retry_count: 2,
          },
        },
      },
      result: generateResult({ usage: usageWithReasoning }),
      startTimeMs: Date.now(),
      options: {},
    });

    expect(event.retry_count).toBe(2);
    expect(event.budget_cost_usd).toBe(0.000001);
    expect(event.budget_exceeded).toBe(true);
  });
});

// Minimal VetchEvent shape for advisory unit tests
function makeEvent(overrides = {}) {
  return {
    error: false,
    cache_read_tokens: 0,
    usage: { text: { input_tokens: 100, output_tokens: 50, total_tokens: 150 } },
    finish_reason: "stop",
    visible_output_chars: 20,
    tool_call_count: 0,
    ...overrides,
  };
}

describe("session advisories", () => {
  it("STALL-001 fires when last 5 non-error calls have ≤5 output tokens", () => {
    const events = Array.from({ length: 5 }, () =>
      makeEvent({ usage: { text: { input_tokens: 100, output_tokens: 2, total_tokens: 102 } } }),
    );
    const last = events.at(-1);
    const codes = detectAdvisories(last, events).map((a) => a.code);
    expect(codes).toContain("STALL-001");
  });

  it("STALL-001 does not fire when output is normal", () => {
    const events = Array.from({ length: 5 }, () => makeEvent());
    const last = events.at(-1);
    const codes = detectAdvisories(last, events).map((a) => a.code);
    expect(codes).not.toContain("STALL-001");
  });

  it("STALL-001 does not fire with fewer than 5 non-error events", () => {
    const events = Array.from({ length: 3 }, () =>
      makeEvent({ usage: { text: { input_tokens: 100, output_tokens: 1, total_tokens: 101 } } }),
    );
    const last = events.at(-1);
    const codes = detectAdvisories(last, events).map((a) => a.code);
    expect(codes).not.toContain("STALL-001");
  });

  it("CACHE-001 fires when >50% of recent events share the same input token count", () => {
    const events = [
      ...Array.from({ length: 5 }, () => makeEvent({ usage: { text: { input_tokens: 500, output_tokens: 50, total_tokens: 550 } } })),
      makeEvent({ usage: { text: { input_tokens: 20, output_tokens: 10, total_tokens: 30 } } }),
    ];
    const last = events.at(-1);
    const codes = detectAdvisories(last, events).map((a) => a.code);
    expect(codes).toContain("CACHE-001");
  });

  it("CACHE-001 does not fire below minimum window size", () => {
    const events = Array.from({ length: 5 }, () =>
      makeEvent({ usage: { text: { input_tokens: 500, output_tokens: 50, total_tokens: 550 } } }),
    );
    const last = events.at(-1);
    const codes = detectAdvisories(last, events).map((a) => a.code);
    expect(codes).not.toContain("CACHE-001");
  });

  it("CACHE-002 fires alongside CACHE-001 when no cache reads observed", () => {
    const events = Array.from({ length: 8 }, () =>
      makeEvent({ usage: { text: { input_tokens: 500, output_tokens: 50, total_tokens: 550 } }, cache_read_tokens: 0 }),
    );
    const last = events.at(-1);
    const codes = detectAdvisories(last, events).map((a) => a.code);
    expect(codes).toContain("CACHE-001");
    expect(codes).toContain("CACHE-002");
  });

  it("CACHE-002 does not fire when cache reads are present", () => {
    const events = Array.from({ length: 8 }, () =>
      makeEvent({ usage: { text: { input_tokens: 500, output_tokens: 50, total_tokens: 550 } }, cache_read_tokens: 400 }),
    );
    const last = events.at(-1);
    const codes = detectAdvisories(last, events).map((a) => a.code);
    expect(codes).toContain("CACHE-001");
    expect(codes).not.toContain("CACHE-002");
  });

  it("ERROR-001 fires on repeated errors", () => {
    const events = Array.from({ length: 5 }, () => makeEvent({ error: true, error_type: "ProviderError" }));
    const codes = detectAdvisories(events.at(-1), events).map((a) => a.code);
    expect(codes).toContain("ERROR-001");
  });

  it("STREAM-001 fires when recent streams are incomplete", () => {
    const events = Array.from({ length: 5 }, () => makeEvent({ is_stream: true, complete: false }));
    const codes = detectAdvisories(events.at(-1), events).map((a) => a.code);
    expect(codes).toContain("STREAM-001");
  });

  it("BUDGET-001 fires when budget_exceeded is true", () => {
    const event = makeEvent({ budget_exceeded: true, budget_cost_usd: 0.01 });
    const codes = detectPerCallAdvisories(event).map((a) => a.code);
    expect(codes).toContain("BUDGET-001");
  });

  it("STALL-001 does not fire across calls without sessionId", async () => {
    const middleware = createVetchMiddleware({
      emissionMode: "await",
    });
    for (let i = 0; i < 5; i += 1) {
      await runGenerate(middleware, { outputTokens: 1 });
    }
    const lastEvents = [];
    const model = withVetch(
      new MockLanguageModelV3({
        provider: "openai",
        modelId: "gpt-4.1-mini",
        doGenerate: generateResult({
          usage: {
            inputTokens: { total: 20, noCache: 20, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 1, text: 1, reasoning: 0 },
          },
        }),
      }),
      {
        emitter: (event) => lastEvents.push(event),
        emissionMode: "await",
      },
    );
    await generateText({ model, prompt: "one more" });
    const codes = lastEvents.at(-1)?.advisories.map((a) => a.code) ?? [];
    expect(codes).not.toContain("STALL-001");
  });

  it("REASONING-001 fires when reasoning models report no reasoning tokens", () => {
    const events = Array.from({ length: 3 }, () =>
      makeEvent({
        model: "o3-mini",
        usage: {
          text: { input_tokens: 100, output_tokens: 50, total_tokens: 150 },
          reasoning: null,
        },
      }),
    );
    const codes = detectAdvisories(events.at(-1), events).map((a) => a.code);
    expect(codes).toContain("REASONING-001");
  });
});
