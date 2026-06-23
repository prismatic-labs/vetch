import { afterEach, describe, expect, it } from "vitest";

import { createVetchEvent } from "../dist/event.js";
import { inferProviderFromBaseUrl } from "../dist/provider-label.js";

// Mirrors Python tests/test_self_hosted_routing.py

function generateResult() {
  return {
    content: [{ type: "text", text: "OK" }],
    finishReason: { unified: "stop", raw: "stop" },
    usage: {
      inputTokens: { total: 1000, noCache: 1000, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 500, text: 500, reasoning: 0 },
    },
    warnings: [],
  };
}

async function eventForBaseUrl(baseURL) {
  return createVetchEvent({
    operation: "generate",
    model: { provider: "openai", modelId: "gpt-4o", baseURL },
    params: {},
    result: generateResult(),
    startTimeMs: Date.now(),
    options: {},
  });
}

describe("inferProviderFromBaseUrl (parity with Python _infer_openai_provider)", () => {
  afterEach(() => {
    delete process.env.VETCH_SELF_HOSTED;
    delete process.env.OLLAMA_HOST;
  });

  const cases = [
    [undefined, null],
    ["https://api.openai.com/v1", "openai"],
    ["https://my-resource.openai.azure.com/", "openai"],
    ["https://eastus.api.cognitive.microsoft.com/", "openai"],
    ["http://localhost:11434/v1", "ollama"],
    ["http://localhost:8000/v1", "self-hosted"],
    ["http://127.0.0.1:8000", "self-hosted"],
    ["http://10.0.0.5:8000/v1", "self-hosted"],
    ["http://192.168.1.9:1234/v1", "self-hosted"],
    ["http://172.16.0.3:8000", "self-hosted"],
    ["https://openrouter.ai/api/v1", "openai-compatible"],
    ["https://api.together.xyz/v1", "openai-compatible"],
    ["https://my-vllm.example.com/v1", "openai-compatible"],
  ];
  for (const [url, expected] of cases) {
    it(`classifies ${url} as ${expected}`, () => {
      expect(inferProviderFromBaseUrl(url)).toBe(expected);
    });
  }

  it("VETCH_SELF_HOSTED forces self-hosted even on a public host", () => {
    process.env.VETCH_SELF_HOSTED = "true";
    expect(inferProviderFromBaseUrl("https://some-public-host.example/v1")).toBe("self-hosted");
  });

  it("OLLAMA_HOST env forces ollama", () => {
    process.env.OLLAMA_HOST = "http://my-ollama:11434";
    expect(inferProviderFromBaseUrl(undefined)).toBe("ollama");
  });
});

describe("cost routing by provider", () => {
  it("official OpenAI is list-priced", async () => {
    const event = await eventForBaseUrl("https://api.openai.com/v1");
    expect(event.provider).toBe("openai");
    expect(event.estimated_cost_usd).toBeGreaterThan(0);
  });

  it("self-hosted (local) bills nothing", async () => {
    const event = await eventForBaseUrl("http://localhost:8000/v1");
    expect(event.provider).toBe("self-hosted");
    expect(event.estimated_cost_usd).toBe(0);
    expect(event.billing_tier).toBe("self-hosted");
    expect(event.estimated_energy_wh).toBeGreaterThan(0);
  });

  it("openai-compatible host leaves cost unknown", async () => {
    const event = await eventForBaseUrl("https://openrouter.ai/api/v1");
    expect(event.provider).toBe("openai-compatible");
    expect(event.estimated_cost_usd).toBeNull();
    expect(event.billing_tier).toBe("unknown");
    expect(event.estimated_energy_wh).toBeGreaterThan(0);
  });

  it("ollama (local :11434) bills nothing", async () => {
    const event = await eventForBaseUrl("http://localhost:11434/v1");
    expect(event.provider).toBe("ollama");
    expect(event.estimated_cost_usd).toBe(0);
    expect(event.billing_tier).toBe("self-hosted");
  });

  it("VETCH_SELF_HOSTED forces cost 0 even on a public host", async () => {
    process.env.VETCH_SELF_HOSTED = "true";
    try {
      const event = await eventForBaseUrl("https://some-public-host.example/v1");
      expect(event.provider).toBe("self-hosted");
      expect(event.estimated_cost_usd).toBe(0);
      expect(event.billing_tier).toBe("self-hosted");
    } finally {
      delete process.env.VETCH_SELF_HOSTED;
    }
  });

  it("a non-OpenAI endpoint never inherits OpenAI's price", async () => {
    const openaiCost = (await eventForBaseUrl("https://api.openai.com/v1")).estimated_cost_usd;
    for (const url of ["http://localhost:8000/v1", "https://openrouter.ai/api/v1"]) {
      const cost = (await eventForBaseUrl(url)).estimated_cost_usd;
      expect(cost).not.toBe(openaiCost);
    }
  });
});
