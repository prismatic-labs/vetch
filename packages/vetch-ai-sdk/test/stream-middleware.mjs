import assert from "node:assert/strict";

import { createVetchMiddleware } from "../dist/middleware.js";

const finishUsage = {
  inputTokens: { total: 42, noCache: 30, cacheRead: 12, cacheWrite: 3 },
  outputTokens: { total: 9, text: 2, reasoning: 7 },
};

const streamChunks = [
  { type: "text-delta", id: "t1", delta: "Hello" },
  {
    type: "tool-call",
    toolCallId: "call_1",
    toolName: "search",
    input: "{}",
  },
  {
    type: "tool-result",
    toolCallId: "call_1",
    toolName: "search",
    result: "{}",
  },
  {
    type: "finish",
    finishReason: "tool-calls",
    usage: finishUsage,
  },
];

function createMockStream(chunks) {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk);
      }
      controller.close();
    },
  });
}

function createMiddlewareOptions(overrides = {}) {
  const events = [];
  return {
    events,
    options: {
      emissionMode: "await",
      modelHint: { provider: "openai", modelId: "gpt-4.1-mini" },
      emitter: (event) => {
        events.push(event);
      },
      ...overrides,
    },
  };
}

async function consumeStream(stream) {
  const received = [];
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    received.push(value);
  }
  return received;
}

async function runWrapStream(chunks, options) {
  const middleware = createVetchMiddleware(options);
  const mockModel = { provider: "openai", modelId: "gpt-4.1-mini" };
  const { stream } = await middleware.wrapStream({
    doStream: async () => ({ stream: createMockStream(chunks) }),
    doGenerate: async () => {
      throw new Error("doGenerate should not be called");
    },
    params: { maxOutputTokens: 256 },
    model: mockModel,
  });
  return consumeStream(stream);
}

{
  const { events, options } = createMiddlewareOptions();
  const received = await runWrapStream(streamChunks, options);

  assert.deepEqual(received, streamChunks);
  assert.equal(events.length, 1);

  const event = events[0];
  assert.equal(event.ai_sdk_operation, "stream");
  assert.equal(event.is_stream, true);
  assert.equal(event.complete, true);
  assert.equal(event.finish_reason, "tool-calls");
  assert.equal(event.visible_output_chars, 5);
  assert.equal(event.tool_call_count, 1);
  assert.equal(event.tool_result_count, 1);
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
  assert.equal(event.cache_read_tokens, 12);
  assert.equal(event.cache_creation_tokens, 3);
}

{
  const { events, options } = createMiddlewareOptions({
    emitter: () => {
      throw new Error("emitter exploded");
    },
    onEmitterError: () => undefined,
    debug: false,
  });
  const received = await runWrapStream(streamChunks, options);

  assert.deepEqual(received, streamChunks);
  assert.equal(events.length, 0);
}

{
  const { events, options } = createMiddlewareOptions();
  const middleware = createVetchMiddleware(options);
  const mockModel = { provider: "openai", modelId: "gpt-4.1-mini" };
  const { stream } = await middleware.wrapStream({
    doStream: async () => ({ stream: createMockStream(streamChunks) }),
    doGenerate: async () => {
      throw new Error("doGenerate should not be called");
    },
    params: {},
    model: mockModel,
  });

  const reader = stream.getReader();
  await reader.read();
  await reader.cancel("user aborted");

  assert.equal(events.length, 1);
  assert.equal(events[0].complete, false);
  assert.ok(events[0].vetch_warnings.includes("stream_cancelled_partial"));
  assert.equal(events[0].visible_output_chars, 5);
}

{
  const { events, options } = createMiddlewareOptions();
  const middleware = createVetchMiddleware(options);
  const mockModel = { provider: "openai", modelId: "gpt-4.1-mini" };
  const { stream } = await middleware.wrapStream({
    doStream: async () => ({ stream: createMockStream(streamChunks) }),
    doGenerate: async () => {
      throw new Error("doGenerate should not be called");
    },
    params: {},
    model: mockModel,
  });

  const reader = stream.getReader();
  await reader.read();
  await reader.cancel("user aborted");
  await reader.cancel("second cancel should not double emit");

  assert.equal(events.length, 1);
}
