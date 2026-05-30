import type {
  VetchEvent,
  VetchAttribution,
  VetchLanguageModel,
  VetchOptions,
  VetchProtocolProgress,
  VetchRequestMetadata,
  VetchStreamObservation,
  VetchTags,
  VetchTextUsage,
  VetchUsage,
} from "./types.js";
import { enrichVetchEvent, resolveModel } from "./calculation.js";
import { VETCH_VERSION } from "./version.js";

interface EventArgs {
  operation: "generate" | "stream" | "error";
  model: unknown;
  params: unknown;
  result?: unknown;
  streamObservation?: VetchStreamObservation;
  startTimeMs: number;
  options: VetchOptions;
  error?: unknown;
}

export function createVetchEvent(args: EventArgs): VetchEvent {
  const modelInfo = getModelInfo(args.model);
  const requestMetadata = getRequestMetadata(args.params);
  const rawUsage = args.streamObservation?.usage ?? getObjectValue(args.result, "usage");
  const textUsage = normalizeTextUsage(rawUsage);
  const finishReason = normalizeFinishReason(
    args.streamObservation?.finishReason ?? getObjectValue(args.result, "finishReason"),
  );
  const visibleOutputChars =
    args.streamObservation?.visibleOutputChars ?? countVisibleOutputChars(args.result);
  const toolCallCount =
    args.streamObservation?.toolCallCount ?? countToolParts(args.result, "tool-call");
  const toolResultCount =
    args.streamObservation?.toolResultCount ?? countToolParts(args.result, "tool-result");
  const cacheReadTokens = getCacheReadTokens(rawUsage);
  const cacheCreationTokens = getCacheCreationTokens(rawUsage);
  const protocolProgress = mergeProtocol(args.options.protocol, requestMetadata.protocol);
  const rawFinishReason = normalizeRawFinishReason(
    args.streamObservation?.rawFinishReason ?? getObjectValue(args.result, "finishReason"),
  );
  const error = args.error ? normalizeError(args.error) : undefined;
  const tags = mergeTags(resolveTags(args.options.tags), requestMetadata.tags);
  const attribution = mergeAttribution(
    resolveAttribution(args.options.attribution),
    requestMetadata.attribution,
  );

  const event: VetchEvent = {
    schema_version: "2",
    vetch_version: VETCH_VERSION,
    event_id: randomId(),
    timestamp: new Date().toISOString(),

    model: modelInfo.model,
    provider: modelInfo.provider,
    model_known: resolveModel(modelInfo.model).known,
    multimodal: hasMultimodalInputs(args.params),

    usage: textUsage ?
      createUsage(rawUsage, textUsage, countImagesInParams(args.params)) :
      null,
    accumulated_chars: args.operation === "stream" ? visibleOutputChars : null,

    estimated_energy_wh: null,
    estimated_carbon_g: null,
    estimated_water_l: null,
    estimated_cost_usd: null,
    estimated_cost_input_usd: null,
    estimated_cost_output_usd: null,
    estimated_cost_cache_write_usd: null,
    estimated_cost_cache_read_usd: null,
    billing_tier: "unknown",

    signal_quality: "live",
    energy_tier: null,
    energy_uncertainty_pct: null,
    energy_p5_wh: null,
    energy_p95_wh: null,
    carbon_p5_g: null,
    carbon_p95_g: null,
    energy_source: "not_calculated",
    energy_override_source: null,
    energy_basis: null,
    grid_intensity_gco2e_kwh: null,
    grid_intensity_timestamp: null,
    grid_intensity_time_of_day: false,
    region: args.options.region ?? null,
    embodied_carbon_g: null,
    pue: null,
    pue_tier: null,
    pue_source: null,

    is_stream: args.operation === "stream",
    is_batch: false,
    is_embedding: false,
    complete: resolveComplete(args),
    latency_ms: Math.max(0, Date.now() - args.startTimeMs),
    visible_output_chars: visibleOutputChars,
    finish_reason: finishReason ?? null,
    requested_max_tokens: getRequestedMaxTokens(args.params),

    tags: Object.keys(tags).length > 0 ? tags : null,
    error: args.operation === "error",
    error_type: error?.name ?? null,

    tracking_disabled: false,
    tracking_degraded: true,
    vetch_warnings: buildVetchWarnings(args),
    usage_estimated: false,
    usage_estimation_method: null,

    budget_energy_wh: null,
    budget_carbon_g: null,
    budget_cost_usd: null,
    budget_exceeded: null,

    cache_read_tokens: cacheReadTokens,
    cache_creation_tokens: cacheCreationTokens,
    cache_hit: cacheReadTokens === null ? null : cacheReadTokens > 0,
    cache_energy_saving_wh: null,
    cache_cost_saving_usd: null,
    cache_carbon_saving_g: null,

    session_id: attribution.sessionId ?? null,
    trace_id: attribution.traceId ?? null,
    span_id: attribution.spanId ?? null,
    parent_span_id: attribution.parentSpanId ?? null,
    request_fingerprint: attribution.requestFingerprint ?? null,

    ai_sdk_operation: args.operation,
    ...(rawFinishReason !== undefined ? { raw_finish_reason: rawFinishReason } : {}),
    tool_call_count: toolCallCount,
    tool_result_count: toolResultCount,
    ...(protocolProgress ? { protocol_progress: protocolProgress } : {}),
    advisories: [],
  };
  const enrichOptions: {
    provider: string;
    region?: string;
    priceMultiplier?: number;
    energyOverride?: import("./calculation.js").EnergyOverride | null;
  } = {
    provider: modelInfo.provider,
  };
  if (args.options.region !== undefined) {
    enrichOptions.region = args.options.region;
  }
  if (args.options.priceMultiplier !== undefined) {
    enrichOptions.priceMultiplier = args.options.priceMultiplier;
  }
  if (args.options.energyOverride !== undefined) {
    enrichOptions.energyOverride = args.options.energyOverride;
  }
  return enrichVetchEvent(event, enrichOptions);
}

export function createStreamObservation(): VetchStreamObservation {
  return {
    visibleOutputChars: 0,
    toolCallCount: 0,
    toolResultCount: 0,
  };
}

export function observeStreamPart(part: unknown, observation: VetchStreamObservation): void {
  const obj = asRecord(part);
  if (!obj) {
    return;
  }

  const type = typeof obj.type === "string" ? obj.type : "";
  const textDelta = firstString(obj.textDelta, obj.delta, obj.text, obj.content);
  if ((type.includes("text") || type === "delta") && textDelta) {
    observation.visibleOutputChars += textDelta.length;
  }

  if (type === "tool-call" || type === "tool-call-start") {
    observation.toolCallCount += 1;
  }

  if (type === "tool-result") {
    observation.toolResultCount += 1;
  }

  const finishReason = normalizeFinishReason(firstDefined(obj.finishReason, obj.finish_reason));
  if (finishReason) {
    observation.finishReason = finishReason;
    observation.rawFinishReason = firstDefined(obj.finishReason, obj.finish_reason);
  }

  if (type === "finish") {
    observation.finished = true;
  }

  const usage = firstDefined(obj.usage, obj.totalUsage);
  if (usage !== undefined) {
    observation.usage = usage;
  }
}

function resolveComplete(args: EventArgs): boolean {
  if (args.operation === "error") {
    return false;
  }
  if (args.operation === "stream") {
    const observation = args.streamObservation;
    if (observation?.cancelled === true) {
      return false;
    }
    return observation?.finished === true;
  }
  return true;
}

function buildVetchWarnings(args: EventArgs): string[] {
  const warnings: string[] = [];
  if (args.operation !== "stream") {
    return warnings;
  }
  const observation = args.streamObservation;
  if (observation?.cancelled === true) {
    warnings.push("stream_cancelled_partial");
  } else if (observation?.finished !== true) {
    warnings.push("stream_incomplete_no_finish");
  }
  return warnings;
}

function getModelInfo(model: unknown): { model: string; provider: string } {
  if (typeof model === "string") {
    const [provider, ...rest] = model.split("/");
    if (rest.length > 0 && provider) {
      return { provider, model: rest.join("/") };
    }
    return { provider: "unknown", model };
  }

  const obj = asRecord(model);
  if (!obj) {
    return { provider: "unknown", model: "unknown" };
  }

  const modelId = firstString(obj.modelId, obj.model, obj.id) ?? "unknown";
  const provider = firstString(obj.provider, obj.providerId, obj.providerName) ?? "unknown";
  if (isGatewayLikeProvider(provider)) {
    const parsed = parseGatewayModelId(modelId);
    if (parsed) {
      return parsed;
    }
  }
  return { provider, model: modelId };
}

const KNOWN_GATEWAY_PROVIDERS = new Set([
  "openai",
  "anthropic",
  "google",
  "vertexai",
  "azure",
  "bedrock",
  "cohere",
  "mistral",
  "groq",
  "xai",
]);

function isGatewayLikeProvider(provider: string): boolean {
  return provider === "unknown" || provider === "gateway" || provider === "ai-gateway";
}

function parseGatewayModelId(modelId: string): { provider: string; model: string } | null {
  const slash = modelId.indexOf("/");
  if (slash <= 0) {
    return null;
  }
  const parsedProvider = modelId.slice(0, slash);
  const parsedModel = modelId.slice(slash + 1);
  if (!parsedProvider || !parsedModel) {
    return null;
  }
  if (!KNOWN_GATEWAY_PROVIDERS.has(parsedProvider)) {
    return null;
  }
  return { provider: parsedProvider, model: parsedModel };
}

function getRequestMetadata(params: unknown): VetchRequestMetadata {
  const obj = asRecord(params);
  const providerMetadata = asRecord(obj?.providerMetadata);
  const providerOptions = asRecord(obj?.providerOptions);
  const vetch = asRecord(providerMetadata?.vetch) ?? asRecord(providerOptions?.vetch);
  if (!vetch) {
    return {};
  }

  const metadata: VetchRequestMetadata = {};
  const tags = asTags(vetch.tags);
  if (tags) {
    metadata.tags = tags;
  }

  const attribution = asAttribution(firstDefined(vetch.attribution, vetch.trace, vetch));
  if (attribution) {
    metadata.attribution = attribution;
  }

  const protocol = asProtocol(firstDefined(vetch.protocol, vetch.protocolProgress));
  if (protocol) {
    metadata.protocol = protocol;
  }

  return metadata;
}

function normalizeTextUsage(value: unknown): VetchTextUsage | null {
  const obj = asRecord(value);
  if (!obj) {
    return null;
  }

  const nestedInputTokens = asRecord(obj.inputTokens);
  const nestedOutputTokens = asRecord(obj.outputTokens);
  if (nestedInputTokens || nestedOutputTokens) {
    const inputTokens = firstNumber(nestedInputTokens?.total, obj.promptTokens, obj.input_tokens);
    const reasoningTokens = getReasoningTokens(value);
    const outputTotal = firstNumberOrUndefined(nestedOutputTokens?.total);
    const outputTokens = firstNumberOrUndefined(nestedOutputTokens?.text) ??
      (outputTotal !== undefined ? Math.max(0, outputTotal - reasoningTokens) : undefined) ??
      firstNumber(obj.completionTokens, obj.output_tokens);
    const totalTokens =
      firstNumberOrUndefined(obj.totalTokens, obj.total_tokens) ?? inputTokens + outputTokens;
    return {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      total_tokens: totalTokens,
    };
  }

  const hasUsageValue = [
    obj.inputTokens,
    obj.promptTokens,
    obj.input_tokens,
    obj.prompt_tokens,
    obj.outputTokens,
    obj.completionTokens,
    obj.output_tokens,
    obj.completion_tokens,
    obj.totalTokens,
    obj.total_tokens,
  ].some((candidate) => candidate !== undefined && candidate !== null);
  if (!hasUsageValue) {
    return null;
  }

  const inputTokens = firstNumber(
    obj?.inputTokens,
    obj?.promptTokens,
    obj?.input_tokens,
    obj?.prompt_tokens,
  );
  const outputTokens = firstNumber(
    obj?.outputTokens,
    obj?.completionTokens,
    obj?.output_tokens,
    obj?.completion_tokens,
  );
  const totalTokens =
    firstNumberOrUndefined(obj?.totalTokens, obj?.total_tokens) ?? inputTokens + outputTokens;
  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: totalTokens,
  };
}

function createUsage(
  rawUsage: unknown,
  textUsage: VetchTextUsage,
  imageCount: number,
): VetchUsage {
  const reasoningTokens = getReasoningTokens(rawUsage);
  return {
    text: textUsage,
    image:
      imageCount > 0 ?
        {
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          image_count: imageCount,
        } :
        null,
    audio: null,
    video: null,
    reasoning:
      reasoningTokens > 0 ?
        { input_tokens: 0, output_tokens: reasoningTokens, total_tokens: reasoningTokens } :
        null,
  };
}

function countImagesInParams(params: unknown, depth = 0): number {
  if (depth > 24) {
    return 0;
  }
  const obj = asRecord(params);
  if (!obj) {
    return 0;
  }
  let count = 0;
  const prompt = firstDefined(obj.prompt, obj.messages);
  count += countImagesInValue(prompt, depth + 1);
  const image = obj.image ?? obj.images;
  if (image !== undefined) {
    count += Array.isArray(image) ? image.length : 1;
  }
  return count;
}

function countImagesInValue(value: unknown, depth: number): number {
  if (depth > 24) {
    return 0;
  }
  if (Array.isArray(value)) {
    return value.reduce((sum, item) => sum + countImagesInValue(item, depth + 1), 0);
  }
  const obj = asRecord(value);
  if (!obj) {
    return 0;
  }
  const type = firstString(obj.type, obj.mediaType, obj.mimeType);
  if (type?.includes("image")) {
    return 1;
  }
  return countImagesInValue(firstDefined(obj.content, obj.parts), depth + 1);
}

function getReasoningTokens(rawUsage: unknown): number {
  const obj = asRecord(rawUsage);
  const outputTokens = asRecord(obj?.outputTokens);
  return firstNumber(outputTokens?.reasoning, obj?.reasoningTokens, obj?.reasoning_tokens);
}

function getCacheReadTokens(rawUsage: unknown): number | null {
  const obj = asRecord(rawUsage);
  const inputTokens = asRecord(obj?.inputTokens);
  return (
    firstNumberOrUndefined(
      inputTokens?.cacheRead,
      obj?.cachedInputTokens,
      obj?.cacheReadTokens,
      obj?.cache_read_tokens,
    ) ?? null
  );
}

function getCacheCreationTokens(rawUsage: unknown): number | null {
  const obj = asRecord(rawUsage);
  const inputTokens = asRecord(obj?.inputTokens);
  return firstNumberOrUndefined(
    inputTokens?.cacheWrite,
    obj?.cacheWriteTokens,
    obj?.cacheCreationTokens,
    obj?.cache_creation_tokens,
  ) ?? null;
}

function countVisibleOutputChars(result: unknown): number {
  const obj = asRecord(result);
  if (!obj) {
    return 0;
  }

  const content = getObjectValue(result, "content");
  if (Array.isArray(content)) {
    return countVisibleContentChars(content);
  }

  const text = firstString(obj.text, obj.responseText);
  if (text) {
    return text.length;
  }

  return countTextishChars(firstDefined(obj.response, obj.output));
}

function countVisibleContentChars(content: readonly unknown[]): number {
  return content.reduce<number>((total, part) => {
    const obj = asRecord(part);
    if (!obj) {
      return total + countTextishChars(part);
    }
    if (obj.type === "text") {
      return total + countTextishChars(obj.text);
    }
    return total;
  }, 0);
}

function countToolParts(result: unknown, type: "tool-call" | "tool-result"): number {
  const content = getObjectValue(result, "content");
  if (Array.isArray(content)) {
    return content.filter((part) => asRecord(part)?.type === type).length;
  }

  const legacyKey = type === "tool-call" ? "toolCalls" : "toolResults";
  return countArrayLike(getObjectValue(result, legacyKey));
}

function countTextishChars(value: unknown, depth = 0): number {
  if (depth > 20) {
    return 0;
  }
  if (typeof value === "string") {
    return value.length;
  }

  if (Array.isArray(value)) {
    return value.reduce((total, item) => total + countTextishChars(item, depth + 1), 0);
  }

  const obj = asRecord(value);
  if (!obj) {
    return 0;
  }

  return countTextishChars(firstDefined(obj.text, obj.content, obj.value), depth + 1);
}

function countArrayLike(value: unknown): number {
  return Array.isArray(value) ? value.length : 0;
}

function normalizeFinishReason(value: unknown): string | undefined {
  if (typeof value !== "string" || value.length === 0) {
    const obj = asRecord(value);
    return firstString(obj?.unified, obj?.type, obj?.reason);
  }
  return value;
}

function normalizeRawFinishReason(value: unknown): unknown {
  const obj = asRecord(value);
  if (obj) {
    return firstDefined(obj.raw, obj.unified, obj.type);
  }
  return value;
}

function normalizeError(error: unknown): { name: string } {
  if (error instanceof Error) {
    return { name: error.name };
  }
  return { name: "Error" };
}

function resolveTags(tags: VetchOptions["tags"]): VetchTags {
  if (!tags) {
    return {};
  }
  return typeof tags === "function" ? sanitizeTags(tags()) : sanitizeTags(tags);
}

function resolveAttribution(attribution: VetchOptions["attribution"]): VetchAttribution {
  if (!attribution) {
    return {};
  }
  return sanitizeAttribution(typeof attribution === "function" ? attribution() : attribution);
}

function mergeAttribution(base: VetchAttribution, extra?: VetchAttribution): VetchAttribution {
  return sanitizeAttribution({ ...base, ...(extra ?? {}) });
}

function asAttribution(value: unknown): VetchAttribution | undefined {
  const obj = asRecord(value);
  if (!obj) {
    return undefined;
  }

  const attribution: VetchAttribution = {};
  assignAttributionField(attribution, "sessionId", firstString(obj.sessionId, obj.session_id));
  assignAttributionField(attribution, "traceId", firstString(obj.traceId, obj.trace_id));
  assignAttributionField(attribution, "spanId", firstString(obj.spanId, obj.span_id));
  assignAttributionField(attribution, "parentSpanId", firstString(obj.parentSpanId, obj.parent_span_id));
  assignAttributionField(
    attribution,
    "requestFingerprint",
    firstString(obj.requestFingerprint, obj.request_fingerprint),
  );
  return Object.keys(attribution).length > 0 ? attribution : undefined;
}

function assignAttributionField(
  target: VetchAttribution,
  key: keyof VetchAttribution,
  value: string | undefined,
): void {
  if (value !== undefined && value !== "") {
    Object.assign(target, { [key]: value });
  }
}

function sanitizeAttribution(value: VetchAttribution): VetchAttribution {
  return Object.fromEntries(
    Object.entries(value).filter(([, fieldValue]) => fieldValue !== undefined && fieldValue !== ""),
  ) as VetchAttribution;
}

function mergeTags(base: VetchTags, extra?: VetchTags): VetchTags {
  return sanitizeTags({ ...base, ...(extra ?? {}) });
}

function mergeProtocol(
  base?: VetchProtocolProgress,
  extra?: VetchProtocolProgress,
): VetchProtocolProgress | undefined {
  const merged = { ...(base ?? {}), ...(extra ?? {}) };
  return Object.keys(merged).length > 0 ? merged : undefined;
}

function asProtocol(value: unknown): VetchProtocolProgress | undefined {
  const obj = asRecord(value);
  if (!obj) {
    return undefined;
  }

  const protocol: VetchProtocolProgress = {};
  setBoolean(protocol, "expectedToolUse", firstBoolean(obj.expectedToolUse, obj.expected_tool_use));
  setNumber(protocol, "validToolCallCount", obj.validToolCallCount, obj.valid_tool_call_count);
  setNumber(protocol, "invalidToolCallCount", obj.invalidToolCallCount, obj.invalid_tool_call_count);
  setBoolean(protocol, "terminalToolCalled", firstBoolean(obj.terminalToolCalled, obj.terminal_tool_called));
  setBoolean(protocol, "acceptedFinalResult", firstBoolean(obj.acceptedFinalResult, obj.accepted_final_result));
  setNumber(protocol, "postTerminalCallCount", obj.postTerminalCallCount, obj.post_terminal_call_count);
  setNumber(protocol, "toolFailureCount", obj.toolFailureCount, obj.tool_failure_count);
  setNumber(protocol, "repairAttemptCount", obj.repairAttemptCount, obj.repair_attempt_count);
  setNumber(protocol, "stepCount", obj.stepCount, obj.step_count);

  const expectedOutputClass = firstString(obj.expectedOutputClass, obj.expected_output_class);
  if (
    expectedOutputClass === "short" ||
    expectedOutputClass === "json" ||
    expectedOutputClass === "tool" ||
    expectedOutputClass === "longform"
  ) {
    protocol.expectedOutputClass = expectedOutputClass;
  }

  return Object.keys(protocol).length > 0 ? protocol : undefined;
}

function setBoolean(
  target: VetchProtocolProgress,
  key: keyof VetchProtocolProgress,
  value: boolean | undefined,
): void {
  if (value !== undefined) {
    Object.assign(target, { [key]: value });
  }
}

function setNumber(
  target: VetchProtocolProgress,
  key: keyof VetchProtocolProgress,
  ...values: unknown[]
): void {
  const value = firstNumberOrUndefined(...values);
  if (value !== undefined) {
    Object.assign(target, { [key]: value });
  }
}

function asTags(value: unknown): VetchTags | undefined {
  const obj = asRecord(value);
  if (!obj) {
    return undefined;
  }
  return sanitizeTags(obj);
}

function sanitizeTags(value: Record<string, unknown>): VetchTags {
  return Object.fromEntries(
    Object.entries(value)
      .filter(([, tagValue]) => tagValue !== undefined && tagValue !== null)
      .map(([key, tagValue]) => [key, String(tagValue)]),
  );
}

function getRequestedMaxTokens(params: unknown): number | null {
  const obj = asRecord(params);
  return firstNumberOrUndefined(obj?.maxOutputTokens, obj?.maxTokens, obj?.max_tokens) ?? null;
}

function hasMultimodalInputs(params: unknown): boolean {
  const obj = asRecord(params);
  const prompt = firstDefined(obj?.prompt, obj?.messages);
  return containsMediaMarker(prompt);
}

function containsMediaMarker(value: unknown, depth = 0): boolean {
  if (depth > 20) {
    return false;
  }
  if (Array.isArray(value)) {
    return value.some((item) => containsMediaMarker(item, depth + 1));
  }
  const obj = asRecord(value);
  if (!obj) {
    return false;
  }
  const type = firstString(obj.type, obj.mediaType, obj.mimeType);
  if (type?.includes("image") || type?.includes("audio") || type?.includes("video")) {
    return true;
  }
  return containsMediaMarker(firstDefined(obj.content, obj.parts), depth + 1);
}

function getObjectValue(value: unknown, key: string): unknown {
  return asRecord(value)?.[key];
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (typeof value === "object" && value !== null) {
    return value as Record<string, unknown>;
  }
  return undefined;
}

function firstDefined(...values: unknown[]): unknown {
  return values.find((value) => value !== undefined && value !== null);
}

function firstString(...values: unknown[]): string | undefined {
  const value = values.find((candidate) => typeof candidate === "string") as string | undefined;
  return value && value.length > 0 ? value : undefined;
}

function firstBoolean(...values: unknown[]): boolean | undefined {
  return values.find((candidate) => typeof candidate === "boolean") as boolean | undefined;
}

function firstNumber(...values: unknown[]): number {
  return firstNumberOrUndefined(...values) ?? 0;
}

function firstNumberOrUndefined(...values: unknown[]): number | undefined {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return undefined;
}

function randomId(): string {
  const crypto = globalThis.crypto as Crypto | undefined;
  if (crypto?.randomUUID) {
    return crypto.randomUUID();
  }
  return `vetch_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
}

export type { VetchLanguageModel };
