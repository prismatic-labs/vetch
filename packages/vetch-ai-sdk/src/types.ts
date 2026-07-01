import type { LanguageModelV3 } from "@ai-sdk/provider";

export type VetchSeverity = "info" | "warning" | "critical";
export type VetchOperation = "generate" | "stream" | "error";
export type VetchSignalQuality = "live" | "delayed" | "blind" | "unknown";
export type VetchModelMatch = "exact" | "alias" | "prefix" | "family" | "fallback";
export type VetchCapabilityKind = "function" | "builtin" | "model" | "agent";

export interface VetchCapabilityRef {
  name: string;
  kind: VetchCapabilityKind;
}

export type VetchTags = Record<string, string>;

export type VetchLanguageModel = LanguageModelV3;

export interface VetchTextUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface VetchImageUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  image_count?: number;
  total_pixels?: number;
}

export interface VetchAudioUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_seconds?: number;
  output_seconds?: number;
}

export interface VetchVideoUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_seconds?: number;
  output_seconds?: number;
}

export interface VetchUsage {
  text: VetchTextUsage;
  image: VetchImageUsage | null;
  audio: VetchAudioUsage | null;
  video: VetchVideoUsage | null;
  reasoning: VetchTextUsage | null;
}

export type VetchEmissionMode = "background" | "await";

export interface VetchProtocolProgress {
  expectedToolUse?: boolean;
  validToolCallCount?: number;
  invalidToolCallCount?: number;
  terminalToolCalled?: boolean;
  acceptedFinalResult?: boolean;
  postTerminalCallCount?: number;
  toolFailureCount?: number;
  repairAttemptCount?: number;
  stepCount?: number;
  expectedOutputClass?: "short" | "json" | "tool" | "longform";
}

export interface VetchAttribution {
  sessionId?: string;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  requestFingerprint?: string;
}

export interface VetchAdvisory {
  code: string;
  severity: VetchSeverity;
  title: string;
  description: string;
  evidence?: Record<string, unknown>;
}

export interface VetchEvent {
  schema_version: "2";
  vetch_version: string;
  event_id: string;
  timestamp: string;

  model: string;
  provider: string;
  model_known: boolean;
  // How the model name resolved against the registry. "exact"/"alias" are
  // high-confidence; "prefix"/"family" are low-confidence proxies (energy_tier
  // floored to 3); "fallback" = no match. Mirrors Python InferenceEvent.
  model_match: VetchModelMatch;
  multimodal: boolean;

  usage: VetchUsage | null;
  accumulated_chars: number | null;

  estimated_energy_wh: number | null;
  estimated_carbon_g: number | null;
  estimated_water_l: number | null;
  estimated_cost_usd: number | null;
  estimated_cost_input_usd: number | null;
  estimated_cost_output_usd: number | null;
  estimated_cost_cache_write_usd: number | null;
  estimated_cost_cache_read_usd: number | null;
  billing_tier: string;

  signal_quality: VetchSignalQuality;
  energy_tier: number | null;
  energy_uncertainty_pct: number | null;
  energy_p5_wh: number | null;
  energy_p95_wh: number | null;
  carbon_p5_g: number | null;
  carbon_p95_g: number | null;
  energy_source: string;
  energy_override_source: string | null;
  energy_basis: string | null;
  grid_intensity_gco2e_kwh: number | null;
  grid_intensity_timestamp: string | null;
  grid_intensity_time_of_day: boolean;
  region: string | null;
  embodied_carbon_g: number | null;
  pue: number | null;
  pue_tier: number | null;
  pue_source: string | null;

  is_stream: boolean;
  is_batch: boolean;
  is_embedding: boolean;
  complete: boolean;
  latency_ms: number;
  visible_output_chars: number;
  finish_reason: string | null;
  requested_max_tokens: number | null;

  tags: VetchTags | null;
  error: boolean;
  error_type: string | null;
  retry_count: number | null;

  tracking_disabled: boolean;
  tracking_degraded: boolean;
  vetch_warnings: string[];
  usage_estimated: boolean;
  usage_estimation_method: string | null;

  budget_energy_wh: number | null;
  budget_carbon_g: number | null;
  budget_cost_usd: number | null;
  budget_exceeded: boolean | null;

  cache_read_tokens: number | null;
  cache_creation_tokens: number | null;
  cache_hit: boolean | null;
  cache_energy_saving_wh: number | null;
  cache_cost_saving_usd: number | null;
  cache_carbon_saving_g: number | null;

  session_id: string | null;
  trace_id: string | null;
  span_id: string | null;
  parent_span_id: string | null;
  request_fingerprint: string | null;

  tools_offered: VetchCapabilityRef[] | null;
  tools_invoked: VetchCapabilityRef[] | null;
  capabilities_invoked: VetchCapabilityRef[] | null;

  ai_sdk_operation: VetchOperation;
  raw_finish_reason?: unknown;
  tool_call_count: number;
  tool_result_count: number;
  protocol_progress?: VetchProtocolProgress;
  advisories: VetchAdvisory[];
}

export interface VetchEmitter {
  (event: VetchEvent): void | Promise<void>;
}

export interface VetchSession {
  record(event: VetchEvent): VetchAdvisory[];
  recentEvents(): readonly VetchEvent[];
}

export interface VetchSessionFactory {
  (sessionId: string): VetchSession;
}

export interface VetchThresholds {
  highOutputTokens?: number;
  highVisibleChars?: number;
  taskModeOutputTokens?: number;
  toolSpinCalls?: number;
  toolFailures?: number;
  repairAttempts?: number;
  protocolVoidWindow?: number;
  protocolVoidMinOutputTokens?: number;
  errorWindow?: number;
  errorFraction?: number;
  consecutiveErrors?: number;
  streamWindow?: number;
  streamIncompleteFraction?: number;
  reasoningWindow?: number;
  reasoningMissingFraction?: number;
}

export interface VetchBudgets {
  energyWh?: number;
  carbonG?: number;
  costUsd?: number;
}

export interface VetchOptions {
  /** Explicitly disable the middleware. Env kill switch also honors VETCH_DISABLED/VETCH_ENABLED. */
  disabled?: boolean;
  /** Opt in to release Easter eggs such as NAPLES-081. Disabled by default. */
  easterEggs?: boolean;
  /** Force provider label on events (e.g. `ollama` for OpenAI-compat local endpoints). */
  providerOverride?: string;
  emitter?: VetchEmitter;
  onAdvisory?: (advisories: VetchAdvisory[], event: VetchEvent) => void | Promise<void>;
  /** Warn-only callback when `budget_exceeded` is true on an event. */
  onBudgetExceeded?: (event: VetchEvent) => void | Promise<void>;
  onEmitterError?: (error: unknown, event: VetchEvent) => void | Promise<void>;
  /** When true (default), emitter/advisory failures do not break the model call. */
  failOpen?: boolean;
  debug?: boolean;
  emissionMode?: VetchEmissionMode;
  emitterTimeoutMs?: number;
  waitUntil?: (promise: Promise<void>) => void;
  session?: VetchSession;
  sessionFactory?: VetchSessionFactory;
  maxSessionCount?: number;
  tags?: VetchTags | (() => VetchTags);
  attribution?: VetchAttribution | (() => VetchAttribution);
  region?: string;
  priceMultiplier?: number;
  budgets?: VetchBudgets | (() => VetchBudgets);
  /** Tier-0 hardware coefficients; auto-loaded from ~/.vetch on Node when omitted. */
  energyOverride?: import("./calculation.js").EnergyOverride | null;
  protocol?: VetchProtocolProgress;
  thresholds?: VetchThresholds;
  /** Called when an LRU-evicted session_id drops advisory state. */
  onSessionEvicted?: (sessionId: string) => void | Promise<void>;
}

export interface VetchRequestMetadata {
  tags?: VetchTags;
  attribution?: VetchAttribution;
  protocol?: VetchProtocolProgress;
  budgets?: VetchBudgets;
  retryCount?: number;
  providerOverride?: string;
}

export interface VetchStreamObservation {
  /** Raw provider usage object from the stream `finish` part (LanguageModelV3Usage shape). */
  usage?: unknown;
  finishReason?: string;
  rawFinishReason?: unknown;
  /** True after a stream `finish` part is observed. */
  finished?: boolean;
  /** True when the observed stream was cancelled before a normal finish. */
  cancelled?: boolean;
  visibleOutputChars: number;
  toolCallCount: number;
  toolResultCount: number;
  /** Tool names observed from stream `tool-call` parts (raw order, may repeat). */
  toolNamesInvoked: string[];
}
