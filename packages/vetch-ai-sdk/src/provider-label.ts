import { isTruthyEnv, readEnv } from "./env.js";

const OPENAI_OFFICIAL_HOST_SUFFIXES = [
  "api.openai.com",
  "openai.azure.com",
  "api.cognitive.microsoft.com",
];

function isPrivateHost(host: string): boolean {
  const h = host.replace(/^\[|\]$/g, ""); // strip IPv6 brackets
  if (h === "localhost" || h === "127.0.0.1" || h === "::1" || h.endsWith(".local")) {
    return true;
  }
  // RFC-1918 IPv4 private ranges.
  if (/^10\./.test(h) || /^192\.168\./.test(h)) {
    return true;
  }
  const m = /^172\.(\d{1,3})\./.exec(h);
  if (m && Number(m[1]) >= 16 && Number(m[1]) <= 31) {
    return true;
  }
  return false;
}

/**
 * Classify an endpoint by base_url into a provider label. Mirrors Python
 * `providers/openai.py::_infer_openai_provider`.
 *
 * Returns "openai" | "ollama" | "self-hosted" | "openai-compatible", or null
 * when there is no base_url to classify (caller defers to the SDK provider).
 */
export function inferProviderFromBaseUrl(baseUrl: string | undefined): string | null {
  if (readEnv("OLLAMA_HOST")?.trim()) {
    return "ollama";
  }
  if (isTruthyEnv(readEnv("VETCH_SELF_HOSTED"))) {
    return "self-hosted";
  }
  if (!baseUrl) {
    return null;
  }
  let host: string;
  try {
    host = new URL(baseUrl).hostname.toLowerCase();
  } catch {
    return null;
  }
  if (!host) {
    return null;
  }
  if (OPENAI_OFFICIAL_HOST_SUFFIXES.some((s) => host === s || host.endsWith("." + s))) {
    return "openai";
  }
  if (baseUrl.includes(":11434") || host === "ollama") {
    return "ollama";
  }
  if (isPrivateHost(host)) {
    return "self-hosted";
  }
  return "openai-compatible";
}

export function extractModelBaseUrl(model: unknown): string | undefined {
  const obj = asRecord(model);
  if (!obj) {
    return undefined;
  }
  const direct = firstString(obj.baseURL, obj.baseUrl, obj.url);
  if (direct) {
    return direct;
  }
  const settings = asRecord(obj.settings);
  return firstString(settings?.baseURL, settings?.baseUrl);
}

export function resolveProviderLabel(args: {
  provider: string;
  model: unknown;
  providerOverride?: string;
}): string {
  if (args.providerOverride?.trim()) {
    return args.providerOverride.trim();
  }
  const inferred = inferProviderFromBaseUrl(extractModelBaseUrl(args.model));
  if (!inferred) {
    return args.provider;
  }
  // Only let a base_url RECLASSIFY into "openai"/"openai-compatible" when the SDK
  // provider is OpenAI-family (or unknown/gateway). Otherwise a Google/Anthropic
  // model that exposes its own baseURL would be mislabelled (and lose pricing).
  // "ollama"/"self-hosted" are applied regardless — a local/private endpoint is
  // not the vendor API no matter which SDK opened it, and undercharging there is
  // the safe direction.
  if (inferred === "ollama" || inferred === "self-hosted") {
    return inferred;
  }
  const base = args.provider.toLowerCase();
  const openAiFamily = base.includes("openai") || base.includes("azure") || base === "unknown";
  if (openAiFamily) {
    return inferred;
  }
  return args.provider;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value) ?
    (value as Record<string, unknown>) :
    undefined;
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return undefined;
}
