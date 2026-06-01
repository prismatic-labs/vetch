import { readEnv } from "./env.js";

/** Align with Python `providers/openai.py::_infer_openai_provider`. */
export function inferProviderFromBaseUrl(baseUrl: string | undefined): string | null {
  if (readEnv("OLLAMA_HOST")?.trim()) {
    return "ollama";
  }
  if (!baseUrl) {
    return null;
  }
  const lower = baseUrl.toLowerCase();
  if (lower.includes("localhost:11434") || lower.includes("127.0.0.1:11434")) {
    return "ollama";
  }
  return null;
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
  if (inferred) {
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
