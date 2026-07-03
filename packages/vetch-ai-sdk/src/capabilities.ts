import { createHmac } from "node:crypto";

import modelCapabilitiesJson from "./registry/model_capabilities.json" with { type: "json" };
import type { VetchCapabilityRef, VetchUsage } from "./types.js";

type ModelCapabilityMap = Record<string, string>;

const MODEL_CAPABILITY_MAP = Object.fromEntries(
  Object.entries(modelCapabilitiesJson as ModelCapabilityMap).filter(
    ([key]) => !key.startsWith("_"),
  ),
) as ModelCapabilityMap;

export function redactCapabilityName(name: string): string {
  const key = process.env.VETCH_REDACTION_KEY;
  if (!key) {
    return name;
  }
  const digest = createHmac("sha256", key).update(name).digest("hex").slice(0, 32);
  return `redacted-${digest}`;
}

function normalizeFunctionTools(names: Iterable<string>): VetchCapabilityRef[] {
  const unique = [...new Set([...names].map((n) => redactCapabilityName(n)).filter(Boolean))].sort();
  return unique.map((name) => ({ name, kind: "function" as const }));
}

function estimateToolJsonTokens(tool: unknown): number {
  try {
    const payload = JSON.stringify(tool);
    return Math.max(1, Math.floor(payload.length / 4));
  } catch {
    return 1;
  }
}

export function extractToolsOfferedWithSizes(params: unknown): {
  refs: VetchCapabilityRef[] | null;
  schemaTokens: Record<string, number> | null;
} {
  const tools = getObjectValue(params, "tools");
  if (!Array.isArray(tools) || tools.length === 0) {
    return { refs: null, schemaTokens: null };
  }
  const names: string[] = [];
  const schemaTokens: Record<string, number> = {};
  for (const tool of tools) {
    const obj = asRecord(tool);
    const fn = obj ? asRecord(obj.function) : null;
    const name =
      (typeof obj?.name === "string" ? obj.name : undefined) ??
      (typeof fn?.name === "string" ? fn.name : undefined);
    if (name) {
      const redacted = redactCapabilityName(name);
      names.push(redacted);
      schemaTokens[redacted] = estimateToolJsonTokens(tool);
    }
  }
  if (names.length === 0) {
    return { refs: null, schemaTokens: null };
  }
  return { refs: normalizeFunctionTools(names), schemaTokens };
}

export function extractToolsOffered(params: unknown): VetchCapabilityRef[] | null {
  return extractToolsOfferedWithSizes(params).refs;
}

export function extractToolsInvoked(
  result: unknown,
  streamObservation?: { toolNamesInvoked?: string[] },
): VetchCapabilityRef[] | null {
  if (streamObservation?.toolNamesInvoked?.length) {
    return normalizeFunctionTools(streamObservation.toolNamesInvoked);
  }
  if (result === undefined) {
    return null;
  }
  const content = getObjectValue(result, "content");
  const names = new Set<string>();
  if (Array.isArray(content)) {
    for (const part of content) {
      const obj = asRecord(part);
      if (obj?.type === "tool-call" && typeof obj.toolName === "string") {
        names.add(obj.toolName);
      }
    }
  }
  const legacyCalls = getObjectValue(result, "toolCalls");
  if (Array.isArray(legacyCalls)) {
    for (const call of legacyCalls) {
      const obj = asRecord(call);
      const fn = obj ? asRecord(obj.function) : null;
      const name =
        (typeof obj?.toolName === "string" ? obj.toolName : undefined) ??
        (typeof fn?.name === "string" ? fn.name : undefined);
      if (name) {
        names.add(name);
      }
    }
  }
  if (names.size === 0) {
    return null;
  }
  return normalizeFunctionTools(names);
}

export function resolveModelCapability(model: string): string | null {
  if (!model) {
    return null;
  }
  if (model in MODEL_CAPABILITY_MAP) {
    return MODEL_CAPABILITY_MAP[model] ?? null;
  }
  const lower = model.toLowerCase();
  let bestPrefix = "";
  let bestCap: string | null = null;
  for (const [prefix, capability] of Object.entries(MODEL_CAPABILITY_MAP)) {
    const prefixLower = prefix.toLowerCase();
    if (lower.startsWith(prefixLower) && prefixLower.length > bestPrefix.length) {
      bestPrefix = prefixLower;
      bestCap = capability;
    }
  }
  return bestCap;
}

export function deriveCapabilitiesInvoked(args: {
  usage: VetchUsage | null;
  model: string;
  isEmbedding?: boolean;
}): VetchCapabilityRef[] | null {
  const refs: VetchCapabilityRef[] = [];
  if (args.isEmbedding) {
    refs.push({ name: "embedding", kind: "model" });
  }
  const usage = args.usage;
  if (usage) {
    for (const [modality, name] of [
      ["image", "image"],
      ["audio", "audio"],
      ["video", "video"],
    ] as const) {
      const block = usage[modality];
      const tokens = (block?.input_tokens ?? 0) + (block?.output_tokens ?? 0);
      if (tokens > 0) {
        refs.push({ name, kind: "model" });
      }
    }
  }
  const cap = resolveModelCapability(args.model);
  if (cap) {
    refs.push({ name: cap, kind: "model" });
  }
  if (refs.length === 0) {
    return null;
  }
  const seen = new Set<string>();
  const unique: VetchCapabilityRef[] = [];
  for (const ref of refs) {
    const key = `${ref.kind}:${ref.name}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(ref);
    }
  }
  return unique.sort((a, b) => `${a.kind}:${a.name}`.localeCompare(`${b.kind}:${b.name}`));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

function getObjectValue(obj: unknown, key: string): unknown {
  const record = asRecord(obj);
  return record ? record[key] : undefined;
}
