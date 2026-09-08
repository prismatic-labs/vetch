import aliasesJson from "./registry/aliases.json" with { type: "json" };
import energyJson from "./registry/energy.json" with { type: "json" };
import pricingJson from "./registry/pricing.json" with { type: "json" };
import wueJson from "./registry/wue.json" with { type: "json" };
import globalAveragesJson from "./sensing/global_averages.json" with { type: "json" };

import { canUseNodeCalibration } from "./node-capability.js";
import type { VetchEvent, VetchModelMatch, VetchSignalQuality, VetchUsage } from "./types.js";
import type { LocalEnergyOverride } from "./local-calibration.js";

const METHODOLOGY_VERSION = "1.5";
const CACHE_READ_ENERGY_FACTOR = 0.15;
const PROMPT_LENGTH_SHORT_THRESHOLD = 1000;
const PROMPT_LENGTH_MEDIUM_THRESHOLD = 5000;

const TIER_UNCERTAINTY_PCT: Record<number, number> = {
  0: 20,
  1: 50,
  2: 100,
  3: 1000,
};

// Provider labels for endpoints that do not bill OpenAI/vendor list prices.
// Mirrors Python calculation.py _SELF_HOSTED_PROVIDERS / _UNKNOWN_PRICE_PROVIDERS.
const SELF_HOSTED_PROVIDERS = new Set(["self-hosted", "ollama"]);
const UNKNOWN_PRICE_PROVIDERS = new Set(["openai-compatible"]);

// Threshold for the tracking-degradation score (mirrors Python
// TRACKING_DEGRADED_THRESHOLD). Fires for unknown models, prefix/family proxies,
// estimated usage, and missing usage; healthy calls stay clean. Score maxes at
// 2.0 — do not set >= 2.0 (old value 2.5 made the flag unreachable).
const TRACKING_DEGRADED_THRESHOLD = 1.0;

// A coefficient may carry a default only when the published spread is narrow
// enough that a wrong pick stays inside the metric's stated uncertainty.
// Published PUE spans 1.09-1.27 (~15% worst case, inside the +/-50% claimed for
// carbon), so a default is an approximation. Published WUE spans 0.02-2.85, so
// there is no DEFAULT_WUE — see calculateWater. Keep in sync with the Python
// DEFAULT_PUE rationale in src/vetch/calculation.py.
const DEFAULT_PUE = 1.2;

// Mirrors PROVIDER_PUE in src/vetch/calculation.py. Values must match Python.
const PROVIDER_PUE: Record<string, number> = {
  google: 1.09,
  vertexai: 1.09,
  azure: 1.12,
  openai: 1.12,
  aws: 1.14,
  anthropic: 1.14,
  bedrock: 1.14,
  deepseek: 1.27,
};

// Provenance of each attribution, not of the figure. 1 = the operator publishes
// this PUE for infrastructure it also operates. 2 = an operator-published PUE
// applied through an undisclosed-hosting assumption. Missing key defaults to 2:
// an unclassified attribution is never treated as disclosed.
const PROVIDER_PUE_TIER: Record<string, number> = {
  google: 1,
  vertexai: 1,
  azure: 1,
  openai: 2,
  aws: 1,
  anthropic: 2,
  bedrock: 1,
  deepseek: 1,
};

const PROVIDER_PUE_SOURCES: Record<string, string> = {
  google: "Google 2026 Environmental Report (FY2025 fleet-wide PUE)",
  vertexai: "Google 2026 Environmental Report (FY2025 fleet-wide PUE)",
  azure: "Microsoft 2025 Environmental Sustainability Report (newest-gen PUE)",
  openai:
    "Microsoft 2025 Environmental Sustainability Report (newest-gen PUE); "
    + "applied on the assumption that OpenAI serves from Azure, which OpenAI "
    + "does not disclose",
  aws: "AWS 2025 Sustainability Report (global average PUE)",
  anthropic:
    "AWS 2025 Sustainability Report (global average PUE); applied on the "
    + "assumption that Anthropic serves from AWS, which Anthropic does not "
    + "disclose",
  bedrock: "AWS 2025 Sustainability Report",
  deepseek: "Jegham et al. (2025) DeepSeek datacenter PUE",
};

// Model families whose names identify a model, not an operator. Open-weight
// checkpoints can be served anywhere, so resolving a host from the name is an
// assumption even when the resolved operator publishes its PUE.
const OPEN_WEIGHT_MODEL_PREFIXES = ["llama-", "mixtral-", "gemma-"];

type EnergyEntry = {
  tier?: number;
  basis?: string;
  architecture?: string;
  embodied_factor?: number;
  wh_per_1k_input?: number;
  wh_per_1k_output?: number;
  prompt_length?: Record<string, {
    wh_per_1k_input: number;
    wh_per_1k_output: number;
  }>;
};

type PricingEntry = {
  usd_per_1k_input: number;
  usd_per_1k_output: number;
  tier_threshold?: number;
  tier_multiplier?: number;
  tier_multiplier_input?: number;
  tier_multiplier_output?: number;
  cache_read_discount?: number;
  cache_creation_premium?: number;
};

type WueEntry = {
  value: number;
  tier: number;
  boundary: string;
  accounting: string;
  basis: string;
};

type EnergyRegistry = Record<string, EnergyEntry>;
type PricingRegistry = Record<string, PricingEntry>;
type AliasRegistry = Record<string, string>;
type WueRegistry = Record<string, WueEntry>;

const ENERGY = energyJson as EnergyRegistry;
const PRICING = pricingJson as PricingRegistry;
const ALIASES = aliasesJson as AliasRegistry;
const WUE_REGISTRY_WARNINGS: string[] = [];
const WUE = Object.fromEntries(
  Object.entries(wueJson as Record<string, unknown>)
    .filter(([key, value]) => {
      if (key.startsWith("_") || value === null || typeof value !== "object") {
        return false;
      }
      const entry = value as Partial<WueEntry>;
      const valid = (
        typeof entry.value === "number"
        && typeof entry.tier === "number"
        && typeof entry.boundary === "string"
        && typeof entry.accounting === "string"
        && typeof entry.basis === "string"
        && Number.isFinite(entry.value)
        && entry.value >= 0
        && [1, 2, 3].includes(entry.tier)
        && ["onsite", "source"].includes(entry.boundary)
        && entry.accounting.length > 0
        && entry.basis.length > 0
      );
      if (!valid) {
        WUE_REGISTRY_WARNINGS.push(`WUE registry entry '${key}' is malformed and was ignored.`);
      }
      return valid;
    }),
) as WueRegistry;
const GLOBAL_AVERAGES = globalAveragesJson as {
  global?: number;
  regions?: Record<string, number>;
  country_defaults?: Record<string, number>;
};

interface EnergyResult {
  energyWh: number;
  tier: number;
  uncertaintyPct: number;
  source: string;
  basis: string | null;
  modelKnown: boolean;
}

interface CarbonResult {
  carbonG: number;
  pue: number;
  pueTier: number;
  pueSource: string;
}

interface WaterResult {
  waterL: number | null;
  wueTier: number | null;
  wueBasis: string | null;
  boundary: string | null;
  accounting: string | null;
  uncertaintyPct: number | null;
}

interface CostResult {
  totalCost: number;
  inputCost: number;
  outputCost: number;
  cacheWriteCost: number;
  cacheReadCost: number;
  billingTier: string;
}

interface GridIntensity {
  intensity: number;
  signalQuality: VetchSignalQuality;
  timestamp: string | null;
}

export interface EnergyOverride {
  wh_per_1k_input?: number | null;
  wh_per_1k_output?: number | null;
  tier?: number;
  source?: string;
  basis?: string;
  wh_per_image?: number | null;
  visual_tokens_per_image?: number;
  intercept_wh?: number | null;
}

interface EnrichOptions {
  provider: string;
  region?: string | null;
  priceMultiplier?: number;
  energyOverride?: EnergyOverride | null;
  wueOverride?: number | null;
  wueBoundary?: string | null;
  waterAccountingKind?: string | null;
  wueUncertaintyPct?: number | null;
}

async function resolveEnergyOverride(
  options: EnrichOptions,
  event: VetchEvent,
): Promise<EnergyOverride | null> {
  if (options.energyOverride !== undefined) {
    return options.energyOverride;
  }
  if (!canUseNodeCalibration()) {
    return null;
  }
  const { loadLocalCalibration } = await import("./local-calibration.js");
  return loadLocalCalibration(options.provider, event.model);
}

export async function enrichVetchEvent(event: VetchEvent, options: EnrichOptions): Promise<VetchEvent> {
  const resolved = resolveModelMatch(event.model);
  event.model_known = resolved.known;
  event.model_match = resolved.precision;
  event.energy_basis = event.energy_basis ?? null;

  let usage = event.usage;
  let text = usage?.text;
  if (!text) {
    const estimatedChars = Math.max(0, event.accumulated_chars ?? event.visible_output_chars ?? 0);
    if (estimatedChars > 0) {
      const outputTokens = Math.max(1, Math.trunc(estimatedChars / 4));
      const inputTokens = outputTokens * 2;
      event.usage = {
        text: {
          input_tokens: inputTokens,
          output_tokens: outputTokens,
          total_tokens: inputTokens + outputTokens,
        },
        image: null,
        audio: null,
        video: null,
        reasoning: null,
      };
      event.usage_estimated = true;
      event.usage_estimation_method = "char_ratio";
      const warning =
        `Token usage estimated from ${estimatedChars} chars (~4 chars/token, en content). ` +
        "Energy uncertainty floored at ±50%.";
      if (!event.vetch_warnings.includes(warning)) {
        event.vetch_warnings.push(warning);
      }
      usage = event.usage;
      text = usage.text;
    }
  }
  if (!text) {
    event.tracking_degraded = true;
    if (!event.vetch_warnings.includes("usage_missing_estimates_not_calculated")) {
      event.vetch_warnings.push("usage_missing_estimates_not_calculated");
    }
    return event;
  }

  const inputTokens = clampInt(text.input_tokens);
  const visibleOutputTokens = clampInt(text.output_tokens);
  const reasoningOutputTokens = clampInt(usage?.reasoning?.output_tokens);
  const energyOutputTokens = visibleOutputTokens + reasoningOutputTokens;
  const cacheReadTokens = clampInt(event.cache_read_tokens);
  const cacheCreationTokens = clampInt(event.cache_creation_tokens);
  const priceMultiplier = options.priceMultiplier ?? 1.0;
  const energyOverride = await resolveEnergyOverride(options, event);

  const nImages = clampInt(usage?.image?.image_count ?? 0);
  const imageInputTokens = clampInt(usage?.image?.input_tokens ?? 0);

  const energy = calculateEnergy(
    inputTokens, energyOutputTokens, event.model, cacheReadTokens,
    nImages, imageInputTokens, energyOverride,
  );
  event.estimated_energy_wh = energy.energyWh;
  event.energy_tier = energy.tier;
  event.energy_uncertainty_pct = energy.uncertaintyPct;
  event.energy_source = energy.source;
  event.energy_basis = energy.basis;
  if (energyOverride) {
    // A warning string is not machine-readable. Record the unpriced axes as a
    // field too, so a consumer can detect an incomplete figure without parsing
    // prose. A null coefficient never means zero cost.
    const markUnattributed = (axis: string, warning: string): void => {
      if (!event.energy_axes_unattributed.includes(axis)) {
        event.energy_axes_unattributed.push(axis);
      }
      if (!event.vetch_warnings.includes(warning)) {
        event.vetch_warnings.push(warning);
      }
    };
    if (energyOverride.wh_per_1k_input === null || energyOverride.wh_per_1k_input === undefined) {
      markUnattributed(
        "wh_per_1k_input",
        "energy_override.wh_per_1k_input is null; not attributing prefill energy",
      );
    }
    if (energyOverride.wh_per_1k_output === null || energyOverride.wh_per_1k_output === undefined) {
      markUnattributed(
        "wh_per_1k_output",
        "energy_override.wh_per_1k_output is null; not attributing decode energy",
      );
    }
  }
  const hasVisualInput = nImages > 0 || imageInputTokens > 0;
  const visualCoefficient = energyOverride?.wh_per_image;
  const hasVisualCoefficient = (
    visualCoefficient !== null
    && visualCoefficient !== undefined
    && Number.isFinite(visualCoefficient)
    && visualCoefficient > 0
  );
  if (hasVisualInput && !hasVisualCoefficient) {
    event.energy_completeness = "text_only";
    const warning =
      `Non-text media input present but no visual energy coefficient for model '${event.model}'; `
      + "estimated_energy_wh covers the text portion only "
      + "(energy_completeness='text_only'). Add a VLM calibration for a complete figure.";
    if (!event.vetch_warnings.includes(warning)) {
      event.vetch_warnings.push(warning);
    }
  }
  // Unpriced axes make the figure a lower bound, so it carries no upper bound.
  if (event.energy_completeness === "text_only" || event.energy_axes_unattributed.length > 0) {
    event.energy_estimate_kind = "lower_bound";
    event.energy_p95_wh = null;
    event.carbon_p95_g = null;
  }
  if (event.usage_estimated && event.energy_uncertainty_pct !== null && event.estimated_energy_wh > 0) {
    event.energy_uncertainty_pct = Math.max(event.energy_uncertainty_pct, 50);
  }

  if (!energy.modelKnown && event.model !== "unknown") {
    const warning = `Model '${event.model}' not in registry, using conservative fallback estimates. Energy/cost estimates may be inaccurate (±100% uncertainty)`;
    if (!event.vetch_warnings.includes(warning)) {
      event.vetch_warnings.push(warning);
    }
  }

  const baselineEnergy = cacheReadTokens > 0 ?
    calculateEnergy(inputTokens, energyOutputTokens, event.model, 0, nImages, imageInputTokens, energyOverride).energyWh :
    null;
  event.cache_energy_saving_wh =
    baselineEnergy === null ? null : Math.max(0, baselineEnergy - energy.energyWh);

  const grid = getCarbonIntensity(options.region ?? event.region);
  event.signal_quality = grid.signalQuality;
  event.grid_intensity_gco2e_kwh = grid.intensity;
  event.grid_intensity_timestamp = grid.timestamp;
  event.grid_intensity_time_of_day = false;

  const carbon = calculateCarbon(energy.energyWh, grid.intensity, event.model, options.provider);
  event.estimated_carbon_g = carbon.carbonG;
  event.pue = carbon.pue;
  event.pue_tier = carbon.pueTier;
  event.pue_source = carbon.pueSource;
  const water = calculateWater(
    energy.energyWh,
    event.model,
    options.provider,
    options.region ?? event.region,
    options.wueOverride,
    options.wueBoundary,
    options.waterAccountingKind,
    options.wueUncertaintyPct,
  );
  if (
    options.wueOverride !== undefined
    && options.wueOverride !== null
    && (!Number.isFinite(options.wueOverride) || options.wueOverride < 0)
  ) {
    const warning = "Invalid wueOverride; expected a finite non-negative liters-per-kWh value.";
    if (!event.vetch_warnings.includes(warning)) {
      event.vetch_warnings.push(warning);
    }
  }
  if (
    options.wueBoundary !== undefined
    && options.wueBoundary !== null
    && !["onsite", "source"].includes(options.wueBoundary.trim().toLowerCase())
  ) {
    const warning = "Invalid wueBoundary; expected 'onsite' or 'source'.";
    if (!event.vetch_warnings.includes(warning)) {
      event.vetch_warnings.push(warning);
    }
  }
  if (
    options.wueUncertaintyPct !== undefined
    && options.wueUncertaintyPct !== null
    && (
      !Number.isFinite(options.wueUncertaintyPct)
      || options.wueUncertaintyPct < 0
    )
  ) {
    const warning = "Invalid wueUncertaintyPct; expected a finite non-negative value.";
    if (!event.vetch_warnings.includes(warning)) {
      event.vetch_warnings.push(warning);
    }
  }
  if (
    options.wueOverride !== undefined
    && options.wueOverride !== null
    && Number.isFinite(options.wueOverride)
    && options.wueOverride >= 0
    && (
      !options.wueBoundary?.trim()
      || !options.waterAccountingKind?.trim()
    )
  ) {
    const warning = (
      "WUE override accounting definition is incomplete; provide wueBoundary "
      + "and waterAccountingKind to enable aggregation and comparison."
    );
    if (!event.vetch_warnings.includes(warning)) {
      event.vetch_warnings.push(warning);
    }
  }
  event.estimated_water_l = water.waterL;
  event.water_tier = water.wueTier;
  event.water_basis = water.wueBasis;
  event.water_boundary = water.boundary;
  event.water_accounting_kind = water.accounting;
  event.water_uncertainty_pct = water.uncertaintyPct;
  if (options.wueOverride === undefined || options.wueOverride === null) {
    for (const warning of WUE_REGISTRY_WARNINGS) {
      if (!event.vetch_warnings.includes(warning)) {
        event.vetch_warnings.push(warning);
      }
    }
  }
  event.embodied_carbon_g = calculateEmbodiedCarbon(inputTokens, energyOutputTokens, event.model);

  if (baselineEnergy !== null && event.estimated_carbon_g !== null) {
    const baselineCarbon = calculateCarbon(
      baselineEnergy,
      grid.intensity,
      event.model,
      options.provider,
    ).carbonG;
    event.cache_carbon_saving_g = Math.max(0, baselineCarbon - event.estimated_carbon_g);
  }

  if (SELF_HOSTED_PROVIDERS.has(options.provider)) {
    // Self-hosted: no per-token API charge (you pay hardware, captured as energy).
    event.estimated_cost_usd = 0;
    event.estimated_cost_input_usd = 0;
    event.estimated_cost_output_usd = 0;
    event.estimated_cost_cache_write_usd = 0;
    event.estimated_cost_cache_read_usd = 0;
    event.billing_tier = "self-hosted";
  } else if (UNKNOWN_PRICE_PROVIDERS.has(options.provider)) {
    // OpenAI-compatible third-party endpoint: don't apply OpenAI's price; leave unknown.
    event.estimated_cost_usd = null;
    event.estimated_cost_input_usd = null;
    event.estimated_cost_output_usd = null;
    event.estimated_cost_cache_write_usd = null;
    event.estimated_cost_cache_read_usd = null;
    event.billing_tier = "unknown";
  } else {
    const cost = calculateCost(
      inputTokens,
      energyOutputTokens,
      event.model,
      cacheReadTokens,
      cacheCreationTokens,
    );
    event.estimated_cost_usd = cost.totalCost * priceMultiplier;
    event.estimated_cost_input_usd = cost.inputCost * priceMultiplier;
    event.estimated_cost_output_usd = cost.outputCost * priceMultiplier;
    event.estimated_cost_cache_write_usd = cost.cacheWriteCost * priceMultiplier;
    event.estimated_cost_cache_read_usd = cost.cacheReadCost * priceMultiplier;
    event.billing_tier = priceMultiplier === 1.0 ? cost.billingTier : `${cost.billingTier}×${priceMultiplier}`;

    if (cacheReadTokens > 0 && event.estimated_cost_usd !== null) {
      const uncachedCost = calculateCost(
        inputTokens,
        energyOutputTokens,
        event.model,
        0,
        cacheCreationTokens,
      ).totalCost;
      event.cache_cost_saving_usd = Math.max(
        0,
        uncachedCost * priceMultiplier - event.estimated_cost_usd,
      );
    }
  }

  const bounds = confidenceBounds(event.estimated_energy_wh, event.energy_uncertainty_pct);
  event.energy_p5_wh = bounds.p5;
  event.energy_p95_wh = bounds.p95;
  const carbonBounds = confidenceBounds(event.estimated_carbon_g, event.energy_uncertainty_pct);
  event.carbon_p5_g = carbonBounds.p5;
  event.carbon_p95_g = carbonBounds.p95;
  // Bounds are populated after the estimate is calculated, so apply the
  // lower-bound contract here as the final write. An incomplete visual figure
  // must never acquire a symmetric upper bound later in this function.
  if (event.energy_completeness === "text_only" || event.energy_axes_unattributed.length > 0) {
    event.energy_estimate_kind = "lower_bound";
    event.energy_p95_wh = null;
    event.carbon_p95_g = null;
  }

  event.tracking_degraded = isTrackingDegraded({
    modelKnown: energy.modelKnown,
    energyTier: energy.tier,
    pueTier: carbon.pueTier,
    signalQuality: grid.signalQuality,
    usageEstimated: event.usage_estimated,
    modelMatch: resolved.precision,
  });

  event.vetch_warnings = event.vetch_warnings.filter(
    (warning) => warning !== "estimates_not_calculated",
  );
  void METHODOLOGY_VERSION;
  return event;
}

export interface ModelMatch {
  resolvedModel: string;
  known: boolean;
  precision: VetchModelMatch;
}

// Deterministic, conservative per-family fallback. Mirrors Python
// calculation.py::_FAMILY_FALLBACK. Bias to the larger (higher-energy) sibling
// when the sub-tier is ambiguous, so a proxy never silently undercounts.
const FAMILY_FALLBACK: Record<string, { large: string; small: string; default: string }> = {
  openai: { large: "gpt-4o", small: "gpt-4o-mini", default: "gpt-4o" },
  anthropic: { large: "claude-opus-4-6", small: "claude-haiku-4-5", default: "claude-sonnet-4-6" },
  google: { large: "gemini-3.1-pro", small: "gemini-3-flash", default: "gemini-3.1-pro" },
  aws: { large: "llama-3-70b", small: "llama-3.1-8b", default: "llama-3-70b" },
  deepseek: { large: "deepseek-r1", small: "deepseek-v3", default: "deepseek-r1" },
};

const FAMILY_LARGE_HINTS = new Set(["pro", "ultra", "opus", "max", "large"]);
const FAMILY_SMALL_HINTS = new Set([
  "flash", "lite", "nano", "mini", "haiku", "small", "tiny",
]);
const PARAM_SIZE_RE = /(?<!\d)(\d+(?:\.\d+)?)\s*b(?![a-z0-9])/g;
const FAMILY_LARGE_PARAM_B = 30;
const FAMILY_SMALL_PARAM_B = 15;

function inferFamilyProvider(modelLower: string): string | null {
  if (["gpt-", "o1", "o3", "o4", "text-davinci", "text-embedding"].some((p) => modelLower.startsWith(p))) {
    return "openai";
  }
  if (modelLower.startsWith("claude-")) {
    return "anthropic";
  }
  if (["gemini-", "gemma-", "palm-"].some((p) => modelLower.includes(p))) {
    return "google";
  }
  if (modelLower.startsWith("llama-") || modelLower.startsWith("mixtral-")) {
    return "aws";
  }
  if (modelLower.startsWith("deepseek-")) {
    return "deepseek";
  }
  return null;
}

function classifyFamilySubtier(modelLower: string): "large" | "small" | "default" {
  const tokens = new Set(modelLower.split(/[-_.]+/));
  const sizes = [...modelLower.matchAll(PARAM_SIZE_RE)].map((m) => parseFloat(m[1]!));
  const maxSize = sizes.length > 0 ? Math.max(...sizes) : null;

  if ([...tokens].some((t) => FAMILY_LARGE_HINTS.has(t)) ||
      (maxSize !== null && maxSize >= FAMILY_LARGE_PARAM_B)) {
    return "large";
  }
  if ([...tokens].some((t) => FAMILY_SMALL_HINTS.has(t)) ||
      (maxSize !== null && maxSize <= FAMILY_SMALL_PARAM_B)) {
    return "small";
  }
  return "default";
}

function familyFallback(modelLower: string): ModelMatch | null {
  const provider = inferFamilyProvider(modelLower);
  const family = provider ? FAMILY_FALLBACK[provider] : undefined;
  if (!family) {
    return null;
  }
  const target = family[classifyFamilySubtier(modelLower)];
  if (!(target in ENERGY)) {
    return null;
  }
  return { resolvedModel: target, known: true, precision: "family" };
}

/** Resolve a model to a registry entry with match precision. Case-insensitive.
 * Mirrors Python calculation.py::resolve_model_match. */
export function resolveModelMatch(model: string): ModelMatch {
  if (typeof model !== "string") {
    return { resolvedModel: String(model), known: false, precision: "fallback" };
  }
  const modelLower = model.toLowerCase();

  if (modelLower in ENERGY) {
    return { resolvedModel: modelLower, known: true, precision: "exact" };
  }
  const alias = ALIASES[modelLower];
  if (alias && alias in ENERGY) {
    return { resolvedModel: alias, known: true, precision: "alias" };
  }

  const parts = modelLower.split("-");
  for (let i = parts.length; i > 0; i -= 1) {
    const prefix = parts.slice(0, i).join("-");
    const degraded = prefix.replace(/(\d+)\.\d+$/, "$1");
    const candidates = degraded === prefix ? [prefix] : [prefix, degraded];
    for (const candidate of candidates) {
      if (candidate in ENERGY) {
        return { resolvedModel: candidate, known: true, precision: "prefix" };
      }
      const prefixAlias = ALIASES[candidate];
      if (prefixAlias && prefixAlias in ENERGY) {
        return { resolvedModel: prefixAlias, known: true, precision: "prefix" };
      }
    }
  }

  const family = familyFallback(modelLower);
  if (family) {
    return family;
  }

  return { resolvedModel: model, known: false, precision: "fallback" };
}

/** Thin back-compat wrapper over resolveModelMatch. */
export function resolveModel(model: string): { resolvedModel: string; known: boolean } {
  const match = resolveModelMatch(model);
  return { resolvedModel: match.resolvedModel, known: match.known };
}

function effectiveTextInputTokens(
  inputTokens: number,
  nImages: number,
  imageInputTokens: number,
  visualTokensPerImage: number | null,
): number {
  const inTok = Math.max(0, inputTokens);
  if (!visualTokensPerImage || visualTokensPerImage <= 0) {
    return inTok;
  }
  let visualTotal = Math.max(0, nImages) * visualTokensPerImage;
  if (imageInputTokens > 0) {
    visualTotal = Math.max(visualTotal, imageInputTokens);
  }
  return Math.max(0, inTok - visualTotal);
}

function calculateEnergy(
  inputTokens: number,
  outputTokens: number,
  model: string,
  cacheReadTokens: number,
  nImages: number = 0,
  imageInputTokens: number = 0,
  energyOverride: EnergyOverride | null = null,
): EnergyResult {
  const inTokens = Math.max(0, inputTokens);
  const outTokens = Math.max(0, outputTokens);
  const resolved = resolveModelMatch(model);

  let whIn: number;
  let whOut: number;
  let tier: number;
  let basis: string | null;
  let source: string;

  if (energyOverride !== null) {
    const whInRaw = energyOverride.wh_per_1k_input;
    const whOutRaw = energyOverride.wh_per_1k_output;
    const whIn = (whInRaw === null || whInRaw === undefined) ? 0 : whInRaw;
    const whOut = (whOutRaw === null || whOutRaw === undefined) ? 0 : whOutRaw;
    tier = energyOverride.tier ?? 1;
    source = energyOverride.source ?? "override";
    basis = energyOverride.basis ?? "User-provided override";
    const vtok =
      energyOverride.visual_tokens_per_image && energyOverride.visual_tokens_per_image > 0
        ? energyOverride.visual_tokens_per_image
        : null;
    const textInTokens = effectiveTextInputTokens(
      inTokens,
      nImages,
      imageInputTokens,
      vtok,
    );
    const cacheTokens = Math.min(Math.max(0, cacheReadTokens), textInTokens);
    const freshTokens = textInTokens - cacheTokens;
    let energyWh = (
      freshTokens * whIn +
      cacheTokens * whIn * CACHE_READ_ENERGY_FACTOR +
      outTokens * whOut
    ) / 1000;
    const whPerImage = energyOverride.wh_per_image;
    if (whPerImage != null && whPerImage > 0) {
      let imageUnits = Math.max(0, nImages);
      if (vtok !== null && imageInputTokens > 0) {
        imageUnits = Math.max(imageUnits, imageInputTokens / vtok);
      }
      if (imageUnits > 0) {
        energyWh += whPerImage * imageUnits;
      }
    }
    const interceptWh = energyOverride.intercept_wh;
    if (interceptWh != null && interceptWh > 0) {
      energyWh += interceptWh;
    }
    return {
      energyWh,
      tier,
      uncertaintyPct: getUncertaintyPct(tier),
      source,
      basis,
      modelKnown: resolved.known,
    };
  } else if (resolved.known) {
    const entry = ENERGY[resolved.resolvedModel]!;
    if (entry.prompt_length) {
      let category = inTokens < PROMPT_LENGTH_SHORT_THRESHOLD ? "short" :
        inTokens < PROMPT_LENGTH_MEDIUM_THRESHOLD ? "medium" :
          "long";
      if (!(category in entry.prompt_length)) {
        category = "medium";
      }
      const promptEntry = entry.prompt_length[category]!;
      whIn = promptEntry.wh_per_1k_input;
      whOut = promptEntry.wh_per_1k_output;
      basis = entry.basis ?? `Vetch registry measured data (${category} prompt)`;
    } else {
      whIn = entry.wh_per_1k_input ?? 1.4;
      whOut = entry.wh_per_1k_output ?? 4.2;
      basis = entry.basis ?? "Vetch registry";
    }
    tier = entry.tier ?? 3;
    source = "registry";

    // A prefix/family proxy is not the same model as the entry it matched, so it
    // must not inherit a measured (tier 0/1) confidence. Floor to Tier 3.
    if ((resolved.precision === "prefix" || resolved.precision === "family") && tier < 3) {
      tier = 3;
      basis = `${basis ?? ""} [Proxy match (${resolved.precision}): '${model}' is not an ` +
        "exact registry entry; confidence downgraded to Tier 3.]";
    }
  } else {
    whIn = 1.4;
    whOut = 4.2;
    tier = 3;
    basis = "Conservative fallback for unknown model";
    source = "fallback";
  }

  const cacheTokens = Math.min(Math.max(0, cacheReadTokens), inTokens);
  const freshTokens = inTokens - cacheTokens;
  const energyWh = (
    freshTokens * whIn +
    cacheTokens * whIn * CACHE_READ_ENERGY_FACTOR +
    outTokens * whOut
  ) / 1000;

  return {
    energyWh,
    tier,
    uncertaintyPct: getUncertaintyPct(tier),
    source,
    basis,
    modelKnown: resolved.known,
  };
}

function calculateCost(
  inputTokens: number,
  outputTokens: number,
  model: string,
  cacheReadTokens: number,
  cacheCreationTokens: number,
): CostResult {
  const resolved = resolveModel(model);
  if (!resolved.known || !(resolved.resolvedModel in PRICING)) {
    return {
      totalCost: 0,
      inputCost: 0,
      outputCost: 0,
      cacheWriteCost: 0,
      cacheReadCost: 0,
      billingTier: "none",
    };
  }

  const entry = PRICING[resolved.resolvedModel]!;
  const tierThreshold = entry.tier_threshold ?? null;
  const tierMultiplierInput = entry.tier_multiplier_input ?? entry.tier_multiplier ?? null;
  const tierMultiplierOutput = entry.tier_multiplier_output ?? entry.tier_multiplier ?? null;
  const cacheReadDiscount = entry.cache_read_discount ?? 0.1;
  const cacheCreationPremium = entry.cache_creation_premium ?? 1.0;
  const longContext = tierThreshold !== null && inputTokens > tierThreshold;
  const inputTierMultiplier = longContext && tierMultiplierInput !== null
    ? tierMultiplierInput
    : 1.0;

  const cacheTokens = Math.min(Math.max(0, cacheReadTokens), Math.max(0, inputTokens));
  const effectiveInput = Math.max(0, inputTokens) - cacheTokens;
  const cacheReadCost = cacheTokens *
    entry.usd_per_1k_input *
    inputTierMultiplier *
    cacheReadDiscount /
    1000;
  const cacheWriteCost = Math.max(0, cacheCreationTokens) *
    entry.usd_per_1k_input *
    inputTierMultiplier *
    cacheCreationPremium /
    1000;
  const inputCost = calculateTieredCost(
    effectiveInput,
    entry.usd_per_1k_input,
    tierThreshold,
    tierMultiplierInput,
    inputTokens,
  ) + cacheReadCost + cacheWriteCost;
  const outputCost = calculateTieredCost(
    Math.max(0, outputTokens),
    entry.usd_per_1k_output,
    tierThreshold,
    tierMultiplierOutput,
    inputTokens,
  );

  return {
    totalCost: inputCost + outputCost,
    inputCost,
    outputCost,
    cacheWriteCost,
    cacheReadCost,
    billingTier: "list",
  };
}

function calculateTieredCost(
  tokens: number,
  baseRatePer1k: number,
  tierThreshold: number | null,
  tierMultiplier: number | null,
  thresholdTokens: number = tokens,
): number {
  if (tierThreshold === null || tierMultiplier === null) {
    return tokens * baseRatePer1k / 1000;
  }
  if (thresholdTokens <= tierThreshold) {
    return tokens * baseRatePer1k / 1000;
  }
  return tokens * baseRatePer1k * tierMultiplier / 1000;
}

function calculateCarbon(
  energyWh: number,
  gridIntensityGco2eKwh: number,
  model: string,
  providerHint: string,
): CarbonResult {
  const [pue, pueTier, pueSource] = getProviderPue(model, providerHint);
  const intensity = sanitizeIntensity(gridIntensityGco2eKwh);
  return {
    carbonG: energyWh * pue * intensity / 1000,
    pue,
    pueTier,
    pueSource,
  };
}

function calculateWater(
  energyWh: number,
  model: string,
  providerHint: string,
  region: string | null | undefined,
  wueOverride: number | null | undefined,
  wueBoundary: string | null | undefined,
  waterAccountingKind: string | null | undefined,
  wueUncertaintyPct: number | null | undefined,
): WaterResult {
  if (wueOverride !== undefined && wueOverride !== null) {
    if (Number.isFinite(wueOverride) && wueOverride >= 0) {
      const boundary = wueBoundary?.trim().toLowerCase();
      const accounting = waterAccountingKind?.trim().toLowerCase();
      const uncertaintyPct = (
        wueUncertaintyPct !== undefined
        && wueUncertaintyPct !== null
        && Number.isFinite(wueUncertaintyPct)
        && wueUncertaintyPct >= 0
      ) ? wueUncertaintyPct : null;
      return {
        waterL: energyWh / 1000 * wueOverride,
        wueTier: 1,
        wueBasis: "explicit override",
        boundary: boundary === "onsite" || boundary === "source" ? boundary : null,
        accounting: accounting || null,
        uncertaintyPct,
      };
    }
    return {
      waterL: null,
      wueTier: null,
      wueBasis: null,
      boundary: null,
      accounting: null,
      uncertaintyPct: null,
    };
  }
  const inferred = inferWueProviderFromModel(model);
  const rawProvider = (providerHint || inferred || "").toLowerCase();
  const provider = normalizeWueProvider(rawProvider);
  let entry: WueEntry | undefined;
  if (provider && region) {
    const regionKey = `${provider}-${region.toLowerCase()}`;
    entry = WUE[regionKey] ?? WUE[provider];
  } else if (provider) {
    entry = WUE[provider];
  }
  if (entry === undefined) {
    return {
      waterL: null,
      wueTier: null,
      wueBasis: null,
      boundary: null,
      accounting: null,
      uncertaintyPct: null,
    };
  }
  return {
    waterL: energyWh / 1000 * entry.value,
    wueTier: entry.tier,
    wueBasis: entry.basis,
    boundary: entry.boundary,
    accounting: entry.accounting,
    // The registry states no uncertainty. Publishers report a WUE, not a band
    // around it; only an explicit wueOverride can carry one.
    uncertaintyPct: null,
  };
}

function normalizeWueProvider(provider: string): string {
  if (
    provider === "google"
    || provider === "vertexai"
    || provider.startsWith("google.")
  ) {
    return "google";
  }
  if (
    provider === "aws"
    || provider === "bedrock"
    || provider === "amazon-bedrock"
    || provider.startsWith("amazon-bedrock.")
  ) {
    return "aws";
  }
  if (provider === "azure" || provider.startsWith("azure.")) {
    return "azure";
  }
  return provider;
}

function calculateEmbodiedCarbon(
  inputTokens: number,
  outputTokens: number,
  model: string,
): number {
  const resolved = resolveModel(model);
  let embodiedFactor = 0.075;
  if (resolved.known) {
    embodiedFactor = ENERGY[resolved.resolvedModel]?.embodied_factor ?? embodiedFactor;
  } else {
    embodiedFactor = estimateEmbodiedFactorByModelName(model);
  }
  return (Math.max(0, inputTokens) + Math.max(0, outputTokens)) / 1000 * embodiedFactor;
}

function getCarbonIntensity(region: string | null | undefined): GridIntensity {
  if (!region) {
    return {
      intensity: numericOrDefault(GLOBAL_AVERAGES.global, 436),
      signalQuality: "unknown",
      timestamp: null,
    };
  }

  const regions = GLOBAL_AVERAGES.regions ?? {};
  if (region in regions) {
    return { intensity: regions[region]!, signalQuality: "blind", timestamp: null };
  }

  const countryCode = extractCountryCode(region);
  const countries = GLOBAL_AVERAGES.country_defaults ?? {};
  if (countryCode && countryCode in countries) {
    return { intensity: countries[countryCode]!, signalQuality: "blind", timestamp: null };
  }

  return {
    intensity: numericOrDefault(GLOBAL_AVERAGES.global, 436),
    signalQuality: "blind",
    timestamp: null,
  };
}

function getProviderPue(model: string, providerHint: string): [number, number, string] {
  // Explicit hint: the caller names the operator, so the only open question is
  // whether that operator publishes its own figure.
  const provider = providerHint.toLowerCase();
  if (provider in PROVIDER_PUE) {
    return [
      PROVIDER_PUE[provider]!,
      PROVIDER_PUE_TIER[provider] ?? 2,
      PROVIDER_PUE_SOURCES[provider] ?? "vendor report",
    ];
  }
  const inferred = inferProviderFromModel(model);
  if (inferred && inferred in PROVIDER_PUE) {
    let tier = PROVIDER_PUE_TIER[inferred] ?? 2;
    let source = PROVIDER_PUE_SOURCES[inferred] ?? "vendor report";
    const lower = model.toLowerCase();
    if (OPEN_WEIGHT_MODEL_PREFIXES.some((prefix) => lower.startsWith(prefix))) {
      tier = Math.max(tier, 2);
      source = `${source}; host assumed from an open-weight model name, `
        + "which does not identify an operator";
    }
    return [PROVIDER_PUE[inferred]!, tier, source];
  }
  return [DEFAULT_PUE, 3, "industry average (no operator resolved)"];
}

function inferProviderFromModel(model: string): string | null {
  const lower = model.toLowerCase();
  if (["gpt-", "o1", "o3", "o4", "text-davinci", "text-embedding"].some((prefix) => lower.startsWith(prefix))) {
    return "openai";
  }
  if (lower.startsWith("claude-")) {
    return "anthropic";
  }
  if (["gemini-", "gemma-", "palm-"].some((prefix) => lower.includes(prefix))) {
    return "google";
  }
  return null;
}

function inferWueProviderFromModel(model: string): string | null {
  const lower = model.toLowerCase();
  if (["gemini-", "palm-"].some((prefix) => lower.startsWith(prefix))) {
    return "google";
  }
  if (["gpt-", "o1", "o3", "o4", "text-davinci", "text-embedding"].some((prefix) => lower.startsWith(prefix))) {
    return "openai";
  }
  if (lower.startsWith("claude-")) {
    return "anthropic";
  }
  return null;
}

function estimateEmbodiedFactorByModelName(model: string): number {
  const lower = model.toLowerCase();
  if (["o1", "o3", "gpt-4", "gpt4", "claude-3-opus"].some((hint) => lower.includes(hint))) {
    return 0.25;
  }
  if (["-7b", "-8b", "small", "mini", "nano"].some((hint) => lower.includes(hint))) {
    return 0.02;
  }
  return 0.075;
}

function isTrackingDegraded(args: {
  modelKnown: boolean;
  energyTier: number;
  pueTier: number;
  signalQuality: VetchSignalQuality;
  usageEstimated: boolean;
  modelMatch: VetchModelMatch;
}): boolean {
  const gridQualityScore: Record<VetchSignalQuality, number> = {
    live: 0,
    delayed: 1,
    blind: 2,
    unknown: 3,
  };
  const score =
    (args.modelKnown ? 0 : 0.6) +
    (args.energyTier / 3) * 0.6 +
    (args.pueTier / 3) * 0.2 +
    (gridQualityScore[args.signalQuality] / 3) * 0.2 +
    (args.usageEstimated ? 0.4 : 0) +
    // A prefix/family proxy is a lower-confidence model match.
    (args.modelMatch === "prefix" || args.modelMatch === "family" ? 0.4 : 0);
  return score > TRACKING_DEGRADED_THRESHOLD;
}

function confidenceBounds(value: number | null, uncertaintyPct: number | null): { p5: number | null; p95: number | null } {
  if (value === null || uncertaintyPct === null) {
    return { p5: null, p95: null };
  }
  const band = value * (uncertaintyPct / 100);
  return { p5: Math.max(value - band, 0), p95: value + band };
}

function getUncertaintyPct(tier: number): number {
  return TIER_UNCERTAINTY_PCT[tier] ?? 1000;
}

function sanitizeIntensity(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  if (!Number.isFinite(value) || value > 2000) {
    return 2000;
  }
  return value;
}

function extractCountryCode(region: string): string | null {
  const lower = region.toLowerCase();
  if (lower.startsWith("us-")) {
    return "US";
  }
  if (lower.startsWith("ca-")) {
    return "CA";
  }
  if (lower.startsWith("sa-")) {
    return "BR";
  }
  if (lower.includes("australia")) {
    return "AU";
  }
  if (lower.includes("northeast") && lower.includes("asia")) {
    return "JP";
  }
  return null;
}

function numericOrDefault(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clampInt(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0, Math.trunc(value));
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return Math.max(0, Math.trunc(parsed));
    }
  }
  return 0;
}

export type { VetchUsage };
