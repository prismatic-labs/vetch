import aliasesJson from "./registry/aliases.json" with { type: "json" };
import energyJson from "./registry/energy.json" with { type: "json" };
import pricingJson from "./registry/pricing.json" with { type: "json" };
import wueJson from "./registry/wue.json" with { type: "json" };
import globalAveragesJson from "./sensing/global_averages.json" with { type: "json" };

import { loadLocalCalibration } from "./local-calibration.js";
import type { VetchEvent, VetchSignalQuality, VetchUsage } from "./types.js";

const METHODOLOGY_VERSION = "1.2";
const CACHE_READ_ENERGY_FACTOR = 0.15;
const PROMPT_LENGTH_SHORT_THRESHOLD = 1000;
const PROMPT_LENGTH_MEDIUM_THRESHOLD = 5000;

const TIER_UNCERTAINTY_PCT: Record<number, number> = {
  0: 20,
  1: 50,
  2: 100,
  3: 1000,
};

const DEFAULT_PUE = 1.2;
const PROVIDER_PUE: Record<string, number> = {
  google: 1.10,
  vertexai: 1.10,
  azure: 1.12,
  openai: 1.12,
  aws: 1.15,
  anthropic: 1.15,
  bedrock: 1.15,
};

const PROVIDER_PUE_SOURCES: Record<string, string> = {
  google: "Google Data Centers Efficiency Report 2023",
  vertexai: "Google Data Centers Efficiency Report 2023",
  azure: "Microsoft Datacenters Sustainability 2024",
  openai: "Microsoft Datacenters Sustainability 2024 (Azure-backed)",
  aws: "AWS Sustainability Report 2024",
  anthropic: "AWS Sustainability Report 2024 (AWS-backed)",
  bedrock: "AWS Sustainability Report 2024",
};

const DEFAULT_WUE = 1.8;
const PROVIDER_WUE: Record<string, number> = {
  google: 1.1,
  vertexai: 1.1,
  azure: 1.7,
  openai: 1.7,
  aws: 2.2,
  anthropic: 2.2,
  bedrock: 2.2,
};

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

type EnergyRegistry = Record<string, EnergyEntry>;
type PricingRegistry = Record<string, PricingEntry>;
type AliasRegistry = Record<string, string>;
type NumberRegistry = Record<string, number>;

const ENERGY = energyJson as EnergyRegistry;
const PRICING = pricingJson as PricingRegistry;
const ALIASES = aliasesJson as AliasRegistry;
const WUE = Object.fromEntries(
  Object.entries(wueJson as Record<string, unknown>)
    .filter(([key, value]) => !key.startsWith("_") && typeof value === "number"),
) as NumberRegistry;
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
  wh_per_1k_input: number;
  wh_per_1k_output: number;
  tier?: number;
  source?: string;
  basis?: string;
  wh_per_image?: number;
  visual_tokens_per_image?: number;
  intercept_wh?: number;
}

interface EnrichOptions {
  provider: string;
  region?: string | null;
  priceMultiplier?: number;
  energyOverride?: EnergyOverride | null;
}

export function enrichVetchEvent(event: VetchEvent, options: EnrichOptions): VetchEvent {
  const resolved = resolveModel(event.model);
  event.model_known = resolved.known;
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
  const energyOverride =
    options.energyOverride ?? loadLocalCalibration(options.provider, event.model);

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
  event.estimated_water_l = calculateWater(
    energy.energyWh,
    event.model,
    options.provider,
    options.region ?? event.region,
  );
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

  if (cacheReadTokens > 0) {
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

  const bounds = confidenceBounds(event.estimated_energy_wh, event.energy_uncertainty_pct);
  event.energy_p5_wh = bounds.p5;
  event.energy_p95_wh = bounds.p95;
  const carbonBounds = confidenceBounds(event.estimated_carbon_g, event.energy_uncertainty_pct);
  event.carbon_p5_g = carbonBounds.p5;
  event.carbon_p95_g = carbonBounds.p95;

  event.tracking_degraded = isTrackingDegraded({
    modelKnown: energy.modelKnown,
    energyTier: energy.tier,
    pueTier: carbon.pueTier,
    signalQuality: grid.signalQuality,
    usageEstimated: event.usage_estimated,
  });

  event.vetch_warnings = event.vetch_warnings.filter(
    (warning) => warning !== "estimates_not_calculated",
  );
  void METHODOLOGY_VERSION;
  return event;
}

export function resolveModel(model: string): { resolvedModel: string; known: boolean } {
  if (model in ENERGY) {
    return { resolvedModel: model, known: true };
  }
  const alias = ALIASES[model];
  if (alias && alias in ENERGY) {
    return { resolvedModel: alias, known: true };
  }

  const parts = model.split("-");
  for (let i = parts.length - 1; i > 0; i -= 1) {
    const prefix = parts.slice(0, i).join("-");
    if (prefix in ENERGY) {
      return { resolvedModel: prefix, known: true };
    }
    const prefixAlias = ALIASES[prefix];
    if (prefixAlias && prefixAlias in ENERGY) {
      return { resolvedModel: prefixAlias, known: true };
    }
  }

  return { resolvedModel: model, known: false };
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
  const resolved = resolveModel(model);

  let whIn: number;
  let whOut: number;
  let tier: number;
  let basis: string | null;
  let source: string;

  if (energyOverride !== null) {
    whIn = energyOverride.wh_per_1k_input;
    whOut = energyOverride.wh_per_1k_output;
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
    if (whPerImage !== undefined && whPerImage > 0) {
      let imageUnits = Math.max(0, nImages);
      if (vtok !== null && imageInputTokens > 0) {
        imageUnits = Math.max(imageUnits, imageInputTokens / vtok);
      }
      if (imageUnits > 0) {
        energyWh += whPerImage * imageUnits;
      }
    }
    const interceptWh = energyOverride.intercept_wh;
    if (interceptWh !== undefined && interceptWh > 0) {
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

  const cacheTokens = Math.min(Math.max(0, cacheReadTokens), Math.max(0, inputTokens));
  const effectiveInput = Math.max(0, inputTokens) - cacheTokens;
  const cacheReadCost = cacheTokens * entry.usd_per_1k_input * cacheReadDiscount / 1000;
  const cacheWriteCost = Math.max(0, cacheCreationTokens) *
    entry.usd_per_1k_input *
    cacheCreationPremium /
    1000;
  const inputCost = calculateTieredCost(
    effectiveInput,
    entry.usd_per_1k_input,
    tierThreshold,
    tierMultiplierInput,
  ) + cacheReadCost + cacheWriteCost;
  const outputCost = calculateTieredCost(
    Math.max(0, outputTokens),
    entry.usd_per_1k_output,
    tierThreshold,
    tierMultiplierOutput,
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
): number {
  if (tierThreshold === null || tierMultiplier === null) {
    return tokens * baseRatePer1k / 1000;
  }
  if (tokens <= tierThreshold) {
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
): number {
  let wue: number;
  const provider = providerHint.toLowerCase();
  if (region && provider) {
    const regionKey = `${provider}-${region.toLowerCase()}`;
    wue = WUE[regionKey] ?? WUE[provider] ?? PROVIDER_WUE[provider] ?? DEFAULT_WUE;
  } else if (provider) {
    wue = WUE[provider] ?? PROVIDER_WUE[provider] ?? DEFAULT_WUE;
  } else {
    const inferred = inferProviderFromModel(model);
    wue = inferred ? WUE[inferred] ?? PROVIDER_WUE[inferred] ?? DEFAULT_WUE : DEFAULT_WUE;
  }
  return energyWh / 1000 * wue;
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
  const provider = providerHint.toLowerCase();
  if (provider in PROVIDER_PUE) {
    return [
      PROVIDER_PUE[provider]!,
      1,
      PROVIDER_PUE_SOURCES[provider] ?? "vendor report",
    ];
  }
  const inferred = inferProviderFromModel(model);
  if (inferred && inferred in PROVIDER_PUE) {
    return [
      PROVIDER_PUE[inferred]!,
      1,
      PROVIDER_PUE_SOURCES[inferred] ?? "vendor report",
    ];
  }
  return [DEFAULT_PUE, 3, "industry average"];
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
    (args.usageEstimated ? 0.4 : 0);
  return score > 2.5;
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
