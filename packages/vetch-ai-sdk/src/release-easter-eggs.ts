import { isTruthyEnv, readEnv } from "./env.js";
import type { VetchAdvisory, VetchOptions } from "./types.js";
import { VETCH_VERSION } from "./version.js";

/** Opt-in release advisories (e.g. NAPLES-081 on v0.8.1). Off by default. */
export function releaseEasterEggs(
  options: Pick<VetchOptions, "easterEggs">,
): VetchAdvisory[] {
  if (VETCH_VERSION !== "0.8.1") {
    return [];
  }
  if (options.easterEggs !== true && !isTruthyEnv(readEnv("VETCH_EASTER_EGGS"))) {
    return [];
  }
  return [
    {
      code: "NAPLES-081",
      severity: "info",
      title: "Release train 081",
      description: "the advice is to get more pizza",
    },
  ];
}
