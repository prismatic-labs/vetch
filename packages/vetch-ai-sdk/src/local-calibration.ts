export interface LocalEnergyOverride {
  wh_per_1k_input: number;
  wh_per_1k_output: number;
  tier?: number;
  source?: string;
  basis?: string;
  wh_per_image?: number;
  visual_tokens_per_image?: number;
  intercept_wh?: number;
}

interface NodeBuiltins {
  fs: {
    existsSync(path: string): boolean;
    readFileSync(path: string, encoding: "utf8"): string;
  };
  os: {
    homedir(): string;
  };
  path: {
    join(...parts: string[]): string;
  };
}

function getBuiltin<T>(name: string): T | null {
  const getBuiltinModule = (
    globalThis as {
      process?: {
        getBuiltinModule?: (specifier: string) => unknown;
      };
    }
  ).process?.getBuiltinModule;
  if (typeof getBuiltinModule !== "function") {
    return null;
  }
  try {
    return getBuiltinModule(name) as T;
  } catch {
    return null;
  }
}

function getNodeBuiltins(): NodeBuiltins | null {
  const fs = getBuiltin<NodeBuiltins["fs"]>("node:fs");
  const os = getBuiltin<NodeBuiltins["os"]>("node:os");
  const path = getBuiltin<NodeBuiltins["path"]>("node:path");
  if (!fs || !os || !path) {
    return null;
  }
  return { fs, os, path };
}

function calibrationModelVariants(model: string): string[] {
  const variants: string[] = [];
  const seen = new Set<string>();
  const add = (name: string): void => {
    if (name && !seen.has(name)) {
      seen.add(name);
      variants.push(name);
    }
  };
  add(model);
  if (model.includes(":")) {
    const idx = model.lastIndexOf(":");
    const base = model.slice(0, idx);
    const tag = model.slice(idx + 1);
    if (tag === "latest") {
      add(base);
    } else {
      add(`${base}:latest`);
    }
  } else {
    add(`${model}:latest`);
  }
  return variants;
}

function safeFileStem(model: string): string {
  return model.replace(/:/g, "_");
}

function readCalibrationFile(
  builtins: NodeBuiltins,
  calibrationDir: string,
  provider: string,
  model: string,
): LocalEnergyOverride | null {
  const filePath = builtins.path.join(calibrationDir, `${provider}_${safeFileStem(model)}.json`);
  if (!builtins.fs.existsSync(filePath)) {
    return null;
  }
  try {
    const data = JSON.parse(builtins.fs.readFileSync(filePath, "utf8")) as Record<string, unknown>;
    if (data.active === false) {
      return null;
    }
    const out: LocalEnergyOverride = {
      wh_per_1k_input: Number(data.wh_per_1k_input),
      wh_per_1k_output: Number(data.wh_per_1k_output),
      tier: typeof data.tier === "number" ? data.tier : 0,
      source: "local_calibration",
      basis: `Hardware-measured on ${String(data.gpu_name ?? "local GPU")}`,
    };
    if (typeof data.wh_per_image === "number") {
      out.wh_per_image = data.wh_per_image;
    }
    if (typeof data.visual_tokens_per_image === "number") {
      out.visual_tokens_per_image = data.visual_tokens_per_image;
    }
    if (typeof data.intercept_wh === "number") {
      out.intercept_wh = data.intercept_wh;
    }
    return out;
  } catch {
    return null;
  }
}

/** Load Tier-0 ~/.vetch calibration (Node.js only; returns null on Edge). */
export function loadLocalCalibration(
  provider: string,
  model: string,
): LocalEnergyOverride | null {
  const builtins = getNodeBuiltins();
  if (builtins === null) {
    return null;
  }
  const calibrationDir = builtins.path.join(builtins.os.homedir(), ".vetch", "calibrations");
  for (const variant of calibrationModelVariants(model)) {
    const loaded = readCalibrationFile(builtins, calibrationDir, provider, variant);
    if (loaded !== null) {
      return loaded;
    }
  }
  return null;
}
