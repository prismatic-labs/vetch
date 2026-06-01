import type { VetchBudgets } from "./types.js";

export function readEnv(name: string): string | undefined {
  return typeof process === "undefined" ? undefined : process.env?.[name];
}

export function isTruthyEnv(value: string | undefined): boolean {
  const normalized = value?.toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes";
}

export function isFalsyEnv(value: string | undefined): boolean {
  const normalized = value?.toLowerCase();
  return normalized === "false" || normalized === "0" || normalized === "no";
}

export function isVetchDisabledFromEnv(): boolean {
  if (isTruthyEnv(readEnv("VETCH_DISABLED"))) {
    return true;
  }
  if (isFalsyEnv(readEnv("VETCH_ENABLED"))) {
    return true;
  }
  return false;
}

/** Mirrors Python `budget.configure_from_env()` for Node runtimes. */
export function readEnvBudgets(): VetchBudgets {
  const budgets: VetchBudgets = {};
  assignEnvBudget(budgets, "costUsd", "VETCH_BUDGET_COST_USD");
  assignEnvBudget(budgets, "energyWh", "VETCH_BUDGET_ENERGY_WH");
  assignEnvBudget(budgets, "carbonG", "VETCH_BUDGET_CARBON_G");
  return budgets;
}

function assignEnvBudget(
  budgets: VetchBudgets,
  key: keyof VetchBudgets,
  envName: string,
): void {
  const raw = readEnv(envName);
  if (raw === undefined || raw.trim() === "") {
    return;
  }
  const value = Number(raw);
  if (Number.isFinite(value) && value >= 0) {
    budgets[key] = value;
  }
}
