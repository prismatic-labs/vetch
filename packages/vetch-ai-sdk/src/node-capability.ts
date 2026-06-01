/** True when Node built-ins are available (not Edge / browser). */
export function canUseNodeCalibration(): boolean {
  const getBuiltinModule = (
    globalThis as {
      process?: {
        getBuiltinModule?: (specifier: string) => unknown;
      };
    }
  ).process?.getBuiltinModule;
  return typeof getBuiltinModule === "function";
}
