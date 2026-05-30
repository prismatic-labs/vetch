import type { VetchEmitter } from "./types.js";

export const noopEmitter: VetchEmitter = () => undefined;

export const consoleJsonEmitter: VetchEmitter = (event) => {
  console.log(JSON.stringify(event));
};

export interface FetchEmitterOptions {
  endpoint: string | URL;
  headers?: Record<string, string> | (() => Record<string, string>);
  /** Bearer token (sent as Authorization: Bearer …) in addition to custom headers. */
  bearerToken?: string | (() => string);
  fetchFn?: typeof fetch;
  timeoutMs?: number;
  /** Retry count after transient failures (default 0). */
  retries?: number;
}

export function createFetchEmitter(options: FetchEmitterOptions): VetchEmitter {
  const fetchImpl = options.fetchFn ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("createFetchEmitter requires fetch to be available");
  }

  const retries = Math.max(0, options.retries ?? 0);

  return async (event) => {
    let lastError: unknown;
    for (let attempt = 0; attempt <= retries; attempt += 1) {
      const controller = new AbortController();
      const timeout = options.timeoutMs === undefined ? 1000 : options.timeoutMs;
      const timeoutId = timeout > 0 ? setTimeout(() => controller.abort(), timeout) : undefined;
      try {
        const response = await fetchImpl(options.endpoint, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            ...resolveAuthHeaders(options),
            ...resolveHeaders(options.headers),
          },
          body: JSON.stringify(event),
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Vetch emitter request failed with HTTP ${response.status}`);
        }
        return;
      } catch (error) {
        lastError = error;
        if (attempt >= retries) {
          throw error;
        }
      } finally {
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
        }
      }
    }
    throw lastError;
  };
}

function resolveAuthHeaders(options: FetchEmitterOptions): Record<string, string> {
  if (!options.bearerToken) {
    return {};
  }
  const token =
    typeof options.bearerToken === "function" ? options.bearerToken() : options.bearerToken;
  return token ? { authorization: `Bearer ${token}` } : {};
}

function resolveHeaders(headers: FetchEmitterOptions["headers"]): Record<string, string> {
  if (!headers) {
    return {};
  }
  return typeof headers === "function" ? headers() : headers;
}
