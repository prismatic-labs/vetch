import { wrapLanguageModel } from "ai";
import type {
  LanguageModelV3,
  LanguageModelV3Middleware,
  LanguageModelV3StreamPart,
} from "@ai-sdk/provider";

import { createIsolatedVetchSession, createVetchSession } from "./advisories.js";
import { isVetchDisabledFromEnv } from "./env.js";
import { noopEmitter } from "./emitter.js";
import { createStreamObservation, createVetchEvent, observeStreamPart } from "./event.js";
import { releaseEasterEggs } from "./release-easter-eggs.js";
import type { VetchEvent, VetchLanguageModel, VetchOptions, VetchSession } from "./types.js";

interface InternalOptions extends VetchOptions {
  modelHint?: unknown;
}

export function withVetch<TModel extends VetchLanguageModel>(
  model: TModel,
  options: VetchOptions = {},
): LanguageModelV3 {
  if (isVetchDisabled(options)) {
    return model;
  }
  return wrapLanguageModel({
    model,
    middleware: createVetchMiddleware({ ...options, modelHint: model }),
  });
}

export function createVetchMiddleware(options: InternalOptions = {}): LanguageModelV3Middleware {
  if (isVetchDisabled(options)) {
    return {
      specificationVersion: "v3",
      async wrapGenerate({ doGenerate }) {
        return doGenerate();
      },
      async wrapStream({ doStream }) {
        return doStream();
      },
    };
  }

  const sessions = new ScopedSessionStore(options);

  return {
    specificationVersion: "v3",

    async wrapGenerate({ doGenerate, params, model }) {
      const startTimeMs = Date.now();
      const modelHint = options.modelHint ?? model;
      try {
        const result = await doGenerate();
        await recordAndEmit(
          await createVetchEvent({
            operation: "generate",
            model: modelHint,
            params,
            result,
            startTimeMs,
            options,
          }),
          sessions,
          options,
        );
        return result;
      } catch (error) {
        await recordAndEmit(
          await createVetchEvent({
            operation: "error",
            model: modelHint,
            params,
            startTimeMs,
            options,
            error,
          }),
          sessions,
          options,
        );
        throw error;
      }
    },

    async wrapStream({ doStream, params, model }) {
      const startTimeMs = Date.now();
      const modelHint = options.modelHint ?? model;
      try {
        const { stream, ...rest } = await doStream();
        const observation = createStreamObservation();
        let finalized = false;
        const finalizeOnce = async (cancelled: boolean): Promise<void> => {
          if (finalized) {
            return;
          }
          finalized = true;
          if (cancelled) {
            observation.cancelled = true;
          }
          await recordAndEmit(
            await createVetchEvent({
              operation: "stream",
              model: modelHint,
              params,
              streamObservation: observation,
              startTimeMs,
              options,
            }),
            sessions,
            options,
          );
        };
        const transformer = {
          transform(
            part: LanguageModelV3StreamPart,
            controller: TransformStreamDefaultController<LanguageModelV3StreamPart>,
          ) {
            observeStreamPart(part, observation);
            controller.enqueue(part);
          },
          async flush() {
            await finalizeOnce(false);
          },
          async cancel() {
            await finalizeOnce(true);
          },
        };
        const observedStream = stream.pipeThrough(
          new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>(
            transformer as Transformer<LanguageModelV3StreamPart, LanguageModelV3StreamPart>,
          ),
        );
        return { ...rest, stream: observedStream };
      } catch (error) {
        await recordAndEmit(
          await createVetchEvent({
            operation: "error",
            model: modelHint,
            params,
            startTimeMs,
            options,
            error,
          }),
          sessions,
          options,
        );
        throw error;
      }
    },
  };
}

async function recordAndEmit(
  event: VetchEvent,
  sessions: ScopedSessionStore,
  options: VetchOptions,
): Promise<void> {
  const delivery = deliverEvent(event, sessions.sessionFor(event), options);
  if (options.failOpen === false || options.emissionMode === "await") {
    await delivery;
    return;
  }

  if (options.waitUntil) {
    options.waitUntil(delivery);
    return;
  }

  void delivery;
}

class ScopedSessionStore {
  private readonly sessions = new Map<string, VetchSession>();
  private readonly maxSessionCount: number;
  private warnedMissingSessionId = false;

  constructor(private readonly options: InternalOptions) {
    this.maxSessionCount = options.maxSessionCount ?? 256;
  }

  sessionFor(event: VetchEvent): VetchSession {
    if (this.options.session) {
      return this.options.session;
    }

    const sessionId = event.session_id;
    if (!sessionId) {
      if (this.options.debug && !this.warnedMissingSessionId) {
        this.warnedMissingSessionId = true;
        console.warn(
          "Vetch: set providerOptions.vetch.attribution.sessionId for rolling advisories " +
            "(STALL, CACHE, ERROR, STREAM, REASONING, PROTO-001). Without it, only per-call advisories run.",
        );
      }
      return createIsolatedVetchSession(
        this.options.thresholds === undefined ? {} : { thresholds: this.options.thresholds },
      );
    }

    const existing = this.sessions.get(sessionId);
    if (existing) {
      this.sessions.delete(sessionId);
      this.sessions.set(sessionId, existing);
      return existing;
    }

    const session = this.createSession(sessionId);
    this.sessions.set(sessionId, session);
    while (this.sessions.size > this.maxSessionCount) {
      const oldest = this.sessions.keys().next().value;
      if (oldest === undefined) {
        break;
      }
      this.sessions.delete(oldest);
      if (this.options.debug) {
        console.warn(`Vetch AI SDK evicted session state for session_id=${oldest}`);
      }
      void Promise.resolve(this.options.onSessionEvicted?.(oldest)).catch(() => undefined);
    }
    return session;
  }

  private createSession(sessionId: string): VetchSession {
    if (this.options.sessionFactory) {
      return this.options.sessionFactory(sessionId);
    }
    return createVetchSession(
      this.options.thresholds === undefined ? {} : { thresholds: this.options.thresholds },
    );
  }
}

async function deliverEvent(
  event: VetchEvent,
  session: VetchSession,
  options: VetchOptions,
): Promise<void> {
  try {
    const advisories = session.record(event);
    event.advisories = dedupeAdvisories([...advisories, ...releaseEasterEggs(options)]);
    const emitter = options.emitter ?? noopEmitter;
    await withTimeout(Promise.resolve(emitter(event)), options.emitterTimeoutMs ?? 1000);
    if (event.advisories.length > 0) {
      await withTimeout(Promise.resolve(options.onAdvisory?.(event.advisories, event)), options.emitterTimeoutMs ?? 1000);
    }
    if (event.budget_exceeded === true) {
      await withTimeout(Promise.resolve(options.onBudgetExceeded?.(event)), options.emitterTimeoutMs ?? 1000);
    }
  } catch (error) {
    await Promise.resolve(options.onEmitterError?.(error, event)).catch(() => undefined);
    if (options.failOpen === false) {
      throw error;
    }
    if (options.debug) {
      console.warn("Vetch AI SDK middleware failed open", error);
    }
  }
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  if (timeoutMs <= 0) {
    return promise;
  }

  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timeoutId = setTimeout(
          () => reject(new Error(`Vetch emitter timed out after ${timeoutMs}ms`)),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    if (timeoutId !== undefined) {
      clearTimeout(timeoutId);
    }
  }
}

function dedupeAdvisories(advisories: VetchEvent["advisories"]): VetchEvent["advisories"] {
  const seen = new Set<string>();
  return advisories.filter((advisory) => {
    if (seen.has(advisory.code)) {
      return false;
    }
    seen.add(advisory.code);
    return true;
  });
}

function isVetchDisabled(options: Pick<VetchOptions, "disabled">): boolean {
  return isVetchDisabledFromEnv() || options.disabled === true;
}
