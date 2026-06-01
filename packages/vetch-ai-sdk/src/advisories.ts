import type { VetchAdvisory, VetchEvent, VetchOptions, VetchSession, VetchThresholds } from "./types.js";

const DEFAULT_THRESHOLDS: Required<VetchThresholds> = {
  highOutputTokens: 1500,
  highVisibleChars: 6000,
  taskModeOutputTokens: 256,
  toolSpinCalls: 8,
  toolFailures: 3,
  repairAttempts: 2,
  protocolVoidWindow: 3,
  protocolVoidMinOutputTokens: 100,
  errorWindow: 5,
  errorFraction: 0.5,
  consecutiveErrors: 3,
  streamWindow: 5,
  streamIncompleteFraction: 0.4,
  reasoningWindow: 3,
  reasoningMissingFraction: 0.8,
};

// Session-level advisory constants
const STALL_MIN_WINDOW = 5;
const STALL_MAX_OUTPUT_TOKENS = 5;
const CACHE_MIN_WINDOW = 6;
const CACHE_REPETITION_FRACTION = 0.5;

export function createVetchSession(options: Pick<VetchOptions, "thresholds"> = {}): VetchSession {
  const thresholds = { ...DEFAULT_THRESHOLDS, ...(options.thresholds ?? {}) };
  return new RollingVetchSession(thresholds);
}

class RollingVetchSession implements VetchSession {
  private readonly events: VetchEvent[] = [];

  constructor(private readonly thresholds: Required<VetchThresholds>) {}

  record(event: VetchEvent): VetchAdvisory[] {
    this.events.push(event);
    if (this.events.length > 40) {
      this.events.shift();
    }
    return detectAdvisories(event, this.events, this.thresholds);
  }

  recentEvents(): readonly VetchEvent[] {
    return this.events;
  }
}

/** Per-call only — used when `attribution.sessionId` is not set (no cross-request bleed). */
class IsolatedVetchSession implements VetchSession {
  constructor(private readonly thresholds: Required<VetchThresholds>) {}

  record(event: VetchEvent): VetchAdvisory[] {
    return detectPerCallAdvisories(event, this.thresholds);
  }

  recentEvents(): readonly VetchEvent[] {
    return [];
  }
}

export function createIsolatedVetchSession(
  options: Pick<VetchOptions, "thresholds"> = {},
): VetchSession {
  const thresholds = { ...DEFAULT_THRESHOLDS, ...(options.thresholds ?? {}) };
  return new IsolatedVetchSession(thresholds);
}

export function detectAdvisories(
  event: VetchEvent,
  recentEvents: readonly VetchEvent[],
  thresholds: Required<VetchThresholds> = DEFAULT_THRESHOLDS,
): VetchAdvisory[] {
  return [
    ...detectPerCallAdvisories(event, thresholds),
    ...detectSessionAdvisories(event, recentEvents, thresholds),
  ];
}

export function detectPerCallAdvisories(
  event: VetchEvent,
  thresholds: Required<VetchThresholds> = DEFAULT_THRESHOLDS,
): VetchAdvisory[] {
  const advisories: VetchAdvisory[] = [];
  const outputTokens = event.usage?.text.output_tokens ?? 0;
  const finishReason = event.finish_reason?.toLowerCase() ?? "";
  const protocol = event.protocol_progress;

  if (event.budget_exceeded === true) {
    advisories.push({
      code: "BUDGET-001",
      severity: "warning",
      title: "Configured budget threshold exceeded",
      description:
        "This call exceeded a configured per-request cost, energy, or carbon threshold. The middleware does not block calls. For rolling session budgets, use Python Vetch set_budget() with session windows.",
      evidence: {
        budget_cost_usd: event.budget_cost_usd,
        budget_energy_wh: event.budget_energy_wh,
        budget_carbon_g: event.budget_carbon_g,
        estimated_cost_usd: event.estimated_cost_usd,
        estimated_energy_wh: event.estimated_energy_wh,
        estimated_carbon_g: event.estimated_carbon_g,
      },
    });
  }

  if (finishReason.includes("max") || finishReason.includes("length")) {
    advisories.push({
      code: "TRUNC-001",
      severity: "warning",
      title: "Generation ended at the token limit",
      description:
        "The model appears to have stopped because it ran out of output budget. This often wastes a call and can hide broken structured output.",
      evidence: {
        finishReason: event.finish_reason,
        outputTokens,
      },
    });
  }

  if (
    outputTokens > 0 &&
    event.visible_output_chars === 0 &&
    event.tool_call_count === 0 &&
    protocol?.expectedOutputClass !== "tool"
  ) {
    advisories.push({
      code: "EMPTY-001",
      severity: "warning",
      title: "Tokens were spent without visible output",
      description:
        "The provider reported output tokens, but the application-visible response was empty. This is a good place to inspect stop sequences and tool protocol handling.",
      evidence: {
        outputTokens,
        finishReason: event.finish_reason,
      },
    });
  }

  if (outputTokens >= thresholds.highOutputTokens || event.visible_output_chars >= thresholds.highVisibleChars) {
    advisories.push({
      code: "BABBLE-001",
      severity: "info",
      title: "High decode volume",
      description:
        "This call produced a lot of output. If the application only needed a short protocol action, this may be decode-side waste.",
      evidence: {
        outputTokens,
        visibleOutputChars: event.visible_output_chars,
      },
    });
  }

  if (
    protocol?.expectedOutputClass !== undefined &&
    protocol.expectedOutputClass !== "longform" &&
    outputTokens >= thresholds.taskModeOutputTokens
  ) {
    advisories.push({
      code: "EXPECTED-LENGTH-001",
      severity: "info",
      title: "Long answer for a compact task mode",
      description:
        "The app marked this as a short, JSON, or tool-oriented step, but the model produced a long decode. A tighter terminal condition may save retries.",
      evidence: {
        expectedOutputClass: protocol.expectedOutputClass,
        outputTokens,
      },
    });
  }

  if (event.tool_call_count >= thresholds.toolSpinCalls) {
    advisories.push({
      code: "TOOL-SPIN-001",
      severity: "warning",
      title: "Heavy tool-call churn",
      description:
        "A single response produced many tool calls. Check whether the agent has a clearer terminal condition or should stop earlier.",
      evidence: {
        toolCallCount: event.tool_call_count,
      },
    });
  }

  if ((protocol?.invalidToolCallCount ?? 0) >= thresholds.toolFailures) {
    advisories.push({
      code: "VOID-001",
      severity: "warning",
      title: "Invalid tool progress",
      description:
        "The application reported repeated invalid tool calls. This is usually invisible to provider-only telemetry and can waste retries.",
      evidence: {
        invalidToolCallCount: protocol?.invalidToolCallCount,
        validToolCallCount: protocol?.validToolCallCount,
      },
    });
  }

  if ((protocol?.toolFailureCount ?? 0) >= thresholds.toolFailures) {
    advisories.push({
      code: "TOOL-TREADMILL-001",
      severity: "warning",
      title: "Repeated failing tool progress",
      description:
        "The application reported repeated tool failures in this workflow. This can become a retrieval treadmill even when token-shape heuristics look normal.",
      evidence: {
        toolFailureCount: protocol?.toolFailureCount,
        stepCount: protocol?.stepCount,
      },
    });
  }

  if ((protocol?.repairAttemptCount ?? 0) >= thresholds.repairAttempts) {
    advisories.push({
      code: "STRUCT-REPAIR-001",
      severity: "warning",
      title: "Structured-output repair loop",
      description:
        "The app reported repeated schema or JSON repair attempts. Consider a smaller schema, clearer tool contract, or lower output cap.",
      evidence: {
        repairAttemptCount: protocol?.repairAttemptCount,
        finishReason: event.finish_reason,
      },
    });
  }

  if ((protocol?.postTerminalCallCount ?? 0) >= 2) {
    advisories.push({
      code: "POSTDONE-DECODE-001",
      severity: "warning",
      title: "Calls continued after terminal progress",
      description:
        "The workflow appears to have accepted or reached a terminal state, but model calls continued afterwards.",
      evidence: {
        acceptedFinalResult: protocol?.acceptedFinalResult,
        postTerminalCallCount: protocol?.postTerminalCallCount,
        outputTokens,
      },
    });
  }

  return advisories;
}

export function detectSessionAdvisories(
  event: VetchEvent,
  recentEvents: readonly VetchEvent[],
  thresholds: Required<VetchThresholds> = DEFAULT_THRESHOLDS,
): VetchAdvisory[] {
  const advisories: VetchAdvisory[] = [];
  const outputTokens = event.usage?.text.output_tokens ?? 0;
  const protocol = event.protocol_progress;

  const recentErrors = recentEvents.slice(-thresholds.errorWindow);
  const errorCount = recentErrors.filter((candidate) => candidate.error).length;
  const consecutiveErrorCount = countTrailing(recentEvents, (candidate) => candidate.error);
  if (
    recentErrors.length >= thresholds.errorWindow &&
    (
      errorCount / recentErrors.length >= thresholds.errorFraction ||
      consecutiveErrorCount >= thresholds.consecutiveErrors
    )
  ) {
    advisories.push({
      code: "ERROR-001",
      severity: "warning",
      title: "Repeated model-call errors",
      description:
        "Recent calls are failing often enough to suggest a retry storm, provider outage, or app-level error loop.",
      evidence: {
        window: recentErrors.length,
        errorCount,
        consecutiveErrors: consecutiveErrorCount,
        lastErrorType: event.error_type,
      },
    });
  }

  const recentStreams = recentEvents
    .filter((candidate) => candidate.is_stream)
    .slice(-thresholds.streamWindow);
  if (recentStreams.length >= thresholds.streamWindow) {
    const incompleteCount = recentStreams.filter((candidate) => !candidate.complete).length;
    if (incompleteCount / recentStreams.length >= thresholds.streamIncompleteFraction) {
      advisories.push({
        code: "STREAM-001",
        severity: "warning",
        title: "Incomplete stream burn",
        description:
          "Several recent streams ended without a normal finish. Cancelled or interrupted streams can still consume tokens and cost.",
        evidence: {
          window: recentStreams.length,
          incompleteCount,
          incompleteFraction: Math.round((incompleteCount / recentStreams.length) * 100) / 100,
        },
      });
    }
  }

  const recentReasoningModels = recentEvents
    .filter((candidate) => isReasoningModel(candidate.model))
    .slice(-thresholds.reasoningWindow);
  if (recentReasoningModels.length >= thresholds.reasoningWindow) {
    const missingReasoningCount = recentReasoningModels.filter(
      (candidate) => (candidate.usage?.reasoning?.output_tokens ?? 0) === 0,
    ).length;
    if (missingReasoningCount / recentReasoningModels.length >= thresholds.reasoningMissingFraction) {
      advisories.push({
        code: "REASONING-001",
        severity: "info",
        title: "Reasoning model returned no reasoning tokens",
        description:
          "A reasoning-capable model is repeatedly returning no reasoning-token telemetry. Check whether reasoning mode is engaged or whether a cheaper non-reasoning model would fit.",
        evidence: {
          window: recentReasoningModels.length,
          missingReasoningCount,
          model: event.model,
        },
      });
    }
  }

  const protocolVoidEvents = recentEvents
    .slice(-thresholds.protocolVoidWindow)
    .filter(
      (candidate) =>
        candidate.protocol_progress?.expectedToolUse === true &&
        candidate.tool_call_count === 0 &&
        (candidate.usage?.text.output_tokens ?? 0) >= thresholds.protocolVoidMinOutputTokens,
    );

  if (
    protocol?.expectedToolUse === true &&
    event.tool_call_count === 0 &&
    protocolVoidEvents.length >= thresholds.protocolVoidWindow
  ) {
    advisories.push({
      code: "PROTO-001",
      severity: "warning",
      title: "Protocol progress disappeared",
      description:
        "The application expected tool progress, but several recent calls spent output tokens without valid tool calls. This is the gap provider wrappers usually cannot see.",
      evidence: {
        window: thresholds.protocolVoidWindow,
        recentNoToolCalls: protocolVoidEvents.length,
        outputTokens,
      },
    });
  }

  const recentNonError = recentEvents.filter((e) => !e.error);
  if (recentNonError.length >= STALL_MIN_WINDOW) {
    const lastN = recentNonError.slice(-STALL_MIN_WINDOW);
    if (lastN.every((e) => (e.usage?.text.output_tokens ?? 0) <= STALL_MAX_OUTPUT_TOKENS)) {
      advisories.push({
        code: "STALL-001",
        severity: "warning",
        title: "Repeated near-zero output — possible stall",
        description:
          `The last ${STALL_MIN_WINDOW} non-error calls each produced ≤${STALL_MAX_OUTPUT_TOKENS} output tokens. ` +
          "This suggests the model is stalling or a stop sequence is triggering prematurely.",
        evidence: {
          window: STALL_MIN_WINDOW,
          maxOutputSeen: Math.max(...lastN.map((e) => e.usage?.text.output_tokens ?? 0)),
        },
      });
    }
  }

  // CACHE-001 / CACHE-002: high input-token repetition in the rolling window
  if (recentEvents.length >= CACHE_MIN_WINDOW) {
    const inputCounts = new Map<number, number>();
    for (const e of recentEvents) {
      const n = e.usage?.text.input_tokens ?? 0;
      inputCounts.set(n, (inputCounts.get(n) ?? 0) + 1);
    }
    const maxCount = Math.max(...inputCounts.values());
    if (maxCount / recentEvents.length > CACHE_REPETITION_FRACTION) {
      advisories.push({
        code: "CACHE-001",
        severity: "info",
        title: "Repeated identical context — consider prompt caching",
        description:
          `More than ${Math.round(CACHE_REPETITION_FRACTION * 100)}% of recent calls share the same input ` +
          "token count, suggesting a large static prefix. Prompt caching would reduce cost and latency.",
        evidence: {
          window: recentEvents.length,
          repetitionFraction: Math.round((maxCount / recentEvents.length) * 100) / 100,
        },
      });

      // CACHE-002: same repetition pattern but no cache reads in the window
      const hasCacheReads = recentEvents.some((e) => (e.cache_read_tokens ?? 0) > 0);
      if (!hasCacheReads) {
        advisories.push({
          code: "CACHE-002",
          severity: "warning",
          title: "Cache opportunity missed — caching not active",
          description:
            "High input-token repetition detected but no cache reads have been observed. " +
            "Enabling prompt caching (Anthropic cache_control, OpenAI prompt caching) would save tokens.",
          evidence: {
            window: recentEvents.length,
            cacheReadsObserved: 0,
          },
        });
      }
    }
  }

  return advisories;
}

function countTrailing<T>(items: readonly T[], predicate: (item: T) => boolean): number {
  let count = 0;
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (!predicate(items[i]!)) {
      break;
    }
    count += 1;
  }
  return count;
}

function isReasoningModel(model: string): boolean {
  if (typeof model !== "string") {
    return false;
  }
  const normalized = model.toLowerCase();
  return (
    /\bo[134](?:-|$)/.test(normalized) ||
    normalized.includes("reasoning") ||
    normalized.includes("thinking") ||
    normalized.includes("deepseek-r1")
  );
}
