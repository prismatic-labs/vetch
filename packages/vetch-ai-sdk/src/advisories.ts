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

export function detectAdvisories(
  event: VetchEvent,
  recentEvents: readonly VetchEvent[],
  thresholds: Required<VetchThresholds> = DEFAULT_THRESHOLDS,
): VetchAdvisory[] {
  const advisories: VetchAdvisory[] = [];
  const outputTokens = event.usage?.text.output_tokens ?? 0;
  const finishReason = event.finish_reason?.toLowerCase() ?? "";
  const protocol = event.protocol_progress;

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

  // --- Session-level advisories (require rolling window) ---

  // STALL-001: last N non-error calls all produced near-zero output
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
