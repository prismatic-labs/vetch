import { openai } from "@ai-sdk/openai";
import { waitUntil } from "@vercel/functions";
import {
  consoleJsonEmitter,
  createFetchEmitter,
  withVetch,
  type VetchMiddlewareOptions,
} from "@prismatic-labs/vetch-ai-sdk";

function buildEmitter(): VetchMiddlewareOptions["emitter"] {
  const endpoint = process.env.VETCH_EVENTS_URL;
  if (!endpoint) {
    return consoleJsonEmitter;
  }
  return createFetchEmitter({
    endpoint,
    bearerToken: process.env.VETCH_COLLECTOR_TOKEN,
    timeoutMs: 2000,
    retries: 1,
  });
}

/** Shared model for API routes - configure once, reuse per request. */
export const vetchModel = withVetch(openai("gpt-4.1-mini"), {
  region: "US-CA",
  tags: { app: "vetch-next-example", route: "api/chat" },
  emitter: buildEmitter(),
  waitUntil: (promise) => waitUntil(promise),
  onAdvisory(advisories) {
    if (advisories.length === 0) {
      return;
    }
    console.warn(
      "[vetch]",
      advisories.map((a) => a.code).join(", "),
    );
  },
});
