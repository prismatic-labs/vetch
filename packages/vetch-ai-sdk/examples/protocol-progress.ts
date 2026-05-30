import { gateway, generateText } from "ai";

import { createVetchSession, withVetch } from "../src/index.js";

const session = createVetchSession({
  thresholds: {
    protocolVoidWindow: 2,
    protocolVoidMinOutputTokens: 20,
  },
});

const model = withVetch(gateway("anthropic/claude-sonnet-4.5"), {
  session,
  tags: {
    app: "agent-demo",
  },
  onAdvisory(advisories) {
    console.warn("Vetch advisories:", advisories.map((advisory) => advisory.code).join(", "));
  },
});

await generateText({
  model,
  prompt:
    "You are inside a retrieval workflow. If you need context, call the search tool. Otherwise say DONE.",
  providerOptions: {
    vetch: {
      tags: {
        workflow: "retrieval",
      },
      protocol: {
        expectedToolUse: true,
        stepCount: 1,
      },
    },
  },
});
