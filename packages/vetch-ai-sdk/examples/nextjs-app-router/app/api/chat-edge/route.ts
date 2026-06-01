import { generateText } from "ai";

import { vetchModel } from "@/src/vetch-model";

// Edge variant of /api/chat. Identical Vetch wiring; runs on the Edge runtime.
// Note: there is no filesystem on Edge, so Tier-0 calibration files under
// ~/.vetch are not read here. Pass `energyOverride` in withVetch (see
// src/vetch-model.ts) if you need Tier-0 energy coefficients on Edge.
export const runtime = "edge";

export async function POST(req: Request) {
  const body = (await req.json()) as { prompt?: string; sessionId?: string };
  const prompt = body.prompt?.trim();
  if (!prompt) {
    return Response.json({ error: "prompt is required" }, { status: 400 });
  }

  const sessionId = body.sessionId ?? crypto.randomUUID();

  const { text } = await generateText({
    model: vetchModel,
    prompt,
    providerOptions: {
      vetch: {
        attribution: {
          sessionId,
          traceId: `next-edge-${sessionId}`,
        },
        tags: {
          env: process.env.NODE_ENV ?? "development",
          runtime: "edge",
        },
      },
    },
  });

  return Response.json({ text, sessionId, runtime: "edge" });
}
