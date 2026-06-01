import { generateText } from "ai";

import { vetchModel } from "@/src/vetch-model";

export const runtime = "nodejs";

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
          traceId: `next-${sessionId}`,
        },
        tags: {
          env: process.env.NODE_ENV ?? "development",
        },
      },
    },
  });

  return Response.json({ text, sessionId });
}
