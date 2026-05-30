import { gateway, generateText } from "ai";

import { withVetch } from "../src/index.js";

const model = withVetch(gateway("openai/gpt-4.1-mini"), {
  tags: {
    app: "vetch-ai-sdk-poc",
    route: "basic",
  },
});

const result = await generateText({
  model,
  prompt: "Write one friendly sentence about measuring invisible AI waste.",
});

console.log(result.text);
