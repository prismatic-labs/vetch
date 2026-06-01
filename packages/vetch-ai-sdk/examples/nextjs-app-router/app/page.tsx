export default function Home() {
  return (
    <main style={{ fontFamily: "system-ui", padding: "2rem", maxWidth: 640 }}>
      <h1>Vetch + Next.js reference</h1>
      <p>
        POST <code>/api/chat</code> with JSON <code>{`{ "prompt": "...", "sessionId": "optional" }`}</code>.
      </p>
      <pre style={{ background: "#f4f4f5", padding: "1rem", overflow: "auto" }}>
        {`curl -s -X POST http://localhost:3000/api/chat \\
  -H 'content-type: application/json' \\
  -d '{"prompt":"Say hello in one sentence."}'`}
      </pre>
      <p>
        See <code>src/vetch-model.ts</code> for <code>waitUntil</code> + emitter setup.
      </p>
    </main>
  );
}
