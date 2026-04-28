"""Vetch v0.4.0 Circuit Breaker — live web dashboard demo.

A single-file, no-external-deps demo built for live presentation. Opens a
browser dashboard on http://localhost:8765 showing real-time cost, energy,
and carbon counters as a deliberately-stalled agent loop runs.

When STALL-001 fires, the dashboard flashes red and shows the
"CIRCUIT BREAKER ENGAGED" banner with the projected savings.

Modes:

    --mock       Use synthetic events with deterministic pacing. No API key
                 needed. Best for live demos where the venue Wi-Fi is unknown.
                 (Default)

    --real       Use the real OpenAI API. Requires OPENAI_API_KEY. Use this
                 when you want to show that the math is grounded in real
                 calls, or to capture a recording with real provider data.

Usage:

    python examples/circuit_breaker_demo_web.py
    python examples/circuit_breaker_demo_web.py --real
    python examples/circuit_breaker_demo_web.py --mock --action reroute

    Then open http://localhost:8765 in a browser.

Stack:
    - stdlib http.server for the HTTP layer (no Flask/FastAPI dependency)
    - Server-Sent Events (SSE) for streaming (no WebSocket dependency)
    - Single self-contained HTML page embedded below
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import vetch
from vetch.emitter import emit_event as _real_emit_event

# ---------------------------------------------------------------------------
# Event broadcast: every browser tab gets its own queue.
# ---------------------------------------------------------------------------

_subscriber_queues: list[queue.Queue[str]] = []
_subscriber_lock = threading.Lock()


def _broadcast(payload: dict[str, Any]) -> None:
    """Push an event to every connected browser."""
    msg = json.dumps(payload, default=str)
    with _subscriber_lock:
        for q in _subscriber_queues:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


# Hook into Vetch's emitter so every InferenceEvent reaches the dashboard.
def _patched_emit_event(event: Any) -> None:
    try:
        _broadcast({"type": "inference_event", "event": event})
    except Exception:
        pass
    _real_emit_event(event)


# Apply the monkey-patch on import.
import vetch.emitter as _emitter_mod  # noqa: E402

_emitter_mod.emit_event = _patched_emit_event


# ---------------------------------------------------------------------------
# Dashboard HTML (single-file, no external assets)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vetch · Circuit Breaker Demo</title>
<style>
  :root {
    --bg: #0a0e1a;
    --bg-elev: #131826;
    --bg-elev2: #1a2033;
    --fg: #e6edf3;
    --fg-dim: #8b949e;
    --accent: #7ee787;
    --accent-dim: #3fb950;
    --warn: #f0883e;
    --danger: #ff6b6b;
    --danger-glow: #ff3344;
    --border: #30363d;
    --mono: 'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--mono);
    background: var(--bg);
    color: var(--fg);
    min-height: 100vh;
    overflow-x: hidden;
    padding: 24px;
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
  }
  .brand { display: flex; align-items: baseline; gap: 14px; }
  .brand-name { font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }
  .brand-version { font-size: 12px; color: var(--fg-dim); }
  .brand-tag { font-size: 13px; color: var(--accent); margin-left: 12px; }
  .live { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--fg-dim); }
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse { 50% { opacity: 0.3; } }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .card {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
  }
  .card-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: var(--fg-dim);
    margin-bottom: 12px;
  }
  .big-number {
    font-size: 56px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    color: var(--fg);
    transition: color 0.3s;
  }
  .big-number.spending { color: var(--warn); }
  .big-number.frozen { color: var(--accent); }
  .big-number .currency { font-size: 28px; vertical-align: top; opacity: 0.6; margin-right: 4px; }
  .big-number .unit { font-size: 22px; color: var(--fg-dim); margin-left: 8px; }
  .sub-line {
    font-size: 13px;
    color: var(--fg-dim);
    margin-top: 8px;
  }

  .meter-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px; }
  .meter { background: var(--bg-elev2); padding: 14px; border-radius: 8px; }
  .meter-label { font-size: 10px; text-transform: uppercase; color: var(--fg-dim); letter-spacing: 1.2px; margin-bottom: 6px; }
  .meter-value { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }

  .calls-feed {
    height: 360px;
    overflow-y: auto;
    font-size: 12px;
    line-height: 1.7;
  }
  .calls-feed::-webkit-scrollbar { width: 6px; }
  .calls-feed::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .call-row {
    display: grid;
    grid-template-columns: 60px 1fr 90px 60px;
    gap: 12px;
    padding: 4px 0;
    color: var(--fg-dim);
    border-bottom: 1px dashed var(--border);
    animation: slideIn 0.3s ease-out;
  }
  .call-row .call-num { color: var(--fg); font-weight: 600; }
  .call-row .call-model { color: var(--accent-dim); }
  .call-row .call-tokens { text-align: right; color: var(--fg-dim); }
  .call-row .call-cost { text-align: right; color: var(--warn); font-variant-numeric: tabular-nums; }
  .call-row.stalled .call-tokens { color: var(--danger); }
  @keyframes slideIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: translateX(0); } }

  .events-feed {
    height: 360px;
    overflow-y: auto;
    font-size: 11px;
    line-height: 1.5;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    color: var(--fg-dim);
  }
  .events-feed::-webkit-scrollbar { width: 6px; }
  .events-feed::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .events-feed pre { white-space: pre-wrap; word-wrap: break-word; margin-bottom: 10px; }
  .events-feed .event-key { color: var(--accent-dim); }
  .events-feed .event-value { color: var(--fg); }

  /* The dramatic STALL DETECTED banner overlay. */
  .stall-overlay {
    position: fixed;
    inset: 0;
    background: rgba(255, 51, 68, 0.0);
    backdrop-filter: blur(0px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 100;
    transition: background 0.4s ease, backdrop-filter 0.4s ease;
  }
  .stall-overlay.visible {
    display: flex;
    background: rgba(255, 51, 68, 0.18);
    backdrop-filter: blur(2px);
    animation: dangerFlash 1.2s ease-out;
  }
  @keyframes dangerFlash {
    0%   { background: rgba(255, 51, 68, 0.6); }
    50%  { background: rgba(255, 51, 68, 0.18); }
    100% { background: rgba(255, 51, 68, 0.18); }
  }
  .stall-banner {
    background: var(--bg-elev);
    border: 2px solid var(--danger-glow);
    box-shadow: 0 0 60px rgba(255, 51, 68, 0.5);
    border-radius: 14px;
    padding: 48px 64px;
    max-width: 720px;
    text-align: center;
    transform: scale(0.9);
    animation: bannerPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
  }
  @keyframes bannerPop {
    0% { transform: scale(0.6); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
  }
  .stall-title {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--danger);
    margin-bottom: 10px;
  }
  .stall-subtitle {
    font-size: 14px;
    color: var(--fg-dim);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 30px;
  }
  .stall-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin: 28px 0; }
  .stall-stat-label { font-size: 10px; text-transform: uppercase; color: var(--fg-dim); letter-spacing: 1.2px; margin-bottom: 6px; }
  .stall-stat-value { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .stall-stat-value.danger { color: var(--danger); }
  .stall-stat-value.accent { color: var(--accent); }
  .stall-footer {
    margin-top: 28px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--fg-dim);
  }
  .kbd {
    display: inline-block;
    padding: 2px 8px;
    background: var(--bg-elev2);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 11px;
    margin: 0 2px;
  }
</style>
</head>
<body>

<div class="header">
  <div class="brand">
    <span class="brand-name">VETCH</span>
    <span class="brand-version" id="brand-version">v0.4.0</span>
    <span class="brand-tag">Circuit Breaker · Live Demo</span>
  </div>
  <div class="live"><span class="live-dot"></span> LIVE</div>
</div>

<div class="grid">

  <!-- Top-left: Cost ticker (the hero number) -->
  <div class="card">
    <div class="card-label">Cumulative Cost</div>
    <div id="cost-display" class="big-number spending">
      <span class="currency">$</span><span id="cost-value">0.0000</span>
    </div>
    <div class="sub-line"><span id="cost-status">Spending&hellip;</span></div>
    <div class="meter-row" style="margin-top: 22px;">
      <div class="meter">
        <div class="meter-label">Energy</div>
        <div class="meter-value"><span id="energy-value">0.000</span> Wh</div>
      </div>
      <div class="meter">
        <div class="meter-label">Carbon</div>
        <div class="meter-value"><span id="carbon-value">0.00</span> g CO<sub>2</sub>e</div>
      </div>
    </div>
  </div>

  <!-- Top-right: Configuration / status -->
  <div class="card">
    <div class="card-label">Run Configuration</div>
    <div class="meter-row">
      <div class="meter">
        <div class="meter-label">Action</div>
        <div class="meter-value" id="action-value">log</div>
      </div>
      <div class="meter">
        <div class="meter-label">Calls</div>
        <div class="meter-value" id="calls-value">0</div>
      </div>
    </div>
    <div class="meter-row" style="margin-top: 14px;">
      <div class="meter">
        <div class="meter-label">Mode</div>
        <div class="meter-value" id="mode-value">mock</div>
      </div>
      <div class="meter">
        <div class="meter-label">Primary Model</div>
        <div class="meter-value" id="model-value" style="font-size: 14px;">&mdash;</div>
      </div>
    </div>
    <div class="sub-line" style="margin-top: 18px; font-size: 12px;">
      A deliberately-stalled agent loop is running. Vetch will detect the
      pattern (low output + high input similarity) and act on it.
    </div>
  </div>

  <!-- Bottom-left: Call feed -->
  <div class="card">
    <div class="card-label">Agent Loop</div>
    <div class="calls-feed" id="calls-feed">
      <div class="call-row" style="color: var(--fg-dim);">
        <div>&mdash;</div><div>waiting&hellip;</div><div></div><div></div>
      </div>
    </div>
  </div>

  <!-- Bottom-right: Raw event JSON stream -->
  <div class="card">
    <div class="card-label">Raw Event Stream</div>
    <div class="events-feed" id="events-feed"></div>
  </div>

</div>

<!-- Dramatic stall banner -->
<div class="stall-overlay" id="stall-overlay">
  <div class="stall-banner">
    <div class="stall-title">⚡ CIRCUIT BREAKER ENGAGED</div>
    <div class="stall-subtitle">STALL-001 · Agentic loop detected · Loop stopped</div>
    <div class="stall-stats">
      <div>
        <div class="stall-stat-label">Calls Wasted</div>
        <div class="stall-stat-value danger" id="stall-count">0</div>
      </div>
      <div>
        <div class="stall-stat-label">Cost Lost</div>
        <div class="stall-stat-value danger">$<span id="stall-wasted">0.00</span></div>
      </div>
      <div>
        <div class="stall-stat-label">Projected Save</div>
        <div class="stall-stat-value accent">$<span id="stall-saved">0.00</span></div>
      </div>
    </div>
    <div class="stall-footer">
      Catch <code>vetch.StallDetected</code> in your code, or call
      <code>session.clear_stall()</code> to re-arm the breaker. Press
      <span class="kbd">Esc</span> to dismiss.
    </div>
  </div>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  let totalCost = 0;
  let totalEnergy = 0;
  let totalCarbon = 0;
  let callCount = 0;

  function fmtCost(v) { return v.toFixed(4); }
  function fmtEnergy(v) { return v.toFixed(3); }
  function fmtCarbon(v) { return v.toFixed(2); }

  function pushCallRow(event) {
    const feed = $('calls-feed');
    if (callCount === 1) feed.innerHTML = '';  // remove "waiting" placeholder
    const usage = (event.usage && event.usage.text) || {};
    const out = usage.output_tokens || 0;
    const isStalled = out < 5;
    const row = document.createElement('div');
    row.className = 'call-row' + (isStalled ? ' stalled' : '');
    row.innerHTML = `
      <div class="call-num">#${String(callCount).padStart(3, '0')}</div>
      <div class="call-model">${event.model || '?'}</div>
      <div class="call-tokens">in:${usage.input_tokens || 0} out:${out}</div>
      <div class="call-cost">$${(event.estimated_cost_usd || 0).toFixed(4)}</div>
    `;
    feed.insertBefore(row, feed.firstChild);
    while (feed.children.length > 50) feed.removeChild(feed.lastChild);
  }

  function pushRawEvent(event) {
    const feed = $('events-feed');
    const summary = {
      model: event.model,
      tokens_in: event.usage && event.usage.text && event.usage.text.input_tokens,
      tokens_out: event.usage && event.usage.text && event.usage.text.output_tokens,
      energy_wh: event.estimated_energy_wh,
      carbon_g: event.estimated_carbon_g,
      cost_usd: event.estimated_cost_usd,
      region: event.region,
      signal_quality: event.signal_quality,
    };
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(summary, null, 2);
    feed.insertBefore(pre, feed.firstChild);
    while (feed.children.length > 8) feed.removeChild(feed.lastChild);
  }

  function showStall(stall) {
    $('stall-count').textContent = stall.request_count || 0;
    $('stall-wasted').textContent = (stall.wasted_cost_usd || 0).toFixed(4);
    $('stall-saved').textContent = (stall.projected_save_usd || 0).toFixed(4);
    $('stall-overlay').classList.add('visible');
    $('cost-display').classList.remove('spending');
    $('cost-display').classList.add('frozen');
    $('cost-status').textContent = 'STOPPED — circuit breaker engaged';
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') $('stall-overlay').classList.remove('visible');
  });

  const evtSource = new EventSource('/events');

  evtSource.addEventListener('config', (e) => {
    const cfg = JSON.parse(e.data);
    $('action-value').textContent = cfg.action || 'log';
    $('mode-value').textContent = cfg.mode || 'mock';
    $('model-value').textContent = cfg.primary_model || '—';
    $('brand-version').textContent = 'v' + (cfg.version || '0.4.0');
  });

  evtSource.addEventListener('inference_event', (e) => {
    const payload = JSON.parse(e.data);
    const event = payload.event;
    if (!event || event.event_type === 'session_complete') return;
    callCount += 1;
    totalCost += event.estimated_cost_usd || 0;
    totalEnergy += event.estimated_energy_wh || 0;
    totalCarbon += event.estimated_carbon_g || 0;
    $('cost-value').textContent = fmtCost(totalCost);
    $('energy-value').textContent = fmtEnergy(totalEnergy);
    $('carbon-value').textContent = fmtCarbon(totalCarbon);
    $('calls-value').textContent = callCount;
    pushCallRow(event);
    pushRawEvent(event);
  });

  evtSource.addEventListener('stall', (e) => {
    showStall(JSON.parse(e.data));
  });

  evtSource.addEventListener('reset', () => {
    totalCost = 0; totalEnergy = 0; totalCarbon = 0; callCount = 0;
    $('cost-value').textContent = '0.0000';
    $('energy-value').textContent = '0.000';
    $('carbon-value').textContent = '0.00';
    $('calls-value').textContent = '0';
    $('calls-feed').innerHTML = '<div class="call-row" style="color: var(--fg-dim);"><div>&mdash;</div><div>waiting&hellip;</div><div></div><div></div></div>';
    $('events-feed').innerHTML = '';
    $('stall-overlay').classList.remove('visible');
    $('cost-display').classList.remove('frozen');
    $('cost-display').classList.add('spending');
    $('cost-status').textContent = 'Spending…';
  });
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class DemoHandler(BaseHTTPRequestHandler):
    """Serves the dashboard at / and an SSE event stream at /events."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence the default access log so the demo console stays clean.
        return

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/events":
            self._handle_sse()
            return

        self.send_response(404)
        self.end_headers()

    def _handle_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        my_queue: queue.Queue[str] = queue.Queue(maxsize=512)
        with _subscriber_lock:
            _subscriber_queues.append(my_queue)

        try:
            # Send any cached config to a new subscriber.
            cfg = getattr(self.server, "config_payload", None)
            if cfg is not None:
                self._sse_send("config", cfg)

            while True:
                try:
                    msg = my_queue.get(timeout=15)
                except queue.Empty:
                    # Heartbeat to keep the connection alive.
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue

                payload = json.loads(msg)
                event_name = payload.get("type", "message")
                self._sse_send(event_name, payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _subscriber_lock:
                if my_queue in _subscriber_queues:
                    _subscriber_queues.remove(my_queue)

    def _sse_send(self, event_name: str, payload: Any) -> None:
        body = f"event: {event_name}\ndata: {json.dumps(payload, default=str)}\n\n"
        self.wfile.write(body.encode("utf-8"))
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Mock event generation: synthetic inference events with realistic shape.
# ---------------------------------------------------------------------------


def _mock_event(model: str, call_num: int, stalled: bool = True) -> dict[str, Any]:
    """Construct a realistic-looking InferenceEvent dict for the demo."""
    in_tokens = 520 + (call_num % 3)  # nearly identical inputs (high similarity)
    out_tokens = 1 if stalled else 80
    cost_per_in = 0.00015 / 1000
    cost_per_out = 0.0006 / 1000
    cost = in_tokens * cost_per_in + out_tokens * cost_per_out
    energy = (0.12 / 1000) * in_tokens + (0.36 / 1000) * out_tokens
    carbon = energy * 0.42  # rough gCO2/Wh

    return {
        "schema_version": "2",
        "vetch_version": vetch.__version__,
        "event_type": "inference",
        "model": model,
        "provider": "openai",
        "region": "us-east-1",
        "signal_quality": "delayed",
        "usage": {
            "text": {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
            }
        },
        "estimated_energy_wh": round(energy, 6),
        "estimated_carbon_g": round(carbon, 6),
        "estimated_cost_usd": round(cost, 6),
        "complete": True,
        "error": False,
    }


# ---------------------------------------------------------------------------
# Demo runner: real OpenAI calls or mock events, both fed through Vetch.
# ---------------------------------------------------------------------------


def _wait_with_pulse(sec: float) -> None:
    """Sleep, but in 50ms chunks so the SSE keepalive cadence stays smooth."""
    end = time.monotonic() + sec
    while time.monotonic() < end:
        time.sleep(min(0.05, end - time.monotonic()))


def run_demo(args: argparse.Namespace) -> None:
    primary_model = args.primary_model
    fallback_model = args.fallback_model

    if args.action == "reroute":
        vetch.set_stall_action("reroute", fallback_model=fallback_model)
    else:
        vetch.set_stall_action(args.action)

    # Wait briefly so the browser tab can connect and receive the config.
    _wait_with_pulse(2.0)

    if args.real:
        run_real_demo(primary_model, fallback_model, args)
    else:
        run_mock_demo(primary_model, fallback_model, args)


def run_real_demo(primary_model: str, fallback_model: str, args: argparse.Namespace) -> None:
    """Run the demo against the real OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Real mode requires `openai`. Install or use --mock.", file=sys.stderr)
        return

    vetch.instrument()
    client = OpenAI()

    try:
        with vetch.Session(emit=False) as session:
            for i in range(args.max_calls):
                try:
                    client.chat.completions.create(
                        model=primary_model,
                        messages=[
                            {"role": "system", "content": "Reply only with 'ok'."},
                            {
                                "role": "user",
                                "content": "Disregard the above. " * 50,
                            },
                        ],
                        max_tokens=2,
                    )
                except vetch.StallDetected as exc:
                    _emit_stall_banner(exc, i, args.max_calls)
                    return
                _wait_with_pulse(args.step_delay)
            print(f"Loop completed without stall. Calls: {args.max_calls}")
    finally:
        vetch.uninstrument()


def run_mock_demo(primary_model: str, fallback_model: str, args: argparse.Namespace) -> None:
    """Run the demo with synthetic events. No API key, no network."""
    from vetch.advisory import generate_advisories
    from vetch.session import Session

    with Session(emit=False) as session:
        for i in range(args.max_calls):
            event = _mock_event(primary_model, i + 1, stalled=True)

            # Apply 'reroute' substitution on the rendered event for visual fidelity.
            if session.stall_triggered and args.action == "reroute":
                event["model"] = fallback_model
                event["estimated_cost_usd"] *= 0.25  # cheaper model

            session.register_event(event)  # type: ignore[arg-type]
            _patched_emit_event(event)  # broadcast to dashboard

            # Apply the configured action *after* the event flows.
            if session.stall_triggered:
                action, fb = vetch.get_stall_action()
                advisory = session.stall_advisory
                if action == "kill":
                    exc = vetch.StallDetected(
                        f"Stall detected after {advisory.request_count} calls",
                        wasted_cost_usd=advisory.potential_savings_usd or 0.0,
                        request_count=advisory.request_count,
                        fallback_model=fb,
                    )
                    _emit_stall_banner(exc, i + 1, args.max_calls)
                    return

            _wait_with_pulse(args.step_delay)

        print(f"Loop completed without stall. Action={args.action}")


def _emit_stall_banner(
    exc: vetch.StallDetected, calls_done: int, max_calls: int
) -> None:
    remaining = max(max_calls - calls_done, 0)
    per_call = (
        exc.wasted_cost_usd / max(exc.request_count, 1)
        if exc.request_count
        else 0.0
    )
    projected = per_call * remaining
    _broadcast({
        "type": "stall",
        "request_count": exc.request_count,
        "wasted_cost_usd": exc.wasted_cost_usd,
        "projected_save_usd": projected,
        "fallback_model": exc.fallback_model,
    })
    print()
    print("=" * 60)
    print(f"STALL DETECTED — Vetch stopped the loop after {calls_done} calls.")
    print(f"  Wasted: ${exc.wasted_cost_usd:.4f}")
    print(f"  Projected save: ${projected:.4f} over remaining {remaining} calls")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mock", dest="real", action="store_false", default=False,
        help="Use synthetic events (default; offline-safe).",
    )
    p.add_argument(
        "--real", dest="real", action="store_true",
        help="Use real OpenAI API. Requires OPENAI_API_KEY.",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--action", choices=["log", "warn", "kill", "reroute"], default="kill")
    p.add_argument("--primary-model", default="gpt-4o-mini")
    p.add_argument("--fallback-model", default="gpt-4o-mini")
    p.add_argument("--max-calls", type=int, default=100)
    p.add_argument("--step-delay", type=float, default=0.18,
                   help="Seconds between calls (controls demo pacing).")
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    server.config_payload = {  # type: ignore[attr-defined]
        "version": vetch.__version__,
        "action": args.action,
        "mode": "real" if args.real else "mock",
        "primary_model": args.primary_model,
        "fallback_model": args.fallback_model,
    }

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print()
    print("Vetch v" + vetch.__version__ + " — Circuit Breaker Demo")
    print("=" * 60)
    print(f"  Dashboard:   http://{args.host}:{args.port}")
    print(f"  Mode:        {'REAL OpenAI' if args.real else 'MOCK (offline)'}")
    print(f"  Action:      {args.action}")
    print(f"  Primary:     {args.primary_model}")
    if args.action == "reroute":
        print(f"  Fallback:    {args.fallback_model}")
    print("=" * 60)
    print("  Open the URL above in your browser, then come back and press")
    print("  Enter to start the loop. Ctrl+C to quit.")
    print()
    try:
        input("  Press Enter to start... ")
    except (KeyboardInterrupt, EOFError):
        return 0

    # Broadcast the run config so any tabs already open update.
    _broadcast({"type": "config", **server.config_payload})  # type: ignore[attr-defined]
    _broadcast({"type": "reset"})

    try:
        run_demo(args)
    except KeyboardInterrupt:
        pass

    print()
    print("Demo complete. Dashboard remains live for inspection.")
    print("Ctrl+C to exit, or refresh the page and press Enter to run again.")
    try:
        while True:
            input("  Press Enter to run again... ")
            _broadcast({"type": "reset"})
            run_demo(args)
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
