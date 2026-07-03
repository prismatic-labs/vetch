# Vetch Quickstart: Local Models

Track energy, cost, and carbon for **locally-hosted LLMs** (Ollama, vLLM, llama.cpp) in 60 seconds.

---

## Why Track Local Models?

**Privacy**: Data never leaves your machine
**Cost**: Electricity vs. API fees (quantify CAPEX vs. OPEX)
**Carbon**: Your grid's carbon intensity, not a data center's
**Control**: Full ownership of infrastructure
**Offline-First**: Works completely offline (no internet required when using local registry)

Vetch measures your **actual local energy consumption** and compares it to cloud alternatives.

---

## Install

### 1. Install Vetch

```bash
pip install vetch
```

### 2. Install a Local Model Server

**Option A: Ollama (Recommended for beginners)**

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version
```

**Option B: vLLM (Advanced, production workloads)**

```bash
pip install vllm
```

**Option C: llama.cpp (C++ implementation)**

See [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) for installation.

---

## Quick Start: Track Ollama Calls

### 1. Start Ollama Server

```bash
# Pull a model (one-time setup)
ollama pull llama3.2:3b

# Verify it works
ollama run llama3.2:3b "Hello, world!"
```

### 2. Track with Vetch (One Line)

Ollama uses an **OpenAI-compatible API**, so Vetch works automatically:

```python
import vetch
from openai import OpenAI

# Configure Vetch for your local grid
# CRITICAL: Use YOUR actual location's region for accurate carbon calculations
vetch.instrument(region="us-west-1", tags={"env": "local", "gpu": "rtx-4090"})

# Point OpenAI client to Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require real keys
)

# Make a call - Vetch tracks it automatically!
response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}]
)

print(response.choices[0].message.content)
# ✅ Energy, cost, and carbon logged automatically
```

**Output example** (logged to stderr by default):

```json
{
  "estimated_cost_usd": 0.0000,
  "estimated_energy_wh": 0.0156,
  "estimated_carbon_g": 0.0078,
  "model": "llama3.2:3b",
  "provider": "ollama",
  "region": "us-west-1",
  "tier": 3,
  "tags": {"env": "local", "gpu": "rtx-4090"}
}
```

> **Note**: Vetch auto-detects `localhost:11434` (and the `OLLAMA_HOST` env var)
> and sets `provider="ollama"` automatically, so Tier-0 Ollama calibrations
> apply even when using the OpenAI-compat API. No extra configuration needed.

---

## Region Configuration (Critical!)

Unlike cloud APIs, local inference uses **your local electricity grid**. Set `region` to your actual location:

```python
# Find your region at https://app.electricitymaps.com/map
# Common examples:
vetch.instrument(region="us-west-1")      # California
vetch.instrument(region="us-east-1")      # Virginia
vetch.instrument(region="eu-west-1")      # Ireland
vetch.instrument(region="eu-central-1")   # Germany
vetch.instrument(region="ap-southeast-1") # Singapore
```

**Why this matters**: California's grid is ~50% renewable. Coal-heavy grids like Poland (`PL`) emit 10x more carbon per kWh. Accurate region = accurate carbon calculations.

### Real-Time Carbon Intensity (Optional but Recommended)

For **live grid data** instead of regional averages, get a free [Electricity Maps API key](https://api-portal.electricitymaps.com/):

```bash
# Set API key (free tier: 100 requests/day)
export ELECTRICITY_MAPS_API_KEY=your-api-key-here

# Vetch will now use real-time carbon intensity for your region
vetch.instrument(region="us-west-1")
```

**Example impact**: On a sunny day in California, solar output can drop carbon intensity from 250g/kWh to 100g/kWh. Real-time data captures this.

**Do I need this?**
- **No** for basic tracking (Vetch uses regional averages by default)
- **Yes** for:
  - Carbon accounting that needs to meet audit standards
  - Green scheduling (run batch jobs when grid is cleanest)
  - Real-time carbon dashboards
  - Comparing time-of-day carbon impact

**Cost**: Free tier (100 requests/day) is sufficient for most local use cases. Vetch caches responses for 15 minutes.

---

## GPU Calibration (Upgrade to Tier 0 Measurements)

Vetch defaults to **Tier 3 (Estimated)** energy values. For **Tier 0 (Measured)** accuracy using your actual GPU:

### Requirements

- NVIDIA GPU
- `pynvml` library:

```bash
pip install nvidia-ml-py3
```

### Best Practices for Calibration

**⚠️ CRITICAL**: Run calibration while your GPU is **completely idle**:
- Close games, browsers, and other GPU-intensive apps
- Stop any mining/rendering processes
- Disable desktop composition if possible (Linux)

**Why**: Background GPU activity inflates power measurements, resulting in inaccurate energy profiles for your model.

**Quantization Matters**: Different quantization levels (Q4, Q8, FP16) have vastly different power draws. Calibrate **each quantization separately**:

```bash
# Example: llama3.2:3b vs llama3.2:3b-q4_0
ollama pull llama3.2:3b        # FP16 (2x memory, ~1.5x power)
ollama pull llama3.2:3b-q4_0   # 4-bit quantized (0.5x memory, ~0.7x power)
```

### Calibrate Your Model

```python
from vetch.calibrate import calibrate_model, format_calibration_result
from openai import OpenAI

# Set up Ollama client
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Define a representative workload
def my_inference():
    response = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "Write a haiku about recursion."}]
    )
    # Return (input_tokens, output_tokens)
    usage = response.usage
    return usage.prompt_tokens, usage.completion_tokens

# Run calibration (measures actual GPU power draw)
result = calibrate_model(
    provider="ollama",
    model="llama3.2:3b",
    workload=my_inference,
    iterations=5  # More iterations = better accuracy
)

# View results
print(format_calibration_result(result))
```

**Output:**

```
Calibration Complete for llama3.2:3b (ollama)
Hardware: NVIDIA GeForce RTX 4090
----------------------------------------
Energy (Input):  0.0234 Wh/1k tokens
Energy (Output): 0.0702 Wh/1k tokens
Confidence:      Tier 0 (Measured)
```

**What happens**: Calibration results are saved to `~/.vetch/calibrations/` and automatically used for future calls with this model.

---

## Understanding Energy in Tangible Terms

**The "Space Heater" Metric**: Energy consumption becomes more intuitive when compared to everyday devices.

After tracking a few calls, check your total energy:

```python
# After running multiple inferences...
stats = vetch.get_session_stats()
total_wh = stats["total_energy_wh"]

# Convert to tangible comparisons
hours_of_100w_bulb = total_wh / 100
hours_of_laptop = total_wh / 50
hours_of_space_heater = total_wh / 1500

print(f"Your LLM usage consumed {total_wh:.2f} Wh")
print(f"That's equivalent to:")
print(f"  - Running a 100W lightbulb for {hours_of_100w_bulb:.1f} hours")
print(f"  - Charging a laptop {hours_of_laptop:.1f} times")
print(f"  - Running a space heater for {hours_of_space_heater*60:.0f} minutes")
```

**Example**: A 70B model running for 1 hour at 300W = 300 Wh:
- Same as running three 100W lightbulbs for 1 hour
- About 6 full laptop charges
- Costs ~$0.04 at $0.13/kWh (US average)
- Emits ~150g CO2e (US grid average)

---

## Headless Server & Homelab Setup

Running Ollama/vLLM on a dedicated server? Configure Vetch for background operation:

### Log to File (for Systemd services)

```bash
# In your service environment file or systemd unit
export VETCH_OUTPUT=/var/log/vetch/inference.jsonl
export VETCH_REGION=us-west-1
```

### Named Pipe for Real-Time Monitoring

```bash
# Create a named pipe
mkfifo /tmp/vetch-pipe

# Start a monitoring process
tail -f /tmp/vetch-pipe | jq '.estimated_energy_wh' &

# Configure Vetch to write to pipe
export VETCH_OUTPUT=/tmp/vetch-pipe
```

### Docker Setup

```dockerfile
# In your Dockerfile
ENV VETCH_OUTPUT=/app/logs/vetch.jsonl
ENV VETCH_REGION=us-east-1
ENV VETCH_CACHE_MODE=memory-only  # For serverless/ephemeral containers

VOLUME ["/app/logs"]
```

---

## Offline-First Operation

Vetch works **completely offline** using a local registry (no internet required):

```bash
# Freeze registry for air-gapped environments
vetch registry freeze --output /opt/vetch/registry.json

# Configure offline mode
export VETCH_REGISTRY_REMOTE=false
export VETCH_REGISTRY_PATH=/opt/vetch/registry.json
```

**Why this matters**:
- **Privacy**: No outbound network calls
- **Security**: Works in firewalled/air-gapped environments
- **Reliability**: No dependency on external services
- **Speed**: Zero cold-start latency from registry fetches

This is ideal for:
- Homelab setups
- Corporate environments with strict firewall rules
- Edge deployments
- High-security/compliance-driven infrastructure

---

## TCO Analysis: CAPEX vs. OPEX

**The CFO Question**: "Should we buy a $30k GPU server or keep paying OpenAI $2k/month?"

Vetch provides the data for **ROI analysis**:

### Step 1: Track Your Current Cloud Spend

```bash
# After running for a week
vetch report --days 7 --tags env=prod

# Example output:
# Total cost (7 days): $487.23
# Projected monthly: $2,088.41
# Projected annual: $25,061.00
```

### Step 2: Estimate Local Alternative

```bash
# Calculate local hardware costs
GPU_SERVER_CAPEX=30000        # One-time: Dell/Supermicro with 4x A100
MONTHLY_POWER_COST=150        # ~1.5 kW * 730 hrs * $0.13/kWh
MONTHLY_MAINTENANCE=200       # Sysadmin time, cooling, etc.

ANNUAL_OPEX=$((12 * (MONTHLY_POWER_COST + MONTHLY_MAINTENANCE)))
BREAK_EVEN_MONTHS=$((GPU_SERVER_CAPEX / (2088 - (MONTHLY_POWER_COST + MONTHLY_MAINTENANCE))))

echo "Break-even: $BREAK_EVEN_MONTHS months"
```

### Step 3: Use Vetch for Real-Time Tracking

```python
import vetch

# Tag by environment to separate prod vs. dev costs
vetch.instrument(region="us-east-1", tags={"env": "prod", "cost_center": "ai-services"})

# At end of quarter, generate financial report
quarterly_cost = vetch.query_usage(days=90)
print(f"Q1 AI Inference Spend: ${quarterly_cost['total_cost_usd']:.2f}")
```

**Common Break-Even Points** (based on Vetch customer data):

| Monthly Cloud Spend | Local Hardware | Break-Even |
|---------------------|----------------|------------|
| $500/mo | $10k (1x A100) | ~14 months |
| $2,000/mo | $30k (4x A100) | ~16 months |
| $10,000/mo | $150k (8x H100) | ~18 months |

**Key Variables**:
- Model quality requirements (can you use Llama 3.2 vs. GPT-4?)
- Peak load patterns (do you need burst capacity?)
- Electricity costs (datacenter rates vs. residential)
- Tax treatment (CAPEX depreciation vs. OPEX expensing)

---

## Verify & Compare

### Check Vetch Status

```bash
vetch status
```

### Estimate Without Running Code

```bash
# Estimate local model
vetch estimate --model llama3.2:3b --input-tokens 1000 --output-tokens 500

# Compare with cloud equivalent
vetch compare --models llama3.2:3b,gpt-4o-mini,claude-3-haiku --tokens 1000
```

**Example comparison output:**

```
Model Comparison (1000 input + 500 output tokens)
---------------------------------------------------
llama3.2:3b       $0.0000   0.0312 Wh   0.0156 gCO2e  [local]
gpt-4o-mini       $0.0002   0.0450 Wh   0.0360 gCO2e  [cloud]
claude-3-haiku    $0.0004   0.0580 Wh   0.0464 gCO2e  [cloud]
```

---

## vLLM Example

For production workloads with vLLM:

```python
import vetch
from openai import OpenAI

# vLLM also uses OpenAI-compatible API
vetch.instrument(region="us-east-1", tags={"env": "prod", "framework": "vllm"})

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✅ Tracked automatically
```

---

## Local vs. Cloud Decision Matrix

| Factor | Local (Ollama/vLLM) | Cloud (OpenAI/Anthropic) |
|--------|---------------------|--------------------------|
| **Privacy** | ✅ Data never leaves machine | ❌ Data sent to third party |
| **Cost** | Electricity (~$0.001-0.01/call) | API fees ($0.0001-0.10/call) |
| **Predictability** | 🎯 Fixed CAPEX + utility bill | 📈 Variable OPEX (bill shock risk) |
| **Latency** | ⚡ Sub-100ms (local GPU) | 🌐 200-1000ms (network + queue) |
| **Quality** | 📊 Varies (Llama 3.2, Mistral) | 🏆 Best-in-class (GPT-4, Claude) |
| **Scalability** | 🖥️ Limited by GPU capacity | ☁️ Infinite (rate-limited) |
| **Carbon** | 🌍 Depends on local grid | 🏭 Depends on data center grid |
| **Offline** | ✅ Works without internet | ❌ Requires connectivity |

**Use local when**: Privacy-critical, high-volume, cost-sensitive, or low-latency requirements.
**Use cloud when**: Need best quality, sporadic usage, or don't want infrastructure overhead.

---

## Per-Call Control

For granular tracking or multi-model setups:

```python
from vetch import wrap

# Track specific calls with different tags
with wrap(region="us-west-1", tags={"model_size": "3b", "task": "summarization"}) as ctx:
    response = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "Summarize this article..."}]
    )

print(f"Energy: {ctx.event['estimated_energy_wh']:.4f} Wh")
print(f"Carbon: {ctx.event['estimated_carbon_g']:.4f} gCO2e")
print(f"Cost:   ${ctx.event['estimated_cost_usd']:.6f}")
```

---

## Session Aggregation (Multi-Step Workflows)

Track energy across multi-turn conversations or RAG pipelines:

```python
import vetch

with vetch.Session(tags={"workflow": "rag-pipeline"}) as session:
    # Step 1: Document retrieval LLM call
    with vetch.wrap(tags={"step": "retrieval"}) as ctx1:
        response1 = client.chat.completions.create(...)

    # Step 2: Answer generation LLM call
    with vetch.wrap(tags={"step": "generation"}) as ctx2:
        response2 = client.chat.completions.create(...)

# Total energy for entire pipeline
print(f"Total energy: {session.total_energy_wh:.4f} Wh")
print(f"Total calls: {session.call_count}")
```

---

## Configure Output

By default, events are logged to `stderr`. Configure with environment variables:

```bash
export VETCH_OUTPUT=/tmp/vetch-local.jsonl  # Log to file
export VETCH_OUTPUT=none                     # Silence output
```

---

## Emergency Kill Switch

```bash
export VETCH_DISABLED=true  # Completely disable Vetch
```

---

## Next Steps

Now that you're tracking local inference:

1. **[Calibrate your models](#gpu-calibration-upgrade-to-tier-0-measurements)** - Upgrade from Tier 3 to Tier 0 accuracy
2. **[Compare costs](README.md#cli-usage)** - Use `vetch compare` to quantify local vs. cloud savings
3. **[Session aggregation](README.md#session-aggregation-agentic-ai)** - Track multi-step agentic workflows
4. **[Green scheduling](README.md#green-signal-api)** - Run batch jobs when grid carbon is lowest
5. **[OTLP export](README.md#otlp-export)** - Send metrics to Grafana/Datadog for dashboards
6. **[Energy methodology](src/vetch/METHODOLOGY.md)** - Understand estimate tiers and uncertainty

---

## Capability observability (agent frameworks)

When your agent framework does not pass `tools=` through a patched provider client
(LangGraph, CrewAI, OpenAI Agents SDK, etc.), supply tool metadata manually via
`capture()`:

```python
import vetch

with vetch.wrap() as ctx:
    response = my_agent.run(...)
    ctx.capture(
        model="gpt-4o",
        provider="openai",
        usage={"text": {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280}},
        tools_offered=[
            {"name": "get_weather", "kind": "function"},
            {"name": "refund_order", "kind": "function"},
        ],
        tools_invoked=[{"name": "get_weather", "kind": "function"}],
        tool_call_count=1,
    )

summary = vetch.get_session_stats().summary()
print(summary["function_tools_never_called"])
# Session total (per-request cost x requests that offered dead tools),
# cache-aware directional estimate. Per-request figure and count are also
# available as wasted_tool_schema_cost_per_request_usd / dead_tool_offer_request_count.
print(summary["wasted_tool_schema_cost_usd"])
```

Note: Vetch's `mcp/` server (`vetch-mcp`) is a FinOps calculator, not instrumentation
of your agent's MCP tool roster — use `capture()` for that path.

---

## Troubleshooting

**Ollama not responding:**

```bash
# Check if server is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

**GPU not detected for calibration:**

```bash
# Install NVIDIA monitoring library
pip install nvidia-ml-py3

# Verify GPU is visible
nvidia-smi
```

**Region unknown errors:**

```bash
# Always set region explicitly
export VETCH_REGION=us-west-1  # Match your actual location
```

**Want more accurate carbon data:**

```bash
# Get a free Electricity Maps API key for real-time grid carbon intensity
export ELECTRICITY_MAPS_API_KEY=your-key-here

# Verify it works
vetch estimate --model llama3.2:3b --input-tokens 100 --output-tokens 50
# Should show "Using live carbon intensity data" in debug output
```

---

**Questions or issues?** [github.com/prismatic-labs/vetch/issues](https://github.com/prismatic-labs/vetch/issues)

**Want to contribute calibration data?** See [CONTRIBUTING.md](CONTRIBUTING.md) for how to share Tier 0 measurements with the community.
