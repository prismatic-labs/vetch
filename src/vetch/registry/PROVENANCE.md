# Vetch Tier 3 Energy Provenance

This document details the methodology and sources used for the Tier 3 (Estimated) values in `energy.json`.

## Methodology Overview

All Tier 3 values are derived using a **Proxy Scaling Model**. Since model providers do not currently publish per-inference energy data, we use three primary signals:

1.  **Parameter Count Proxies**: We map models to their estimated parameter counts (e.g., GPT-4 ~1.8T MoE, Llama-3-70B).
2.  **Hardware Efficiency Baselines**: We assume inference on NVIDIA H100 or A100 GPUs. Efficiency is calculated using a baseline of ~0.2 - 0.5 Joules per parameter per token, adjusted for MoE (Mixture of Experts) activation ratios.
3.  **The 1:3 Input-to-Output Heuristic**:
    *   **Input (Prefill)**: KV-cache population. This is compute-bound but highly parallelizable.
    *   **Output (Decode)**: Autoregressive generation. This is memory-bandwidth bound and requires one full forward pass per token.
    *   *Source*: Based on architectural analysis in *Pope et al. (2022) "Efficiently Scaling Transformer Inference"*.

## Model-Specific Baselines

### GPT-4 Family
*   **Source**: Estimates from *SemiAnalysis* blog and *Luccioni et al. (2023) "Power Hungry Processing"*.
*   **Logic**: Scaled from Bloom (175B) measurements, adjusted for the MoE architecture which reduces active parameters per token while maintaining high memory overhead.

### Claude 3 Family
*   **Source**: Benchmarked latency scaling relative to GPT-4.
*   **Logic**: Energy is correlated with latency under similar load. Claude 3 Opus displays latency patterns consistent with ~1.5x the compute intensity of GPT-4 Turbo.

### Llama 3 Family
*   **Source**: Meta's published training energy, reverse-scaled for inference ratios.
*   **Logic**: Uses official 70B parameter count as the anchor point for the registry's linear scaling.

## Uncertainty Margin
**Confidence Level: Low (order of magnitude)**

### The 1.2x Hardware Overhead Multiplier

Registry values represent **IT equipment energy** (GPU + server overhead), *not* total facility energy. PUE is applied separately in `calculate_carbon()`.

To account for non-GPU server components, we apply a **1.2x Hardware Overhead Multiplier** to raw GPU benchmarks:

*   **Networking, CPU host coordination, and memory-bandwidth idle draw**: 1.2x

This multiplier covers the energy consumed by server components beyond the GPU itself (CPUs, DRAM, NVLink, host networking). It does **not** include data center cooling or power distribution — those are accounted for by the PUE factor applied during carbon calculation.

**PUE (Power Usage Effectiveness)** is applied separately in `calculate_carbon()` with a configurable default of 1.2, based on:
*   Google: 1.09 (fleet average, 2024)
*   AWS: 1.14 (Jiang et al. 2025)
*   Azure: 1.12 (Jiang et al. 2025)
*   Industry average: ~1.58 (Uptime Institute 2023)

Override via `VETCH_DEFAULT_PUE` environment variable.

### Model Assumptions and Limitations

*   Energy per token is modeled as **independent of context length**. In reality, output token energy increases with KV-cache size (longer inputs → more expensive outputs). This is a known simplification.
*   **Batch size effects** are not modeled. Higher batch sizes improve throughput efficiency but are provider-controlled.

These values are intended for **relative comparison** and **trend analysis**, not audit-grade reporting.

## References

*   Pope, R. et al. (2022). "Efficiently Scaling Transformer Inference." MLSys 2023. arXiv:2211.05102
*   Luccioni, A.S. et al. (2023). "Power Hungry Processing: Watts Driving the Cost of AI Deployment?" FAccT 2024. arXiv:2311.16863
*   Jiang, S. et al. (2025). "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference." arXiv:2505.09598
*   Uptime Institute (2023). Global Data Center Survey.
