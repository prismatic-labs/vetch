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
*   **Source**: Estimates from *SemiAnalysis* and *Luccioni et al. (2023)*.
*   **Logic**: Scaled from Bloom (175B) measurements, adjusted for the MoE architecture which reduces active parameters per token while maintaining high memory overhead.

### Claude 3 Family
*   **Source**: Benchmarked latency scaling relative to GPT-4.
*   **Logic**: Energy is correlated with latency under similar load. Claude 3 Opus displays latency patterns consistent with ~1.5x the compute intensity of GPT-4 Turbo.

### Llama 3 Family
*   **Source**: Meta's published training energy, reverse-scaled for inference ratios.
*   **Logic**: Uses official 70B parameter count as the anchor point for the registry's linear scaling.

## Uncertainty Margin
**Confidence Level: Low (±10x)**

### The 1.3x System Multiplier
To ensure our estimates are defensible for FinOps but realistic for engineers, we apply a **1.3x System Multiplier** to raw GPU benchmarks (like those from Epoch AI). This is a cumulative (multiplicative) calculation:

*   **Hardware Overhead (1.2x)**: Networking, CPU host coordination, and memory-bandwidth idle draw.
*   **PUE (1.1x)**: Average data center cooling and power conversion.

**Total Calculation**: $1.2 \times 1.1 = 1.32$ (rounded to **1.3x** for the registry).

These values are intended for **relative comparison** and **trend analysis**.
