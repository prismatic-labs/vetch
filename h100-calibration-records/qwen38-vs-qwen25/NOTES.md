# Qwen3.8-27B vs Qwen2.5-32B — H100 energy (2026-08-14)

Same-box comparison on Vast.ai H100 SXM 80GB, vLLM 0.27.1, bf16, `max-model-len=8192`, unique prompts.

## Batch=1 Tier-0 grid (preserved)

File: `../self-hosted__qwen-qwen3.8-27b__h100-sxm-80gb__vllm__bf16-5cb685a8.json`

| field | value |
|---|---|
| wh_per_1k_output | 1.9021 |
| samples | 22 |
| idle drift | 1.0% |
| fit R² | 0.9943 |
| active | False (negative-intercept quality gate; audit artifact) |

Qwen3.8 served with `--default-chat-template-kwargs '{"enable_thinking": false}'`.

## Concurrency sweep (`vetch calibrate-cuda --batched`, experimental)

64 requests/level, 64 out tokens, C ∈ {1,2,4,8,16,32}.

| C | Qwen3.8-27B | Qwen2.5-32B | 3.8 / 2.5 |
|---|---:|---:|---:|
| 1 | 1.9746 | 2.4436 | 0.808 |
| 2 | 1.0076 | 1.2458 | 0.809 |
| 4 | 0.5188 | 0.6418 | 0.808 |
| 8 | 0.2854 | 0.3358 | 0.850 |
| 16 | 0.1706 | 0.1944 | 0.877 |
| 32 | 0.1078 | 0.1209 | 0.892 |

Amortization fits `Wh/1k_out ≈ a/C + b`:

| model | a | b | R² | C1→C32 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 1.9273 | 0.0451 | 1.0000 | 18.3x |
| Qwen2.5-32B | 2.4018 | 0.0423 | 1.0000 | 20.2x |

At batch-of-one, Qwen3.8-27B is 19.2% lower than Qwen2.5-32B (1.975 vs 2.444 Wh/1k).

## Artifacts

- `fig6_qwen38_vs_qwen25.png` — fig6-style comparison plot
- `curve.csv` — concurrency points
- `remote/` — per-C JSON records + run logs
- `qwen38_tier0.txt` — Tier-0 CLI transcript

Batched records are **not** Tier-0. The batch=1 grid record is the preserved measurement.
