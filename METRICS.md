# METRICS — HW2 (DATA266, SID4 = 2733)

Personal parameters (Sec 0.1): SID4=2733, SEED=2733, SLICE=733, HP_ID=3, CLS_A=3, CLS_B=7.
HW1 is the only homework with an HP_ID mapping, so HW2 only uses SEED and SLICE; HP_ID/CLS_A/CLS_B are stated but unused.

## Task 1 — Word2Vec transfer learning

Fine-tuned on an 8000-review IMDB subset starting at index SLICE=733.

| word   | cosine_sim(pretrained, fine-tuned) |
|--------|------------------------------------:|
| review | 0.498 |
| plot   | 0.641 |
| screen | 1.000 |
| score  | 1.000 |
| cast   | 1.000 |

Most shifted: review. Least shifted: cast (tied exactly with score, screen at 1.000000 — see AI_USE.md).

## Task 2 — RAG pipeline

| question | top-3 contains answer | first relevant rank |
|---|---|---|
| Who directed Inception? | Yes | 1 |
| In what year was The Matrix released? | Yes | 2 |
| Who directed The Godfather? | Yes | 2 |
| What is Parasite about? | Yes | 2 |
| Who composed the musical score for Jurassic Park? | Yes | 1 |

Retrieval Success Rate = 5/5 = 1.00

## Task 3 — Optimization techniques

CPU-only (no GPU on this Colab runtime); GPU columns N/A. Same model/data, batch=256, steps=150, 3 repeated trials per config after warm-up.

| technique | time_sec (mean±std) | peak RSS Δ MB (mean±std) | final loss (mean±std) |
|---|---:|---:|---:|
| tensor_creation_cpu_direct | 3.459 | — | — |
| tensor_creation_cpu_create_then_move | 2.123 | — | — |
| tensor_creation_gpu | N/A | N/A | N/A |
| weight_init_default | 3.196 ± 0.681 | 0.000 ± 0.000 | 0.6926 ± 0.0003 |
| weight_init_xavier | 2.005 ± 0.423 | 0.000 ± 0.000 | 0.4105 ± 0.0337 |
| weight_init_large_normal | 1.893 ± 0.411 | 0.016 ± 0.022 | NaN (diverged) |
| activation_checkpoint_off | 7.283 ± 2.617 | 0.000 ± 0.000 | 0.6932 ± 0.0001 |
| activation_checkpoint_on | 9.626 ± 4.048 | 0.025 ± 0.023 | 0.6932 ± 0.0001 |
| grad_accum_off_batch256 | 3.631 ± 0.363 | 0.000 ± 0.000 | 0.6926 ± 0.0003 |
| grad_accum_on_4x64 | 2.846 ± 0.506 | 0.016 ± 0.022 | 0.6926 ± 0.0003 |
| mixed_precision_off_fp32 | 3.752 ± 0.938 | 0.000 ± 0.000 | 0.6926 ± 0.0003 |
| mixed_precision_on_bf16 | 20.851 ± 2.195 | 0.000 ± 0.000 | 0.6926 ± 0.0003 |

Key findings: large_normal init diverges to NaN (real failure). Mixed precision bf16 is ~5.5x slower than fp32 on this CPU (no native BF16 support) — opposite of expected GPU behavior. Gradient accumulation matched single-batch loss almost exactly, confirming numerical equivalence.