# Metrics – HW10 (K = 5)

| Comparison | Setting | P@5 | R@5 | MRR |
|---|---|---|---|---|
| Reranking | Without | 0.550 | 0.629 | 0.875 |
| | With | **0.775** | **0.919** | 0.750 |
| Chunking | S1 Fixed-size | 0.625 | 0.729 | 0.875 |
| | S2 Sentence-aware | **0.700** | **0.819** | 0.750 |
| Memory | Without | 0.275 | 0.396 | 0.438 |
| | With | **0.400** | **0.529** | **0.875** |

**Best:** S2 + reranking (P@5 0.80, R@5 0.938). Memory: C2 0.00 → 0.60 P@5, C1 0.55 → 0.20 (dilution).
