# HW12 Metrics

SID4=2733 · SEED=2733 · SLICE=733 · HP_ID=3 · CLS_A=3 · CLS_B=7
Hardware: 1× Tesla T4 (Kaggle) · torch 2.15.0 · transformers 5.23.0 · gptqmodel 7.12.0

## Part 1 — GPTQ

Model: `facebook/opt-125m` @ `27dcfa74d334bc871f3234de431e71c6eeba5dd6`
GPTQ 4-bit, group 128, calibrated on Wikitext2 train. PPL on Wikitext2 test. Latency: 128-token prompt + 64 new tokens, 5 warm-up + 30 runs.

### Table HW11.1 — GPTQ comparison on the same LLM/task

| Configuration | Wikitext2 PPL | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Base LLM (fp16) | 23.69 | 548.4 / 576.0 | 0.248 | 238.9 |
| GPTQ-quantized LLM | 28.41 | 1012.6 / 1035.0 | 0.147 | 122.7 |

### Findings
- **Quality:** PPL +20% — a small model has little redundancy to absorb 4-bit error.
- **Size:** ~2× smaller, not 4×, because the fp16 embedding table (~77 MB) isn't quantized.
- **Memory:** −41%, matching the weight savings.
- **Latency:** ~1.85× slower — dequantization overhead dominates on a tiny model, and the fast Marlin kernel needs an Ampere GPU (T4 is Turing), so ExLlamaV2 was used.

### Error analysis
- Marlin kernel failed to compile on the T4 → forced ExLlamaV2.
- An earlier run showed GPTQ memory at 269 MB because a failed load left tensors on the GPU; a clean restart gave the true 150 MB.

**Takeaway:** GPTQ helps memory/storage-bound deployment here, not latency-bound deployment.

## Part 2 — Distillation
*(to be added)*

## Quantization vs distillation
*(to be added)*
