# HW12 Metrics

SID4=2733 · SEED=2733 · SLICE=733 · HP_ID=3 · CLS_A=3 · CLS_B=7
Hardware: 1× Tesla T4 (Kaggle), GPU 0 only.
Part 1: torch 2.15.0 · transformers 5.23.0 · gptqmodel 7.12.0 — Part 2: torch 2.11.0 · transformers 5.20.2 · datasets 4.8.5

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

Teacher: provided BERT-base (`teacher_model`, 109.5M params), loaded locally. Logits extracted for all 120,000 AG News training samples (teacher train acc 98.95%).
Student: `distilbert/distilbert-base-uncased` (67.0M params), fresh from the same checkpoint with SEED for every run.
Loss: `loss = α·CE(student, labels) + (1−α)·KL(student/T ‖ teacher/T)·T²` (T² keeps gradient scale comparable across T — Hinton et al.).
Fixed settings: 1 epoch, batch 64, lr 5e-5, warmup 6%, wd 0.01, max_len 128, fp16. Best T/α chosen on a 5,000-sample validation split from train; test accuracy reported for every run.
Training log: `RUN_LOG_HW12.txt` (Part 2 section). Loss trends: `HW12_loss_trends.png`.

### Temperature sweep (α = 0.5)

| T | Val acc | Test acc |
|---|---|---|
| **1** | **0.9418** | 0.9433 |
| 2 | 0.9384 | 0.9433 |
| 5 | 0.9406 | 0.9432 |
| 10 | 0.9416 | 0.9418 |

Best T = 1. All four are within 0.3 pts — about the noise level of a 5k validation set, so temperature had little effect here.

### Loss-weighting sweep (T = 1, three fresh students)

| α | Val acc | Test acc |
|---|---|---|
| 0.25 | 0.9406 | 0.9430 |
| **0.5** | **0.9418** | **0.9433** |
| 0.75 | 0.9408 | 0.9420 |

Best student: T = 1, α = 0.5. The α = 0.5 run reproduced the T-sweep run exactly (identical loss at every logged step) — SEED makes training deterministic.

### Table HW11.2 — the separate distillation comparison on AG News

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Teacher (BERT-base) | 0.9443 | 13.57 / 15.05 | 0.779 | 418.6 |
| Best distilled student | 0.9434 | 6.72 / 7.17 | 0.514 | 256.1 |

Latency = batch-1, 300 test texts after 20 warm-ups, fp16. Full test set (7,600): teacher 12.2 s, student 6.7 s.

**Student vs teacher:** keeps 99.9% of teacher accuracy (−0.09 pts) while being ~2× faster, using 34% less GPU memory, and 39% less disk and parameters.

### Where the compression hurts (per-class accuracy)

| Class | Teacher | Student | Change (pts) |
|---|---|---|---|
| World | 0.9489 | 0.9447 | −0.42 |
| Sports | 0.9884 | 0.9884 | 0.00 |
| **Business** | 0.9137 | 0.9053 | **−0.84** |
| Sci/Tech | 0.9263 | 0.9353 | +0.89 |

Business degrades most. 55 Business articles are teacher-right / student-wrong; the student sends 45 of them to Sci/Tech.

Five examples (true = Business, teacher correct, student wrong):
1. *Two top Nokia executives resign …* → Sci/Tech
2. *Botswana miners 'face dismissal' …* → World
3. *Netflix, TiVo sign VoD alliance …* → Sci/Tech
4. *Guilty Plea Seen in Computer Associates Case …* → Sci/Tech
5. *Nortel Downsizes Again … eliminate 3,500 jobs …* → Sci/Tech

**What they have in common:** four of five are business events (resignations, deals, lawsuits, layoffs) at technology companies. The student keys on tech vocabulary ("networks", "mobile phone", "computer") over the business framing, shifting the Business/Sci-Tech boundary toward Sci/Tech — which is also why Sci/Tech accuracy rose.

### Checkpoint
Best student (T1, α0.5): `best_student_T1_a0.5.zip` (236 MB) — link: https://drive.google.com/file/d/1yv0aqi4rCNr59v8GNHTbqF2zsG0TPx5j/view?usp=drive_link
Reproduce: run `HW12_part2_distill.ipynb` top to bottom on a Kaggle T4 (`train_student(T=1, alpha=0.5, tag="T1_a0.5")`).

## Quantization vs distillation

GPTQ compresses an existing model after training in under a minute, while distillation trains a new, smaller architecture (~5 min per student here). On the T4, GPTQ halved disk size (238.9 → 122.7 MB) and cut peak memory by 41%, but generation became ~1.85× slower because weights are dequantized on the fly without Marlin support. Distillation removed 39% of the parameters, which halved latency (p50 13.57 → 6.72 ms) and cut memory by 34% and disk by 39%. Each lost a little quality on its own task, but the two quality numbers measure different tasks and are not comparable. For a latency-bound deployment I would consider distillation first, because fewer layers mean less compute on any hardware; for a memory-bound deployment I would consider GPTQ first, because it shrinks weights without retraining and can also be applied on top of a distilled model.
