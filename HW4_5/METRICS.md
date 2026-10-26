# HW4.5 Summary 



## Setup

- **Dataset:** `EleutherAI/twitter-sentiment` (train split), the same as HW4.a
- **Rows used:** 733 to 5732 (5,000 consecutive rows starting at SLICE = 733)
- **Split:** 70/15/15 train/validation/test (3,500 / 750 / 750), SEED = 2733, same as HW4.a
- **Model:** bert-base-uncased, max length 128, AdamW lr 2e-5, batch 32, 3 epochs
- **GPU:** Google Colab Tesla T4 (14.6 GB). One GPU used, as allowed in the instructor's announcement. UUIDs are in `uuid/t4_uuid.txt` and `RUN_LOG.txt`.

## HW4.a baseline

Peak memory 3.73 GB · training time 221.63 s · test accuracy 0.7960

## Best configurations (Tesla T4)

| Goal | Configuration | Result |
|---|---|---|
| Lowest peak memory | Gradient checkpointing | 2.09 GB (vs 3.73 GB baseline) |
| Highest throughput | AMP + 8-bit optimiser | 162.17 samples/s (vs 51.96 baseline) |
| Fastest wall-clock | AMP + 8-bit optimiser | 68.95 s for 3 epochs (vs 217.41 s baseline) |

## Table HW4.5.1 — Tesla T4

| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |
|---|---:|---:|---:|---:|
| Baseline (HW4 settings) | 128 | 51.96 | 3.73 | 0.7920 |
| Mixed precision | 256 | 160.38 | 3.01 | 0.7920 |
| Gradient checkpointing | 512 | 35.73 | 2.09 | 0.7907 |
| Gradient accumulation (32 × 8) | 128 physical (effective 256) | 50.65 | 4.17 | 0.7973 |
| 8-bit optimiser | 128 | 49.78 | 3.09 | 0.7920 |
| Best combination (AMP + 8-bit) | 256 | 162.17 | 2.40 | 0.8027 |

## Part A note

Only one GPU (Tesla T4) was used

## Part C

Part C skipped which was optional

## Part E — Analysis

**1. GPU-hours for one full fine-tune**

"Full fine-tune" here means training on the whole dataset instead of my 5,000-row slice, with the same 70% training split and 3 epochs.

- Full dataset: 1,568,627 rows → 70% for training = 1,098,038 rows
- 3 epochs = 1,098,038 × 3 = **3,294,114 training samples**
- Time = samples ÷ measured throughput

| Configuration | Throughput | Time | GPU-hours |
|---|---:|---:|---:|
| Baseline (HW4 settings, fp32) | 51.96 samp/s | 3,294,114 ÷ 51.96 = 63,397 s | **≈ 17.6 h** |
| AMP + 8-bit optimiser | 162.17 samp/s | 3,294,114 ÷ 162.17 = 20,313 s | **≈ 5.6 h** |

Validation passes add a few percent on top. Check: my 5k slice took 217 s at baseline, and the full data is about 314× larger (1,098,038 ÷ 3,500), which gives the same ~17.6 h estimate after removing validation time.

**2. Configuration I would use**

**AMP + 8-bit optimiser, batch size 32, 3 epochs.**

- It was the fastest (3.1× the baseline) and had the best accuracy (0.8027), so nothing is lost.
- It cuts a full fine-tune from about 17.6 to about 5.6 GPU-hours.
- Peak memory is only 2.40 GB, so it fits easily on any lab GPU (T4 up to 32 GB).
- Batch 32 is enough: in Part A, throughput barely improves past 32.
- Gradient checkpointing is not needed. It saves memory we don't need and costs 31% speed.
- Gradient accumulation is not needed, because the batches we want already fit.
