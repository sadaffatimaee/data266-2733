# HW6 Metrics

## Personal Parameters

- SID4: 2733
- SEED: 2733
- SLICE: 733
- HP_ID: 3
- CLS_A: 3
- CLS_B: 7

## Test Accuracy

| Part | Method | Test Accuracy |
|---|---|---:|
| A | Supervised ResNet-18 with 500 labeled images | 38.34% |
| B | Rotation SSL with frozen linear classifier | 37.66% |
| C | SimCLR with frozen linear classifier | 47.23% |

## Part D: Nearest Neighbors

- Query indices: 10, 150, and 300
- Embedding dimension: 512
- Similarity metric: cosine similarity
- Neighbors retrieved per query: 5
- Encoders compared: supervised, rotation SSL, and SimCLR

Part B and Part C used 12 pretraining epochs for computational efficiency. Each linear classifier was trained for 15 epochs with the encoder frozen.
