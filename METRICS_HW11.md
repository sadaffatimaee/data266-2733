# HW11 Metrics

SID4 = 2733 | SEED = 2733 | Model: CLIP ViT-B/32 (openai/clip-vit-base-patch32) | GPU: Tesla T4

## Part 1: Similarity to "a photo of a dog" (100 images per class)

| Class | Mean | Std | Min | Max |
|---|---|---|---|---|
| dog | 0.2594 | 0.0120 | 0.2209 | 0.2855 |
| cat | 0.2394 | 0.0133 | 0.2057 | 0.2681 |
| deer | 0.2344 | 0.0137 | 0.2059 | 0.2743 |
| frog | 0.2291 | 0.0132 | 0.1941 | 0.2599 |
| bird | 0.2253 | 0.0133 | 0.1982 | 0.2751 |
| horse | 0.2201 | 0.0141 | 0.1920 | 0.2648 |
| airplane | 0.2035 | 0.0156 | 0.1521 | 0.2318 |
| automobile | 0.2015 | 0.0112 | 0.1726 | 0.2254 |
| truck | 0.1951 | 0.0100 | 0.1711 | 0.2228 |
| ship | 0.1951 | 0.0115 | 0.1691 | 0.2236 |

Top 20 matches: 18 dogs, 1 deer, 1 bird. Bottom 20: all vehicles.

## Part 2: Zero-shot CIFAR-10 (10,000 test images)

Prompt: "a photo of a {class}". **Overall accuracy: 88.80%**

| Class | Accuracy |
|---|---|
| automobile | 97.6% |
| horse | 97.6% |
| ship | 95.5% |
| airplane | 90.1% |
| truck | 90.1% |
| bird | 88.8% |
| dog | 88.4% |
| cat | 86.0% |
| frog | 77.4% |
| deer | 76.5% |

### Confusion matrix (rows = true, columns = predicted)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **airplane** | 901 | 12 | 51 | 0 | 0 | 2 | 1 | 1 | 25 | 7 |
| **automobile** | 3 | 976 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 18 |
| **bird** | 9 | 4 | 888 | 21 | 23 | 12 | 6 | 36 | 1 | 0 |
| **cat** | 3 | 7 | 20 | 860 | 8 | 79 | 8 | 11 | 4 | 0 |
| **deer** | 3 | 1 | 31 | 17 | 765 | 20 | 6 | 154 | 3 | 0 |
| **dog** | 0 | 1 | 21 | 69 | 7 | 884 | 2 | 13 | 3 | 0 |
| **frog** | 1 | 8 | 75 | 79 | 13 | 32 | 774 | 9 | 9 | 0 |
| **horse** | 3 | 1 | 5 | 4 | 3 | 8 | 0 | 976 | 0 | 0 |
| **ship** | 25 | 12 | 2 | 0 | 0 | 2 | 0 | 0 | 955 | 4 |
| **truck** | 0 | 89 | 0 | 0 | 0 | 0 | 0 | 1 | 9 | 901 |

### Top confusions
| True → Predicted | Count |
|---|---|
| deer → horse | 154 |
| truck → automobile | 89 |
| cat → dog | 79 |
| frog → cat | 79 |
| frog → bird | 75 |
| dog → cat | 69 |
