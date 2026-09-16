Model Architecture Metrics
 Total Parameters: 425777, Hidden Dimension: 128, Number of Attention Heads: 4, Number of Decoder Layers: 2, Sequence Length: 128, Vocabulary Size: 49.

Training Configuration and Results

 Optimizer: Adam, Learning Rate: 3e-4, Batch Size: 32, Epochs: 7, Final Training Loss: ~2.03.

Sampling Parameters Tested
 Greedy Decoding uses deterministic argmax, Temperature Sampling tests T values of 0.3, 1.0, and 1.8, and Top-k Sampling tests k values of 3 and 20 at T = 1.0.