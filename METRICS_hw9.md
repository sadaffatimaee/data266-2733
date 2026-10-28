# Homework 9 Metrics

## VAE Reconstruction

- VAE model: `stabilityai/sd-vae-ft-mse`
- Latent scaling factor: `0.18215`
- Original image shape: `(8, 3, 64, 64)`
- Latent shape: `(8, 4, 8, 8)`
- Reconstructed image shape: `(8, 3, 64, 64)`
- Reconstruction MSE: `0.007643`

## Forward Diffusion

- Diffusion steps: `1000`
- Beta schedule: `[0.0001, 0.02]`

| Timestep | Reconstruction MSE |
|---:|---:|
| 0 | 0.006317 |
| 200 | 0.187685 |
| 500 | 0.825087 |
| 800 | 1.173333 |
| 1000 | 1.223222 |

## Model Architecture

- Input: 4-channel VAE latent
- Base channels: `64`
- Channel multipliers: `(1, 2, 4)`
- Sinusoidal time embedding: Yes
- Residual blocks: Yes
- GroupNorm and SiLU: Yes
- Self-attention: `16×16` and bottleneck
- Cross-attention conditioning: Yes
- Class embeddings: 10 classes plus null label
- Label-drop probability: `0.1`
- Trainable parameters: `7,748,100`

## Training

- Epochs: `50`
- Batch size: `128`
- Learning rate: `2e-5`
- Final training loss: `0.107927`

## Sampling

- Guidance scale: `3.0`
- Reverse diffusion steps: `1000`
- Generated samples: `64`
- Generated image shape: `(64, 3, 64, 64)`
- Labels: Fashion-MNIST classes `0–9` repeated

## Implementation Notes

Fashion-MNIST images were converted to 3 channels so they could be processed by the pretrained RGB VAE. The VAE produces 4-channel latent tensors, so the diffusion U-Net operates on 4-channel latent inputs.

The requested display timestep `t=1000` is mapped to schedule index `999`, because PyTorch indices for a 1000-step schedule range from `0` through `999`.
