# Metrics and Results

## VAE Reconstruction

- Latent shape: `(8, 4, 8, 8)`
- Reconstructed image shape: `(8, 3, 64, 64)`
- Reconstruction MSE: `0.007643`
- Latent scaling factor: `0.18215`

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

## Training

- Epochs: `50`
- Batch size: `128`
- Learning rate: `2e-5`
- Label-drop probability: `0.1`
- Final average training loss: `0.107927`

## Sampling

- Guidance scale: `3.0`
- Generated images: `64`
- Sample grid labels: `0–9 repeated`
- Generated image shape: `(64, 3, 64, 64)`

- ## Implementation Notes

The Fashion-MNIST images were converted to 3 channels so they could be processed by the pretrained RGB VAE. The VAE produces 4-channel latent tensors, so the diffusion U-Net operates on 4-channel latent inputs.

For the required display timestep `t=1000`, the code maps it to schedule index `999`, because PyTorch tensor indices range from `0` through `999` for a 1000-step schedule.
