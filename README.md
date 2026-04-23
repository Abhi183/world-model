# world-model

A world model for **CarRacing-v3**, built from scratch in PyTorch. Inspired by Ha & Schmidhuber's 2018 paper, updated for 2026 hardware.

## What it does

Learns a compressed simulator of CarRacing by combining three neural networks:

1. **VAE** — compresses 64×64 RGB frames to a 32-dim latent `z`
2. **MDN-RNN** — predicts next latent `z_{t+1}` given current latent + action (the "dream")
3. **Controller** — tiny policy trained inside the dreamed rollouts, not the real env

Once trained, the controller drives the car in the real environment using nothing but the compressed model's imagination.

## Stack

- PyTorch 2.11 (MPS / Apple Silicon)
- Gymnasium 1.3 (CarRacing-v3, Box2D)
- Python 3.12

## Layout

```
world-model/
├── src/world_model/     # models: vae.py, mdn_rnn.py, controller.py
├── scripts/             # collect_rollouts.py, train_vae.py, ...
├── configs/             # hyperparameters
├── data/                # rollouts (gitignored)
├── checkpoints/         # trained weights (gitignored)
└── outputs/             # GIFs, plots (gitignored)
```

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/sanity_check.py
```

## References

- [World Models (Ha & Schmidhuber, 2018)](https://worldmodels.github.io/)
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)
