# Mini-Genie

A minimal implementation of [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) for educational purposes.

Genie learns to generate playable 2D environments from video — without action labels. You can prompt it with an image and "play" by selecting latent actions.

## Architecture

Three models work together:

1. **Video Tokenizer** (VQ-VAE): Compresses 64×64 frames → 16×16 discrete tokens
2. **Latent Action Model (LAM)**: Discovers actions from unlabeled video
3. **Dynamics Model**: Predicts next frame tokens given current tokens + action

## Setup

```bash
uv sync
```

For data collection, you need procgen which is only up until 3.10. I installed in on Lambda Labs and ran the data collection on a rented GPU.

## Training

### Full pipeline:

```bash
# 1. Collect CoinRun data (~30 min)
# 2. Train tokenizer (~2-3 hours)
# 3. Train LAM + dynamics (~3-4 hours)
uv run python scripts/train.py --device cuda
```

### Run phases separately:

```bash
# Just collect data
uv run python scripts/train.py --phase collect

# Just train tokenizer
uv run python scripts/train.py --phase tokenizer --tokenizer_steps 50000

# Just train dynamics (requires trained tokenizer)
uv run python scripts/train.py --phase dynamics --dynamics_steps 50000
```

## Play

After training:

```bash
# Interactive mode
uv run python scripts/play.py --output_dir outputs --use_dataset_frame

# Batch generation
uv run python scripts/play.py --output_dir outputs --actions "2,2,2,5,5,5"
```

Actions 0-7 have learned meanings (e.g., left, right, jump) — you discover them by playing!

## Project Structure

```
mini-genie/
├── configs/
│   └── default.py          # All hyperparameters
├── genie/
│   ├── data/
│   │   └── coinrun.py      # Data collection & dataset
│   ├── models/
│   │   ├── video_tokenizer.py
│   │   ├── lam.py
│   │   └── dynamics.py
│   └── train/
│       ├── train_tokenizer.py
│       └── train_dynamics.py
└── scripts/
    ├── train.py            # Main training entrypoint
    └── play.py             # Interactive play
```

## Simplifications from Paper

| Paper | Mini-Genie | Why |
|-------|------------|-----|
| ST-Transformer tokenizer | CNN-based | Simpler, faster |
| 11B parameters | ~30M parameters | Fits on single GPU |
| 16 frame context | 8 frame context | Less memory |
| 300k + 200k steps | 50k + 50k steps | Faster iteration |

## References

- [Genie Paper](https://arxiv.org/abs/2402.15391) - Bruce et al., 2024
- [MaskGIT](https://arxiv.org/abs/2202.04200) - Chang et al., 2022 (used for dynamics model)
- [VQ-VAE](https://arxiv.org/abs/1711.00937) - van den Oord et al., 2017 (used for tokenizer)