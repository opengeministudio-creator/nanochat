# nanochat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-6e56cf.svg)](https://github.com/astral-sh/uv)

![nanochat logo](dev/nanochat.png)

nanochat is a compact, hackable LLM training stack that runs end-to-end on a single machine. It covers tokenizer training, base pretraining, SFT/RL chat tuning, evaluation, and local chat interfaces (CLI + web).

## What’s new (last 13 pushes)

- **Configurable MoE MLPs** in the GPT model for base training experiments.
- **Harmony-style tokenizer alignment** (special tokens + split pattern) and related runtime/training updates.
- **Optional `system` role support** for JSON conversation data and SFT pipelines.
- **Single-GPU speedrun script** (`runs/speedrun-single-gpu.sh`).
- **Project knowledge docs** for identity/synthetic-data voice work:
  - `knowledge/self-knowledge.md`
  - `knowledge/Soul.md`
- **Expanded tests** covering MoE and system-role/tokenizer behavior.

## Quick start

### 1) Environment

```bash
uv sync
source .venv/bin/activate
```

### 2) Fastest full pipeline (8xH100 reference)

```bash
bash runs/speedrun.sh
```

### 3) Single-GPU variant

```bash
bash runs/speedrun-single-gpu.sh
```

### 4) Chat with a trained checkpoint

```bash
python -m scripts.chat_web
# or
python -m scripts.chat_cli
```

## Core scripts

- **Tokenizer**: `scripts/tok_train.py`, `scripts/tok_eval.py`
- **Base model**: `scripts/base_train.py`, `scripts/base_eval.py`
- **Chat tuning**: `scripts/chat_sft.py`, `scripts/chat_rl.py`, `scripts/chat_eval.py`
- **Inference**: `scripts/chat_cli.py`, `scripts/chat_web.py`

## Data + conversation format notes

- Custom JSON conversations are handled in `tasks/customjson.py`.
- SFT conversations now support an **optional `system` role**.
- For identity/personality-oriented synthetic data generation, see `dev/gen_synthetic_data.py` and the `knowledge/` docs.

## Repo layout (condensed)

```text
.
├── nanochat/              # model, tokenizer, runtime, utils
├── scripts/               # train/eval/chat entrypoints
├── runs/                  # reproducible run scripts (incl. speedrun single-GPU)
├── tasks/                 # datasets/task adapters
├── tests/                 # unit tests (engine, MoE, system-role, tokenizer)
├── knowledge/             # project voice/identity docs
└── dev/                   # misc project assets + helper scripts
```

## Leaderboard + docs

- GPT-2 speedrun context and rules: `dev/LEADERBOARD.md`
- Community discussions: https://github.com/karpathy/nanochat/discussions

## Contributing

nanochat aims to be small, readable, and easy to fork. Keep changes practical and benchmark-oriented, especially around “time-to-GPT-2” and quality-per-dollar.

If you submit AI-assisted contributions, disclose significant LLM-generated portions.

## License

MIT
