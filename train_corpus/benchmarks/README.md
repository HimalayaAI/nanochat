# Benchmarks

This folder contains lightweight benchmark scripts we can run repeatedly during
base training and SFT iterations.

## Indic Eval (Nepali)

`indic_eval_ne.py` runs open-ended Nepali evaluation on:
- dataset: `Cognitive-Lab/Aya_Indic_Eval`
- config: `npi` (Nepali; `--dataset-config ne` is auto-mapped to `npi`)
- split: `test`

### Base checkpoint (nanochat)

```bash
uv run python -m train_corpus.benchmarks.indic_eval_ne \
  --backend nanochat \
  --source base \
  --model-tag d15_harl_fulltokens_sdpa_bs32 \
  --max-examples 200 \
  --prompt-style plain
```

### SFT checkpoint (nanochat)

```bash
uv run python -m train_corpus.benchmarks.indic_eval_ne \
  --backend nanochat \
  --source sft \
  --model-tag sft_harl_ne_v1 \
  --max-examples 200 \
  --prompt-style chat
```

### HF-exported model

```bash
uv run python -m train_corpus.benchmarks.indic_eval_ne \
  --backend hf \
  --hf-model himalaya-ai/himalayagpt-0.5b \
  --trust-remote-code \
  --max-examples 200
```

Outputs are written to `data/benchmarks/indic_eval_ne/`:
- `*.predictions.jsonl`
- `*.summary.json`

## Smoke tests

`smoke_test_nanochat.py` is a quick pre-SFT checkpoint generation sanity check.
