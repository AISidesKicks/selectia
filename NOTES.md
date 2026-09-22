# Notes

## Original idea

The original idea is based on the AISidesKicks experiment with YESMOM: yes/no logit-based
experiments on small (under 500M) LFM2.5 models.

That is, instead of getting a model to generate a "yes" or "no" token, read the logits for the
`yes` / `no` tokens directly at a fixed prompt slot and use their ratio as a calibrated
probability. The decider recipe (see `plans/INTEND.md`) generalizes that readout from two tokens
to the full typed-decision label set, and YESMOM narrows it back to the binary head for the
350M / 230M LFM2.5 bases.

## Phase 0: label capacity (probed 2026-09-22)

`scripts/probe_lfm.py` loads every base and renders a toy narrow / wide / Noul prompt through the
slot readout. Result: open question 3 is settled, no cap needed.

| model | vocab | label capacity | A..J single tokens | pad / bos / eos |
|---|---|---|---|---|
| LFM2.5-2.6B-Base | 125017 | 255 | yes | 124893 / 124894 / 124900 |
| LFM2.5-1.2B-Base | 64400 | 255 | yes | 0 / 1 / 7 |
| LFM2.5-350M-Base | 64400 | 255 | yes | 0 / 1 / 7 |
| LFM2.5-230M-Base | 64402 | 255 | yes | 0 / 1 / 7 |

All four expose `.model(...).last_hidden_state` and `.lm_head`, all are tied-embedding models
(`lm_head.weight is embed_tokens.weight`), the invalid options mask to `-inf`, probabilities sum to
1, and the readout is bit-for-bit deterministic. So `--max_options 255` (and therefore the full wide
label table) works at both vocab sizes.

The first pass ran on CPU because the local RTX 4070 was fully occupied by other processes (27 MiB
free, 4.67 GiB + 2.34 GiB held by two foreign PIDs). Nothing was stopped or evicted. Once those
processes exited, the same probe ran again with `--device cuda` and all four bases load and read out
on the GPU as well, with the same capacities and the same determinism.

LFM2's short conv falls back to the reference PyTorch kernel (`causal_conv1d` is not installed).
That is correct but slow; `pixi add --pypi causal-conv1d` is the optional speedup. The numbers below
are therefore a conservative upper bound on the wall clock.

## Training cost on one RTX 4070 (12 GB)

Measured with `scripts/bench_train.py` (bf16, `max_ctx 1536`, `max_tokens 16384`, AdamW `(0.9,0.95)`,
`accum 2`). "ckpt" is `--grad_ckpt`. Throughput counts padded tokens, the same shape of work the
trainer does. The token budgets are the plan's numbers (staged core ~130M, full mixture ~455M) plus a
first estimate for the YESMOM subset (~60M); the YESMOM figure is a guess until `data/tasks.pkl` and
the real mixture exist, so treat that column as a placeholder to refresh.

| model | settings | batch | tokens/s | s / optimizer step | peak VRAM | core (130M tok) | yesmom (60M tok) | full (455M tok) |
|---|---|---|---|---|---|---|---|---|
| 230M | ckpt | 11x1408 | 19385 | 1.60 | 3.1 GB | 1.9 h | 0.9 h | 6.5 h |
| 230M | no ckpt | 5x1408 | 25460 | 0.55 | 6.9 GB | 1.4 h | 0.7 h | 5.0 h |
| 350M | ckpt | 11x1408 | 14109 | 2.20 | 4.3 GB | 2.6 h | 1.2 h | 9.0 h |
| 1.2B | ckpt, no optimizer | 11x1408 | 5280 | 5.87 | 7.2 GB | 6.8 h | 3.2 h | 23.9 h |
| 2.6B | ckpt, no optimizer | 1x1408 | 2457 | 1.15 | 11.3 GB | 14.7 h | 6.8 h | 51.4 h |

What does not fit on this card:

- 230M at `max_tokens 16384` **without** `--grad_ckpt` is already OOM at 11.6 GB used. Gradient
  checkpointing is not a 2.6B-only concern, it is the default for any full-size batch here.
- 1.2B with AdamW is OOM, and it is OOM at `max_tokens 4096` too. The batch size is not the problem:
  the weights (2.4 GB bf16) + grads (2.4 GB) + AdamW state (2 x 4 bytes x 1.17B = 9.4 GB) are about
  14 GB before any activation. A 12 GB card needs 8-bit Adam, an SGD/AdaFactor run, or offload.
- 2.6B with AdamW is OOM. Params + grads are 10.4 GB bf16 and the AdamW state is 20.8 GB, so a full
  fine-tune wants a 40-80 GB card or sharding, matching the plan's "80GB-class" note. Forward plus
  backward alone already takes 11.3 GB for a single 1408-token sequence.

What this means for the plan: YESMOM (230M then 350M) trains locally, 230M in about two hours for
the staged budget and 350M in about three. The full-decision 1.2B and 2.6B runs need a bigger GPU or
an 8-bit optimizer, which is exactly the trade the educational write-up can show.

Not measured: the tokenization pass over the real mixture and the `decider_lfm.data.core` download
(~95 public datasets). Both are one-off and need `data/tasks.pkl` first.

## What is validated so far

- `pytest tests` passes (19 tests): the request/answer layer, the rule data, the prompt layouts and the
  label-capacity assertions, all against the real LFM2.5 tokenizers.
- Two smoke runs on the 230M base, one full-decision and one `--yesmom`, each complete a real
  optimizer loop, save the weights and write `decider_config.json` (`yesmom: true` and
  `isolated_levels: false` for the binary model, `isolated_levels: true` otherwise).
- Both runtime contracts load from the saved folder: the full model answers a Choice plus a Noul
  question, the YESMOM model returns `{"noul": p}` and rejects a Choice question.

Not yet done: building `data/tasks.pkl`, the staged `core` and `yesmom` mixtures, and any real
training or calibration run.
