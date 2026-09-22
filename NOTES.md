# Notes

## Original idea

The original idea is based on the AISidesKicks experiment with YESMOM: yes/no logit-based
experiments on small (under 500M) LFM2.5 models.

That is, instead of getting a model to generate a "yes" or "no" token, read the logits for the
`yes` / `no` tokens directly at a fixed prompt slot and use their ratio as a calibrated
probability. The System One recipe (see `plans/INTEND.md`) generalizes that readout from two tokens
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

Not measured: the tokenization pass over the real mixture and the `selectia.data.core` download
(~95 public datasets). Both are one-off and need `data/tasks.pkl` first.

## JevBench public items, zero-shot (measured 2026-09-22)

[JevBench](https://github.com/fstandhartinger/jevbench) (MIT, Benchmark Heaven's own benchmark) is the
leaderboard the question was about, and it is reproducible in part: it ships its harness and 231 of its
534 decisions (`datasets/public/{original,easy,hard}.jsonl`). The other 146 imported and 109 held-out
hard items are not published, so **the official JevBench Score cannot be reproduced** from the repo.
What follows is per-tier accuracy / Brier / ECE on the public items, computed with JevBench's own
`score_task` and `summarize` via `selectia/bench/jevbench_public.py`, so the conventions are theirs.

**No selectia checkpoint is trained yet**, so these runs are the LFM2.5 bases read through our readout
zero-shot. That is the floor the fine-tunes have to beat, not a selectia result.

| model | all 231 | original (72) | easy (48) | hard (111) | Brier (hard) | ECE (hard) | p50 (hard) | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-230M-Base | 0.307 | 0.306 | 0.333 | 0.297 | 0.888 | 0.354 | 0.016 s | 0.63 GB |
| LFM2.5-350M-Base | 0.307 | 0.319 | 0.313 | 0.297 | 0.745 | 0.196 | 0.018 s | 0.93 GB |
| LFM2.5-1.2B-Base | 0.338 | 0.347 | 0.313 | 0.342 | 0.731 | 0.210 | 0.034 s | 2.68 GB |
| LFM2.5-2.6B-Base | 0.541 | 0.528 | 0.917 | 0.387 | 0.852 | 0.316 | 0.071 s | 5.79 GB |

For reference, `decider-2b` (the closest published entrant, a trained 1.9B Qwen3.5 readout) scores easy
1.000, hard 0.473, with hard Brier 0.806 and ECE 0.322 in `results/v1.2/additions/decider-2b.json`. So
the 2.6B base is already close on the hard tier (0.387 vs 0.473) and far off on easy (0.917 vs 1.000);
the 1.2B base is roughly at chance.

Two things worth reading twice:

- **Accuracy is not monotone in size on the hard tier, but calibration is.** The 2.6B is more accurate
  than the 1.2B and *worse calibrated* (hard Brier 0.852 vs 0.731, ECE 0.316 vs 0.210). It is
  confidently wrong. That is the failure mode temperature scaling is for, and it is why JevBench's
  v1.3 score is chance-corrected and carries a separate Calibration axis.
- **A single fitted temperature fixes most of the calibration.** Fitted on the public items by NLL
  (in-sample, so optimistic, and a diagnostic only, not part of the official score):

  | model | fitted T | Brier raw to fitted | ECE raw to fitted |
  |---|---|---|---|
  | 230M | 2.50 | 0.960 to 0.755 | 0.403 to 0.192 |
  | 350M | 2.50 | 0.830 to 0.724 | 0.249 to 0.133 |
  | 1.2B | 2.50 | 0.758 to 0.685 | 0.217 to 0.081 |
  | 2.6B | 2.30 | 0.648 to 0.585 | 0.208 to 0.064 |

  Every base wants T around 2.3 to 2.5, i.e. all four are systematically overconfident, and ECE drops
  by 2.5x to 3x. Note this is a single scalar on an untrained readout; it does not fix accuracy.

Where the 2.6B base is strong and weak (public items, by family): tool_selection 1.00, extraction 0.88,
intent 0.83, fact 0.83 against long_policy 0.26, temporal_numeric 0.27, multi_hop 0.33, ordinal 0.33,
ambiguous 0.14. In other words the short explicit-label tasks are nearly solved and everything that
needs a long state or ordinal reasoning is not. Paraphrase agreement is 0.833 for both big models while
the both-correct rate is 0.306 (1.2B) and 0.444 (2.6B): the models are consistently wrong, not unstable.

Throughput and cost basis (hard tier, mean 1178 input tokens per decision, matching decider-2b's 1170):

| model | decisions/s | tok/s | JevBench Speed axis (hard, with their x2 + 0.15 s self-host adjustment) |
|---|---|---|---|
| 230M | 54.5 | 33386 | 93.3 |
| 350M | 38.5 | 23582 | 92.8 |
| 1.2B | 22.3 | 13652 | 89.5 |
| 2.6B | 13.0 | 7710 | 86.1 |

`decider-2b`'s published Speed comes out around 83.2 under the same formula, but that latency was
measured over the internet from Germany to a RunPod GPU in Canada while ours is local with no network,
so the two are not directly comparable and we do not claim a speed win. Cost is not computed: a local
GPU has no provider tariff, and JevBench reports `null` rather than zero for exactly that reason. At
1178 input tokens per hard decision the arithmetic is easy to redo for any hosted price.

**Option order.** JevBench's `labels` order and its `criteria` order disagree in 119 of the 231 public
items, and the harness has two conventions: `openai_compat` iterates `labels`, `typesafe` (which
`decider-2b` ran under) passes `criteria` through unchanged. Their README measured one small model
swinging from 72% to 21% when option order was reversed, so both were run:

| model | order | all | original | easy | hard |
|---|---|---|---|---|---|
| 1.2B | labels | 0.338 | 0.347 | 0.313 | 0.342 |
| 1.2B | criteria | 0.329 | 0.347 | 0.271 | 0.342 |
| 2.6B | labels | 0.541 | 0.528 | 0.917 | 0.387 |
| 2.6B | criteria | 0.537 | 0.500 | 0.938 | 0.387 |

The effect is small here (under 5 points overall, and the hard tier is identical), unlike the 51-point
swing JevBench reported for a prompted baseline. Worth stating plainly rather than assuming: the
readout is trained against positions, so position sensitivity is a property to measure, not to assume
away.

Caveats to keep attached to these numbers: public items only; zero-shot bases, not selectia; the fitted
temperature is in-sample; local latency; and the hard tier's 111 public items are not the full 220, so
even the hard-tier figure is a subset.

## What is validated so far

- `pytest tests` passes (19 tests): the request/answer layer, the rule data, the prompt layouts and the
  label-capacity assertions, all against the real LFM2.5 tokenizers.
- Two smoke runs on the 230M base, one full-decision and one `--yesmom`, each complete a real
  optimizer loop, save the weights and write `selectia_config.json` (`yesmom: true` and
  `isolated_levels: false` for the binary model, `isolated_levels: true` otherwise).
- Both runtime contracts load from the saved folder: the full model answers a Choice plus a Noul
  question, the YESMOM model returns `{"noul": p}` and rejects a Choice question.

Not yet done: building `data/tasks.pkl`, the staged `core` and `yesmom` mixtures, and any real
training or calibration run.
