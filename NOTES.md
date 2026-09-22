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
  14 GB before any activation. 8-bit Adam fixes this, see below.
- 2.6B with AdamW is OOM. Params + grads are 10.4 GB bf16 and the AdamW state is 20.8 GB, so a full
  fine-tune wants a 40-80 GB card or sharding, matching the plan's "80GB-class" note. Forward plus
  backward alone already takes 11.3 GB for a single 1408-token sequence.

### 8-bit Adam: it saves the 1.2B, not the 2.6B

Measured with `bitsandbytes` 0.50.2 (`pixi add --pypi bitsandbytes`). The 8-bit optimizer state is
**2.03 bytes/param** measured on this build (`state1`/`state2` uint8 plus block absmax and qmap), against
8 bytes/param for fp32 AdamW, so a 1.17B model drops from 9.4 GB of optimizer state to 2.4 GB.

| model | optimizer | batch | peak VRAM | tokens/s | verdict |
|---|---|---|---|---|---|
| 1.2B | AdamW8bit + ckpt | 11x1408 | **9.57 GB** | 4997 | fits, ~2 GB headroom |
| 2.6B | AdamW8bit + ckpt | 1x1408 | - | - | OOM (needs ~15.7 GB) |
| 2.6B | PagedAdamW8bit + ckpt | 1x1408 | - | - | CUDA illegal memory access, not usable |

So the 1.2B full fine-tune is now a local job: 130M tokens of staged `core` mixture is about 7.2 h,
the YESMOM subset about 3.3 h, the full 455M mixture about 25 h, at 16k padded tokens per micro-batch.
The throughput cost of 8-bit over no optimizer at all is about 5% (4997 vs 5280 tokens/s), which is
cheap. `train.py` takes `--optim adamw8bit` and a 1.2B smoke run completes on the 4070.

The 2.6B does not fit: params + grads are 10.79 GB bf16 before any optimizer state, and 8-bit state adds
5.47 GB, so its fixed cost alone is 16.26 GB. `PagedAdamW8bit`, which would page that state to CPU,
raises a CUDA illegal memory access on this bitsandbytes/torch/driver combination (0.50.2 / 2.14+cu130 /
CUDA 13.2); worth retrying after a version bump, not worth chasing now.

### How big a GPU

Full fine-tune, bf16 weights and grads, gradient checkpointing, activation cost measured on the 4070
(2.6B: 1408 tokens cost 0.55 GB over the fixed cost; 1.2B: 0.162 GB per 1000 tokens in the 8-bit run).

| model | optimizer | fixed (params + grads + state) | + activations | smallest card |
|---|---|---|---|---|
| 1.2B | AdamW8bit | 7.06 GB | 0.162 GB / 1k tok | **12 GB** (measured 9.57 GB at 16k tok) |
| 2.6B | AdamW8bit | 16.26 GB | 0.392 GB / 1k tok | **24 GB** (17.9 GB at 4k, 19.5 at 8k, 22.7 at 16k) |
| 2.6B | AdamW fp32 | 32.36 GB | 0.392 GB / 1k tok | 40 GB tight, 48 GB comfortable |

Read that as:

- **1.2B: any 12 GB card**, which is this machine. 8-bit Adam is mandatory (fp32 AdamW needs 14.04 GB
  fixed and OOMs here).
- **2.6B: 24 GB minimum** (RTX 3090/4090, L4, A10G), and only with 8-bit Adam, which saves 16.1 GB of
  state (21.58 down to 5.47 GB). At 24 GB keep `max_tokens` at 8k for headroom or 16k to use it all.
- **16 GB cards do not work at all** for the 2.6B under any optimizer: the fixed 16.26 GB exceeds the
  card before a single activation.
- **fp32 AdamW for the 2.6B wants 40 GB or more** (A100 40 GB tight, 48 GB L40S/A6000 comfortable,
  80 GB A100/H100 with no constraints). That is the plan's "80GB-class" note, and 8-bit Adam is what
  brings the requirement down to a rentable single 24 GB card.
- Rented time at the measured 2457 tokens/s for the 2.6B: the staged `core` budget of 130M tokens is
  about 15 h, the full 455M mixture about 53 h, on a 4070-class card. A 4090 has about twice the memory
  bandwidth, so roughly half that.

### An RTX 5090 (32 GB, sm_120) with driver 580.76.05 and CUDA 13.0

Checked against this environment rather than assumed:

- `torch 2.14.0+cu130` reports `torch.cuda.get_arch_list()` = `sm_75, sm_80, sm_86, sm_90, sm_100,
  sm_120`. **sm_120 is Blackwell**, so this exact wheel runs on a 5090 with no rebuild.
- `bitsandbytes 0.50.2` ships `libbitsandbytes_cuda130.so` alongside `cuda118` through `cuda132`, so the
  8-bit optimizer has a CUDA 13.0 binary for that driver.
- Driver 580.x is the CUDA 13.0 family and Blackwell needs 570 or newer, so 580.76.05 is the right
  pairing for a `cu130` wheel; Ubuntu 24.04 is fine for the DKMS module.
- 32 GB changes the batch, not the optimizer choice: the 2.6B with fp32 AdamW needs 32.36 GB of fixed
  state and would still not fit a 32 GB card, so **8-bit Adam stays mandatory**. With 8-bit Adam the
  fixed cost is 16.26 GB, leaving about 15 GB of activation budget, which is `max_tokens` around 32k
  for the 2.6B and 64k for the 1.2B. Memory bandwidth is about 3.5x the 4070's, so expect roughly 3x
  throughput: staged `core` in about 5 h and the full mixture in about 18 h for the 2.6B.

Two things to verify on the box, because they cannot be checked from here: that `AdamW8bit` steps
cleanly on sm_120 (the `PagedAdamW8bit` illegal-memory-access above is this bitsandbytes/torch/driver
combination, so retest rather than assume), and whether `causal_conv1d` has a Blackwell build, since
without it LFM2's short conv falls back to the reference PyTorch kernel. Also note the 5090 is a 575 W
card, which is a PSU and cooling decision, not a software one.

### RTX PRO 6000 Blackwell (96 GB, sm_120)

Same compute capability as the 5090, so the same `cu130` torch wheel and the same driver family apply.
The difference is that 96 GB removes the memory constraints instead of just relaxing them:

| config | fixed | 16k tok | 32k tok | 64k tok | 128k tok |
|---|---|---|---|---|---|
| 2.6B, fp32 AdamW + ckpt | 32.36 GB | 38.8 | 45.2 | 58.1 | 83.7 |
| 1.2B, fp32 AdamW + ckpt | 14.04 GB | 16.6 | 19.4 | 24.7 | 35.3 |

So on 96 GB: **8-bit Adam becomes optional rather than required**, the plan's exact 16k-token config
runs with 57 GB to spare, and the batch can grow to the model's full 128k position limit. No sharding,
no offload, no optimizer substitution - the upstream recipe verbatim.

Keep gradient checkpointing on anyway. Measured on the 230M, dropping it costs about **29x** the
activation memory per token (0.020 GB/1k with it, 0.587 GB/1k without), so a 96 GB card is better spent
on a longer batch than on turning checkpointing off. Bandwidth is the same class as the 5090 (~1.79
TB/s), so 96 GB buys headroom, not speed: throughput stays roughly 3x this 4070.

Two caveats on the card itself: it ships in a Max-Q (300 W) and a Workstation Edition (600 W) variant
with a real clock difference, and 600 W is a PSU and cooling decision. If the 2.6B is the target and
buying is on the table, this is the no-constraints option; a rented 24 GB card plus 8-bit Adam reaches
the same place more cheaply for a one-off run.

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

## The first real checkpoint: selectia-1.2b-teacher (2026-09-22)

The first trained selectia model, and it needed no dataset downloads: `scripts/make_teacher_data.py`
turns the shipped `teacher_data/` into 102,580 training rows (18.0k packed/single custom questions,
65.6k isolated rows, 9.3k commands, 9.7k routing) and 4,020 held-out eval rows, and the 1.2B was
fine-tuned on it with 8-bit Adam and gradient checkpointing.

Run: `LFM2.5-1.2B-Base`, 1 epoch, 706 optimizer steps, 14.1M tokens, lr 1e-5 with 200-step warmup,
`max_tokens 12288`, `max_ctx 1536`, `schema_first_prob 0.5`, `none_prob 0` (no clean cross-task label
pool exists in a single-source mixture, see the `none_augment` guard). **55 minutes on one RTX 4070**,
5200 tokens/s, 10 GB allocated, CE 1.196 to 0.33, final grad norm ~20.

On its own held-out teacher sets: accuracy 0.912, NLL 0.230, Brier 0.126, **ECE 0.053**, AURC 0.019.

On JevBench's public items, against the same base model zero-shot:

| metric (public items) | 1.2B base | selectia-1.2b-teacher |
|---|---|---|
| all 231 | 0.338 | **0.628** |
| original (72) | 0.347 | **0.708** |
| easy (48) | 0.313 | **1.000** |
| hard (111) | 0.342 | **0.414** |
| Brier, all | 0.758 | 0.558 |
| ECE, all | 0.217 | 0.193 |
| Brier, hard | 0.731 | 0.894 |
| ECE, hard | 0.210 | 0.343 |
| ECE after a fitted T = 2.5 | 0.081 | 0.097 |
| paraphrase: agree / both correct | 0.833 / 0.306 | 0.722 / 0.583 |
| latency p50 (hard) | 0.034 s | 0.035 s |

Read it as:

- **Accuracy nearly doubled from teacher data alone, with zero public datasets.** The easy tier went to
  a perfect 1.000, matching `decider-2b`'s published easy 1.000, and the hard tier reached 0.414 against
  `decider-2b`'s 0.473 - a 1.2B trained on 14M tokens is within 6 points of a 1.9B trained on the full
  ~455M-token mixture.
- **Calibration improved overall and got worse on the hard tier.** ECE fell 0.217 to 0.193 across the
  public items, but on hard items it rose 0.210 to 0.343 while accuracy improved: the model learned to be
  confident on the teacher's task shapes and is confidently wrong on long out-of-distribution items. This
  is the same size-versus-calibration split seen across the bases, now caused by training rather than by
  size, and it is the strongest argument for the plan's held-out calibration gate.
- **One fitted temperature still rescues most of it** (hard-tier ECE 0.343 raw to 0.097 at T = 2.5), and
  every model so far has wanted T around 2.3 to 2.5.
- **Paraphrase behaviour flipped.** Agreement fell (0.833 to 0.722) while the both-correct rate rose
  (0.306 to 0.583). The base was consistently wrong; the fine-tune is right more often and slightly less
  self-consistent.

Per-family, the gains are where the teacher data lives: tool_selection 1.00, fact 1.00, extraction 0.92,
intent 0.88, tradeoff 0.83, against judge_hard 0.41, long_policy 0.32, multi_hop 0.28, temporal_numeric
0.27, ambiguous 0.14. Everything needing a long state is still unsolved.

Caveats: this checkpoint's `selectia_config.json` sets `schema_first` and `isolated_levels` true, so the
benchmark ran it in the schema-first layout while the base runs used state-first, which is a confound
worth removing before quoting the delta as pure training gain. It is a partial model (teacher data only,
no public tasks, no rules, no contrastive), it is not published, and `decider-2b`'s hard figure is on the
full 220-item tier while ours is the 111 public items.

## The staged core mixture is built (2026-09-22)

`python -m selectia.data.core --jobs 6` finished in about 9 minutes with 6 workers plus Xet transfer:
`data/tasks.pkl`, 787 MB, **968,970 train examples across 97 eval tasks** (120,424 eval examples). Two
of the 99 registry entries did not build: `games` needs `data/mario.pkl` that is not shipped, and
`trec_fine` died with "Dataset scripts are no longer supported, but found trec.py" because `CogComp/trec`
is a legacy loading script. `trec_fine` is fixed by parsing the same 500-item test set from
`SetFit/TREC-QC`'s raw `TREC_10.label`, which is script-free.

`python -m selectia.data.mixture --mode core` then produced:

| artifact | contents |
|---|---|
| `data/mixture_core.pkl` | 695,795 train examples, ~187M tokens (sampled estimate), 97 eval sets |
| `data/probes.pkl` | 73 probe sets, 35,296 examples (loaded as `({}, sets)`, the same shape as a mixture) |

The largest training slices are `custom+iso` (59.5k), `helpsteer2+iso` (49.0k), `json_state+fmt` (48.8k),
`hate_speech_scales+iso` (45.1k), `custom+fmt` (18.0k) and `routing+fmt` (9.7k).

At the 5200 tokens/s measured on the 1.2B teacher run, 187M tokens is **about 10 hours** on this 4070.
It is running now as `runs/selectia_core`: 8,975 optimizer steps at `max_tokens 12288 --accum 2`, 5,146
tokens/s, 11.4 GB VRAM, ETA about 11.5 hours.

One thing had to be fixed before it would run. `make_items` kept every token id as a Python list, which
for 186M tokens is 6 to 7 GB of int objects held for the whole run; the host hit 24 of 31 GB with only
6 GB free and was climbing about 1.3 GB per minute during tokenization. Storing `ids` as `array('i')`
instead took the process to a flat 5.5 GB RSS, since `collate` and `batches_by_tokens` only need `len()`
and a buffer. Worth knowing for any future mixture of this size: the memory is in the token ids, not in
the weights or the activations.

## YESMOM: both sizes trained (2026-09-22)

The Noul-only binary head, trained on `data/mixture_yesmom.pkl` (279,032 rows, 38.9M tokens, balanced
139,516 yes / 139,516 no). One epoch, 1,427 optimizer steps, lr 1e-5 with 200-step warmup,
`max_tokens 16384 --accum 2`, gradient checkpointing, fp32 AdamW, `--max_options 2 --yesmom`. Both
checkpoints write `"yesmom": true`, `"max_options": 2`, `"isolated_levels": false`.

| model | wall clock | throughput | peak VRAM |
|---|---|---|---|
| `yesmom-230m` | about 35 min | 21,000 tok/s | 4 GB |
| `yesmom-350m` | about 47 min | 14,100 tok/s | 5 GB |

### JevBench, Noul items only

A YESMOM model rejects anything that is not a Noul question, and only **74 of JevBench's 231 public items
are Noul** (139 are Choice, 18 Score). So the only fair comparison is on those 74, with the bases scored
on the same subset:

| model | acc | Brier | ECE |
|---|---|---|---|
| LFM2.5-230M-Base | 0.527 | 0.738 | 0.383 |
| **yesmom-230m** | 0.473 | **0.557** | **0.179** |
| LFM2.5-350M-Base | 0.514 | 0.623 | 0.221 |
| **yesmom-350m** | 0.446 | **0.554** | **0.151** |
| LFM2.5-1.2B-Base | 0.568 | 0.540 | 0.149 |
| LFM2.5-2.6B-Base | 0.581 | 0.690 | 0.318 |
| selectia-1.2b-teacher | 0.635 | 0.666 | 0.289 |

Read this carefully: **n = 74, so accuracy carries a 95% interval of about plus or minus 0.11.** Every
accuracy in that column is statistically indistinguishable from every other. The one thing that is not
noise is calibration: Brier falls for both models against their own bases (0.738 to 0.557, 0.623 to
0.554) and ECE more than halves (0.383 to 0.179, 0.221 to 0.151). The binary fine-tune sharpened the
Noul head without moving measurable accuracy on 74 items. Do not quote the accuracy column as a win.

### The YESMOM regression sets, where the signal actually is

The mixture's own eval half is 19 Noul sets, several with thousands of items, so these are not
noise-limited. Balanced accuracy equals accuracy on every set (majority class 0.50 to 0.53), so neither
model is exploiting the class skew.

| set | n | yesmom-230m | yesmom-350m |
|---|---|---|---|
| `abstain_binary` | 12,666 | 0.720 | **0.841** |
| `offtopic_binary` | 11,013 | 0.673 | **0.838** |
| `counterfactual` | 1,500 | 0.902 | 0.902 |
| `toxic_chat` | 3,000 | 0.957 | 0.955 |
| `civil_comments` | 7,500 | 0.775 | **0.846** |
| `qqp` | 1,500 | 0.563 | **0.706** |
| `mrpc` | 408 | 0.684 | 0.684 |
| `boolq` | 1,500 | 0.632 | 0.631 |
| `custom_noul` | 1,077 | 0.524 | **0.602** |
| `multirc` | 1,500 | 0.462 | **0.574** |
| `ade` (held out) | 1,500 | 0.565 | **0.655** |
| `strategyqa` (held out) | 687 | 0.489 | 0.518 |
| `synth` | 150 | 0.560 | 0.567 |
| `wiki_qa` | 879 | 0.496 | 0.485 |
| `wic` | 638 | 0.500 | 0.500 |
| `msmarco_rel` | 1,446 | 0.488 | 0.488 |
| `tweet_hate` | 1,500 | 0.447 | 0.456 |
| `paws` (held out) | 1,500 | 0.439 | 0.439 |

Three things fall out of this:

- **The two gates that are the product both improve a lot with size**: abstain 0.720 to 0.841 and
  off-topic 0.673 to 0.838, on 12k and 11k items. With ECE 0.068 and 0.049 respectively, the binary head
  is well calibrated exactly where a gate has to be. This is the number worth publishing for YESMOM.
- **Counterfactual (0.902) and toxic_chat (0.955 to 0.957) are near solved**, and civil_comments improves
  0.775 to 0.846.
- **A cluster sits at or below chance for both sizes**: `paws` 0.439, `tweet_hate` 0.45, `wic` 0.500,
  `msmarco_rel` 0.488, `wiki_qa` 0.49. These are the same-meaning and hateful-nuance judgements. Since
  both sizes fail them identically, they are not size-limited at 230M to 350M; they are shape-limited.
  The training data teaches the verifier shape ("does the proposed answer fit?"), and these eval sets ask
  a direct question about the state instead.

### Runtime contract

Validated on both checkpoints: a `bool` schema and a plain `no`/`yes` question are accepted, and a
non-Noul 2-option question, a `choice` schema, and a 3-option question are all rejected with `ValueError`.
The `yesmom: true` flag is what enforces this, which is why the training run needs `--yesmom`; without it
the head is still binary but the runtime would offer it multi-option questions.

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
