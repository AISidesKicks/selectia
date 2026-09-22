# selectia

One-pass typed decisions on small open-weight models. Give the model a **state** and a set of **typed
questions**; it returns a probability distribution per question from a single forward pass. No decoding,
no parsing, no generated text, and no answer outside the options you defined.

An educational reproduction of the "System One" model class (TypeSafe's *Jev*) and of
[Mapika/decider](https://github.com/Mapika/decider), moved from Qwen3.5 onto Liquid AI's
[LFM2.5](https://huggingface.co/LiquidAI) base weights. The readout, prompt layout, data mixture and
training loop are ported from that upstream project (Apache-2.0, see `NOTICE`); **selectia** is this
fork's own name. Findings live in `NOTES.md`, the plan in `plans/INTEND.md`.

## At a glance

| | |
|---|---|
| **Two variants** | full typed decisions (Choice, Score, Noul) on 2.6B / 1.2B, and YESMOM (Noul only) on 350M / 230M |
| **First result** | 55 minutes of fine-tuning on one RTX 4070 doubled JevBench public accuracy, 0.338 to **0.628**, with the easy tier at a perfect **1.000** |
| **Cheapest card** | the 1.2B full fine-tune fits **12 GB** with 8-bit Adam; YESMOM needs 3-4 GB |
| **Not yet** | the 2.6B run (needs 24 GB+) and any published weights |

**Contents:** [Two variants](#two-variants) - [The three question types](#the-three-question-types) -
[Quick start](#quick-start) - [How it works](#how-it-works) - [Results](#results) -
[Training on one GPU](#training-on-one-gpu) - [Status](#status) - [Layout](#layout) -
[Licensing](#licensing) - [Releases](#releases)

## Two variants

| | **full typed decisions** | **YESMOM** |
|---|---|---|
| base models | `LFM2.5-2.6B-Base`, `LFM2.5-1.2B-Base` | `LFM2.5-350M-Base`, `LFM2.5-230M-Base` |
| question types | **Choice, Score, Noul** | **Noul only** (yes/no) |
| per forward pass | one state + N typed questions | one state + one yes/no question |
| output | a distribution per question over its own options | `P(yes)` and the derived boolean |
| rendering | narrow `(A)..(J)`, or wide single-token labels up to 255 | two option tokens, no label table |
| why this shape | reading rubrics and recalling knowledge needs capacity | a verification is a much smaller function |
| intended use | routing, classification, rubric scoring, incident triage | cheap on-device gating: policy checks, safety, entailment, answerability, form-fill `skip` |
| fits on | the 1.2B on a 12 GB card with 8-bit Adam; the 2.6B on 24 GB with 8-bit Adam, or 40 GB+ with fp32 AdamW | 3-4 GB of VRAM, trains on one RTX 4070 |

YESMOM is not a different architecture. It is the `Noul` question type as the *only* supported type, so
the readout collapses to the `no` / `yes` option tokens and the tiny bases never have to carry a 255-wide
label table. Keeping it Noul-only is deliberate: mixing Choice and Score into a 230M model dilutes the
binary calibration that is the whole product.

## The three question types

`choice`, `score` and `noul`, exactly the primitives from the System One definition. Ids are never shown
to the model; `instructions` and every description may be a string or any JSON value.

```json
{"type": "choice", "instructions": "Which department should handle this?",
 "criteria": {"billing": "charges and refunds", "technical": null, "sales": {"not_for": "support"}}}

{"type": "score",  "instructions": "How urgent is this?",
 "criteria": ["none", "low", "medium", "high"]}

{"type": "noul",   "instructions": "Does this request need a refund?",
 "criteria": {"true": "money back", "false": "anything else"}}
```

- **Choice** options come from the `criteria` keys; a `null` description renders as a bare label.
- **Score** takes 2 to 10 levels, or a `{"0": ..., "1": ...}` legend.
- **Noul** criteria are optional.

Answers come back typed and normalised:

```json
{"type": "choice", "choice": "billing", "confidence": 0.97, "certainty": 0.91,
 "probabilities": {"billing": 0.97, "technical": 0.02, "sales": 0.01}}

{"type": "score", "score": 1.3, "legend": {"0": "none", "1": "low", "2": "medium", "3": "high"},
 "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4, "3": 0.0}}

{"type": "noul", "noul": 0.95}
```

Score levels are scored one at a time when the model's config turns on `isolated_levels` (our full models
do), so a level never sees its own number or its neighbours. `fit_mass` then tells you when no level, or
several, fit.

## Quick start

```python
from selectia.infer import Selectia

d = Selectia("runs/selectia_1.2b_teacher/model", device="cuda")

state = "My card was charged twice for the same order and I want the extra charge refunded."

# Jev-shaped: noul / choice / score, with criteria
out = d.system_one(state, {
    "team": {"type": "choice", "instructions": "Which department should handle this?",
             "criteria": {"billing": "charges and refunds", "technical": null, "sales": null}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?",
               "criteria": {"true": "needs action today", "false": "anything else"}},
})
print(out["answers"]["team"]["choice"], out["answers"]["team"]["confidence"])
print(out["answers"]["urgent"]["noul"])            # P(yes)

# same thing in the compact JSON shape
print(d.decide_json(state, {"team": {"type": "choice", "options": ["billing", "technical", "sales"]}}))
```

`system_one` returns `{"model", "answers": {id: answer}, "usage"}`. The runtime reads temperature, layout
and flags from the model folder's `selectia_config.json`, which `train.py` writes. A model with
`"yesmom": true` accepts only `noul` questions and rejects everything else.

## How it works

One forward pass, one readout. Each question is rendered into the same prompt with an answer slot,
`... Answer 2: (`, and the hidden state at that slot is projected onto the `lm_head` rows of the
single-token option labels:

```python
h  = lm.model(input_ids=..., attention_mask=...).last_hidden_state
hs = h[slot_batch, slot_idx]                    # hidden state at " (" for every question
W  = lm.lm_head.weight[letters]                 # one row per single-token label
logits = F.linear(hs, W).float().masked_fill(arange(W.shape[0]) >= nopts[:, None], -inf)
```

So the distribution is *constrained to your options by construction*: invalid options are masked to
`-inf` before the softmax, and nothing is ever generated. A fitted temperature from `selectia_config.json`
is the calibration control. Because there is no decoding, all N questions cost one forward pass, and
question order does not change any answer.

Two consequences worth knowing: each option label must be a **single token** in the tokenizer (all four
LFM2.5 tokenizers supply the full 255, see `NOTES.md`), and these are full fine-tunes, not prompted
general models, which is what separates this from a chat model with a grammar constraint.

## Results

JevBench is Benchmark Heaven's decision-model benchmark. Only 231 of its 534 decisions are public, so the
official JevBench Score cannot be reproduced from its repository. Everything below is scored with
JevBench's own `score_task` and `summarize`, on the public items only.

### The bases, zero-shot

The LFM2.5 bases read through our readout with **no training at all**. This is the floor the fine-tunes
have to beat, not a selectia result.

| model | all 231 | easy | hard | Brier (hard) | ECE (hard) | p50 (hard) | peak VRAM |
|---|---|---|---|---|---|---|---|
| LFM2.5-230M-Base | 0.307 | 0.333 | 0.297 | 0.888 | 0.354 | 0.016 s | 0.63 GB |
| LFM2.5-350M-Base | 0.307 | 0.313 | 0.297 | 0.745 | 0.196 | 0.018 s | 0.93 GB |
| LFM2.5-1.2B-Base | 0.338 | 0.313 | 0.342 | 0.731 | 0.210 | 0.034 s | 2.68 GB |
| LFM2.5-2.6B-Base | 0.541 | 0.917 | 0.387 | 0.852 | 0.316 | 0.071 s | 5.79 GB |

Two findings worth keeping:

- The 2.6B is more accurate than the 1.2B **and worse calibrated**. It is confidently wrong, which is why
  the score carries a Calibration axis separate from Intelligence.
- One fitted temperature fixes most of it: every base wants T around 2.3 to 2.5 and ECE drops 2.5x to 3x.

### The first trained checkpoint

`selectia-1.2b-teacher` is 55 minutes of fine-tuning on the shipped `teacher_data/` alone, with no public
datasets, on one RTX 4070.

| public items | 1.2B base | selectia-1.2b-teacher |
|---|---|---|
| all 231 | 0.338 | **0.628** |
| original (72) | 0.347 | **0.708** |
| easy (48) | 0.313 | **1.000** |
| hard (111) | 0.342 | **0.414** |
| Brier / ECE, all | 0.758 / 0.217 | 0.558 / 0.193 |
| Brier / ECE, hard | 0.731 / 0.210 | 0.894 / 0.343 |
| ECE, hard, after one fitted T | 0.081 | 0.097 |
| paraphrase agree / both correct | 0.833 / 0.306 | 0.722 / 0.583 |

For scale, the closest published entrant `decider-2b` (a trained 1.9B readout) scores easy 1.000 and hard
0.473. So a 1.2B trained on 14M tokens is within 6 points of a 1.9B trained on roughly 455M.

Accuracy roughly doubles and the easy tier becomes perfect, but **hard-tier calibration gets worse while
hard-tier accuracy improves**: the model learns to be confident on the shapes it saw. That gap is the
argument for the held-out calibration gate in `plans/INTEND.md`.

Full tables, per-family breakdowns, option-order sensitivity and the latency and token numbers are in
`NOTES.md`.

## Training on one GPU

Measured on an RTX 4070 (12 GB), bf16, `max_ctx 1536`.

| run | settings | peak VRAM | time for the staged budget |
|---|---|---|---|
| YESMOM 230M | fp32 AdamW | 3.1 GB | about 2 h |
| YESMOM 350M | fp32 AdamW | 4.3 GB | about 3 h |
| 1.2B full decisions | `--optim adamw8bit --grad_ckpt` | 9.6 GB | about 7 h |
| 2.6B full decisions | - | does not fit | needs 24 GB with 8-bit Adam, or 40 GB+ with fp32 |

8-bit Adam is the difference: 2.03 bytes per parameter of optimizer state instead of 8, which is 9.4 GB
down to 2.4 GB on the 1.2B, for about a 5% throughput cost. The 2.6B stays out of reach on a 12 GB card
under any optimizer, because its weights plus grads alone are 10.79 GB before any activation.

### Running it

```bash
pixi run python scripts/probe_lfm.py                    # Phase 0: load the bases, record label capacity
pixi run python -m pytest tests -q                      # CPU tests, no model needed

# what a training step costs on your card, before committing to a long run
pixi run python scripts/bench_train.py --model LiquidAI/LFM2.5-1.2B-Base --grad_ckpt --optim adamw8bit

# data, then train, then measure
pixi run python -m selectia.data.core --jobs 6 --out data/tasks.pkl
pixi run python -m selectia.data.mixture --mode core --out data/mixture_core.pkl --probes data/probes.pkl
scripts/train.sh core                                   # staged full-decision run (1.2B, 8-bit Adam)
scripts/evaluate.sh runs/selectia_core/model             # accuracy, NLL, Brier, ECE, AURC, probes
pixi run python -m selectia.bench.jevbench_public --src scratch/jevbench-src \
    --models runs/selectia_core/model --device cuda --fit-temperature
```

`train.sh` picks the base from the mode (`core` is the 1.2B, `full` the 2.6B, `yesmom` the 230M) and can be
pointed at another one with a second argument. The `core` recipe already uses `--grad_ckpt --optim
adamw8bit`, which is what makes it fit 12 GB; override with `OPTIM=adamw` on a 40 GB+ card.

Everything runs inside the `sclt` pixi environment (Python 3.12, torch, transformers 5, numpy<2). Run
`pixi shell` once, or prefix with `pixi run`.

The JevBench harness is not vendored. Clone it and pass `--src`, as above:

```bash
git clone --depth 1 https://github.com/fstandhartinger/jevbench scratch/jevbench-src
```

## Status

Work in progress, and honest about it.

**Done**

- The port (`selectia/`), the prompt and readout layer, the label-capacity work, the staged `core` and
  Noul-only `yesmom` data modes, the Phase 0 probe, and 19 passing CPU tests.
- `data/tasks.pkl`: 968,970 train examples from 97 of the 99 registered public datasets.
- `data/mixture_core.pkl`: 695,795 train examples, roughly 187M tokens, plus 73 probe sets.
- The first trained checkpoint, `selectia-1.2b-teacher`, benchmarked above.
- A JevBench harness (`selectia/bench/jevbench_public.py`) that scores our readout with JevBench's code.
- Measured memory budgets for every size on a 12 GB card.

**Not done**

- No published weights, and no 2.6B run.
- The staged `core` mixture is built but not yet trained on. At the measured 5200 tokens/s, 187M tokens is
  about **10 hours** on the 4070.
- Two registry tasks do not build: `games` needs an unshipped `data/mario.pkl`, and `trec_fine` was fixed
  by reading its raw test file (see `NOTES.md`).

## Layout

```
selectia/       the port: prompt.py, model.py, systemone.py, infer.py, train.py, evaluate.py,
                data/ (task registry, mixture, augmentations), probes/, bench/
scripts/        probe_lfm.py, bench_train.py, gpu_watch.py, make_teacher_data.py,
                train.sh, evaluate.sh, stage_release.py, upload_hf.py
tests/          CPU tests for the request/answer layer, the prompt layouts and the rule data
plans/          INTEND.md, the implementation plan
NOTES.md        findings, label capacity, benchmark results, memory budgets
teacher_data/   the shipped teacher-written decisions
.env.example    HF write token for pushing releases; copy to .env, which is gitignored
```

## Licensing

Two licences, deliberately split, because code and weights are not the same thing:

- **Code** is Apache-2.0, including the modules ported from Mapika/decider (`selectia/`). Full text:
  `CODE-LICENSE`. Attribution: `NOTICE`.
- **Model derivatives** of the LFM2.5 base models are under the **LFM Open License v1.0**, which is not an
  OSI open-source licence: it limits commercial use above a revenue threshold (10M USD of annual revenue)
  and requires redistributing the licence text. Full text: `MODEL-LICENSE`.

The base weights are downloaded from Hugging Face and their licence travels with them. See `LICENSE` for
the pointer to both.

## Releases

Planned Hugging Face repositories, nothing published yet:

| variant | repository | base |
|---|---|---|
| full typed decisions | `AISidesKicks/selectia-2.6b` | `LiquidAI/LFM2.5-2.6B-Base` |
| full typed decisions | `AISidesKicks/selectia-1.2b` | `LiquidAI/LFM2.5-1.2B-Base` |
| YESMOM | `AISidesKicks/selectia-yesmom-350m` | `LiquidAI/LFM2.5-350M-Base` |
| YESMOM | `AISidesKicks/selectia-yesmom-230m` | `LiquidAI/LFM2.5-230M-Base` |

`scripts/stage_release.py <model dir> runs/release/selectia-2.6b` builds the upload folder (weights,
tokenizer, `selectia_config.json`, the inference-only subset of `selectia/`, `MODEL-LICENSE`, card), and
`scripts/upload_hf.py` pushes it. `create_repo(exist_ok=True)` in that script can create the repo itself,
so pre-creating them on the Hub is only needed if your token cannot create repositories.

Pushing needs a write token. Copy the template and fill it in, then load it into your shell:

```bash
cp .env.example .env        # put your write token in HF_TOKEN; .env is gitignored
set -a; source .env; set +a
```
