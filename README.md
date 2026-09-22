# selectia

One-pass typed decisions on small open-weight models. Give the model a **state** and a set of **typed
questions**; it returns a calibrated probability distribution per question from a single forward pass.
No decoding, no parsing, no generated text, and no answer outside the options you defined.

This is an educational reproduction of the "System One" model class (TypeSafe's *Jev*) and of
[Mapika/decider](https://github.com/Mapika/decider), moved from Qwen3.5 onto Liquid AI's
[LFM2.5](https://huggingface.co/LiquidAI) base weights. The readout, prompt layout, data mixture and
training loop are ported from that upstream project (Apache-2.0, see `NOTICE`); **selectia** is this
fork's own name. The write-ups live in `NOTES.md` and `plans/INTEND.md`.

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
| fits on | a 40-80 GB class GPU for a full fine-tune (12 GB is not enough for the 1.2B) | 3-4 GB of VRAM, trains on one RTX 4070 |

YESMOM is not a different architecture. It is the `Noul` question type as the *only* supported
type, so the readout collapses to the `no` / `yes` option tokens and the tiny bases never have to
carry a 255-wide label table. Keeping it Noul-only is deliberate: mixing Choice and Score into a 230M
model dilutes the binary calibration that is the whole product.

## The three question types

`choice`, `score` and `noul`, exactly the primitives from the System One definition. Ids are never
shown to the model; `instructions` and every description may be a string or any JSON value.

```python
{"type": "choice", "instructions": "Which department should handle this?",
 "criteria": {"billing": "charges and refunds", "technical": None, "sales": {"not_for": "support"}}}

{"type": "score",  "instructions": "How urgent is this?",
 "criteria": ["none", "low", "medium", "high"]}          # 2..10 levels, or a {"0": ..., "1": ...} legend

{"type": "noul",   "instructions": "Does this request need a refund?",
 "criteria": {"true": "money back", "false": "anything else"}}   # criteria are optional
```

Answers come back typed and normalised:

```python
{"type": "choice", "choice": "billing", "confidence": 0.97, "certainty": 0.91,
 "probabilities": {"billing": 0.97, "technical": 0.02, "sales": 0.01}}

{"type": "score", "score": 1.3, "legend": {"0": "none", "1": "low", "2": "medium", "3": "high"},
 "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4, "3": 0.0}}      # plus level_fit and fit_mass when isolated

{"type": "noul", "noul": 0.95}
```

Score levels are scored one at a time when the model's config turns on `isolated_levels` (our full
models do), so a level never sees its own number or its neighbours, and `fit_mass` tells you when no
level, or several, fit.

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
`-inf` before the softmax, and nothing is ever generated. A fitted temperature from
`selectia_config.json` is the calibration control. Because there is no decoding, all N questions cost
one forward pass, and question order does not change any answer.

Two consequences worth knowing. Each option label must be a **single token** in the tokenizer (all
four LFM2.5 tokenizers supply the full 255, see `NOTES.md`), and the models are full fine-tunes, not
prompted general models, which is what separates this from a chat model with a grammar constraint.

## Status

Work in progress, and honest about it.

- Done: the port (`selectia/`), the prompt/readout/label-capacity work, the staged `core`
  and Noul-only `yesmom` data modes, the Phase 0 probe, tests (19 passing), and measured training
  budgets on one RTX 4070.
- Done: two smoke runs on the 230M base (full-decision and `--yesmom`) complete a real optimizer loop,
  save the weights, and write a `selectia_config.json` that the runtime reads back.
- Not done: no trained or released weights yet, so no accuracy or ECE numbers. The next step is
  building `data/tasks.pkl` (`python -m selectia.data.core`) and the real mixture.

On one 12 GB GPU: YESMOM trains (230M about 2 h, 350M about 3 h for the staged token budget). The
1.2B and 2.6B full-decision runs need a bigger card or an 8-bit optimizer. The measurements, and what
does not fit, are in `NOTES.md`.

## Running it

```bash
pixi run python scripts/probe_lfm.py                    # Phase 0: load the bases, record label capacity
pixi run python -m pytest tests -q                      # CPU tests, no model needed
pixi run python scripts/bench_train.py --model LiquidAI/LFM2.5-230M-Base --grad_ckpt
scripts/train.sh yesmom                                 # staged Noul-only run on the 230M base
scripts/train.sh core                                   # staged full-decision run
scripts/evaluate.sh runs/selectia_core/model             # accuracy, NLL, Brier, ECE, AURC, probes
```

Everything runs inside the `sclt` pixi environment (Python 3.12, torch, transformers 5, numpy<2). Run
`pixi shell` once, or prefix with `pixi run`.

## Layout

```
selectia/   the port: prompt.py, model.py, systemone.py, infer.py, train.py, evaluate.py,
               data/ (the task registry, the mixture and the augmentations), probes/
scripts/       probe_lfm.py, bench_train.py, train.sh, evaluate.sh, stage_release.py, upload_hf.py
tests/         CPU tests for the request/answer layer, the prompt layouts and the rule data
plans/         INTEND.md, the implementation plan
NOTES.md       findings, the label-capacity table and the training budget measurements
```

## Licensing

Two licences, deliberately split, because code and weights are not the same thing:

- **Code** in this repository is Apache-2.0, including the modules ported from Mapika/decider
  (`selectia/`). Full text: `CODE-LICENSE`. Attribution: `NOTICE`.
- **Model derivatives** of the LFM2.5 base models are under the **LFM Open License v1.0**, which is
  not an OSI open-source licence: it limits commercial use above a revenue threshold (10M USD of
  annual revenue) and requires redistributing the licence text. Full text: `MODEL-LICENSE`.

The base weights are downloaded from Hugging Face and their licence travels with them. See `LICENSE`
for the pointer to both.

## Releases

Planned Hugging Face repositories, nothing published yet:

| variant | repository | base |
|---|---|---|
| full typed decisions | `AISidesKicks/selectia-2.6b` | `LiquidAI/LFM2.5-2.6B-Base` |
| full typed decisions | `AISidesKicks/selectia-1.2b` | `LiquidAI/LFM2.5-1.2B-Base` |
| YESMOM | `AISidesKicks/selectia-yesmom-350m` | `LiquidAI/LFM2.5-350M-Base` |
| YESMOM | `AISidesKicks/selectia-yesmom-230m` | `LiquidAI/LFM2.5-230M-Base` |

`scripts/stage_release.py <model dir> runs/release/selectia-2.6b` builds the upload folder (weights,
tokenizer, `selectia_config.json`, the inference-only subset of `selectia/`, `MODEL-LICENSE`, card),
and `scripts/upload_hf.py` pushes it. `create_repo(exist_ok=True)` in that script can create the repo
itself, so pre-creating them on the Hub is only needed if your token cannot create repositories.

Pushing needs a write token. Copy the template and fill it in, then load it into your shell:

```bash
cp .env.example .env        # put your write token in HF_TOKEN; .env is gitignored
set -a; source .env; set +a
```
