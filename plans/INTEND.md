# INTEND: decider-style one-pass typed decisions on LFM2.5 (2.6B / 1.2B / 350M / 230M)

## Goal

Reproduce the `Mapika/decider` "System One" recipe on Liquid AI LFM2.5 base models across four sizes, in two product forms:

**Full typed decisions** (all question types: Choice / Score / Noul):

- `LiquidAI/LFM2.5-2.6B-Base`
- `LiquidAI/LFM2.5-1.2B-Base`

**YESMOM** (tiny binary yes/no specialists; see below):

- `LiquidAI/LFM2.5-350M-Base`
- `LiquidAI/LFM2.5-230M-Base`

The full model reads a **state** plus a set of **typed questions** and returns, from one forward pass, a calibrated probability distribution per question over the options. No decoding, no parsing, no generated text. YESMOM is the same readout narrowed to a single Noul question (`P(yes)`) per forward pass. Reuse decider's public wire format, prompt layouts, data mixture, training loop and evaluation so results are comparable to `decider-2b`.

## Why LFM2.5 fits the recipe

decider's core is architecture-agnostic: run the backbone, take the hidden state at each `Answer k: (` slot, project it onto one label token per option (`A..J`, then wide `A..Z`/two-letter), and softmax over the valid options. That only needs `last_hidden_state` and `lm_head` rows, both present on `Lfm2ForCausalLM`.

Use the **Base** checkpoints, not Instruct. decider trains from `Qwen3.5-*-Base`; the LFM2.5-2.6B instruct model is post-trained to always emit `<think>` and is optimized for generation, which is irrelevant (and possibly harmful) for a hidden-state readout.

## Confirmed target architectures (from HF configs)

| | LFM2.5-2.6B-Base | LFM2.5-1.2B-Base | LFM2.5-350M-Base | LFM2.5-230M-Base |
|---|---|---|---|---|
| role | full typed decisions | full typed decisions | YESMOM (Noul) | YESMOM (Noul) |
| layers | 30 (22 conv + 8 attn) | 16 (10 conv + 6 attn) | 16 (10 conv + 6 attn) | 14 (8 conv + 6 attn) |
| hidden | 2048 | 2048 | 1024 | 1024 |
| vocab | 128000 | 65536 | 65536 | 65536 |
| embeddings | tied | tied | tied | tied |
| max positions | 128000 | 128000 | 128000 | 128000 |
| pad / bos / eos | 124893 / 124894 / 124900 | 0 / 1 / 7 | 0 / 1 / 7 | 0 / 1 / 7 |
| attn | 32 heads / 8 KV | 32 / 8 | 16 / 8 | 16 / 8 |
| conv | `conv_L_cache=3`, dim 2048 | same | `conv_L_cache=3`, dim 1024 | same |

## YESMOM: what it is

YESMOM is a family of tiny **yes/no-only** models built on the 350M and 230M LFM2.5 bases. It is not a new architecture: it is decider's `Noul` question type as the *only* supported question, which means:

- one state + one question per forward pass, output `P(yes)` (plus the derived boolean);
- no label-table/large-option path needed at all - the readout is the two `no`/`yes` option tokens;
- no Options block to render, so the prompt is much shorter;
- training data is the subset of decider's mixture whose questions are `noul`: `boolq`, `squad2`, `paws`, `civil_comments`, `aegis2`, plus teacher-written `noul` custom questions and the abstain/off-topic probes.

Because the target is far narrower, the 230M/350M models avoid the knowledge-recall weakness that limits the small full-decision models: a yes/no verification is a much smaller function to learn. The expected use is cheap, on-device, high-volume gating (policy checks, safety checks, entailment, answerability, form-fill `skip` decisions) served as a binary head.

YESMOM is Noul-only on the 350M and 230M bases: one state + one yes/no question per forward pass, output `P(yes)`, no Choice/Score and no label table. The 1.2B/2.6B remain full typed decision models; no Noul-only variants are built at those sizes.

Keep the 350M/230M strictly Noul-only even though the tokenizer supports the format. Mixing in Choice/Score at that capacity dilutes the binary calibration and is not the product.

## Key differences vs decider's Qwen3.5 targets (risks)

1. **Hybrid conv+attention backbone, not delta-net.** decider's `schema_engine.py`/`engine.py` cache "attention K/V of the full-attention layers + conv/recurrent state of the delta-net layers". For LFM2 the cache must hold GQA K/V for `full_attention` layers and the short-conv state (`conv_L_cache=3`) for `conv` layers. The prompt readout itself is unaffected; only the served engine is.
2. **Tokenizer / label table.** `label_table` is tokenizer-driven but assumes `A..Z` and enough two-letter uppercase strings are single tokens. Must verify per tokenizer. The 65536-vocab 1.2B may not supply 255 single-token labels; if not, cap `MAX_OPTIONS` (e.g. 50-100) or fall back to the 10-option narrow rendering.
3. **Tied embeddings.** `lm_head.weight` aliases `embed_tokens.weight`; the letter readout indexes embedding rows. Verify gradients flow and no double-init issues.
4. **No chat template / BOS handling.** decider feeds raw text (`Context:\n...\nQuestion: ...`). Do not use `apply_chat_template`. Check whether the LFM2.5 tokenizer prepends BOS and keep the renderer tokenizer-agnostic.
5. **`transformers` version skew.** 2.6B config declares `transformers_version: 5.9.0`, 1.2B declares `4.57.2`. Pin one version that loads both and test.
6. **Gradient checkpointing.** decider enables HF gradient checkpointing unconditionally. LFM2's custom conv kernel may not support it identically; make it conditional and test memory vs. correctness.
7. **License.** LFM Open License v1.0 (not Apache-2.0). Affects release naming/terms.

## Proposed approach

Fork decider's tokenizer-agnostic pieces rather than rewriting. Everything below the readout (`prompt.py`, `model.py`, `systemone.py`, `infer.py`, `evaluate.py`, `data/`) already works off `AutoTokenizer`/`AutoModelForCausalLM` and needs only small guards. The Qwen-specific engine is deferred.

### Phase 0 - Environment + feasibility spike

- Environment: pixi workspace `sclt` (Python 3.12, see section A). Add `torch`, `transformers>=5`, `datasets`, `accelerate`, `scikit-learn`, `numpy<2`, `huggingface_hub`, `pytest` with `pixi add`. Do not add `flash-linear-attention` (LFM2's short conv is native).
- Load all four bases with `AutoModelForCausalLM`, confirm `.model(...)` returns `last_hidden_state` and `.lm_head` exists.
- Tokenizer probe: count single-token labels for `A..Z` and two-letter combos; record the achievable `MAX_OPTIONS` per tokenizer (128k-vocab 2.6B vs the 65536-vocab 1.2B/350M/230M).
- Smoke test `slot_logits`: one `Example` -> logits `[N, MAX_OPTIONS]`, mask invalid options; and a Noul-only smoke test for the two small bases.
- **Exit criteria:** all four models load, readout produces finite logits, label capacity recorded.

### Phase 1 - Generalize the decider modules

- Vendor: `prompt.py`, `model.py`, `systemone.py`, `infer.py`, `evaluate.py`, `report.py`, `data/` (registry, `augment.py`, `mixture.py`, teacher generators), `probes/`, `bench/public_suite.py`, `tests/`.
- `model.py`: keep `slot_logits` unchanged; make `grad_ckpt` conditional; add a tied-embedding assertion; keep letter buffer registration.
- `prompt.py`: keep `label_table` tokenizer-driven; add the Phase 0 capacity assertion and a configurable `MAX_OPTIONS`; verify the narrow `\n(A) text` path tokenizes identically for the LFM2 tokenizer.
- Drop/guard the Qwen-specific `engine.py`, `schema_engine.py`, `fp8.py` until Phase 5; default inference path is the eager `DecisionModel` path in `infer.py` (`use_graphs=False`).
- Keep `systemone.py` unchanged (wire format, isolated levels, `annotate_indices`, `render_state`).
- Port `tests/` (CPU-only) and adapt expected strings for the LFM2 tokenizer.

### Phase 2 - Data

- Reuse decider's mixture and ~95-task registry (`decider/data/mixture.py`). Two options:
  - (a) full mixture (~1.47M examples / ~455M tokens), matching `scripts/train.sh full`;
  - (b) staged: core decision formats first (~100-150M tokens), then scale.
- **Recommend (b) staged** for the first working model: routing/choice/score/noul, custom questions, abstain augmentation, described options, JSON states, isolated levels. Scale to full only after eval gates pass.
- Teacher data: first reuse decider's shipped `teacher_data/` (label descriptions, custom questions, routing messages, situations) instead of regenerating; regenerate with a locally run teacher only if a gap appears.
- Keep `max_ctx=1536` and the same 24-28 held-out tasks for calibration comparability.
- Note token-budget differences from the 128k vs 65536 vocab (more tokens per example on the 2.6B tokenizer).
- **YESMOM subset:** filter the mixture to Noul-type questions (state + one `noul` question), plus teacher-written Noul custom questions and the abstain/off-topic probes. Balances yes/no, includes described criteria (`{"true": ..., "false": ...}`) and bare `no`/`yes`. Keep the full-decision mixture out of YESMOM. Consider oversampling the hard abstain/off-topic cases so the tiny binary head calibrates at the extremes.
- **Optional cross-size data reuse:** the same Noul subset, tokenized per tokenizer, feeds both the 350M and 230M runs; no separate data curation.

### Phase 3 - Training

**Full typed models (2.6B, 1.2B):**

- Reuse `train.py` unchanged except `--model`, grad-ckpt handling, and device/bucket sizing.
- Hyperparameters: AdamW, `lr=1e-5`, `wd=0`, `betas=(0.9,0.95)`, 200 warmup + cosine, `max_tokens=16384`, `accum=2`, `brier_w=0`, `label_smooth=0`, `none_prob` augmentation on, `schema_first_prob=0.5` (so the cacheable layout is trained even though state-first stays default).
- Full fine-tune both backbones; 2.6B needs gradient checkpointing and ~80GB-class memory or sharding; 1.2B is lighter.
- Save `model/` + `decider_config.json` (`temperature`, `isolated_levels`, `schema_first`, `version`), plus `hist.json` and per-step CE/eval logs.
- **Exit criteria:** in-task ECE < 0.06 after temperature fit; held-out accuracy clearly above base-model zero-shot.

**YESMOM (350M, 230M):**

- Same `train.py`, same readout, but `max_options`/label path irrelevant; the only question type is `noul`. `schema_first_prob` can be 0 (state-first is fine; the prompt is short).
- Smaller budgets: these fit on a single modest GPU and train in a fraction of the time. Higher LR is usually tolerable at this scale; start at the same `1e-5` and sweep `{3e-5, 1e-5, 3e-6}` on the 230M first.
- Do **not** add a Brier term initially; a single fitted temperature on in-task Noul data is the calibration control.
- Save `decider_config.json` with a `yesmom: true` flag so the runtime exposes only the binary head.
- **Exit criteria:** held-out Noul accuracy clearly above the base model zero-shot; post-temperature ECE < 0.05.

### Phase 4 - Evaluation

- Reuse `evaluate.py` / `report.py`: accuracy, NLL, Brier, ECE, AURC, selective accuracy per task; temperature fit on in-task, checked on held-out.
- Probes: question independence, isolated Score levels, abstain battery, custom-question battery.
- Baselines: `Qwen3.5-*-Base` zero-shot, `decider-2b` published numbers (in-task ~0.81 / held-out ~0.74), and the LFM2.5 sizes against each other.
- Optional external: Bespoke public suite (`bench/public_suite.py`), JevBench public items.
- Suggested gates (adjust after first run): 2.6B held-out acc >= 0.70, ECE <= 0.10; 1.2B held-out acc >= 0.65, ECE <= 0.12.
- **YESMOM eval:** accuracy, NLL, Brier and ECE on the Noul tasks only. Reference points from `decider-2b` on Bespoke's public suite: boolq 0.803, squad2 0.776, paws 0.720, civil_comments 0.840, aegis2 0.728. Gates: 350M >= 0.75 macro Noul accuracy and ECE <= 0.05; 230M >= 0.70 and ECE <= 0.07 (adjust after first run). Add a threshold sweep (selective yes/no at a confidence gate) since YESMOM's product use is gating.
- **YESMOM size check:** 230M vs 350M is the interesting comparison - decide whether the 230M is accurate enough to ship or whether 350M is the floor.

### Phase 5 - Serving engine (later, optional)

- Stage 1: eager bf16 `serve.py` + `/v1/systemone` and `/decide`; no graphs/FP8.
- Stage 2: schema cache. Implement a generic `prepare` that stores per-layer state: GQA K/V for `full_attention` layers and conv state for `conv` layers (instead of delta-net recurrent state). Reuse decider's read-only-prefix + suffix-graph design.
- Stage 3: CUDA graphs + FP8.
- Keep state-first as default (schema-first costs ~1.5 points on fixed label sets).
- **YESMOM serving is much simpler:** the prompt is a single fixed-shape binary question, so a plain CUDA graph over `(batch, length)` buckets is enough; no schema cache or label table. Expose a dedicated `/yes` route (state + question -> `{noul: p}`) plus the standard `noul` form of `/v1/systemone`.

### Phase 6 - Release

- Port `scripts/stage_release.py` / `upload_hf.py`; publish `decider-lfm25-2.6b`, `decider-lfm25-1.2b`, `yesmom-350m`, `yesmom-230m` with model cards stating the LFM Open License v1.0 and the measured calibration.
- Include the tokenizer probe result and `MAX_OPTIONS` used (full models); note YESMOM is Noul-only.

## Proposed repository layout

```
INTEND.md                      this plan
decider_lfm/
  prompt.py model.py systemone.py infer.py train.py evaluate.py report.py
  data/  probes/  bench/  serve.py
scripts/  train.sh evaluate.sh serve.sh stage_release.sh
tests/
```

## Ordered task list (implementation agent)

0. License layout (section H, do before vendoring): move `LICENSE` -> `CODE-LICENSE`, add `MODEL-LICENSE` (LFM Open License v1.0), replace `LICENSE` with a pointer.
1. Phase 0 spike: env, load all four bases, tokenizer/label probe, `slot_logits` smoke test. Record `MAX_OPTIONS`.
2. Vendor decider modules; make `model.py`/`prompt.py` tokenizer-agnostic; disable grad-ckpt if unsupported; port CPU tests.
3. Build the staged full-decision subset and the Noul-only subset; verify one training step on `--train_cap 2000`.
4. Train YESMOM 230M end-to-end first (cheapest, validates the whole pipeline); fit temperature; run Noul eval.
5. Train YESMOM 350M; compare against 230M; decide the shipping floor.
6. Train 1.2B full-decision; fit temperature; run held-out eval + probes.
7. Train 2.6B with gradient checkpointing; same eval + probes.
8. Compare both full models to base zero-shot and `decider-2b`; tune `none_prob`/`schema_first_prob` if ECE or abstain lags.
9. Scale data to the full mixture if gates hold; re-eval.
10. (Optional) eager serve + schema cache + graphs/FP8; dedicated YESMOM `/yes` route.
11. Release artifacts + model cards for the four models.

## Validation

- CPU unit tests for the request/answer layer and both prompt layouts, adapted to the LFM2 tokenizer.
- Tokenizer probe assertions (`A..J` present as single tokens; wide label capacity >= configured `MAX_OPTIONS`).
- Determinism check: same input -> same probabilities across runs; independence check (reordering questions does not change answers).
- Calibration check: ECE on held-out tasks after a single in-task temperature.
- YESMOM check: `P(yes)` calibration and accuracy on held-out Noul tasks, plus a confidence-gate threshold sweep.
- Smoke training run before any long run.

## Open questions / decisions to confirm

1. ~~**YESMOM scope:**~~ **Resolved:** Noul-only, one yes/no question per forward pass, on 350M and 230M only. 1.2B/2.6B stay full typed decision models.
2. **Scope of first milestone:** staged core-format model (recommended) vs. full decider mixture from the start.
3. **1.2B wide labels:** if the 65536-vocab tokenizer cannot supply 255 single-token labels, cap `MAX_OPTIONS` or keep narrow-only. Decided by Phase 0 probe; irrelevant for the Noul-only small models.
4. **Teacher data:** reuse decider's shipped `teacher_data/` (recommended) vs. regenerate with a locally run teacher.
5. **Instruct checkpoints:** base only (recommended) vs. also fine-tune the `<think>` instruct variants for comparison.
6. **RL/calibration stage (decider v10):** out of scope initially; revisit only after supervised models are calibrated.
7. **Release license:** LFM Open License v1.0 terms for redistributed fine-tunes.

---

# Implementation plan (verified against upstream 2026-09-22)

Actionable companion to the intent above. Derived from the planning handover plus a direct read of upstream
`Mapika/decider` at commit `c4daaac28af9fea95d627015cffa2dd5a5926ee6` (file tree and source of `prompt.py`,
`model.py`, `train.py`, `infer.py`, `systemone.py`, `scripts/train.sh`, `pyproject.toml`).

## A. Bootstrap: pixi env, docker isolation, scratch dirs

Repo state before work: `CODE-LICENSE`, `MODEL-LICENSE`, `LICENSE`, `NOTICE`, `pixi.toml`, `pixi.lock`, `plans/`.
No vendored decider code yet. The implementation session needs permissions to clone, to add pixi dependencies,
and to run docker under the project prefix.

### Interpreter and Python env (pixi, per AGENTS.md)

- This repo is already a pixi workspace: `pixi.toml` `name = "sclt"`, env `default`,
  `channels = ["conda-forge"]`, `platforms = ["linux-64"]`, `python = "==3.12"`. 3.12 is settled by the
  workspace; do not create a separate venv.
- Run everything inside `pixi shell` (or prefix with `pixi run`). Before installing, confirm you are in the
  project workspace; the project name is `sclt`.
- Add deps with `pixi add ...` (conda-forge); use `pixi add --pypi ...` or `[pypi-dependencies]` for packages not
  on conda-forge. Suggested set: `torch`, `transformers>=5`, `numpy<2`, `huggingface_hub`, `datasets`,
  `accelerate`, `scikit-learn`, `pytest`. Do NOT add `flash-linear-attention` (Qwen delta-net only; LFM2's short
  conv is native). Inspect the preinstalled toolset with `pixi list` and suggest expansions rather than swapping
  packages.
- decider's `scripts/train.sh` hardcodes `PY=${PY:-.venv312/bin/python}`. There is no `.venv312` here; run it as
  `PY=$(which python) scripts/train.sh ...` from the pixi shell, or change the script's default.
- Clone into scratch, not `/tmp`:
  `git clone https://github.com/Mapika/decider "$SCRATCH/decider-src"` and pin commit
  `c4daaac28af9fea95d627015cffa2dd5a5926ee6`.

### Scratch dirs (per AGENTS.md)

- `$SCRATCH` = `$PIXI_PROJECT_ROOT/scratch` (persists through crashes) - clone, dataset cache, logs, run artifacts.
- `$TMPDIR` = `/run/user/$UID/pixi_tmp/sclt` (tmpfs) - throwaway temp files only.
- Both are declared in `[activation.env]`, so they exist once the pixi shell is active.

### Docker (per AGENTS.md)

- Every container, volume, network, image tag created for this project uses the `slct-` prefix, including temp
  and test resources.
- Never stop, delete, or prune containers/volumes/networks you did not create; that needs explicit HITL approval.
- Phase 5 serving and any CUDA-graph/FP8 work runs in `slct-`-prefixed containers unless run directly on the host.

### AGENTS.md contradictions to fix (implementation agent)

- AGENTS.md says `pixi info | grep Name` should return `selectia`, but `pixi.toml` sets `name = "sclt"` (and the
  AGENTS heading says pixi `sclt`). Trust `sclt`; fix the AGENTS.md line.
- AGENTS.md says `$SCRATCH` is gitignored except `.gitkeep`, but `.gitignore` only ignores `.pixi/*`. Add
  `scratch/*` and `!scratch/.gitkeep`, and create `scratch/.gitkeep`.
- AGENTS.md says `$TMPDIR` is created by the pixi `default` task, but `pixi.toml` has no tasks; it comes from
  `[activation.env]`. Run `mkdir -p "$TMPDIR"` before first use.
- Deliberate name split: pixi workspace `sclt` vs docker prefix `slct-`. Keep as is, do not "fix" one to match.

Vendor into `decider_lfm/`:
- core: `prompt.py model.py systemone.py infer.py evaluate.py report.py metrics.py card_table.py train.py`
- data: `data/{__init__,core,mixture,rules,augment,tasks_agents,tasks_extended,tasks_heldout,teacher_contrastive,teacher_labels,teacher_questions,teacher_situations}.py`
- `probes/`, `bench/public_suite.py`, `tests/{test_prompt,test_rules,test_systemone}.py`, shipped `teacher_data/`, adapted `scripts/`
- keep upstream `LICENSE`/attribution (code is Apache-2.0).
- Defer to Phase 5: `engine.py schema_engine.py fp8.py serve.py`, `bench/{engine,latency,loadtest,schema}.py`, `games/`, `vision/`, `moe/`.

## B. File-level changes (the actual port)

**`decider_lfm/prompt.py`**
- `label_table` ends with `assert len(out) == MAX_OPTIONS and len({...}) == MAX_OPTIONS`, which hard-fails on
  the 65536-vocab tokenizers. Replace with: collect all single-token labels, `assert len(out) >= WIDE_MIN` (11),
  `assert len({i for _, i in out}) == len(out)`, return the **actual** capacity (do not pad to 255).
- Decouple head width from the module constant; keep `MAX_OPTIONS` only as an upper bound to validate `--max_options`.
- Wide branch of `build`/`_options_ids`: assert `max_options <= label_capacity(tok)`, else fall back to narrow.
- `letter_ids` A..J assertion stays (narrow path + first 10 label rows).
- Narrow rendering is literal text (`tok.encode("".join(lines))`, `"\n(A) option"`) so it is tokenizer-agnostic;
  only the wide path needs real single-token labels.
- Add `label_capacity(tok)`; cache is keyed by `id(tok)` — never share across sizes (2.6B 128k vocab vs other three 65536).

**`decider_lfm/model.py`**
- `slot_logits`: use `torch.arange(W.shape[0])` for the mask width (`W = self.lm.lm_head.weight[self.letters]`),
  removing the global-constant coupling.
- `grad_ckpt=True` is unconditional today -> default `False`; enable only for 2.6B, and test on it separately.
- Add tied-embedding assertion: `self.lm.lm_head.weight.data_ptr() == self.lm.model.embed_tokens.weight.data_ptr()`.
- Phase 0 must confirm `hasattr(self.lm, "model")` and `hasattr(self.lm, "lm_head")` on `Lfm2ForCausalLM`.
- `collate` unchanged.

**`decider_lfm/train.py`**
- Add `--grad_ckpt` (default False), `--max_options` (default probed capacity), `--device`, `--yesmom`.
- **Write `decider_config.json` into `out/model`** at save time. Upstream `train.py` saves only LM+tokenizer and
  never writes it, but `Decider` reads temperature/flags only from that file — it is the runtime contract.
  Full models: `temperature`, `isolated_levels`, `schema_first`, `version`; YESMOM adds `yesmom: true`.
- Keep loss/schedule exactly (CE over `golds>=0`, AdamW `(0.9,0.95)`, clip 1.0, warmup+cosine, `brier_w=0`).

**`decider_lfm/infer.py`**
- Eager path is default (`use_graphs=False`); guard `engine`/`schema_engine` imports so the port runs before Phase 5.
- Replace `MAX_OPTIONS` validation with `label_capacity(tok)`; honor `cfg["yesmom"]` (expose only `noul`).
- Keep `neutralize_options` (literal `"none of the above"` -> `"not listed here"`).

**`systemone.py`** unchanged. **tests**: adapt expected strings for the LFM2 tokenizer; add a capacity test
(`A..J` single tokens and `capacity >= configured MAX_OPTIONS` for the wide path).

## C. Phase 0 deliverable

`scripts/probe_lfm.py`, for each `LiquidAI/LFM2.5-{2.6B,1.2B,350M,230M}-Base`:
1. load with `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`;
2. assert `.model(...).last_hidden_state` and `.lm_head` exist;
3. run `label_table(tok)`, print capacity and whether `A..J` are single tokens;
4. render a toy `Example` narrow (`max_options=10`) and wide (`max_options=capacity`), run `slot_logits`;
5. Noul-only smoke test for 350M/230M.

Record capacity per tokenizer; it fixes `--max_options` for the 1.2B/2.6B runs.

## D. Data

- Full: `python -m decider_lfm.data.core --out data/tasks.pkl` then
  `python -m decider_lfm.data.mixture --base data/tasks.pkl --mode full ...` (mirrors `scripts/train.sh full`;
  `core.py` downloads ~95 public datasets — network/disk heavy, cache it).
- Staged milestone (recommended): `mixture.py` only has `full`/`delta`, so **add a `core` mode** (routing/choice/
  score/noul, custom questions, abstain augmentation, described options, JSON states, isolated levels) and a
  `yesmom` mode filtering to `noul` (`boolq`, `squad2`, `paws`, `civil_comments`, `aegis2`) + teacher noul +
  abstain/off-topic probes (oversample the hard abstain/off-topic cases). This is new code, not upstream.
- Reuse shipped `teacher_data/` instead of regenerating; tokenize per tokenizer; the same Noul subset feeds both 350M and 230M.

## E. Training commands

Full (2.6B / 1.2B) — upstream `train.sh` uses `--max_ctx 16384`; this plan keeps 1536 for memory:

```
python -m decider_lfm.train --model LiquidAI/LFM2.5-2.6B-Base \
  --data data/mixture_core.pkl --out runs/full_2.6b \
  --epochs 1 --lr 1e-5 --warmup 200 --max_tokens 16384 --accum 2 \
  --max_ctx 1536 --max_options <capacity> --none_prob 0.1 --schema_first_prob 0.5 \
  --brier_w 0 --grad_ckpt
```

YESMOM (230M first, then 350M): same but `--model ...-230M-Base --yesmom --schema_first_prob 0 --max_options 2`
(no `--grad_ckpt`); sweep `lr in {3e-5,1e-5,3e-6}` on 230M first.

## F. Gates

As in the intent section, plus: tokenizer-probe assertions, determinism/reordering-independence check, a smoke
step on `--train_cap 2000` before any long run, and a `decider_config.json` presence check before eval.

## G. Updated risk table

| risk | mitigation |
|---|---|
| `label_table` 255-assert | relax to `>= WIDE_MIN`, capacity-driven head width |
| 65536-vocab wide labels | only 1.2B needs wide; cap `MAX_OPTIONS` to probed capacity or stay narrow-only |
| `lm.model` / `lm.lm_head` path | Phase 0 assertions before any training |
| grad ckpt on LFM2 conv | default off; test on 2.6B only |
| transformers skew (5.9 / 4.57 / 5.0rc1 / 5.2) | pin one `transformers>=5`; load all four in Phase 0 |
| tied embeddings | data_ptr assertion; verify gradients flow, no double-init |
| `flash-linear-attention` | drop unless proven necessary |
| BOS / chat template | inspect tokenizer; renderer stays tokenizer-agnostic, no `apply_chat_template` |
| license | LFM Open License v1.0 for weights; Apache-2.0 for ported code |

## H. Licensing layout (do first)

Goal: split code and model licensing so the repo is unambiguous before any weights are released.

Target files:
- `CODE-LICENSE` - the current root `LICENSE` moved unchanged (Apache License 2.0, 201 lines). Stays the license for all code, including ported `Mapika/decider` modules.
- `MODEL-LICENSE` - the Liquid AI LFM Open License v1.0 full text, fetched from `LiquidAI/LFM2.5-2.6B-Base` `LICENSE` (HF card `license_name: lfm1.0`). Applies to fine-tuned/derivative weights and to any redistributed base-model files.
- `LICENSE` - new short pointer (not a license body) stating the split.
- `NOTICE` (recommended) - attribution for ported Apache-2.0 decider code.

Commands (implementation agent):
```
git mv LICENSE CODE-LICENSE
curl -L https://huggingface.co/LiquidAI/LFM2.5-2.6B-Base/raw/main/LICENSE -o MODEL-LICENSE
```
Then write `LICENSE` as:
```
Licensing
=========

1. Code in this repository is licensed under the Apache License, Version 2.0.
   Full text: CODE-LICENSE     SPDX-License-Identifier: Apache-2.0

2. Model derivatives of the Liquid AI LFM2.5 base models (fine-tuned weights and
   any files shipped with them, e.g. decider_config.json) are licensed under the
   LFM Open License v1.0.
   Full text: MODEL-LICENSE
   The LFM Open License v1.0 is not an OSI open-source license: it limits
   commercial use above a revenue threshold and requires redistributing the
   license text.

Base models are downloaded from Hugging Face (LiquidAI/LFM2.5-*-Base) and their
license travels with them. Ported code from Mapika/decider retains its Apache-2.0
attribution (see NOTICE).
```

Verification:
- `git status` shows the `LICENSE -> CODE-LICENSE` rename, plus new `MODEL-LICENSE` and rewritten `LICENSE`.
- Confirm the LFM license is byte-identical across all four model repos (2.6B/1.2B/350M/230M) before shipping one `MODEL-LICENSE`.
- `grep -rn "LICENSE" --include=*.md .` currently hits only the plan; point the Phase 6 model-card text at `MODEL-LICENSE`.
- No source file other than `LICENSE` references the license, so no code changes are needed.

Open decision: the Apache appendix still reads `Copyright [yyyy] [name of copyright owner]` (template, no holder). Recommendation: add `NOTICE` crediting Mapika/decider and leave the appendix templated until a holder is chosen, rather than inventing one.

Implementation note: this needs file edits and `git mv`, so it must run on an implementation-capable agent, not this plan agent.
