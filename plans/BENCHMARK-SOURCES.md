# Plan: external Jev benchmarks, calibration upgrades and the data-licence blocker

Follow-up to the temperature work. The question was what is usable in
`Luni/laya-jev-benchmark` and `Meanblock/JEV-CPU/benchmarks`, whether the Hub has more Jev
benchmarks, and whether any of it should go into `NOTES.md` and `README.md`.

Short answer: yes to all three, and one finding outranks all of them. Our staged `core` mixture trains
on `Tobi-Bueck/customer-support-tickets`, which is **CC-BY-NC-4.0**, so `selectia-core-1.2b` inherits a
non-commercial restriction. That is the same trap `Luni/laya-jev-benchmark` documents for its own
checkpoint, and it changes the release plan rather than the docs.

## What the two sources actually are

### `Luni/laya-jev-benchmark` (dataset, Apache-2.0, 186 downloads)

A single-author, reproducible head-to-head of **Laya** (`convaiinnovations/laya`, Apache-2.0, 421M)
against Jev, measured on one RTX 5090 on 2026-09-19. Ships `bench/bench_phish.py`, `bench/platt.py`,
`bench/probe.py`, `bench/eval.py`, `results/RESULTS.md` and raw logs.

Usable parts:

- **Two public item sets on which Jev has published numbers.** `AreLit/PhishNChips` (2,000 emails, 1,000
  phishing / 1,000 legitimate) and `LocalLLaMA/typed-decisions` (400 cases, 2,000 decisions). Neither
  model was trained on them.
- **`bench/platt.py`: Platt scaling with a bias term, fitted on one half and scored on the other.** The
  argument is precise and matters to us: temperature scaling is `softmax(z/T)`, multiplicative with no
  bias term, so it can push confidence toward 0.5 but never across the decision boundary. When a model
  ranks well but sits entirely on one side of the threshold, only a bias term fixes it. Their case:
  Laya raw accuracy 0.505 at recall 0.012 with AUROC 0.678 against Jev's 0.689, so the ranking is nearly
  identical and only the threshold is wrong. Fitting `sigma(a*logit + b)` on 1,000 emails lifts accuracy
  to 0.611 with temperature unable to do it.
- **`bench/probe.py`: 11 assertions** over grounding, contradiction, option-rename stability and
  overconfidence. Examples: `P(needs a Human) + P(a Bot can resolve it)` should be about 1.0, and
  `P(Phishing) + P(legitimate Sender)` should be about 1.0. Laya answered both "no" at confidence 0.94
  and 0.97, giving sums of 0.09 and 1.73.
- **Reference points.** Jev published on PhishNChips: accuracy 0.626, ECE 0.154, AUROC 0.689, recall
  0.432, p50 239 ms. Jev 1.13.0 on typed-decisions: accuracy 0.727, soft accuracy 0.580, ECE 0.144,
  Brier 0.148, 710 ms per case. Claude Haiku 4.5 on PhishNChips: 0.813, ECE 0.097, AUROC 0.837. Their
  own fine-tune reference: 180k items, three epochs, **55 minutes**, held-out 0.838 with macro ECE 0.166.
- **Their stated limitations**, worth copying into our own notes: ECE computed against argmax correctness
  is wrong on soft targets; the probe suite is 11 self-written assertions over two examples and is not a
  benchmark; option renaming still moves the verdict after fine-tuning; the fine-tune made phishing worse.

### `Meanblock/JEV-CPU` (model, MIT, Qwen3-0.6B, "semif")

A reranker-style decision model with an unusually complete benchmark directory.

Usable parts:

- **Fixtures**: `benchmarks/data/authored144.jsonl` (144 owned labelled items), `shape777.jsonl` (777
  rows, 6.7 MB, SHA-256 pinned) and `perturbations108.jsonl`, a stability fixture derived from 36
  originals with a manifest.
- **A metric contract**: `benchmarks/manifests/metric-contract.json`, `evaluation-matrix.jsonl` (706 row
  ids frozen) and `source-selection.jsonl`.
- **`benchmarks/decision_vs_generation.py`**: measures the typed-decision path against a generated
  baseline that is asked for "an ordered JSON array of yes/no strings". This is a direct test of the
  claim we make in our README and have never measured.
- **Statistics**: `evaluate.py` reports hard-label accuracy, **balanced accuracy**, F1, probability
  metrics and **paired source-group bootstrap intervals**.
- External sets: WANLI (CC-BY-4.0), "Every" (public), and a TypeSafe-102 set that is not redistributed.

## The Hub landscape

**Benchmarks and evaluation data**

| repo | licence | size | what it gives us |
|---|---|---|---|
| `Praveenrajus/jev-bench` | other | 22 configs, 166,054 rows, 22,773 test records, 4 calibration-gold configs | real human labels reformatted as System One questions, with human label distributions, and `jev-1.13.0` scored on every test record |
| `ZefanCai/Open-Jev` | CC0-1.0 | 12 frozen configs | synthetic controlled typed-decision tasks with reference labels |
| `ZefanCai/Open-Jev-v1.1` | other | 326,619 records, 147,139 train | the 27B training mixture projection |
| `SargeDev/jev-distill-corpus-v3` | Apache-2.0 | 740,957 rows | a calibrated typed-decision corpus in the noul/choice/score schema, built for training small local System One judges |
| `tasksource/tasksource-jev` | other | 100K to 1M | Tasksource classification and MC recast as runtime-defined decisions, multilingual |
| `vagmi/jevlite_dataset` | CC-BY-SA-4.0 | 5,866 questions / 978 states | teacher soft labels rather than a single label |
| `ctaxnagomi/INSTRUCT_JEV` | MIT | under 1K | instruction corpus built from the TypeSafe docs |
| `Luni/laya-jev-benchmark` | Apache-2.0 | 2 item sets | above |

**Models**, roughly 15 open reproductions, all created 2026-09-17 to 2026-09-23. Two architectural
families, which is the useful observation: **BERT-class cross-encoders** (`com-kotobalabs/open-jev-deberta-v3-large`,
`mobarmg/jev-schema-scorer-deberta-v3-large`, `tasksource/modernbert-tasksource-jev`,
`argos1111/modernbert-ja-310m-jev`) and **small causal LMs with a non-generative readout**
(`chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF`, `ZefanCai/Open-Jev-2B/9B/27B-v1.1`,
`lostargon/Tiny-Jev`, `samatv256/mini-Jev`, `NicolaiMTLassen/bonzi-*-jev`). Ours is the second family, so
the cross-encoders are the honest comparison for "is a 1.2B causal readout worth it".

## Work, in priority order

### 0. Data-licence audit before anything is published (blocker)

`Tobi-Bueck/customer-support-tickets` is `license:cc-by-nc-4.0` and is used by the `support_tickets`
task at `selectia/data/core.py:155`. `facebook/anli`, the other dataset Luni flags, is **not** in our
registry, so this is the one known offender.

1. Audit all 97 tasks in the registry for their licence, recording the HF `license:` tag per task.
2. Decide per offender: drop the task and rebuild the mixture, or disclose on the model card.
3. Update `README.md` licensing: our story is currently "Apache-2.0 code plus LFM weights". If any
   training task is non-commercial, the released weights carry that too and the README must say so.

### 1. Record the landscape (the direct ask)

- `NOTES.md`: a "Related work and external benchmarks" section with the tables above, the published Jev
  numbers, and Luni's stated limitations quoted rather than paraphrased.
- `README.md`: one short paragraph plus a comparison table so a reader can see our numbers next to Jev's
  published ones, with the caveat that Jev's figures come from other item sets.
- Add the "n=74 means plus or minus 0.11" caveat next to every JevBench subset number, since this is the
  first thing a careful reader will ask.

### 2. Platt and bias calibration, not just temperature

This is the highest-value technical adoption, because our YESMOM gates have exactly the pathology Luni
describes and we already measured the symptom (mean confidence 0.686 against accuracy 0.563).

- Extend `scripts/fit_temperature.py` into a calibration fitter that compares, on a held-out half:
  raw, temperature `softmax(z/T)`, and **Platt `sigma(a*z + b)`** for binary outputs.
- Report accuracy, Brier, ECE and **AUROC** before and after, so a threshold-only fix is visible as
  accuracy moving while AUROC stays flat.
- Only apply a bias term where the output is genuinely binary (`noul`, and the YESMOM gates). For
  `choice` and `score` a bias per label is a different, riskier model; keep temperature there.
- Persist the chosen form in `selectia_config.json` (`"calibration": {"kind": "platt", "a": ..., "b": ...}`)
  and have `Selectia` apply it, so the runtime contract stays in one file.
- Guard against the failure we already found: fit and select by **NLL or Brier**, never by ECE, because a
  large T flattens everything and minimises ECE while destroying the information.

### 3. A third benchmark with published Jev numbers

- Add `AreLit/PhishNChips` and `LocalLLaMA/typed-decisions` as evaluation-only sources, run all four
  checkpoints, and report accuracy, ECE, AUROC and Brier beside Jev's published 0.626/0.154/0.689 and
  0.727/0.144/0.148.
- Evaluate only. Do not train on them: PhishNChips is the basis of a licence argument and typed-decisions
  is the source of Luni's own non-commercial restriction.
- This is also the natural place for the Platt experiment, because that is where Jev's threshold
  behaviour is documented.

### 4. Adopt the probe suite

- New `selectia/probes/assertions.py` implementing the 11 assertion families: complementarity sums
  (`P(human) + P(bot) ~= 1`), grounding (facts stated in the state must be answered high), contradiction
  (opposite questions must not both be low), option-rename stability, and overconfidence.
- Fix the two fixtures they used, one support ticket and one phishing email, so the output is comparable
  line by line with their table, and report a pass count like theirs (Laya 7 of 11, their fine-tune 1 of 11).
- Keep the honest framing: 11 assertions over two examples is a smoke test, not a benchmark.

### 5. Rename and perturbation stability

We measured option *order* (labels vs criteria, small effect) but not option *renaming*. The mixture
already has the machinery: `named` and `opaque` renderings.

- Evaluate the same held-out items under named and opaque labels and report the verdict flip rate.
- `Meanblock`'s `build_perturbations.py` and `perturbations108.jsonl` give a ready-made perturbation
  methodology if we want more than renaming.

### 6. Decision-versus-generation baseline

- Port the idea from `benchmarks/decision_vs_generation.py`: ask the same base model to emit an ordered
  JSON array of `"yes"`/`"no"` for the same items, and compare accuracy, format-failure rate and latency
  against our one-pass readout.
- This converts our README claim ("no decoding, no parsing") from an assertion into a measured one, and it
  is cheap: a few hundred items.

### 7. Statistics

- Adopt paired source-group bootstrap intervals from `Meanblock`'s `evaluate.py` for our per-set numbers.
- This directly fixes the weakest point in the current notes: JevBench subsets of n=74 are reported as
  point estimates.

### 8. Candidate training data (later, gated on step 0)

`SargeDev/jev-distill-corpus-v3` (Apache-2.0, 740,957 rows, exactly our schema) and
`Praveenrajus/jev-bench` (human labels, licence "other") are the two credible additions. Treat them as a
separate ablation with a licence check first, not as a change to the decider port, since our story is
that the mixture is ported from upstream.

## Out of scope

- Reproducing every open Jev model. The useful comparison is one cross-encoder family representative, and
  only if the README needs the positioning.
- `ZefanCai/Open-Jev` and `Open-Jev-v1.1` as training data: synthetic and controlled, so they would make
  our held-out numbers look better for reasons that are not real.
- Any change to the 2.6B plan. It stays blocked on hardware.

## Validation

- The licence audit produces a machine-readable per-task licence table committed next to the registry.
- Calibration: the fitter prints raw against fitted for accuracy, Brier, ECE and AUROC on a held-out
  half, and the degenerate all-flatten solution is visible as ECE improving while Brier worsens.
- PhishNChips and typed-decisions numbers land beside Jev's published ones with the item sets pinned by
  revision.
- The probe suite prints a pass count per checkpoint and reproduces Luni's table shape.
- Decision-versus-generation reports a format-failure rate for the generated path, which is the number
  that makes our "no parsing" claim falsifiable.
- All of it on the 4070, with `scripts/gpu_watch.py` confirming the card is actually working.
