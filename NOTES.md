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

Two environment notes for a GPU run:

- The probe ran on CPU because the local RTX 4070 was fully occupied by other processes. Nothing was
  stopped or evicted.
- LFM2's short conv falls back to the reference PyTorch kernel (`causal_conv1d` is not installed).
  That is correct but slow; `pixi add --pypi causal-conv1d` is the optional speedup for training on
  a free GPU.
