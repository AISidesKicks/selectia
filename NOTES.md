# Notes

## Original idea

The original idea is based on the AISidesKicks experiment with YESMOM: yes/no logit-based
experiments on small (under 500M) LFM2.5 models.

That is, instead of getting a model to generate a "yes" or "no" token, read the logits for the
`yes` / `no` tokens directly at a fixed prompt slot and use their ratio as a calibrated
probability. The decider recipe (see `plans/INTEND.md`) generalizes that readout from two tokens
to the full typed-decision label set, and YESMOM narrows it back to the binary head for the
350M / 230M LFM2.5 bases.
