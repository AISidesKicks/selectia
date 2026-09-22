# Upstream

`selectia/` is a port of [Mapika/decider](https://github.com/Mapika/decider) at commit
`c4daaac28af9fea95d627015cffa2dd5a5926ee6`, licensed Apache-2.0 (see `CODE-LICENSE` and `NOTICE`).
`selectia` is the product name for this LFM2.5 fork; the upstream project keeps its own name wherever
it is cited. The Qwen3.5 delta-net engine is intentionally not vendored yet.

## What changed in the port

- **Naming**: the package, the runtime class (`Decider` -> `Selectia`), the config file
  (`decider_config.json` -> `selectia_config.json`) and the model repo names all moved to `selectia`.
  Every `decider.*` import became `selectia.*`.
- **`prompt.py`**: `label_table` no longer hard-asserts `MAX_OPTIONS == 255`. It returns the tokenizer's
  actual single-token label capacity (asserting `>= WIDE_MIN`), and `label_capacity(tok)` exposes it.
  `MAX_OPTIONS` is now only a ceiling for `--max_options`. The label/option caches key on
  `(name_or_path, vocab size)` instead of `id(tok)`, which could be recycled after a tokenizer was collected.
- **`model.py`**: head width follows the label capacity (`torch.arange(W.shape[0])`), `grad_ckpt` defaults to
  `False`, tied embeddings and the `lm.model` / `lm.lm_head` paths are asserted at construction, and a
  `pad_id(tok)` helper covers tokenizers without a pad token.
- **`train.py`**: adds `--grad_ckpt`, `--device`, `--yesmom`, `--isolated_levels`, `--temperature`, `--version`;
  `--max_options 0` means "auto = tokenizer capacity". It writes `selectia_config.json` into the saved model
  folder, which upstream never did but the runtime class reads at load time.
- **`infer.py`**: option validation uses `label_capacity`, and `selectia_config.json`'s `yesmom` flag restricts
  the runtime to Noul (no/yes) questions.
- **Dropped until Phase 5**: `engine.py`, `schema_engine.py`, `fp8.py`, `serve.py`, `bench/{engine,latency,loadtest,schema}.py`,
  `games/`, `vision/`, `moe/`. The eager `DecisionModel` path is the default.

Everything else (`systemone.py`, `evaluate.py`, `report.py`, `metrics.py`, `data/`, `probes/`) is upstream
code with the import rename only.
