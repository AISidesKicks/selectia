"""Phase 0 spike: load every LFM2.5 base, verify the readout paths and record the label capacity.

    python scripts/probe_lfm.py [--models LiquidAI/LFM2.5-350M-Base] [--no-weights] [--device auto]

For each base this checks:
  1. the tokenizer's single-token label table (capacity, A..J single tokens, pad/bos/eos);
  2. `AutoModelForCausalLM` exposes `.model(...).last_hidden_state` and `.lm_head`, tied embeddings included;
  3. a toy narrow and a toy wide `Example` produce finite `[N, capacity]` slot logits with the invalid options masked;
  4. a Noul-only (yes/no) smoke test for the 350M / 230M bases;
  5. determinism: the same input twice gives bit-identical probabilities.

Writes a JSON report (default `$SCRATCH/probe_lfm.json`) that fixes `--max_options` for the training runs.
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer                                                          # noqa: E402
from decider_lfm.infer import Q, Example                                                        # noqa: E402
from decider_lfm.model import DecisionModel, collate, pad_id                                    # noqa: E402
from decider_lfm.prompt import MAX_OPTIONS, WIDE_MIN, build, label_capacity, label_table, letter_ids   # noqa: E402

BASES = ["LiquidAI/LFM2.5-2.6B-Base", "LiquidAI/LFM2.5-1.2B-Base",
         "LiquidAI/LFM2.5-350M-Base", "LiquidAI/LFM2.5-230M-Base"]
YESMOM = {"LiquidAI/LFM2.5-350M-Base", "LiquidAI/LFM2.5-230M-Base"}


class _Keep:
    def shuffle(self, x): pass

    def sample(self, xs, k): return xs[:k]


def probe_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    names, ids, open_ids = label_table(tok)
    cap = label_capacity(tok)
    letters_single = all(tok.encode(L, add_special_tokens=False) == [ids[j]] for j, L in enumerate("ABCDEFGHIJ"))
    return tok, dict(model=name, vocab=len(tok), capacity=cap, wide_min=WIDE_MIN, max_options=MAX_OPTIONS,
                     a_to_j_single=letters_single, first_wide_label=names[10] if cap > 10 else None,
                     open_ids=open_ids, pad=tok.pad_token_id, bos=tok.bos_token_id, eos=tok.eos_token_id,
                     letter_ids=letter_ids(tok))


def _on(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def probe_readout(m, tok, cap, noul_only):
    """Toy narrow / wide / noul prompts through slot_logits. Returns a dict of shapes and checks."""
    ex_narrow = Example("The user reports a duplicate charge on their invoice.",
                        [Q("Which team should handle this?", ["billing", "technical", "sales"], 0)])
    n_wide = max(WIDE_MIN, min(cap, 60))
    wide_opts = [f"category {i}" for i in range(n_wide)]
    ex_wide = Example("The user reports a duplicate charge on their invoice.",
                      [Q("Which category?", wide_opts, n_wide - 1)])
    ex_noul = Example("A customer says they were charged twice for one order.",
                      [Q("Is this a duplicate charge?", ["no", "yes"], 1)])

    out = {}
    for tag, ex, mo in [("narrow", ex_narrow, 10), ("wide", ex_wide, cap), ("noul", ex_noul, 10)]:
        if noul_only and tag != "noul":
            continue
        it = build(ex, tok, _Keep(), max_options=mo)
        b = _on(collate([it], pad_id(tok)), next(m.parameters()).device)
        logits = m.slot_logits(b["input_ids"], b["attention_mask"], b["slot_idx"], b["slot_batch"], b["nopts"])
        probs = torch.softmax(logits.float(), -1)[0]
        finite = int(torch.isfinite(logits[0, : it["nopts"][0]]).sum())
        out[tag] = dict(nopts=it["nopts"][0], logits=tuple(logits.shape), finite_options=finite,
                        masked_tail_finite=bool(torch.isfinite(logits[0, it["nopts"][0]:]).any()),
                        p_sum=round(float(probs.sum()), 6), p_max=round(float(probs.max()), 6))
        assert logits.shape[0] == 1 and logits.shape[1] == cap, f"{tag}: unexpected logits shape {tuple(logits.shape)}"
        assert finite == it["nopts"][0], f"{tag}: valid options are not all finite"
        assert not torch.isfinite(logits[0, it["nopts"][0]:]).any(), f"{tag}: invalid options were not masked"
        assert abs(float(probs.sum()) - 1.0) < 1e-4, f"{tag}: probabilities do not sum to 1"
    if "noul" in out:
        assert out["noul"]["nopts"] == 2 and out["noul"]["finite_options"] == 2
    return out


@torch.no_grad()
def probe_determinism(m, tok):
    ex = Example("Determinism check.", [Q("Pick one.", ["a", "b", "c"], 1)])
    it = build(ex, tok, _Keep(), max_options=10)
    b = _on(collate([it], pad_id(tok)), next(m.parameters()).device)
    a = torch.softmax(m.slot_logits(b["input_ids"], b["attention_mask"], b["slot_idx"], b["slot_batch"], b["nopts"]).float(), -1)
    c = torch.softmax(m.slot_logits(b["input_ids"], b["attention_mask"], b["slot_idx"], b["slot_batch"], b["nopts"]).float(), -1)
    return bool(torch.equal(a, c))


def load_model(name, dtype, device):
    m = DecisionModel(name, dtype=dtype, grad_ckpt=False).to(device).eval()
    assert hasattr(m.lm, "model") and hasattr(m.lm, "lm_head")
    tied = bool(getattr(m.lm.config, "tie_word_embeddings", False))
    same = m.lm.lm_head.weight.data_ptr() == m.lm.model.embed_tokens.weight.data_ptr()
    dev = next(m.parameters()).device
    ids = torch.tensor([[1, 2, 3]], device=dev)
    h = m.lm.model(input_ids=ids, attention_mask=torch.ones(1, 3, dtype=torch.long, device=dev)).last_hidden_state
    return m, dict(tie_word_embeddings=tied, tied_ptr_equal=same, last_hidden_state=tuple(h.shape), hidden=h.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=BASES)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu (auto falls back to cpu on OOM)")
    ap.add_argument("--weights", default=True, action=argparse.BooleanOptionalAction, help="load the backbone (tokenizer probe is always run)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "scratch"), "probe_lfm.json"))
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)

    report = {"dtype": a.dtype, "torch": torch.__version__, "models": {}}
    for name in a.models:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        tok, trep = probe_tokenizer(name)
        print(f"[tokenizer] vocab={trep['vocab']} capacity={trep['capacity']} A..J single={trep['a_to_j_single']} "
              f"pad={trep['pad']} bos={trep['bos']} eos={trep['eos']}", flush=True)
        rep = dict(tokenizer=trep)
        if a.weights:
            dev = a.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            if dev == "cuda":
                torch.cuda.empty_cache()
            try:
                m, mrep = load_model(name, dtype, dev)
            except Exception as e:                                  # OOM and friends: fall back, never abort the probe
                if dev == "cpu":
                    raise
                print(f"[load] {type(e).__name__} on {dev}: {str(e).splitlines()[0]}; falling back to cpu", flush=True)
                torch.cuda.empty_cache(); dev = "cpu"
                m, mrep = load_model(name, dtype, dev)
            rep["load"] = dict(mrep, device=dev)
            rep["readout"] = probe_readout(m, tok, trep["capacity"], noul_only=name in YESMOM)
            rep["deterministic"] = probe_determinism(m, tok)
            print(f"[load] device={dev} hidden={mrep['hidden']} tied={mrep['tie_word_embeddings']} "
                  f"ptr_equal={mrep['tied_ptr_equal']} deterministic={rep['deterministic']}", flush=True)
            print(f"[readout] {json.dumps(rep['readout'])}", flush=True)
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rep["seconds"] = round(time.time() - t0, 1)
        report["models"][name] = rep

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n[report] {a.out}", flush=True)


if __name__ == "__main__":
    main()
