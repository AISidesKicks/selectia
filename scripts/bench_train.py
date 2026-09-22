"""Measure training throughput and peak GPU memory for one LFM2.5 size, so a README can state realistic budgets
instead of guessing.

    python scripts/bench_train.py --model LiquidAI/LFM2.5-230M-Base
    python scripts/bench_train.py --model LiquidAI/LFM2.5-2.6B-Base --grad_ckpt --steps 6

Synthetic prompts are built with the real `prompt.py` (a state of about `--ctx` tokens plus two questions), padded to
the same 64-token buckets `train.py` uses, so the numbers are the same shape of work.  Reports tokens/s, seconds per
optimizer step, peak allocated memory, and the wall clock projected for a token budget.

Add `--optim none` to measure forward+backward memory without the AdamW state, which is what tells you whether a size
fits a given card.
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selectia.infer import Q, Example                                     # noqa: E402
from selectia.model import DecisionModel, collate, pad_id                 # noqa: E402
from selectia.prompt import build                                         # noqa: E402
from selectia.train import loss_fn                                        # noqa: E402

FILLER = "The customer reported an issue with the order and asked for help. "
QUESTION_TAIL = 200                       # room for the two question blocks and both answer slots
BUDGETS = {"core (staged)": 130e6, "yesmom (noul subset)": 60e6, "full mixture": 455e6}


class _Keep:
    def shuffle(self, x): pass

    def sample(self, xs, k): return xs[:k]


def synthetic_example(tok, target_len, i):
    ctx, n = "", 0
    while len(tok.encode(ctx, add_special_tokens=False)) < max(32, target_len - QUESTION_TAIL):
        ctx += FILLER; n += 1
        if n > 4000:
            break
    qs = [Q("Which team should handle this?", ["billing", "technical", "sales"], i % 3),
          Q("Is this urgent?", ["no", "yes"], i % 2)]
    return Example(ctx, qs, "bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-230M-Base")
    ap.add_argument("--ctx", type=int, default=1536, help="target context tokens (the plan keeps max_ctx 1536)")
    ap.add_argument("--max_tokens", type=int, default=16384, help="tokens per micro-batch (B*T)")
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation, for the optimizer-step timing")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit", "adamw8bit_paged", "none"],
                    help="adamw8bit needs bitsandbytes: ~2 bytes/param of optimizer state instead of 8. "
                         "adamw8bit_paged also pages that state to CPU, trading PCIe traffic for VRAM")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--max_options", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = DecisionModel(a.model, dtype=getattr(torch, a.dtype), grad_ckpt=a.grad_ckpt).to(a.device)
    m.train()
    tok = m.tok
    print(f"[bench] {a.model} grad_ckpt={a.grad_ckpt} optim={a.optim} dtype={a.dtype} device={a.device}", flush=True)
    print(f"[bench] weights {sum(p.numel() for p in m.parameters())/1e6:.0f}M params", flush=True)

    probe = build(synthetic_example(tok, a.ctx, 0), tok, _Keep(), max_options=a.max_options, max_ctx_tokens=a.ctx)
    T = ((len(probe["ids"]) + 63) // 64) * 64
    B = max(1, a.max_tokens // T)
    b = collate([build(synthetic_example(tok, a.ctx, i), tok, _Keep(), max_options=a.max_options, max_ctx_tokens=a.ctx) for i in range(B)], pad_id(tok))
    b = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in b.items()}
    rows, cols = b["input_ids"].shape
    print(f"[bench] batch {rows} x {cols} = {rows*cols} padded tokens, {len(b['slot_idx'])} slots", flush=True)

    params = [p for p in m.parameters() if p.requires_grad]
    if a.optim == "adamw":
        opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    elif a.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    elif a.optim == "adamw8bit_paged":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        opt = None
    t0 = None
    for s in range(a.warmup + a.steps):
        if s == a.warmup:
            torch.cuda.synchronize(); t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        logits = m(b)
        loss, _ = loss_fn(logits, b["golds"], b["nopts"])
        loss.backward()
        if opt is not None:
            opt.step(); opt.zero_grad(set_to_none=True)
        else:
            for p in params:
                p.grad = None
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_step = rows * cols
    tps = tok_step * a.steps / dt
    sec_step = dt / a.steps
    peak = torch.cuda.max_memory_allocated() / 1e9
    rep = dict(model=a.model, params_m=round(sum(p.numel() for p in m.parameters()) / 1e6),
               batch=f"{rows}x{cols}", padded_tokens_per_step=tok_step, accum=a.accum, grad_ckpt=a.grad_ckpt,
               optim=a.optim, dtype=a.dtype, tokens_per_s=round(tps), s_per_micro_step=round(sec_step, 3),
               s_per_optim_step=round(sec_step * a.accum, 3), peak_gb=round(peak, 2))
    rep["projected_hours"] = {k: round(v / tps / 3600, 2) for k, v in BUDGETS.items()}
    print("[bench] " + json.dumps(rep, indent=1), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rep) + "\n")


if __name__ == "__main__":
    main()
