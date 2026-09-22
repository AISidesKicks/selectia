"""Stage 1: proper-scoring-rule fine-tune (CE, optionally + Brier) on the multi-task decision mixture."""
import argparse, json, math, os, pickle, random, time

# Measured on the 4070: without this the allocator leaves ~0.7 GB reserved-but-unallocated and a run that
# fits by the numbers OOMs in backward. Set before torch's first CUDA allocation, which reads it lazily.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np, torch, torch.nn.functional as F
from selectia.model import DecisionModel, collate, pad_id
from selectia.prompt import build, MAX_OPTIONS, label_capacity
from selectia.evaluate import run_eval, aggregate
from selectia import data as D


from selectia.data.augment import none_augment, build_label_pool


def make_items(train, tok, rng, max_ctx, none_prob=0.0, max_options=10, schema_first_prob=0.0):
    from array import array
    items = []; pool = build_label_pool(train) if none_prob > 0 else None
    for i, e in enumerate(train):
        it = build(none_augment(e, rng, none_prob, pool), tok, rng, max_options=max_options, max_ctx_tokens=max_ctx,
                   layout="schema_first" if rng.random() < schema_first_prob else "state_first"); it["task"] = e.task; it["ex_id"] = i
        # 4 bytes per token id instead of a ~36-byte int object each. The staged core mixture is ~186M tokens, which
        # as Python lists is 6-7 GB held for the whole run; as array('i') it is under 1 GB. `collate` and
        # `batches_by_tokens` only call len() and torch.tensor() on this, both of which take a buffer.
        it["ids"] = array("i", it["ids"])
        items.append(it)
    return items


def batches_by_tokens(items, max_tokens, rng, bucket=64):
    """Deterministic shape buckets: T is padded to a multiple of `bucket`, and every
    batch in a T-bucket has exactly B = max_tokens // T rows (one ragged batch per bucket).
    Keeps the number of distinct (B, T) shapes ~= 24 so kernels compile once."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, it in enumerate(items):
        n = len(it["ids"]); bk = bucket if n <= 2048 else 512 if n <= 8192 else 2048      # few distinct long shapes
        T = ((n + bk - 1) // bk) * bk
        groups[T].append(i)
    out = []
    for T, idx in groups.items():
        rng.shuffle(idx)
        B = max(1, max_tokens // T)
        for c in range(0, len(idx), B):
            out.append(idx[c:c + B])
    rng.shuffle(out)
    return out


def loss_fn(logits, golds, nopts, brier_w=0.0, label_smooth=0.0):
    ok = golds >= 0
    logits, golds, nopts = logits[ok], golds[ok], nopts[ok]
    ce = F.cross_entropy(logits, golds, label_smoothing=label_smooth)
    loss = ce
    if brier_w > 0:
        p = torch.softmax(logits, -1); p = torch.nan_to_num(p)
        onehot = F.one_hot(golds, p.shape[1]).float()
        loss = loss + brier_w * ((p - onehot) ** 2).sum(1).mean()
    return loss, ce.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--data", default="data/tasks.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=16384, help="tokens per micro-batch (B*T)")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max_ctx", type=int, default=1536)
    ap.add_argument("--brier_w", type=float, default=0.0)
    ap.add_argument("--label_smooth", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_limit", type=int, default=300)
    ap.add_argument("--train_cap", type=int, default=0, help="cap total train examples (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample_every_epoch", action="store_true")
    ap.add_argument("--none_prob", type=float, default=0.0, help="prob. of none-of-the-above augmentation per question")
    ap.add_argument("--schema_first_prob", type=float, default=0.0, help="share of examples rendered questions-first (cacheable schema prefix)")
    ap.add_argument("--max_options", type=int, default=0, help="options kept per question (0 = auto: the tokenizer's label capacity, capped at 255; 10 = original narrow protocol)")
    ap.add_argument("--grad_ckpt", action="store_true", help="activation checkpointing (needed for the 2.6B)")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit", "adamw8bit_paged"],
                    help="adamw8bit is what fits the 1.2B on a 12 GB card; the paged variant crashes on this "
                         "bitsandbytes/torch build, see NOTES.md")
    ap.add_argument("--device", default="", help="cuda | cpu (default: cuda when available)")
    ap.add_argument("--yesmom", action="store_true", help="YESMOM: Noul-only binary head; marks selectia_config.json")
    ap.add_argument("--isolated_levels", default=None, action=argparse.BooleanOptionalAction, help="Score levels judged one per row (default on; off for --yesmom)")
    ap.add_argument("--temperature", type=float, default=1.0, help="value written to selectia_config.json (fit it in Phase 4)")
    ap.add_argument("--version", default="dev", help="version tag written to selectia_config.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(f"{a.out}/train.log", "a")
    def log(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    log("[args]", json.dumps(vars(a)))
    torch.manual_seed(a.seed); rng = random.Random(a.seed)

    train, evals = D.load_cache(a.data)
    if a.train_cap:
        rng.shuffle(train); train = train[:a.train_cap]
    evals_small = {k: v[:a.eval_limit] for k, v in evals.items()}
    if a.yesmom and a.none_prob > 0:
        raise SystemExit("--yesmom trains a binary head: run with --none_prob 0")
    if not a.yesmom and train and all(len(q.options) == 2 and q.options[0].lower().startswith("no") and q.options[1].lower().startswith("yes")
                                      for e in train[:2000] for q in e.qs):
        # forgetting --yesmom still trains a correct binary head but writes a config that lets the runtime
        # offer multi-option questions to it, so say so loudly rather than silently shipping the wrong contract
        log("[warn] the mixture is Noul-only but --yesmom was not passed: the saved selectia_config.json will "
            "not mark the model as binary. Add --yesmom to write the runtime contract.")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DecisionModel(a.model, grad_ckpt=a.grad_ckpt).to(device)
    tok = model.tok
    cap = label_capacity(tok)
    a.max_options = a.max_options or min(MAX_OPTIONS, cap)
    log(f"[model] {a.model} on {device}; label capacity {cap}; max_options {a.max_options}; grad_ckpt {a.grad_ckpt}")
    if a.yesmom and a.max_options > 2:
        log("[warn] --yesmom: max_options is irrelevant (noul questions always carry 2 options)")
    iso = (not a.yesmom) if a.isolated_levels is None else bool(a.isolated_levels)

    def save():
        """Weights + the runtime contract.  The runtime reads temperature/layout/flags only from
        selectia_config.json, which upstream train.py never wrote."""
        model.lm.save_pretrained(f"{a.out}/model"); tok.save_pretrained(f"{a.out}/model")
        cfg = dict(version=a.version, temperature=a.temperature, model=a.model, max_options=a.max_options,
                   max_ctx=a.max_ctx, isolated_levels=iso, schema_first=a.schema_first_prob > 0)
        if a.yesmom:
            cfg["yesmom"] = True
        with open(f"{a.out}/model/selectia_config.json", "w") as f:
            json.dump(cfg, f, indent=1)
        log("[save]", f"{a.out}/model")
    log(f"[data] train examples {len(train)}; tokenizing...")
    t0 = time.time(); items = make_items(train, tok, rng, a.max_ctx, a.none_prob, a.max_options, a.schema_first_prob)
    ntok = sum(len(it["ids"]) for it in items); nq = sum(len(it["slots"]) for it in items)
    log(f"[data] {len(items)} items, {nq} questions, {ntok/1e6:.1f}M tokens, tokenized in {time.time()-t0:.0f}s")

    if a.optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    else:                                   # 8-bit state: ~2 bytes/param instead of 8, which is what makes the 1.2B fit 12 GB
        import bitsandbytes as bnb
        cls = bnb.optim.AdamW8bit if a.optim == "adamw8bit" else bnb.optim.PagedAdamW8bit
        opt = cls(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
        log(f"[optim] {a.optim} (bitsandbytes {bnb.__version__})")
    steps_per_epoch = math.ceil(len(batches_by_tokens(items, a.max_tokens, random.Random(0))) / a.accum)
    total = int(steps_per_epoch * a.epochs)
    log(f"[sched] {steps_per_epoch} optimizer steps/epoch, {total} total")
    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup))))

    step, micro, ep = 0, 0, 0
    hist = []
    model.train()
    t0 = time.time(); ce_acc, n_acc = 0.0, 0; t_last = t0; tok_acc = 0
    while step < total:
        if ep > 0 and a.resample_every_epoch:
            items = make_items(train, tok, rng, a.max_ctx, a.none_prob, a.max_options, a.schema_first_prob)
        for bidx in batches_by_tokens(items, a.max_tokens, rng):
            if step >= total:
                break
            b = collate([items[i] for i in bidx], pad_id(tok))
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            logits = model(b)
            loss, ce = loss_fn(logits, b["golds"], b["nopts"], a.brier_w, a.label_smooth)
            (loss / a.accum).backward()
            ce_acc += ce.item(); n_acc += 1; micro += 1; tok_acc += b["input_ids"].numel()
            if micro % a.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    el = time.time() - t0; now = time.time(); tps = tok_acc / (now - t_last)
                    log(f"[train] step {step}/{total} ep {ep} ce {ce_acc/n_acc:.4f} gn {gn:.2f} lr {lr_at(step):.2e} {el/60:.1f}min eta {(total-step)*(now-t_last)/20/60:.0f}min {tps:.0f}tok/s mem {torch.cuda.max_memory_allocated()/1e9:.0f}GB")
                    ce_acc, n_acc = 0.0, 0; t_last = now; tok_acc = 0
                if step == total:                       # save before the final eval so an eval crash cannot lose the run
                    save()
                if step % a.eval_every == 0 or step == total:
                    res, _ = run_eval(model, evals_small, log=log); agg = aggregate(res)
                    log(f"[eval-agg] step {step} " + json.dumps(agg))
                    hist.append(dict(step=step, agg=agg, results=res)); json.dump(hist, open(f"{a.out}/hist.json", "w"), indent=1)
                    model.train()
        ep += 1
    save()
    log("[done] saved", f"{a.out}/model")


if __name__ == "__main__":
    main()
