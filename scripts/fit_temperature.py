"""Fit one temperature on held-out items, with an honest split, and report what it buys.

    python scripts/fit_temperature.py --model runs/selectia_yesmom_230m/model --data data/mixture_yesmom.pkl
    python scripts/fit_temperature.py --model ... --data ... --split-by task --limit 4000

The earlier fits were in-sample: the temperature was fitted on the same items it was scored on, and the
search grid stopped at 2.5 so four models sat on the ceiling. This script fits on half the items by NLL and
reports Brier, ECE and accuracy on the other half, with a grid out to 50 so the bound cannot be the answer.
It also prints the per-item logit spread, which is what a single scalar is really compensating for.
"""
import argparse, collections, json, math, os, random, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selectia.infer import Selectia                      # noqa: E402
from selectia.data.core import load_cache                # noqa: E402

GRID = [round(0.25 + 0.25 * i, 2) for i in range(200)]    # 0.25 .. 50.0, coarse pass
FINE = [round(0.01 * i, 2) for i in range(1, 1001)]       # 0.01 .. 10.0, refinement window


def softmax(z):
    m = max(z); ex = [math.exp(x - m) for x in z]; s = sum(ex)
    return [x / s for x in ex]


def ece_bins(pairs, bins=15):
    b = [[0, 0.0, 0] for _ in range(bins)]
    for c, ok in pairs:
        i = min(bins - 1, int(min(max(c, 0.0), 1.0) * bins)); b[i][0] += 1; b[i][1] += c; b[i][2] += 1 if ok else 0
    n = sum(x[0] for x in b)
    return sum(x[0] / n * abs(x[1] / x[0] - x[2] / x[0]) for x in b if x[0]) if n else 0.0


def report(rows, T):
    """rows: list of (logprobs dict, gold). Brier, ECE and accuracy after scaling by T."""
    bs, pairs, acc = [], [], []
    for lp, gold in rows:
        labs = list(lp); p = softmax([lp[l] / T for l in labs])
        bs.append(sum((p[i] - (1.0 if l == gold else 0.0)) ** 2 for i, l in enumerate(labs)))
        j = max(range(len(p)), key=p.__getitem__)
        pairs.append((p[j], labs[j] == gold)); acc.append(labs[j] == gold)
    return dict(brier=float(np.mean(bs)), ece=float(ece_bins(pairs)), acc=float(np.mean(acc)), mean_conf=float(np.mean([c for c, _ in pairs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="a mixture pkl; its eval half is used")
    ap.add_argument("--limit", type=int, default=2000, help="max items per eval set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-items", type=int, default=20000, help="cap the total after sampling")
    ap.add_argument("--cross-records", default="", help="a jevbench records json, used as a second, out-of-distribution "
                                                        "source to measure whether a temperature transfers across distributions")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    _, evals = load_cache(a.data)
    d = Selectia(a.model, device=a.device)
    print(f"[fit] {d.name} on {a.device} | yesmom {d.yesmom} | layout schema_first={d.schema_first}", flush=True)

    items = []
    for t, exs in sorted(evals.items()):
        for e in exs[:a.limit]:
            for q in e.qs:
                items.append((t, e.context, str(q.text), [str(o) for o in q.options], int(q.gold)))
    rng = random.Random(a.seed); rng.shuffle(items)
    if len(items) > a.max_items:
        items = items[:a.max_items]
    print(f"[fit] {len(items)} items over {len(evals)} eval sets", flush=True)

    rows, by_task, spreads = [], collections.defaultdict(list), []
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        reqs = [(c, [{"question": qt, "options": opts}]) for _, c, qt, opts, _ in chunk]
        try:
            res = d.decide_batch(reqs)
        except Exception as ex:
            print(f"[fit] batch failed: {type(ex).__name__}: {ex}", flush=True)
            continue
        for (t, _, _, opts, gold), answers in zip(chunk, res):
            probs = (answers[0] or {}).get("probs") or {}      # one request per item, one question per request
            if len(probs) != len(opts) or opts[gold] not in probs:
                continue
            lp = {k: math.log(max(float(v), 1e-12)) for k, v in probs.items()}
            rows.append((lp, opts[gold])); by_task[t].append((lp, opts[gold]))
            z = list(lp.values())
            if len(z) > 1:
                spreads.append(float(np.std(z)))
        if (i // 16) % 40 == 0:
            print(f"  {i + len(chunk)}/{len(items)}", flush=True)
    if not rows:
        raise SystemExit("[fit] no scorable items")

    half = len(rows) // 2
    fit_rows, test_rows = rows[:half], rows[half:]
    def nll_at(T, rows_):
        tot = 0.0
        for lp, g in rows_:
            labs = list(lp); p = softmax([lp[l] / T for l in labs])
            tot -= math.log(max(p[labs.index(g)], 1e-12))
        return tot

    best, best_t = None, 1.0
    for T in GRID:                                         # coarse, wide enough that the bound is not the answer
        nll = nll_at(T, fit_rows)
        if best is None or nll < best:
            best, best_t = nll, T
    lo, hi = max(0.01, best_t - 0.5), best_t + 0.5
    for T in [t for t in FINE if lo <= t <= hi]:           # refine around the coarse winner
        nll = nll_at(T, fit_rows)
        if nll < best:
            best, best_t = nll, T
    raw, fitted = report(test_rows, 1.0), report(test_rows, best_t)
    print(f"\n[fit] split {len(fit_rows)} fit / {len(test_rows)} test")
    print(f"[fit] fitted T = {best_t:.2f}  (grid max {GRID[-1]}){'  <-- PINNED AT CEILING' if best_t >= GRID[-1] else ''}")
    print(f"[fit] test Brier  raw {raw['brier']:.4f} -> fitted {fitted['brier']:.4f}")
    print(f"[fit] test ECE    raw {raw['ece']:.4f} -> fitted {fitted['ece']:.4f}")
    print(f"[fit] test acc    raw {raw['acc']:.4f} -> fitted {fitted['acc']:.4f}  (unchanged: T cannot move the argmax)")
    print(f"[fit] mean confidence raw {raw['mean_conf']:.3f} vs accuracy {raw['acc']:.3f} -> overconfidence {raw['mean_conf'] - raw['acc']:+.3f}")
    print(f"[fit] per-item logit spread sd {float(np.std(spreads)):.2f}, mean {float(np.mean(spreads)):.2f} over {len(spreads)} items")
    if a.cross_records:
        cross = []
        for r in json.load(open(a.cross_records)):
            p = r.get("probs") or {}
            if p and r.get("gold") in p:
                cross.append(({k: math.log(max(float(v), 1e-12)) for k, v in p.items()}, r["gold"]))
        ch = len(cross) // 2
        T_in = best_t
        T_out = min((nll_at(T, cross[:ch]), T) for T in GRID)[1] if cross else 1.0
        print(f"\n[transfer] in-domain fit T = {T_in:.2f} on {len(fit_rows)} items")
        print(f"[transfer] out-of-distribution fit T = {T_out:.2f} on {len(cross)} items")
        print(f"[transfer] a big T flattens every confidence towards uniform, which lowers ECE without adding"
              f" information. Brier and NLL punish that, so read the Brier row.")
        print(f"[transfer] {'source':9s} {'n':>6s} | {'ECE@1':>7s} {'ECE@in':>7s} {'ECE@ood':>8s} | "
              f"{'Brier@1':>8s} {'Brier@in':>9s} {'Brier@ood':>10s}")
        for name, rows_ in [("in-domain", test_rows), ("jevbench", cross)]:
            if not rows_:
                continue
            r1, ri, ro = report(rows_, 1.0), report(rows_, T_in), report(rows_, T_out)
            print(f"[transfer] {name:9s} {len(rows_):6d} | {r1['ece']:7.3f} {ri['ece']:7.3f} {ro['ece']:8.3f} | "
                  f"{r1['brier']:8.3f} {ri['brier']:9.3f} {ro['brier']:10.3f}")
    worst = sorted(((report(v, 1.0)['ece'], k, len(v)) for k, v in by_task.items()), reverse=True)[:6]
    print("[fit] worst ECE by set (raw):", ", ".join(f"{k} {e:.3f} (n={n})" for e, k, n in worst), flush=True)
    if a.out:
        json.dump(dict(model=a.model, fitted_T=best_t, pinned=best_t >= GRID[-1], raw=raw, fitted=fitted,
                       n_fit=len(fit_rows), n_test=len(test_rows), logit_spread_sd=float(np.std(spreads))),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
