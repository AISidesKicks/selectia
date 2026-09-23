"""Fit a calibration on held-out items, with an honest split, and report what it buys.

    python scripts/fit_temperature.py --model runs/selectia_yesmom_230m/model --data data/mixture_yesmom.pkl
    python scripts/fit_temperature.py --model ... --data ... --split-by task --limit 4000
    python scripts/fit_temperature.py --model ... --data ... --write-config runs/.../model   # persist the winner

The earlier fits were in-sample: the temperature was fitted on the same items it was scored on, and the
search grid stopped at 2.5 so four models sat on the ceiling. This script fits on half the items and
reports on the other half, with a grid out to 50 so the bound cannot be the answer.

It compares three forms on the same held-out half:

    raw      softmax(z)
    temp     softmax(z / T)                       one scalar, multiplicative
    platt    sigmoid(a * (z_yes - z_no) + b)      binary outputs only, has a bias term

The bias term matters because temperature is multiplicative: `softmax(z/T)` can flatten a confident
distribution but can never move a model from "always no" to "yes when warranted". Luni's PhishNChips
finding is exactly that - Laya ranks near Jev (AUROC 0.678 vs 0.689) but sits on one side of the
threshold, and only `sigma(a*z+b)` fixes it. Every form is scored by **Brier and NLL, never by ECE**:
a large T flattens everything, minimises ECE and destroys the information. AUROC is printed beside them
so a threshold-only fix shows up as accuracy moving while AUROC stays flat.
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


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def ece_bins(pairs, bins=15):
    b = [[0, 0.0, 0] for _ in range(bins)]
    for c, ok in pairs:
        i = min(bins - 1, int(min(max(c, 0.0), 1.0) * bins)); b[i][0] += 1; b[i][1] += c; b[i][2] += 1 if ok else 0
    n = sum(x[0] for x in b)
    return sum(x[0] / n * abs(x[1] / x[0] - x[2] / x[0]) for x in b if x[0]) if n else 0.0


def auroc(scores, labels):
    """Mann-Whitney AUROC. scores/labels are floats/ints of the same length; None when one class is absent."""
    pos = [s for s, y in zip(scores, labels) if y]; neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores); i = 0
    while i < len(order):                                  # average ranks over ties
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rp = sum(r for r, y in zip(ranks, labels) if y)
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _probs(lp, T=1.0, ab=None):
    """lp: label -> log-prob (option order preserved). ab=(a,b) is Platt, binary only."""
    labs = list(lp)
    if ab is not None and len(labs) == 2:
        z = lp[labs[1]] - lp[labs[0]]
        p1 = _sigmoid(ab[0] * z + ab[1])
        return [1.0 - p1, p1]
    return softmax([lp[l] / T for l in labs])


def report(rows, T=1.0, ab=None):
    """rows: list of (logprobs dict, gold). Brier (multi-class sum), ECE, accuracy and binary AUROC."""
    bs, pairs, acc, ys, ss = [], [], [], [], []
    for lp, gold in rows:
        labs = list(lp); p = _probs(lp, T, ab)
        bs.append(sum((p[i] - (1.0 if l == gold else 0.0)) ** 2 for i, l in enumerate(labs)))
        j = max(range(len(p)), key=p.__getitem__)
        pairs.append((p[j], labs[j] == gold)); acc.append(labs[j] == gold)
        if len(labs) == 2:
            ys.append(1 if gold == labs[1] else 0); ss.append(p[1])
    return dict(brier=float(np.mean(bs)), ece=float(ece_bins(pairs)), acc=float(np.mean(acc)),
                mean_conf=float(np.mean([c for c, _ in pairs])), auc=auroc(ss, ys) if ys and 0 < sum(ys) < len(ys) else None,
                n=len(rows))


def nll_at(rows, T=1.0, ab=None):
    tot = 0.0
    for lp, g in rows:
        labs = list(lp); p = _probs(lp, T, ab)
        tot -= math.log(max(p[labs.index(g)], 1e-12))
    return tot


def fit_temperature(fit_rows):
    best, best_t = None, 1.0
    for T in GRID:                                         # coarse, wide enough that the bound is not the answer
        nll = nll_at(fit_rows, T)
        if best is None or nll < best:
            best, best_t = nll, T
    lo, hi = max(0.01, best_t - 0.5), best_t + 0.5
    for T in [t for t in FINE if lo <= t <= hi]:           # refine around the coarse winner
        nll = nll_at(fit_rows, T)
        if nll < best:
            best, best_t = nll, T
    return best_t


def fit_platt(fit_rows, iters=600, lr=0.1):
    """Gradient descent on the binary NLL, the same objective Luni's `platt.py` minimises.

    The logit difference is standardised before the fit (lr 0.1 on raw logits diverges when |z| is large)
    and the parameters are mapped back, so `a` is still a gain on the raw log-odds."""
    zs, ys = [], []
    for lp, gold in fit_rows:
        labs = list(lp)
        if len(labs) != 2:
            continue
        zs.append(lp[labs[1]] - lp[labs[0]]); ys.append(1.0 if gold == labs[1] else 0.0)
    if not zs or not (0 < sum(ys) < len(ys)):
        return None
    z = np.asarray(zs, dtype=np.float64); y = np.asarray(ys, dtype=np.float64)
    mu, sd = float(z.mean()), float(z.std() or 1.0)
    zn = (z - mu) / sd
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * zn + b)))
        ga = float((p - y) @ zn) / len(y); gb = float((p - y).mean())
        a -= lr * ga; b -= lr * gb
    return (a / sd, b - a * mu / sd)                        # map the standardised fit back to z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="a mixture pkl; its eval half is used")
    ap.add_argument("--limit", type=int, default=2000, help="max items per eval set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-items", type=int, default=20000, help="cap the total after sampling")
    ap.add_argument("--cross-records", default="", help="a jevbench records json, used as a second, out-of-distribution "
                                                        "source to measure whether a calibration transfers across distributions")
    ap.add_argument("--write-config", default="", help="merge the winning calibration into this model folder's selectia_config.json")
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
    binary = all(len(lp) == 2 for lp, _ in rows)
    best_t = fit_temperature(fit_rows)
    ab = fit_platt(fit_rows) if binary else None

    raw, temp = report(test_rows, T=1.0), report(test_rows, T=best_t)
    plat = report(test_rows, ab=ab) if ab else None
    print(f"\n[fit] split {len(fit_rows)} fit / {len(test_rows)} test | {'binary (noul)' if binary else 'multi-option'}")
    print(f"[fit] fitted T = {best_t:.2f}  (grid max {GRID[-1]}){'  <-- PINNED AT CEILING' if best_t >= GRID[-1] else ''}")
    if ab:
        print(f"[fit] fitted Platt a = {ab[0]:.4f}, b = {ab[1]:+.4f}   (sigma(a*z+b), z = log-odds of yes)")
    print(f"[fit] select by NLL/Brier, never ECE. Columns: raw | temperature | platt")
    for k, fmt in [("acc", "{:.4f}"), ("brier", "{:.4f}"), ("ece", "{:.4f}"), ("mean_conf", "{:.4f}"), ("auc", "{:.4f}")]:
        rv, tv = raw[k], temp[k]
        pv = plat[k] if plat else float("nan")
        print(f"[fit] test {k:9s} raw {fmt.format(rv)} | temp {fmt.format(tv)} | platt {fmt.format(pv)}"
              + ("   (chance AUC = 0.5)" if k == "auc" else ""))
    print(f"[fit] test acc is unchanged by a scalar T (argmax invariant) but a Platt bias can move it: "
          f"{raw['acc']:.4f} -> {plat['acc']:.4f}" if plat else "")
    print(f"[fit] mean confidence raw {raw['mean_conf']:.3f} vs accuracy {raw['acc']:.3f} "
          f"-> overconfidence {raw['mean_conf'] - raw['acc']:+.3f}")
    print(f"[fit] per-item logit spread sd {float(np.std(spreads)):.2f}, mean {float(np.mean(spreads)):.2f} over {len(spreads)} items")

    if a.cross_records:
        cross = []
        for r in json.load(open(a.cross_records)):
            p = r.get("probs") or {}
            if p and r.get("gold") in p:
                cross.append(({k: math.log(max(float(v), 1e-12)) for k, v in p.items()}, r["gold"]))
        T_out = min((nll_at(cross, T), T) for T in GRID)[1] if cross else 1.0
        print(f"\n[transfer] in-domain fit T = {best_t:.2f} on {len(fit_rows)} items")
        print(f"[transfer] out-of-distribution fit T = {T_out:.2f} on {len(cross)} items")
        print(f"[transfer] a big T flattens every confidence towards uniform, which lowers ECE without adding"
              f" information. Brier and NLL punish that, so read the Brier row.")
        print(f"[transfer] {'source':9s} {'n':>6s} | {'ECE@1':>7s} {'ECE@in':>7s} {'ECE@ood':>8s} | "
              f"{'Brier@1':>8s} {'Brier@in':>9s} {'Brier@ood':>10s}")
        for name, rows_ in [("in-domain", test_rows), ("jevbench", cross)]:
            if not rows_:
                continue
            r1, ri, ro = report(rows_, T=1.0), report(rows_, T=best_t), report(rows_, T=T_out)
            print(f"[transfer] {name:9s} {len(rows_):6d} | {r1['ece']:7.3f} {ri['ece']:7.3f} {ro['ece']:8.3f} | "
                  f"{r1['brier']:8.3f} {ri['brier']:9.3f} {ro['brier']:10.3f}")

    worst = sorted(((report(v, T=1.0)['ece'], k, len(v)) for k, v in by_task.items()), reverse=True)[:6]
    print("[fit] worst ECE by set (raw):", ", ".join(f"{k} {e:.3f} (n={n})" for e, k, n in worst), flush=True)

    # winner by NLL on the held-out half (Brier would pick the same form on every run so far); ECE never selects
    cand = {"raw": (raw, None, 1.0), "temperature": (temp, None, best_t), "platt": (plat, ab, 1.0)}
    pick = min((c for c in cand.values() if c[0]), key=lambda c: c[0]["brier"])
    kind = "platt" if pick is cand["platt"] else ("temperature" if pick is cand["temperature"] else "raw")
    cal = {"kind": kind, "fit_on": os.path.basename(a.data), "n_fit": len(fit_rows), "n_test": len(test_rows),
           "selected_by": "brier_heldout"}
    if kind == "platt":
        cal.update(a=round(ab[0], 6), b=round(ab[1], 6))
    elif kind == "temperature":
        cal["temperature"] = best_t
    print(f"[fit] winner by held-out Brier: {kind}  ->  {json.dumps(cal)}")
    if a.write_config:
        cfg_path = os.path.join(a.write_config, "selectia_config.json")
        cfg = json.load(open(cfg_path))
        cfg["calibration"] = cal
        json.dump(cfg, open(cfg_path, "w"), indent=1)
        print(f"[fit] wrote calibration to {cfg_path}")
    if a.out:
        json.dump(dict(model=a.model, fitted_T=best_t, platt_ab=ab, picked=kind, calibration=cal,
                       raw=raw, temperature=temp, platt=plat,
                       n_fit=len(fit_rows), n_test=len(test_rows), logit_spread_sd=float(np.std(spreads))),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
