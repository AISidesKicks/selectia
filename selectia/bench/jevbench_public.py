"""selectia on JevBench's public items (github.com/fstandhartinger/jevbench, MIT).

    python -m selectia.bench.jevbench_public --src scratch/jevbench-src \
        --models LiquidAI/LFM2.5-1.2B-Base LiquidAI/LFM2.5-2.6B-Base \
        --device cuda --out scratch/jevbench

JevBench is Benchmark Heaven's own decision-model benchmark. Its task set is 534 decisions in four
tiers; only 231 are public (`datasets/public/{original,easy,hard}.jsonl`), the other 146 imported and
109 hard items are held out, so the official JevBench Score cannot be reproduced from this repository.
What this script produces is the per-tier accuracy / Brier / ECE on the public items, computed with
JevBench's own `score_task` and `summarize` so the conventions are theirs, not ours.

Two things to know before reading the numbers:

* **No selectia checkpoint is trained yet.** Point `--models` at the LFM2.5 bases and this measures the
  readout zero-shot, which is the floor the fine-tunes have to beat. `decider-2b` (the closest published
  entrant) scores easy 1.000 / standard 0.854 / judge 0.774 / hard 0.473 in `results/v1.2/`.
* **Option order.** JevBench's `labels` order and its `criteria` order disagree in 119 of the 231 public
  items, and the two adapter conventions in their harness differ: `openai_compat` iterates `labels`,
  `typesafe` (what `decider-2b` ran under) passes `criteria` through unchanged. JevBench's own README
  measured a small model swinging from 72% to 21% on the same items when option order was reversed, so
  `--order` runs both and the difference is reported rather than hidden.
"""
import argparse, json, math, os, sys, time

import torch

from selectia.infer import Selectia

TIERS = ["original", "easy", "hard"]
NOUL = ["no", "yes"]


def load_jevbench(src):
    """Import the harness so scoring is theirs. Returns (Task, score_task, metric, summarize)."""
    src = os.path.abspath(src)
    if src not in sys.path:
        sys.path.insert(0, src)
    from jevbench.tasks import Task                                    # noqa: E402
    from jevbench.scoring import score_task                             # noqa: E402
    from jevbench.summarize import metric, summarize                    # noqa: E402
    return Task, score_task, metric, summarize


def load_tasks(src, tiers):
    rows = []
    for t in tiers:
        path = os.path.join(src, "datasets", "public", f"{t}.jsonl")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            r["_tier"] = t
            rows.append(r)
    return rows


def order_options(crit, labels, order):
    """The exact label set, ordered either as JevBench lists it or as the criteria map does."""
    labels = [str(x) for x in labels]
    if order == "labels":
        return labels
    keys = [str(k) for k in crit] if isinstance(crit, dict) else []
    kept = [k for k in keys if k in labels]
    return kept + [l for l in labels if l not in kept]


def build_question(row, order):
    """The JevBench question spec, restricted to the item's exact label set.

    `render_question` derives the option order from the criteria map, so the criteria map is rebuilt in
    the requested order and every label that has no description is passed as a bare name. Labels that the
    criteria map mentions but the label set does not are dropped: JevBench scores over the exact label set
    and an answer covering extra options is not a valid distribution over it.
    """
    q = row["question"]
    qtype, crit = q["type"], q.get("criteria")
    if qtype == "score":
        levels = list(crit) if isinstance(crit, (list, tuple)) else []
        if not levels and isinstance(crit, dict):
            levels = [crit[k] for k in sorted(crit, key=lambda x: float(x))]
        return {"type": "score", "instructions": q["instructions"], "criteria": levels}
    if qtype == "noul":
        c = crit if isinstance(crit, dict) else {}
        return {"type": "noul", "instructions": q["instructions"],
                "criteria": {"false": c.get("false", c.get(False)), "true": c.get("true", c.get(True))}}
    opts = order_options(crit, row["labels"], order)
    desc = crit if isinstance(crit, dict) else {}
    return {"type": "choice", "instructions": q["instructions"],
            "criteria": {lab: desc.get(lab) for lab in opts}}


def answer_probs(qtype, ans, labels):
    """selectia's typed answer -> a probability map over exactly `labels`."""
    if qtype == "noul":
        p = float(ans["noul"])
        return {"no": 1.0 - p, "yes": p}
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        raise ValueError("answer carries no probabilities")
    out = {str(k): float(v) for k, v in probs.items()}
    missing = [l for l in labels if l not in out]
    if missing:
        raise ValueError(f"distribution misses {missing}")
    return {l: out[l] for l in labels}


def run_model(path, rows, tasks_by_id, score_task, order, device, dtype, isolated, fit_temperature, limit=0, log=print):
    d = Selectia(path, device=device, dtype=dtype)
    log(f"[model] {path} on {device}; name={d.name} yesmom={d.yesmom} max_options={d.max_options} isolated={isolated}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    records, n_tok = [], 0
    t_all = time.perf_counter()
    for i, row in enumerate(rows, 1):
        if limit and i > limit:
            break
        qtype = row["question"]["type"]
        labels = [str(x) for x in row["labels"]]
        q = build_question(row, order)
        t0 = time.perf_counter()
        try:
            out = d.system_one(row["state"], {"decision": q}, isolated=isolated, max_state_tokens=32768)
            ans = out["answers"]["decision"]
            probs = answer_probs(qtype, ans, labels)
            usage = out.get("usage") or {}
            err, ok = None, True
        except Exception as e:
            probs, usage, err, ok = None, {}, f"{type(e).__name__}: {e}", False
        wall = time.perf_counter() - t0
        n_tok += int(usage.get("input_tokens") or 0)
        sc = score_task(probs or {}, tasks_by_id[row["id"]]) if ok else {"valid": False, "strict_valid": False,
                                                                         "renormalized": False, "correct": False, "predicted": None}
        records.append(dict(task_id=row["id"], family=row["family"], split=row["split"], group=row.get("group"),
                            tier=row["_tier"], gold=row["_gold"], ok=ok, error=err, valid=sc.get("valid", False),
                            strict_valid=sc.get("strict_valid", False), renormalized=sc.get("renormalized", False),
                            correct=sc.get("correct"), predicted=sc.get("predicted"), ordinal_ev=sc.get("ordinal_ev"),
                            probs=sc.get("probs"), probs_source="native", model=d.name, latency_s=wall,
                            usage=usage, cost_usd=None, cost_basis="local_gpu_no_provider_tariff", status_code=None))
        if i % 25 == 0:
            log(f"  {i}/{len(rows)} {time.perf_counter() - t_all:.0f}s")
    total_s = time.perf_counter() - t_all
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    lat = sorted(r["latency_s"] for r in records)
    runtime = dict(model=path, device=device, dtype=str(dtype).replace("torch.", ""), order=order,
                   isolated=bool(isolated), n=len(records), total_s=round(total_s, 2),
                   decisions_per_s=round(len(records) / total_s, 3) if total_s else None,
                   input_tokens=n_tok, tokens_per_s=round(n_tok / total_s, 1) if total_s else None,
                   mean_input_tokens=round(n_tok / len(records), 1) if records else None,
                   peak_gpu_gb=round(peak, 2), latency_p50_s=round(lat[len(lat) // 2], 4) if lat else None,
                   latency_p95_s=round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 4) if lat else None)
    if fit_temperature:
        runtime["temperature"] = fit_and_report(records, log)
    del d
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, runtime


def _softmax(logits):
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [x / s for x in ex]


def fit_and_report(records, log, grid=None):
    """One scalar temperature fitted on the public items (NLL), then ECE/Brier after scaling.

    JevBench fits its reranker temperatures on public items only, and the fitted value is a diagnostic
    here, not part of the official score.
    """
    grid = grid or [round(0.5 + 0.05 * i, 2) for i in range(41)]
    scorable = [r for r in records if r.get("probs") and r.get("correct") is not None and r.get("valid")]
    if not scorable:
        return None
    best, best_t = None, 1.0
    for T in grid:
        nll = 0.0
        for r in scorable:
            labs = list(r["probs"])
            gold = _gold(r)
            if gold is None:
                continue
            p = _softmax([math.log(max(r["probs"][l], 1e-12)) / T for l in labs])
            nll -= math.log(max(p[labs.index(gold)], 1e-12))
        if best is None or nll < best:
            best, best_t = nll, T
    out = dict(fitted_temperature=best_t, n=len(scorable))
    for tag, T in [("raw", 1.0), ("fitted", best_t)]:
        bs, pairs = [], []
        for r in scorable:
            labs = list(r["probs"])
            gold = _gold(r)
            if gold is None:
                continue
            p = _softmax([math.log(max(r["probs"][l], 1e-12)) / T for l in labs])
            bs.append(sum((p[i] - (1.0 if l == gold else 0.0)) ** 2 for i, l in enumerate(labs)))
            j = max(range(len(p)), key=p.__getitem__)
            pairs.append((p[j], labs[j] == gold))
        out[f"brier_{tag}"] = round(sum(bs) / len(bs), 4)
        out[f"ece_{tag}"] = round(ece(pairs), 4)
    return out


def _gold(r):
    """The gold label of a record (score items carry the level index as a string)."""
    return r.get("gold")


def ece(pairs, bins=10):
    """Top-label ECE in 10 equal-width bins, JevBench's convention (empty bins absent)."""
    b = [[0, 0.0, 0] for _ in range(bins)]
    for conf, ok in pairs:
        i = min(bins - 1, int(min(max(conf, 0.0), 1.0) * bins))
        b[i][0] += 1; b[i][1] += conf; b[i][2] += 1 if ok else 0
    n = sum(x[0] for x in b)
    return sum(x[0] / n * abs(x[1] / x[0] - x[2] / x[0]) for x in b if x[0]) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="scratch/jevbench-src", help="clone of fstandhartinger/jevbench")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--order", default="labels", choices=["labels", "criteria"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--isolated", action="store_true", help="score Score levels one row at a time")
    ap.add_argument("--fit-temperature", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N items")
    ap.add_argument("--out", default="scratch/jevbench")
    a = ap.parse_args()

    Task, score_task, metric, summarize = load_jevbench(a.src)
    tiers = a.tiers.split(",")
    rows = load_tasks(a.src, tiers)
    rows = [dict(r, _gold=str(r["expected"])) for r in rows]        # score levels score as their index string
    tasks = [Task(id=r["id"], family=r["family"], state=r["state"], question=r["question"], labels=r["labels"],
                  expected=r["expected"], split=r["split"], group=r.get("group"), provenance=r.get("provenance") or {})
             for r in rows]
    tasks_by_id = {t.id: t for t in tasks}
    print(f"[data] {len(tasks)} public items, tiers {tiers}, order={a.order}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    report = dict(order=a.order, tiers=tiers, n_items=len(tasks), models={})
    for model in a.models:
        recs, runtime = run_model(model, rows, tasks_by_id, score_task, a.order, a.device, getattr(torch, a.dtype),
                                  a.isolated, a.fit_temperature, a.limit)
        summary = summarize(tasks, recs)
        tiersum = {}
        for t in tiers:
            tr = [r for r in recs if r["tier"] == t]
            tiersum[t] = metric([tasks_by_id[r["task_id"]] for r in tr], tr)
        report["models"][model] = dict(runtime=runtime, summary=summary, tiers=tiersum)
        tag = os.path.basename(model.rstrip("/")) or model.replace("/", "_")
        json.dump(dict(runtime=runtime, summary=summary, tiers=tiersum),
                  open(os.path.join(a.out, f"{tag}__{a.order}.json"), "w"), indent=1)
        json.dump(recs, open(os.path.join(a.out, f"{tag}__{a.order}__records.json"), "w"), indent=1)
        print(f"[{model}] accuracy={summary.get('accuracy')} macro={summary.get('macro_accuracy')} "
              f"brier={summary.get('brier_mean')} ece={summary.get('ece')} "
              f"valid={summary.get('schema_validity')} p50={runtime['latency_p50_s']}s "
              f"peak={runtime['peak_gpu_gb']}GB {runtime['decisions_per_s']}/s", flush=True)
        for t, s in tiersum.items():
            print(f"    {t:9s} n={s.get('n_scorable')} acc={s.get('accuracy')} brier={s.get('brier_mean')} "
                  f"ece={s.get('ece')} mae={s.get('ordinal_mae')}", flush=True)
    json.dump(report, open(os.path.join(a.out, f"report__{a.order}.json"), "w"), indent=1)
    print(f"[report] {a.out}/report__{a.order}.json", flush=True)


if __name__ == "__main__":
    main()
