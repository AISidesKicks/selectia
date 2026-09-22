"""Teacher-only decisions from the shipped `teacher_data/`: real data, no dataset downloads.

    python scripts/make_teacher_data.py --out data/mixture_teacher.pkl [--seed 6]

The training rows come from the same `teacher_sets()` the full mixture uses (packed and single custom
questions, routing, commands, isolated rows), and the eval sets are the held-out domains' custom
questions and routing rows, the same slicing `probes()` uses. What is missing against the full mixture is
the 99 converted public datasets. That makes this a partial but entirely real training set: enough to
validate a run end to end and to warm a checkpoint while `python -m selectia.data.core` builds
`data/tasks.pkl`, which needs no GPU.
"""
import argparse, collections, os, pickle, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selectia import data as D
from selectia import systemone as S1
from selectia.data.teacher_questions import to_example
from selectia.data.mixture import HELD_DOMAINS, load_teacher, teacher_sets


def teacher_eval(recs, routes):
    """Held-out domains only, named the way the upstream probes name them (custom_<type>, routing_<group>)."""
    out = collections.defaultdict(list)
    for r in recs:
        if r["domain"] not in HELD_DOMAINS:
            continue
        ex = to_example(r, D, S1, "custom")
        for q, m in zip(ex.qs, r["questions"]):
            out[f"custom_{m['type']}"].append(D.Example(ex.context, [q], f"custom_{m['type']}"))
    for r in routes:
        if r["domain"] in HELD_DOMAINS:
            ex = to_example(r, D, S1, "routing")
            k = f"routing_{r['group']}"
            out[k].append(D.Example(ex.context, ex.qs, k))
    return {k: v[:1500] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mixture_teacher.pkl")
    ap.add_argument("--seed", type=int, default=6)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    recs, routes, commands = load_teacher()
    parts = teacher_sets(recs, routes, rng, commands)
    train = [e for v in parts.values() for e in v]
    rng.shuffle(train)
    evals = teacher_eval(recs, routes)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump((train, evals), f)
    print("[teacher] records", len(recs), "routes", len(routes), "commands", len(commands))
    print("[teacher] train", len(train), {k: len(v) for k, v in parts.items()})
    print("[teacher] evals", {k: len(v) for k, v in evals.items()}, "total", sum(len(v) for v in evals.values()))


if __name__ == "__main__":
    main()
