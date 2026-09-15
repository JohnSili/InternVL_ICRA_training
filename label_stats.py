#!/usr/bin/env python3
"""Статистика разметки для статьи, без модели: классы gt по агентам и задачам, ручная разметка и её
согласие с gt.

    python3 label_stats.py --root ~/Simpler/trajectories --ann-root ~/Simpler/gt --out data/paper/labels

Пишет в --out label_counts.json и labeler_vs_human.jsonl: по строке на эпизод с обеими метками,
gt — ручная метка, pred — метка gt-разметчика. paper_tables.py строит по ним таблицу данных и строку
«gt-разметчик» в таблице человеческого теста.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import CLASSES, load_episodes  # noqa: E402


def nested_counts(eps, src):
    out = defaultdict(lambda: defaultdict(Counter))
    for e in eps:
        if e[src]:
            out[e["agent"]][e["task"]][e[src]["cls"]] += 1
    return {a: {t: dict(c) for t, c in sorted(tasks.items())} for a, tasks in sorted(out.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("VLA_META_ROOT"), help="кадры и meta с ручной разметкой")
    ap.add_argument("--ann-root", default=os.environ.get("VLA_ANN_ROOT"), help="gt-разметка <agent>/<name>_auto.json")
    ap.add_argument("--out", default="data/paper/labels")
    args = ap.parse_args()
    if not args.root or not args.ann_root:
        sys.exit("нужны --root и --ann-root (или VLA_META_ROOT и VLA_ANN_ROOT)")
    eps = load_episodes(os.path.expanduser(args.root), os.path.expanduser(args.ann_root))
    both = sorted((e for e in eps if e["manual"] and e["auto"]), key=lambda e: e["id"])
    counts = {
        "episodes": len(eps),
        "unlabeled": {src: sum(not e[src] for e in eps) for src in ("auto", "manual")},
        "auto": nested_counts(eps, "auto"),
        "manual": nested_counts(eps, "manual"),
        "auto_confidence": {c: dict(Counter(str(e["auto"].get("confidence")) for e in eps
                                            if e["auto"] and e["auto"]["cls"] == c)) for c in CLASSES},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "label_counts.json"), "w") as f:
        json.dump(counts, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "labeler_vs_human.jsonl"), "w") as f:
        for e in both:
            f.write(json.dumps({"id": e["id"], "agent": e["agent"], "task": e["task"],
                                "gt": e["manual"]["cls"], "pred": e["auto"]["cls"]}) + "\n")

    agree = sum(e["manual"]["cls"] == e["auto"]["cls"] for e in both)
    print(f"эпизодов {len(eps)}: без gt-метки {counts['unlabeled']['auto']}, "
          f"с ручной меткой {len(eps) - counts['unlabeled']['manual']}")
    for agent, tasks in counts["auto"].items():
        c = Counter()
        for tc in tasks.values():
            c.update(tc)
        print(f"  gt, {agent}: " + " ".join(f"{k}={c.get(k, 0)}" for k in CLASSES))
    print(f"обе метки у {len(both)} эпизодов, совпадают {agree} ({100 * agree / max(len(both), 1):.1f}%)")
    print(f"saved {args.out}/{{label_counts.json,labeler_vs_human.jsonl}}")


if __name__ == "__main__":
    main()
