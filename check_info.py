#!/usr/bin/env python3
"""Аудит разметки траекторий: что размечено, чем, и заполнены ли stages.

    python3 check_annotation.py [/data/trajectories]

Читает <root>/<agent>/meta/*_meta.json целиком (annotation.stages не всегда попадает
в первые HEAD_BYTES, поэтому голову файла тут не хватит).
"""

import json
import os
import sys
from collections import Counter, defaultdict

CLASS_KEYS = ["A", "B", "C", "D", "E"]
LEGACY_SUCCESS = "success"
ERROR_STAGES = ["approach", "grasp", "transport", "placement"]


def scan(root):
    rows = []
    if any(fn.endswith("_meta.json") for fn in os.listdir(root)):
        # root сам является папкой meta/ (плоский режим: json лежат прямо в root)
        base = os.path.basename(os.path.normpath(root))
        label = os.path.basename(os.path.dirname(os.path.normpath(root))) if base == "meta" else base
        agent_dirs = [(label, root)]
    else:
        agent_dirs = [(agent, os.path.join(root, agent, "meta")) for agent in sorted(os.listdir(root))]
    for agent, meta_dir in agent_dirs:
        if not os.path.isdir(meta_dir):
            continue
        for fn in sorted(os.listdir(meta_dir)):
            if not fn.endswith("_meta.json"):
                continue
            try:
                with open(os.path.join(meta_dir, fn)) as f:
                    meta = json.load(f)
            except (OSError, ValueError) as e:
                print(f"  !! битый файл {agent}/{fn}: {e}")
                continue
            ann = meta.get("annotation")
            ann = ann if isinstance(ann, dict) else {}
            rows.append(
                {
                    "agent": agent,
                    "class_key": meta.get("class_key", ""),
                    "n_steps": len(meta.get("actions") or []),
                    "has_ann": bool(ann),
                    "stages": ann.get("stages"),
                    "terminal_frame": ann.get("terminal_frame"),
                    "error_stage": ann.get("error_stage"),
                    "observation": ann.get("observation"),
                    "cause": ann.get("cause"),
                    "recovery": bool(ann.get("recovery")),
                }
            )
    return rows


def pct(n, total):
    return f"{n:5d} ({100.0 * n / total:5.1f}%)" if total else f"{n:5d}"


def dist(title, counter, total):
    print(f"\n{title}")
    for key, n in sorted(counter.items(), key=lambda kv: -kv[1]):
        print(f"  {str(key):40s} {pct(n, total)}")


def stages_valid(stages, n_steps):
    """Непрерывность и покрытие. n_steps=0 -> длину не проверяем."""
    if not isinstance(stages, dict) or set(stages) != set(ERROR_STAGES):
        return False
    prev_end = -1
    for stage in ERROR_STAGES:
        span = stages[stage]
        if not isinstance(span, list) or len(span) != 2:
            return False
        start, end = span
        if start != prev_end + 1 or start > end:
            return False
        prev_end = end
    return n_steps == 0 or prev_end in (n_steps - 1, n_steps)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "LABLER_ROOT", "/data/trajectories"
    )
    rows = scan(root)
    total = len(rows)
    if not total:
        print(f"в {root} не найдено meta-json")
        return
    print(f"root: {root}\nвсего эпизодов: {total}")

    # --- 1. что размечено -------------------------------------------------
    labeled = [r for r in rows if r["class_key"] in CLASS_KEYS]
    legacy = [r for r in rows if r["class_key"] == LEGACY_SUCCESS]
    empty = [r for r in rows if r["class_key"] not in CLASS_KEYS + [LEGACY_SUCCESS]]
    print(f"\nразмечено вручную (A-E):   {pct(len(labeled), total)}")
    print(f"legacy '{LEGACY_SUCCESS}':          {pct(len(legacy), total)}")
    print(f"без класса:                {pct(len(empty), total)}")

    orphan = [r for r in labeled if not r["has_ann"]]
    if orphan:
        print(f"  !! class_key есть, annotation нет: {len(orphan)}")

    if not labeled:
        print("\nразмеченных эпизодов нет — дальше считать нечего")
        return

    n = len(labeled)
    by_agent = Counter(r["agent"] for r in labeled)
    dist("по агентам (размеченные):", by_agent, n)
    dist("по class_key:", Counter(r["class_key"] for r in labeled), n)

    # --- 2. stages: везде или выборочно ----------------------------------
    print("\n=== stages ===")
    with_stages = [r for r in labeled if r["stages"] is not None]
    print(f"stages заполнены: {pct(len(with_stages), n)}")

    print("\nпо классам (заполнено / всего):")
    per_class = defaultdict(lambda: [0, 0])
    for r in labeled:
        per_class[r["class_key"]][1] += 1
        if r["stages"] is not None:
            per_class[r["class_key"]][0] += 1
    for key in CLASS_KEYS:
        filled, tot = per_class[key]
        share = f"{100.0 * filled / tot:5.1f}%" if tot else "    -"
        print(f"  {key}  {filled:5d} / {tot:5d}   {share}")

    print("\nпо агентам (заполнено / всего):")
    per_agent = defaultdict(lambda: [0, 0])
    for r in labeled:
        per_agent[r["agent"]][1] += 1
        if r["stages"] is not None:
            per_agent[r["agent"]][0] += 1
    for agent in sorted(per_agent):
        filled, tot = per_agent[agent]
        print(f"  {agent:20s} {filled:5d} / {tot:5d}   {100.0 * filled / tot:5.1f}%")

    if with_stages:
        bad = [r for r in with_stages if not stages_valid(r["stages"], r["n_steps"])]
        print(f"\nневалидные stages (разрывы/не покрывают эпизод): {len(bad)}")
        for r in bad[:5]:
            print(f"  {r['agent']} {r['class_key']} n_steps={r['n_steps']} {r['stages']}")

        # длительность фаз в кадрах — видно, есть ли вырожденные (нулевые) фазы
        print("\nмедианная длина фаз (кадры) и доля вырожденных:")
        for stage in ERROR_STAGES:
            lens = sorted(
                r["stages"][stage][1] - r["stages"][stage][0] + 1
                for r in with_stages
                if isinstance(r["stages"].get(stage), list)
            )
            if not lens:
                continue
            med = lens[len(lens) // 2]
            degen = sum(1 for x in lens if x <= 1)
            print(f"  {stage:12s} med={med:4d}  <=1 кадра: {pct(degen, len(lens))}")

    # --- 3. остальные поля ------------------------------------------------
    no_terminal = [r for r in labeled if r["terminal_frame"] is None]
    print(f"\nбез terminal_frame: {pct(len(no_terminal), n)}")

    errors = [r for r in labeled if r["class_key"] != "E"]
    if errors:
        dist("error_stage (только ошибки):", Counter(r["error_stage"] for r in errors), len(errors))
        dist("observation (только ошибки):", Counter(r["observation"] for r in errors), len(errors))
        dist("cause (только ошибки):", Counter(r["cause"] for r in errors), len(errors))

    rec = [r for r in labeled if r["recovery"]]
    print(f"\nrecovery=True: {pct(len(rec), n)}")
    if rec:
        dist("  recovery по class_key:", Counter(r["class_key"] for r in rec), len(rec))

    # --- 4. длины ---------------------------------------------------------
    lens = sorted(r["n_steps"] for r in labeled if r["n_steps"])
    if lens:
        q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
        print(
            f"\nдлина траектории (len(actions)): "
            f"min={lens[0]} p25={q(0.25)} med={q(0.5)} p75={q(0.75)} max={lens[-1]}"
        )


if __name__ == "__main__":
    main()