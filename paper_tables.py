#!/usr/bin/env python3
"""Таблицы статьи и парные тесты McNemar по прогонам evaluate.py, без модели и GPU.

    python3 paper_tables.py                                      # data/paper/eval -> печать и paper_tables.md рядом
    python3 paper_tables.py --eval-dir server/data/paper/eval --run-dir server/work_dirs/paper

Прогон описывается своими metrics.json и run_config.json (модель, LoRA, стратегия кадров, прунинг), а не
именем папки. Прогоны с --limit и пересчёты --from-predictions пропускаются. Сравнения парные: берутся
общие id двух прогонов, точный двусторонний тест McNemar по эпизодам, где прав ровно один из двух.
"""

import argparse
import glob
import json
import math
import os
import statistics

CLASSES = "ABCDE"
STRATEGIES = ("dense_sparse", "uniform", "surrounding")
DEFAULT_MAX_FRAMES = 20


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def model_name(m):
    base = os.path.basename(os.path.normpath(m["model"]))
    return f"{base} + LoRA ({os.path.basename(os.path.normpath(m['lora']))})" if m.get("lora") else base


def load_runs(eval_dir, data_fs, keep_limited):
    runs, skipped = [], []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*", "metrics.json"))):
        d = os.path.dirname(path)
        name = os.path.basename(d)
        with open(path) as f:
            m = json.load(f)
        cfg_path = os.path.join(d, "run_config.json")
        args = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                args = json.load(f).get("args", {})
        pred_path = os.path.join(d, "predictions.jsonl")
        if "data" not in m or not os.path.exists(pred_path):
            skipped.append((name, "нет описания прогона или predictions.jsonl"))
            continue
        if args.get("limit") and not keep_limited:
            skipped.append((name, f"--limit {args['limit']}"))
            continue
        prune = m.get("prune") or {}
        runs.append({
            "dir": name,
            "split": os.path.splitext(os.path.basename(m["data"]))[0],
            "model": model_name(m), "ft": bool(m.get("lora")),
            "fs": m.get("frame_selection") or data_fs, "reselected": bool(m.get("frame_selection")),
            "max_frames": m.get("max_frames") or DEFAULT_MAX_FRAMES,
            "ratio": prune.get("token_ratio"), "random": bool(prune.get("token_random")), "topk": prune.get("topk"),
            "metrics": m["overall"]["metrics"], "majority": m["overall"]["majority"],
            "frames": m.get("mean_frames"), "tokens": m.get("mean_visual_tokens"),
            "preds": {p["id"]: p for p in load_jsonl(pred_path)},
        })
    return runs, skipped


def mcnemar(a, b):
    """-> (прав только a, прав только b, p, общих id). a, b: прогоны."""
    ids = sorted(set(a["preds"]) & set(b["preds"]))
    right = lambda r, i: r["preds"][i]["pred"] == r["preds"][i]["gt"]
    for i in ids:
        assert a["preds"][i]["gt"] == b["preds"][i]["gt"], f"{i}: разные метки в {a['dir']} и {b['dir']}"
    only_a = sum(right(a, i) and not right(b, i) for i in ids)
    only_b = sum(right(b, i) and not right(a, i) for i in ids)
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n)
    return only_a, only_b, p, len(ids)


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fmt_test(a, b):
    only_a, only_b, p, n = mcnemar(a, b)
    return f"{only_a}/{only_b}, " + ("p<0.001" if p < 0.001 else f"p={p:.3f}")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pick(runs, **kw):
    return [r for r in runs if all(r[k] == v for k, v in kw.items())]


def full_run(runs, split, model, data_fs):
    """Полные токены, стратегия данных; прогон без пересборки кадров предпочтительнее."""
    cand = [r for r in pick(runs, split=split, model=model, fs=data_fs, ratio=None, topk=None)
            if r["max_frames"] == DEFAULT_MAX_FRAMES]
    cand.sort(key=lambda r: r["reselected"])
    return cand[0] if cand else None


def models_of(runs):
    return sorted({r["model"] for r in runs}, key=lambda m: (" + LoRA" in m, m))


def section_main(runs, data_fs, split, note):
    base = [full_run(runs, split, m, data_fs) for m in models_of(runs)]
    base = [r for r in base if r]
    if not base:
        return None
    ref = max(base, key=lambda r: r["metrics"]["n"])
    maj = ref["majority"]
    header = ["модель", "n", "macro-F1", "bal-acc", "acc"] + [f"recall {c}" for c in CLASSES]
    rows = [[f"majority (всегда {maj['class']})", maj["metrics"]["n"], f3(maj["metrics"]["macro_f1"]),
             f3(maj["metrics"]["balanced_accuracy"]), f3(maj["metrics"]["accuracy"])] + ["—"] * len(CLASSES)]
    for r in base:
        m = r["metrics"]
        rows.append([r["model"], m["n"], f3(m["macro_f1"]), f3(m["balanced_accuracy"]), f3(m["accuracy"])]
                    + [f3(m["recall"].get(c)) for c in CLASSES])
    out = [f"## {split}: полные токены, кадры {data_fs}", note, "", table(header, rows)]
    tests = [f"- {a['model']} против {b['model']}: прав только первый / только второй = {fmt_test(a, b)}"
             for a in base if a["ft"] for b in base if not b["ft"]]
    if tests:
        out += ["", "McNemar:"] + tests
    return "\n".join(o for o in out if o is not None)


def section_strategies(runs, data_fs, split):
    rows = []
    for model in models_of(runs):
        ref = full_run(runs, split, model, data_fs)
        for fs in STRATEGIES:
            cand = sorted([r for r in pick(runs, split=split, model=model, fs=fs, ratio=None, topk=None)
                           if r["max_frames"] == DEFAULT_MAX_FRAMES], key=lambda r: r["reselected"])
            if not cand:
                continue
            r, m = cand[0], cand[0]["metrics"]
            vs = fmt_test(r, ref) if ref and r is not ref else "эталон"
            rows.append([model, fs, m["n"], f3(m["macro_f1"]), f3(m["balanced_accuracy"]), f3(m["accuracy"]),
                         f"{r['frames']:.1f}" if r["frames"] else "—", f"{r['tokens']:.0f}" if r["tokens"] else "—", vs])
    if not rows:
        return None
    return "\n".join([f"## {split}: стратегии отбора кадров (бюджет {DEFAULT_MAX_FRAMES})",
                      f"Последний столбец: McNemar против {data_fs} той же модели, прав только эта стратегия / только {data_fs}.",
                      "", table(["модель", "кадры", "n", "macro-F1", "bal-acc", "acc", "кадров", "токенов", "McNemar"], rows)])


def section_pruning(runs, data_fs, split):
    out = []
    for model in models_of(runs):
        ref = full_run(runs, split, model, data_fs)
        pruned = [r for r in pick(runs, split=split, model=model, topk=None) if r["ratio"] is not None]
        if not ref or not pruned:
            continue
        rows = [["все токены", "1.0", f"{ref['tokens']:.0f}", f3(ref["metrics"]["macro_f1"]), f3(ref["metrics"]["accuracy"]),
                 "—", "—", "—"]]
        for ratio in sorted({r["ratio"] for r in pruned}, reverse=True):
            div = next(iter(pick(pruned, ratio=ratio, random=False)), None)
            rnd = next(iter(pick(pruned, ratio=ratio, random=True)), None)
            for r, label in ((div, "DivPrune"), (rnd, "случайные")):
                if r is None:
                    continue
                d = r["metrics"]["macro_f1"] - ref["metrics"]["macro_f1"]
                vs_rand = fmt_test(div, rnd) if r is div and rnd else "—"
                rows.append([label, f"{ratio:g}", f"{r['tokens']:.0f}", f3(r["metrics"]["macro_f1"]),
                             f3(r["metrics"]["accuracy"]), f"{d:+.3f}", fmt_test(r, ref), vs_rand])
        out += [f"### {model}", "", table(["токены", "доля", "токенов", "macro-F1", "acc", "Δ macro-F1",
                                           "McNemar против всех токенов", "DivPrune против случайных"], rows), ""]
    if not out:
        return None
    return "\n".join([f"## {split}: прунинг визуальных токенов",
                      "McNemar: прав только эта строка / только сравниваемый прогон.", ""] + out)


def section_topk(runs, data_fs, split):
    rows = []
    for model in models_of(runs):
        ref = full_run(runs, split, model, data_fs)
        for r in pick(runs, split=split, model=model, ratio=None):
            if not r["topk"] or not ref:
                continue
            ks = [p["n_frames"] for p in r["preds"].values()]
            chosen, dropped = [], []
            for p in r["preds"].values():
                scores, k = p.get("frame_scores") or [], len(p.get("topk_frames") or [])
                # topk_frames хранит номера кадров, а не позиции в пуле; Top-K берёт k кадров с наибольшим баллом,
                # поэтому выбранные — первые k по убыванию балла (при равных баллах — раньше по времени)
                if scores and k <= len(scores):
                    top = set(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k])
                    chosen += [x for i, x in enumerate(scores) if i in top]
                    dropped += [x for i, x in enumerate(scores) if i not in top]
            d = r["metrics"]["macro_f1"] - ref["metrics"]["macro_f1"]
            rows.append([model, f"{r['fs']}, пул {r['max_frames']}", r["topk"],
                         f"{statistics.mean(ks):.1f} ({min(ks)}–{max(ks)})", f"{r['tokens']:.0f}",
                         f3(r["metrics"]["macro_f1"]), f3(r["metrics"]["accuracy"]), f"{d:+.3f}", fmt_test(r, ref),
                         f"{statistics.mean(chosen):.3f} / {statistics.mean(dropped):.3f}" if chosen and dropped else "—"])
    if not rows:
        return None
    return "\n".join([f"## {split}: Top-K кадров",
                      "Балл кадра — доля его токенов, оставленная DivPrune в пуле. Последний столбец: средний балл "
                      "выбранных / отброшенных кадров. McNemar против всех кадров стратегии данных.", "",
                      table(["модель", "пул", "K", "кадров, среднее (мин–макс)", "токенов", "macro-F1", "acc", "Δ macro-F1",
                             "McNemar", "балл выбранных / отброшенных"], rows)])


def section_token_stats(runs, split):
    rows = []
    for model in models_of(runs):
        for r in sorted([r for r in pick(runs, split=split, model=model, topk=None) if r["ratio"] is not None],
                        key=lambda r: (-r["ratio"], r["random"])):
            cvs, zero, total = [], 0, 0
            for p in r["preds"].values():
                kept = p.get("kept_per_frame") or []
                if not kept:
                    continue
                mean = statistics.mean(kept)
                if mean > 0:
                    cvs.append(statistics.pstdev(kept) / mean)
                zero += sum(k == 0 for k in kept)
                total += len(kept)
            if total:
                rows.append([model, "случайные" if r["random"] else "DivPrune", f"{r['ratio']:g}",
                             f"{statistics.mean(cvs):.2f}", f"{100 * zero / total:.1f}%"])
    if not rows:
        return None
    return "\n".join([f"## {split}: распределение оставленных токенов по кадрам",
                      "CV — коэффициент вариации числа токенов между кадрами эпизода (0 — поровну), среднее по эпизодам.", "",
                      table(["модель", "токены", "доля", "CV по кадрам", "кадров без токенов"], rows)])


def section_checkpoints(run_dir):
    rows = []
    for path in glob.glob(os.path.join(run_dir, "checkpoint-*", "val_eval", "metrics.json")):
        ck = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            m = json.load(f)["overall"]["metrics"]
        rows.append((int(ck.split("-")[-1]), ck, m))
    if not rows:
        return None
    rows.sort()
    best_path = os.path.join(run_dir, "best_checkpoint.txt")
    best = os.path.basename(open(best_path).read().strip()) if os.path.exists(best_path) else None
    return "\n".join(["## Отбор чекпоинта по val", "", table(
        ["чекпоинт", "macro-F1", "bal-acc", "acc", ""],
        [[ck, f3(m["macro_f1"]), f3(m["balanced_accuracy"]), f3(m["accuracy"]), "выбран" if ck == best else ""]
         for _, ck, m in rows])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default="data/paper/eval")
    ap.add_argument("--run-dir", default=None, help="папка обучения с checkpoint-*/val_eval: таблица отбора чекпоинта")
    ap.add_argument("--data-fs", default=None, help="стратегия кадров данных; по умолчанию из ../prepare_config.json")
    ap.add_argument("--out", default=None, help="по умолчанию <eval-dir>/paper_tables.md")
    ap.add_argument("--keep-limited", action="store_true", help="не пропускать прогоны с --limit (смоук)")
    args = ap.parse_args()
    eval_dir = os.path.expanduser(args.eval_dir)
    data_fs = args.data_fs
    if data_fs is None:
        cfg = os.path.join(eval_dir, "..", "prepare_config.json")
        if not os.path.exists(cfg):
            raise SystemExit(f"нет {cfg}: укажите --data-fs")
        with open(cfg) as f:
            data_fs = json.load(f)["frame_selection"]
    runs, skipped = load_runs(eval_dir, data_fs, args.keep_limited)
    if not runs:
        raise SystemExit(f"в {eval_dir} нет прогонов evaluate.py")

    sections = [
        section_checkpoints(os.path.expanduser(args.run_dir)) if args.run_dir else None,
        section_strategies(runs, data_fs, "val"),
        section_main(runs, data_fs, "heldout", "Test: 400 эпизодов, 50 на пару агент × задача, метки gt."),
        section_main(runs, data_fs, "heldout_human",
                     "Человеческий тест: 141 эпизод с ручной разметкой, только PutCarrotOnPlate; согласие gt с ручной 66.7%."),
        section_strategies(runs, data_fs, "heldout"),
        section_pruning(runs, data_fs, "heldout"),
        section_topk(runs, data_fs, "heldout"),
        section_token_stats(runs, "heldout"),
    ]
    text = "\n\n".join(s for s in sections if s)
    if skipped:
        text += "\n\nПропущены:\n" + "\n".join(f"- {n}: {why}" for n, why in skipped)
    print(text)
    out = args.out or os.path.join(eval_dir, "paper_tables.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
