#!/usr/bin/env python3
"""Таблицы статьи и парные тесты McNemar по прогонам evaluate.py, без модели и GPU.

    python3 paper_tables.py                                      # data/paper/eval -> печать и paper_tables.md рядом
    python3 paper_tables.py --eval-dir server/data/paper/eval --run-dir server/work_dirs/paper

Если рядом есть <eval-dir>/../labels от label_stats.py, добавляются таблица данных и строка gt-разметчика на
человеческом тесте. Время на эпизод берётся из логов очереди paper_evals*.log (по умолчанию в корне проекта,
на три уровня выше eval-dir): последняя строка прогресса [n/n] Ts каждого прогона, без загрузки модели.

Прогон описывается своими metrics.json и run_config.json (модель, LoRA, стратегия кадров, прунинг), а не
именем папки. Прогоны с --limit и пересчёты --from-predictions пропускаются. Сравнения парные: берутся
общие id двух прогонов, точный двусторонний тест McNemar по эпизодам, где прав ровно один из двух.
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import Counter

CLASSES = "ABCDE"
STRATEGIES = ("dense_sparse", "uniform", "surrounding")
DEFAULT_MAX_FRAMES = 20

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import metrics  # noqa: E402  те же метрики, что в metrics.json


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def model_name(m):
    base = os.path.basename(os.path.normpath(m["model"]))
    return f"{base} + LoRA ({os.path.basename(os.path.normpath(m['lora']))})" if m.get("lora") else base


def load_timings(paths):
    """Имя папки прогона -> секунд на эпизод по логам paper_evals.sh."""
    out = {}
    for path in paths:
        cur = None
        with open(path, errors="replace") as f:
            for line in f:
                m = re.match(r"=== \d\d:\d\d (.+?)\s*$", line)
                if m:
                    cur = os.path.basename(os.path.normpath(m.group(1)))
                    continue
                m = re.match(r"\[(\d+)/(\d+)\] (\d+)s", line)
                if cur and m and m.group(1) == m.group(2):
                    out[cur] = int(m.group(3)) / int(m.group(2))
    return out


def labeler_run(path):
    """Строка «gt-разметчик против людей»: gt — ручная метка, pred — метка gt-разметчика."""
    preds = {p["id"]: p for p in load_jsonl(path)}
    ys = list(preds.values())
    return {"dir": os.path.basename(path), "split": "heldout_human", "model": "gt-разметчик", "ft": False,
            "metrics": metrics([p["gt"] for p in ys], [p["pred"] for p in ys]), "preds": preds,
            "frames": None, "tokens": None, "sec": None}


def load_runs(eval_dir, data_fs, keep_limited, timings):
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
            "sec": timings.get(name),
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


def fsec(x):
    return "—" if x is None else f"{x:.2f}"


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


def section_main(runs, data_fs, split, note, labeler=None):
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
    if labeler:
        m = labeler["metrics"]
        rows.append([labeler["model"], m["n"], f3(m["macro_f1"]), f3(m["balanced_accuracy"]), f3(m["accuracy"])]
                    + [f3(m["recall"].get(c)) for c in CLASSES])
    out = [f"## {split}: полные токены, кадры {data_fs}", note, "", table(header, rows)]
    tests = [f"- {a['model']} против {b['model']}: прав только первый / только второй = {fmt_test(a, b)}"
             for a in base if a["ft"] for b in base + ([labeler] if labeler else []) if not b["ft"]]
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
                         f"{r['frames']:.1f}" if r["frames"] else "—", f"{r['tokens']:.0f}" if r["tokens"] else "—",
                         fsec(r["sec"]), vs])
    if not rows:
        return None
    return "\n".join([f"## {split}: стратегии отбора кадров (бюджет {DEFAULT_MAX_FRAMES})",
                      f"Последний столбец: McNemar против {data_fs} той же модели, прав только эта стратегия / только {data_fs}.",
                      "", table(["модель", "кадры", "n", "macro-F1", "bal-acc", "acc", "кадров", "токенов", "с/эпизод", "McNemar"], rows)])


def section_pruning(runs, data_fs, split):
    out = []
    for model in models_of(runs):
        ref = full_run(runs, split, model, data_fs)
        pruned = [r for r in pick(runs, split=split, model=model, topk=None) if r["ratio"] is not None]
        if not ref or not pruned:
            continue
        rows = [["все токены", "1.0", f"{ref['tokens']:.0f}", fsec(ref["sec"]), f3(ref["metrics"]["macro_f1"]),
                 f3(ref["metrics"]["accuracy"]), "—", "—", "—"]]
        for ratio in sorted({r["ratio"] for r in pruned}, reverse=True):
            div = next(iter(pick(pruned, ratio=ratio, random=False)), None)
            rnd = next(iter(pick(pruned, ratio=ratio, random=True)), None)
            for r, label in ((div, "DivPrune"), (rnd, "случайные")):
                if r is None:
                    continue
                d = r["metrics"]["macro_f1"] - ref["metrics"]["macro_f1"]
                vs_rand = fmt_test(div, rnd) if r is div and rnd else "—"
                rows.append([label, f"{ratio:g}", f"{r['tokens']:.0f}", fsec(r["sec"]), f3(r["metrics"]["macro_f1"]),
                             f3(r["metrics"]["accuracy"]), f"{d:+.3f}", fmt_test(r, ref), vs_rand])
        out += [f"### {model}", "", table(["токены", "доля", "токенов", "с/эпизод", "macro-F1", "acc", "Δ macro-F1",
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
                         f"{statistics.mean(ks):.1f} ({min(ks)}–{max(ks)})", f"{r['tokens']:.0f}", fsec(r["sec"]),
                         f3(r["metrics"]["macro_f1"]), f3(r["metrics"]["accuracy"]), f"{d:+.3f}", fmt_test(r, ref),
                         f"{statistics.mean(chosen):.3f} / {statistics.mean(dropped):.3f}" if chosen and dropped else "—"])
    if not rows:
        return None
    return "\n".join([f"## {split}: Top-K кадров",
                      "Балл кадра — доля его токенов, оставленная DivPrune в пуле. Последний столбец: средний балл "
                      "выбранных / отброшенных кадров. McNemar против всех кадров стратегии данных.", "",
                      table(["модель", "пул", "K", "кадров, среднее (мин–макс)", "токенов", "с/эпизод", "macro-F1", "acc", "Δ macro-F1",
                             "McNemar", "балл выбранных / отброшенных"], rows)])


def section_efficiency(runs, data_fs, timing_path):
    """Время из отдельного замера (timing.log цепочки final), качество — из полных прогонов на test."""
    sec = load_timings([timing_path])
    ft = [m for m in models_of(runs) if " + LoRA" in m]
    if not sec or not ft:
        return None
    model = ft[-1]
    ref = full_run(runs, "heldout", model, data_fs)
    if not ref:
        return None
    full_times = [sec[k] for k in ("full", "full_again") if k in sec]
    base_t = statistics.mean(full_times) if full_times else None
    first = lambda cand: next(iter(cand), None)
    variants = [("все токены", ref, base_t),
                ("uniform", first(r for r in pick(runs, split="heldout", model=model, fs="uniform", ratio=None, topk=None)
                                  if r["max_frames"] == DEFAULT_MAX_FRAMES), sec.get("uniform"))]
    for ratio in (0.5, 0.2):
        variants.append((f"DivPrune {ratio:g}", first(pick(runs, split="heldout", model=model, ratio=ratio, random=False, topk=None)),
                         sec.get(f"tok{ratio:g}")))
        variants.append((f"случайные {ratio:g}", first(pick(runs, split="heldout", model=model, ratio=ratio, random=True, topk=None)),
                         sec.get(f"rand{ratio:g}")))
    variants.append(("Top-K", first(r for r in pick(runs, split="heldout", model=model, ratio=None) if r["topk"]), sec.get("topk20")))
    rows = [[label, f"{r['frames']:.1f}", f"{r['tokens']:.0f}", f"{t:.2f}", f"{t / base_t:.2f}×" if base_t else "—",
             f3(r["metrics"]["macro_f1"]), f3(r["metrics"]["accuracy"]), "—" if r is ref else fmt_test(r, ref)]
            for label, r, t in variants if r is not None and t is not None]
    drift = (f" Все токены в начале и в конце замера: {full_times[0]:.2f} и {full_times[1]:.2f} с/эпизод."
             if len(full_times) == 2 else "")
    return "\n".join([f"## Эффективность: {model}",
                      "Время — отдельный замер на первых 100 эпизодах test, прогоны по очереди без других задач на карте, "
                      f"с чтением кадров, без загрузки модели; прогрев отброшен.{drift} Качество — на всех 400 эпизодах.", "",
                      table(["вариант", "кадров", "токенов", "с/эпизод", "к всем токенам", "macro-F1", "acc",
                             "McNemar против всех токенов"], rows)])


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


def section_groups(runs, data_fs, split):
    """Дообученная модель не должна держаться на одном агенте или задаче: метрики по группам."""
    rows, header, keys = [], None, None
    for model in models_of(runs):
        r = full_run(runs, split, model, data_fs)
        if not r:
            continue
        preds = list(r["preds"].values())
        if keys is None:
            keys = ([("agent", v) for v in sorted({p.get("agent", "?") for p in preds})]
                    + [("task", v) for v in sorted({p.get("task", "?") for p in preds})])
            header = ["модель"] + [f"{v} (n={sum(p.get(k) == v for p in preds)})" for k, v in keys]
        cells = []
        for k, v in keys:
            sub = [p for p in preds if p.get(k) == v]
            m = metrics([p["gt"] for p in sub], [p["pred"] for p in sub]) if sub else None
            cells.append(f"{m['macro_f1']:.3f} / {m['accuracy']:.3f}" if m else "—")
        rows.append([model] + cells)
    if not rows:
        return None
    preds = list(full_run(runs, split, models_of(runs)[0], data_fs)["preds"].values())
    maj = []
    for k, v in keys:
        c = Counter(p["gt"] for p in preds if p.get(k) == v)
        maj.append(f"— / {max(c.values()) / sum(c.values()):.3f} ({c.most_common(1)[0][0]})" if c else "—")
    rows.append(["majority внутри группы"] + maj)
    return "\n".join([f"## {split}: по агентам и задачам, полные токены",
                      "В ячейке macro-F1 / accuracy; macro-F1 по классам, которые есть в группе. Классы сильно зависят от "
                      "агента, поэтому точность сравнивать с majority внутри той же группы.", "", table(header, rows)])


def section_dataset(counts, labeler):
    out = ["## Данные"]
    if counts:
        rows, total, total_manual = [], Counter(), 0
        for agent, tasks in sorted(counts["auto"].items()):
            c = Counter()
            for tc in tasks.values():
                c.update(tc)
            n = sum(c.values())
            manual = sum(sum(tc.values()) for tc in counts["manual"].get(agent, {}).values())
            rows.append([agent, n] + [f"{c[k]} ({100 * c[k] / n:.1f}%)" for k in CLASSES] + [manual])
            total.update(c)
            total_manual += manual
        n = sum(total.values())
        rows.append(["всего", n] + [f"{total[k]} ({100 * total[k] / n:.1f}%)" for k in CLASSES] + [total_manual])
        task_rows = [[agent, task, sum(tc.values())] + [tc.get(k, 0) for k in CLASSES]
                     for agent, tasks in sorted(counts["auto"].items()) for task, tc in sorted(tasks.items())]
        out += ["", f"Классы по gt-разметке; без gt-метки {counts['unlabeled']['auto']} эпизодов.", "",
                table(["агент", "эпизодов"] + list(CLASSES) + ["с ручной меткой"], rows), "",
                table(["агент", "задача", "эпизодов"] + list(CLASSES), task_rows)]
    if labeler:
        m = labeler["metrics"]
        out += ["", f"Согласие gt-разметки с ручной на {m['n']} эпизодах: accuracy {m['accuracy']:.3f}, "
                    f"macro-F1 {m['macro_f1']:.3f}. Строки — ручная метка, столбцы — gt.", "",
                table(["ручная \\ gt"] + list(CLASSES), [[CLASSES[i]] + m["confusion"][i] for i in range(len(CLASSES))])]
    return "\n".join(out)


PRIOR_TAUS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def train_priors(path):
    with open(path) as f:
        c = Counter(json.loads(l)["cls"] for l in f if l.strip())
    n = sum(c.values())
    return {k: c.get(k, 0) / n for k in CLASSES}


def adjust(preds, priors, tau):
    """Logit adjustment: класс = argmax_c log p_c − τ·log π_c; при равенстве — более ранняя буква."""
    out = {}
    for i, p in preds.items():
        score = {c: math.log(max(p["probs"][c], 1e-6)) - tau * math.log(max(priors[c], 1e-6)) for c in CLASSES}
        out[i] = dict(p, pred=max(CLASSES, key=lambda c: (score[c], -CLASSES.index(c))))
    return out


def section_prior(runs, data_fs, run_dir, priors):
    """Поправка дообученной модели на дисбаланс классов без переобучения; τ выбирается только на val."""
    title = "## Поправка на частоту классов (logit adjustment)"
    best_path = os.path.join(run_dir, "best_checkpoint.txt")
    if not os.path.exists(best_path):
        return f"{title}\nпропущено: нет {best_path}"
    with open(best_path) as f:
        ck = os.path.basename(f.read().strip())
    val_path = os.path.join(run_dir, ck, "val_eval", "predictions.jsonl")
    model = next((m for m in models_of(runs) if m.endswith(f"({ck})")), None)
    if not os.path.exists(val_path):
        return f"{title}\nпропущено: нет {val_path}, τ не на чем выбрать"
    if model is None:
        return f"{title}\nпропущено: среди прогонов нет модели с лучшим чекпоинтом {ck}"
    val = {p["id"]: p for p in load_jsonl(val_path)}
    grid = []
    for tau in PRIOR_TAUS:
        ys = list(adjust(val, priors, tau).values())
        grid.append((tau, metrics([p["gt"] for p in ys], [p["pred"] for p in ys])["macro_f1"]))
    best_tau = max(grid, key=lambda x: (round(x[1], 9), -x[0]))[0]  # при равенстве — меньший τ
    rows = []
    for split in ("heldout", "heldout_human"):
        ref = full_run(runs, split, model, data_fs)
        if not ref:
            continue
        preds = adjust(ref["preds"], priors, best_tau)
        ys = list(preds.values())
        adj = dict(ref, preds=preds, metrics=metrics([p["gt"] for p in ys], [p["pred"] for p in ys]),
                   dir=f"{ref['dir']}+tau{best_tau:g}")
        for label, r in (("без поправки", ref), (f"τ={best_tau:g}", adj)):
            m = r["metrics"]
            rows.append([split, label, f3(m["macro_f1"]), f3(m["balanced_accuracy"]), f3(m["accuracy"])]
                        + [f3(m["recall"].get(c)) for c in CLASSES] + [fmt_test(adj, ref) if r is adj else "—"])
    if not rows:
        return None
    return "\n".join([
        f"## {model}: поправка на частоту классов (logit adjustment)",
        "score_c = log p_c − τ·log π_c, π — доли классов в train: " + ", ".join(f"{c} {priors[c]:.3f}" for c in CLASSES) + ".",
        f"τ выбран по macro-F1 на val ({len(val)} эпизодов) чекпоинта {ck}: "
        + ", ".join(f"τ={t:g} → {v:.3f}" for t, v in grid) + f"; выбран τ={best_tau:g}. На test и человеческом тесте "
        "τ не подбирался. McNemar: прав только вариант с поправкой / только без.", "",
        table(["сплит", "вариант", "macro-F1", "bal-acc", "acc"] + [f"recall {c}" for c in CLASSES] + ["McNemar"], rows)])


def section_checkpoints(run_dir, train_log=None):
    rows = {}
    for path in glob.glob(os.path.join(run_dir, "checkpoint-*", "val_eval", "metrics.json")):
        ck = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            m = json.load(f)["overall"]["metrics"]
        rows[ck] = [ck, f3(m["macro_f1"]), f3(m["balanced_accuracy"]), f3(m["accuracy"])]
    if train_log and os.path.exists(train_log):  # чекпоинты, удалённые после обучения, остались только в логе train.sh
        with open(train_log, errors="replace") as f:
            for line in f:
                mm = re.search(r"(checkpoint-\d+)\s+val macro-F1 = ([\d.]+)", line)
                if mm and mm.group(1) not in rows:
                    rows[mm.group(1)] = [mm.group(1), f3(float(mm.group(2))), "—", "—"]
    if not rows:
        return None
    best_path = os.path.join(run_dir, "best_checkpoint.txt")
    best = os.path.basename(open(best_path).read().strip()) if os.path.exists(best_path) else None
    ordered = sorted(rows.values(), key=lambda r: int(r[0].split("-")[-1]))
    return "\n".join(["## Отбор чекпоинта по val", "Прочерки — чекпоинты, от которых остался только лог train.sh.", "",
                      table(["чекпоинт", "macro-F1", "bal-acc", "acc", ""], [r + ["выбран" if r[0] == best else ""] for r in ordered])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default="data/paper/eval")
    ap.add_argument("--run-dir", default=None, help="папка обучения с checkpoint-*/val_eval: таблица отбора чекпоинта")
    ap.add_argument("--data-fs", default=None, help="стратегия кадров данных; по умолчанию из ../prepare_config.json")
    ap.add_argument("--out", default=None, help="по умолчанию <eval-dir>/paper_tables.md")
    ap.add_argument("--labels", default=None, help="папка label_stats.py; по умолчанию <eval-dir>/../labels")
    ap.add_argument("--log", action="append", default=None,
                    help="лог paper_evals.sh для времени на эпизод в таблицах прогонов; по умолчанию не берётся: в очереди "
                         "прогоны делили карту, и время несравнимо")
    ap.add_argument("--timing", default=None, help="timing.log отдельного замера; по умолчанию в корне проекта")
    ap.add_argument("--train-log", default=None, help="лог train.sh с val macro-F1 чекпоинтов; по умолчанию train_paper.log в корне")
    ap.add_argument("--train", default=None, help="train.jsonl для долей классов в поправке; по умолчанию <eval-dir>/../train.jsonl")
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
    root = os.path.join(eval_dir, "..", "..", "..")
    logs = args.log or []
    timing_path = args.timing or os.path.join(root, "timing.log")
    train_log = args.train_log or os.path.join(root, "train_paper.log")
    runs, skipped = load_runs(eval_dir, data_fs, args.keep_limited, load_timings(logs))
    labels_dir = args.labels or os.path.join(eval_dir, "..", "labels")
    counts_path = os.path.join(labels_dir, "label_counts.json")
    labeler_path = os.path.join(labels_dir, "labeler_vs_human.jsonl")
    counts = None
    if os.path.exists(counts_path):
        with open(counts_path) as f:
            counts = json.load(f)
    labeler = labeler_run(labeler_path) if os.path.exists(labeler_path) else None
    train_path = args.train or os.path.join(eval_dir, "..", "train.jsonl")
    priors = train_priors(train_path) if os.path.exists(train_path) else None
    if not runs:
        raise SystemExit(f"в {eval_dir} нет прогонов evaluate.py")

    sections = [
        section_checkpoints(os.path.expanduser(args.run_dir), train_log) if args.run_dir else None,
        section_dataset(counts, labeler) if counts or labeler else None,
        section_strategies(runs, data_fs, "val"),
        section_main(runs, data_fs, "heldout", "Test: эпизоды из списка held-out, метки gt."),
        section_main(runs, data_fs, "heldout_human",
                     "Человеческий тест: эпизоды с ручной разметкой, метки людей. Строка gt-разметчика — согласие "
                     "автоматической разметки с людьми, это не модель.", labeler),
        section_groups(runs, data_fs, "heldout"),
        section_prior(runs, data_fs, os.path.expanduser(args.run_dir), priors) if args.run_dir and priors else None,
        section_strategies(runs, data_fs, "heldout"),
        section_pruning(runs, data_fs, "heldout"),
        section_topk(runs, data_fs, "heldout"),
        section_efficiency(runs, data_fs, timing_path) if os.path.exists(timing_path) else None,
        section_token_stats(runs, "heldout"),
    ]
    text = "\n\n".join(s for s in sections if s)
    if logs:
        text += "\n\nВремя на эпизод в таблицах прогонов из логов очереди (несравнимо между прогонами): " + ", ".join(logs)
    if skipped:
        text += "\n\nПропущены:\n" + "\n".join(f"- {n}: {why}" for n, why in skipped)
    print(text)
    out = args.out or os.path.join(eval_dir, "paper_tables.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
