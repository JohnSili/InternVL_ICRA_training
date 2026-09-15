#!/usr/bin/env python3
"""Рисунки и статистика для статьи: отбор кадров относительно момента ошибки и токены DivPrune.

    python3 paper_figures.py --data data/paper/heldout.jsonl --ann-root ~/Simpler/gt \\
        --lora $(cat work_dirs/paper/best_checkpoint.txt) --out figures_out

1. Покрытие момента первой ошибки. Для каждого эпизода с ошибкой (A-D) берётся terminal_frame автоматической
   разметки и расстояние от него до ближайшего кадра каждого правила отбора. Доли по классам и точность модели
   при покрытии и без него (по predictions.jsonl прогонов в --eval-dir). Рисунок: два эпизода, смены команды
   гриппера, момент ошибки и кадры правил. -> fsr_coverage.json, fig_fsr_timeline.{pdf,png}
2. Токены DivPrune на кадре ошибки. По сохранённым kept_per_frame прогонов прунинга: доля токенов на кадре,
   ближайшем к ошибке, против средней доли в эпизоде; случайный выбор как контроль. -> divprune_failure_share.json
3. Какие токены оставлены. Для одного эпизода считает признаки модели и рисует оставленные токены DivPrune и
   случайного выбора с той же долей, как в evaluate.py. Нужна GPU; --no-gpu пропускает. -> fig_tokens.{pdf,png}

Сводка всех чисел — figures_summary.md.
"""

import argparse
import json
import math
import os
import statistics
import sys
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsr  # noqa: E402
from prepare_data import pick_frames  # noqa: E402

FAILURES = "ABCD"
POLICY = {"INTACT-pi0-scratch-bridge": "Pi0", "openvla-7b": "OpenVLA"}
# (подпись, стратегия, tail, суффикс папки прогона после <split>_<tag>)
RULES = [("dense+sparse", "dense_sparse", 0, ""), ("uniform", "uniform", 0, "_uniform"),
         ("surr(2)", "surrounding", 0, "_surrounding"), ("surr(2)+tail(5)", "surrounding", 5, "_surrounding_tail5")]
NEAR = 2  # кадр «покрывает» ошибку, если он не дальше NEAR шагов от неё


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def frame_number(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def failure_frame(ann_root, episode_id):
    agent, name = episode_id.split("/", 1)
    for p in (os.path.join(ann_root, agent, f"{name}_auto.json"), os.path.join(ann_root, f"{name}_auto.json")):
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f).get("terminal_frame")
    return None


def load_preds(eval_dir, name):
    path = os.path.join(eval_dir, name, "predictions.jsonl")
    return {p["id"]: p for p in load_jsonl(path)} if os.path.exists(path) else None


def plt_module():
    try:
        import matplotlib
    except ImportError:
        sys.exit("нет matplotlib: uv pip install --python .venv/bin/python matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42})
    return plt


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"saved {out}/{name}.{{pdf,png}}")


# ----------------------------------------------------------------------------
# 1. покрытие момента ошибки
# ----------------------------------------------------------------------------

def coverage(failures, eval_dir, split, tag):
    preds = {label: load_preds(eval_dir, f"{split}_{tag}{sfx}") for label, _, _, sfx in RULES}
    rows = []
    for rec, tf in failures:
        for label, fs, tail, _ in RULES:
            idx = pick_frames(rec, fs, tail=tail)
            p = (preds[label] or {}).get(rec["id"])
            rows.append({"id": rec["id"], "cls": rec["cls"], "rule": label, "dist": min(abs(i - tf) for i in idx),
                         "frames": len(idx), "correct": None if p is None else p["pred"] == p["gt"]})
    summary = {}
    for label, _, _, _ in RULES:
        summary[label] = {"predictions": preds[label] is not None}
        for cls in list(FAILURES) + ["all"]:
            sub = [r for r in rows if r["rule"] == label and (cls == "all" or r["cls"] == cls)]
            if not sub:
                continue
            near = [r for r in sub if r["dist"] <= NEAR]
            far = [r for r in sub if r["dist"] > NEAR]
            acc = lambda rs: (sum(r["correct"] for r in rs) / len(rs)) if rs and rs[0]["correct"] is not None else None
            summary[label][cls] = {"n": len(sub), "covered": len(near) / len(sub), "exact": sum(r["dist"] == 0 for r in sub) / len(sub),
                                   "median_dist": statistics.median(r["dist"] for r in sub),
                                   "acc_covered": acc(near), "n_covered": len(near), "acc_uncovered": acc(far), "n_uncovered": len(far)}
    return rows, summary


def pick_examples(failures, rows):
    """Эпизод C и эпизод D, где dense+sparse покрывает ошибку, а surr(2) нет; при наличии предсказаний —
    ещё и где модель права с dense+sparse и ошибается с surr(2)."""
    by = {(r["id"], r["rule"]): r for r in rows}
    out = []
    for cls in ("C", "D"):
        cands = [(rec, tf) for rec, tf in failures if rec["cls"] == cls]
        def score(item):
            rec = item[0]
            ds, su = by[(rec["id"], "dense+sparse")], by[(rec["id"], "surr(2)")]
            return (ds["dist"] <= NEAR and su["dist"] > NEAR, ds["correct"] is True and su["correct"] is False, su["dist"])
        if cands:
            out.append(max(sorted(cands, key=lambda x: x[0]["id"]), key=score))
    return out


def timeline_figure(examples, out):
    plt = plt_module()
    fig, axes = plt.subplots(len(examples), 1, figsize=(3.5, 1.25 * len(examples)), squeeze=False)
    for ax, (rec, tf) in zip(axes[:, 0], examples):
        rows = [("gripper change", [t for t, _ in rec["flips"]])]
        rows += [(label, pick_frames(rec, fs, tail=tail)) for label, fs, tail, _ in RULES]
        for y, (label, xs) in enumerate(reversed(rows)):
            if label == "gripper change":
                ax.scatter(xs, [y] * len(xs), marker="|", s=50, color="black", linewidths=1)
            else:
                ax.scatter(xs, [y] * len(xs), s=5, color="tab:blue")
        ax.axvline(tf, color="tab:red", ls="--", lw=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([label for label, _ in reversed(rows)])
        ax.set_xlim(-1, rec["n_frames"])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(f"{POLICY.get(rec['agent'], rec['agent'])}, {rec['instruction']}: class {rec['cls']}, "
                     f"first failure at step {tf} (dashed)")
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    save(fig, out, "fig_fsr_timeline")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2. токены DivPrune на кадре ошибки
# ----------------------------------------------------------------------------

def failure_share(failures, eval_dir, split, tag):
    out = {}
    for method, prefix in (("DivPrune", "tok"), ("random", "rand")):
        for ratio in (0.7, 0.5, 0.2):
            preds = load_preds(eval_dir, f"{split}_{tag}_{prefix}{ratio:g}")
            if preds is None:
                continue
            ratios, above, per_cls = [], 0, Counter()
            for rec, tf in failures:
                p = preds.get(rec["id"])
                if p is None or "kept_per_frame" not in p:
                    continue
                frames = [frame_number(path) for path in rec["image"]]
                kept = p["kept_per_frame"]
                assert len(kept) == len(frames), rec["id"]
                j = min(range(len(frames)), key=lambda k: abs(frames[k] - tf))
                mean = statistics.mean(kept)
                if mean == 0:
                    continue
                ratios.append(kept[j] / mean)
                above += kept[j] > mean
                per_cls[rec["cls"]] += 1
            if ratios:
                out[f"{method} {ratio:g}"] = {"n": len(ratios), "mean_ratio": statistics.mean(ratios),
                                              "median_ratio": statistics.median(ratios), "share_above_mean": above / len(ratios)}
    return out


# ----------------------------------------------------------------------------
# 3. какие токены оставлены
# ----------------------------------------------------------------------------

def token_masks(rec, model_name, lora, device, ratio, seed):
    """Оставленные токены DivPrune и случайного выбора по кадрам, теми же вызовами, что в evaluate.predict."""
    import torch
    from PIL import Image
    from evaluate import VIT_CHUNK, build_transform, load_model
    tok, model = load_model(model_name, lora, device)
    transform = build_transform(model.config.force_image_size or 448)
    with torch.inference_mode():
        pix = torch.stack([transform(Image.open(p)) for p in rec["image"]]).to(device, torch.bfloat16)
        vit = torch.cat([model.extract_feature(pix[i:i + VIT_CHUNK]) for i in range(0, pix.shape[0], VIT_CHUNK)])
    n_frames, tpf = vit.shape[0], vit.shape[1]
    flat = vit.reshape(-1, vit.shape[-1]).float()
    kept = {"DivPrune": fsr.kept_by_frame(fsr.divprune(flat, ratio), n_frames, tpf),
            "random": fsr.kept_by_frame(fsr.random_prune(flat.shape[0], ratio, zlib.crc32(f"{seed}/{rec['id']}".encode())),
                                        n_frames, tpf)}
    return kept, tpf


def token_figure(rec, tf, kept, tpf, ratio, out):
    """Кадры эпизода с затемнёнными отброшенными токенами и доля оставленных токенов по кадрам."""
    import numpy as np
    from PIL import Image
    plt = plt_module()
    n_frames = len(rec["image"])
    side = math.isqrt(tpf)
    assert side * side == tpf, tpf
    frames = [frame_number(p) for p in rec["image"]]
    jf = min(range(n_frames), key=lambda k: abs(frames[k] - tf))
    cols = sorted({0, n_frames // 4, jf, (3 * n_frames) // 4, n_frames - 1})
    size = 448
    cell = size // side
    fig = plt.figure(figsize=(7.0, 3.3))
    grid = fig.add_gridspec(3, len(cols), height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.05)
    for row, method in enumerate(("DivPrune", "random")):
        for c, j in enumerate(cols):
            ax = fig.add_subplot(grid[row, c])
            img = np.asarray(Image.open(rec["image"][j]).convert("RGB").resize((size, size)), dtype=float)
            mask = np.zeros((side, side), dtype=bool)
            for t in kept[method][j]:
                mask[t // side, t % side] = True
            img[~np.kron(mask, np.ones((cell, cell), dtype=bool))] *= 0.25
            ax.imshow(img.astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            mark = (" (failure)" if frames[j] == tf else " (near failure)") if j == jf else ""
            title = f"step {frames[j]}{mark}" if row == 0 else ""
            ax.set_title((title + "\n" if title else "") + f"{len(kept[method][j])} tokens")
            if c == 0:
                ax.set_ylabel(f"{method}, r={ratio:g}")
    ax = fig.add_subplot(grid[2, :])
    for method in ("DivPrune", "random"):
        ax.plot(frames, [len(k) / tpf for k in kept[method]], marker="o", ms=2, lw=1, label=method)
    ax.axvline(tf, color="tab:red", ls="--", lw=1, label="first failure")
    ax.set_xlabel("step")
    ax.set_ylabel("retained share")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, out, "fig_tokens")
    plt.close(fig)
    return {"id": rec["id"], "cls": rec["cls"], "failure_frame": tf, "ratio": ratio, "frames": frames,
            "kept_divprune": [len(k) for k in kept["DivPrune"]], "kept_random": [len(k) for k in kept["random"]]}


def fmt(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="jsonl test из prepare_data.py (кадры dense+sparse, flips)")
    ap.add_argument("--ann-root", required=True, help="автоматическая разметка <agent>/<name>_auto.json с terminal_frame")
    ap.add_argument("--eval-dir", default=None, help="папки прогонов evaluate.py; по умолчанию <dir(data)>/eval")
    ap.add_argument("--model", default="OpenGVLab/InternVL3-2B")
    ap.add_argument("--lora", default=None, help="чекпоинт дообученной модели; от него же имена папок прогонов")
    ap.add_argument("--out", default="figures_out")
    ap.add_argument("--no-gpu", action="store_true", help="без рисунка токенов")
    ap.add_argument("--episode", default=None, help="id эпизода для рисунка токенов; по умолчанию пример класса C")
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    data, ann_root = os.path.expanduser(args.data), os.path.expanduser(args.ann_root)
    eval_dir = os.path.expanduser(args.eval_dir) if args.eval_dir else os.path.join(os.path.dirname(os.path.abspath(data)), "eval")
    lora = os.path.expanduser(args.lora) if args.lora else None
    split = os.path.splitext(os.path.basename(data))[0]
    tag = os.path.basename(os.path.normpath(lora)) if lora else "base"
    os.makedirs(args.out, exist_ok=True)

    recs = load_jsonl(data)
    failures, missing = [], 0
    for rec in recs:
        if rec["cls"] not in FAILURES:
            continue
        tf = failure_frame(ann_root, rec["id"])
        if tf is None:
            missing += 1
            continue
        failures.append((rec, int(tf)))
    if not failures:
        sys.exit(f"нет эпизодов с ошибкой и terminal_frame в {ann_root}")
    print(f"эпизодов с ошибкой: {len(failures)} (без terminal_frame: {missing}), классы {dict(Counter(r['cls'] for r, _ in failures))}")

    rows, cov = coverage(failures, eval_dir, split, tag)
    share = failure_share(failures, eval_dir, split, tag)
    examples = pick_examples(failures, rows)
    timeline_figure(examples, args.out)
    tokens = None
    if not args.no_gpu:
        target = next(((r, tf) for r, tf in failures if r["id"] == args.episode), None) if args.episode else examples[0]
        if target is None:
            sys.exit(f"эпизод {args.episode} не найден среди эпизодов с ошибкой")
        kept, tpf = token_masks(target[0], args.model, lora, args.device, args.ratio, args.seed)
        tokens = token_figure(*target, kept, tpf, args.ratio, args.out)

    with open(os.path.join(args.out, "fsr_coverage.json"), "w") as f:
        json.dump({"near": NEAR, "summary": cov, "rows": rows}, f, indent=1)
    with open(os.path.join(args.out, "divprune_failure_share.json"), "w") as f:
        json.dump({"share": share, "token_example": tokens}, f, indent=1)

    lines = [f"# Рисунки и статистика ({split}, модель {tag})", "",
             f"## Покрытие момента первой ошибки: доля эпизодов, где правило берёт кадр не дальше {NEAR} шагов от ошибки", "",
             "| правило | " + " | ".join(list(FAILURES) + ["все"]) + " | медиана расстояния (все) | точность: покрыто / не покрыто (все) |",
             "|---" * (len(FAILURES) + 4) + "|"]
    for label, _, _, _ in RULES:
        s = cov[label]
        cells = [f"{fmt(s[c]['covered'])} (n={s[c]['n']})" if c in s else "—" for c in list(FAILURES) + ["all"]]
        a = s["all"]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {a['median_dist']:g} | "
                     f"{fmt(a['acc_covered'])} (n={a['n_covered']}) / {fmt(a['acc_uncovered'])} (n={a['n_uncovered']}) |")
    lines += ["", "Точность по классу C: покрыто / не покрыто", ""]
    for label, _, _, _ in RULES:
        c = cov[label].get("C")
        if c:
            lines.append(f"- {label}: {fmt(c['acc_covered'])} (n={c['n_covered']}) / {fmt(c['acc_uncovered'])} (n={c['n_uncovered']})")
    lines += ["", "## Доля токенов на кадре, ближайшем к ошибке, относительно средней по эпизоду", "",
              "| отбор | n | среднее отношение | медиана | доля эпизодов выше среднего |", "|---|---|---|---|---|"]
    for key, v in share.items():
        lines.append(f"| {key} | {v['n']} | {v['mean_ratio']:.2f} | {v['median_ratio']:.2f} | {v['share_above_mean']:.2f} |")
    lines += ["", "## Примеры на рисунке отбора кадров", ""] + [f"- {r['id']} (класс {r['cls']}, ошибка на шаге {tf})" for r, tf in examples]
    if tokens:
        lines += ["", f"## Рисунок токенов: {tokens['id']} (класс {tokens['cls']}, ошибка на шаге {tokens['failure_frame']}, доля {tokens['ratio']:g})"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "figures_summary.md"), "w") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
