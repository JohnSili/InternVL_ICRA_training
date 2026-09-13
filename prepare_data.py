#!/usr/bin/env python3
"""meta-json -> jsonl для InternVL (train / val / heldout + meta.json).

    python3 prepare_data.py --root /data/trajectories --out data/cls
    python3 prepare_data.py --root ~/Simpler/trajectories --ann-root ~/Simpler/gt --out data/cls
    python3 prepare_data.py --root ... --out data/obs --target obs --balance --drop-recovery

Кадры и actions всегда из --root (<агент>/{meta,frames}). Метки: без --ann-root из meta (ручная
разметка), с --ann-root из <агент>/<имя>_auto.json; held-out берёт тот же источник, что train, а
--holdout-src manual переключает его на ручную разметку. Печатается согласие двух разметок.
--holdout-per-group N нарезает held-out по N эпизодов на каждую пару (агент, задача).

Три набора: heldout (фиксированный список id в <out>/heldout.txt, создаётся один раз
и потом только читается), val (отбор чекпоинта) и train. Все стратифицированы по классу.
Промпт живёт только здесь: evaluate.py берёт его из jsonl, поэтому zero-shot и
дообученная модель гарантированно видят один и тот же текст. Стратегии отбора кадров
(select_frames) тоже живут здесь, evaluate.py их импортирует для ablation.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

N_FRAMES = 16
KEY_WINDOW = 2  # ±кадров вокруг смены команды гриппера (surr2)
MIN_UNIFORM = 4  # столько кадров всегда берём равномерно по эпизоду, даже если ключевых много
DENSE = 8  # dense_sparse: подряд идущих кадров вокруг первого закрытия гриппера
FRAME_SELECTIONS = ("surr2", "uniform", "dense_sparse")
CLASSES = "ABCDE"
OBSERVATIONS = [
    "gripper_approaches_wrong_object",
    "gripper_misses_target_position",
    "gripper_closes_beside_object",
    "gripper_closes_without_securing_object",
    "gripper_contacts_object_without_grasping",
    "object_slips_during_grasp",
    "end_effector_contacts_obstacle",
    "end_effector_moves_away_from_target",
    "object_released_outside_target",
    "object_remains_in_gripper",
    "object_dropped_during_transport",  # из gt-разметки, в словаре ТЗ его нет
]

# Эталонный текст human-хода (16 x "Frame k: <image>" + инструкция + описание классов) лежит в
# prompt_cls.txt / prompt_obs.txt рядом со скриптом; test_pipeline.py сверяет jsonl с ним побуквенно.
PROMPT_FILES = {t: os.path.join(os.path.dirname(os.path.abspath(__file__)), f"prompt_{t}.txt") for t in ("cls", "obs")}


def load_prompt(target):
    with open(PROMPT_FILES[target]) as f:
        text = f.read()
    assert text.count("<image>") == N_FRAMES, f"{PROMPT_FILES[target]}: {text.count('<image>')} x <image>, ожидалось {N_FRAMES}"
    assert "{instruction}" in text, f"{PROMPT_FILES[target]}: нет плейсхолдера {{instruction}}"
    return text


def agent_dirs(root):
    root = os.path.abspath(root)
    if os.path.basename(root) == "meta":
        a = os.path.dirname(root)
        return [(os.path.basename(a), root, os.path.join(a, "frames"))]
    if os.path.isdir(os.path.join(root, "meta")):
        return [(os.path.basename(root), os.path.join(root, "meta"), os.path.join(root, "frames"))]
    out = []
    for a in sorted(os.listdir(root)):
        m = os.path.join(root, a, "meta")
        if os.path.isdir(m):
            out.append((a, m, os.path.join(root, a, "frames")))
    return out


def as_label(class_key, ann, extra=None):
    """Единый вид метки из любого источника. None, если класс не из A-E."""
    if class_key not in set(CLASSES):  # set, а не строка: "" in "ABCDE" истинно, а None in "ABCDE" падает
        return None
    ann = ann or {}
    lab = {"cls": class_key, "obs": ann.get("observation") or "none", "recovery": bool(ann.get("recovery"))}
    if extra:
        lab.update(extra)
    return lab


def read_auto(ann_root, agent, name):
    """Метка из <ann_root>/<agent>/<name>_auto.json (или из плоского <ann_root>/<name>_auto.json).

    В _auto.json поля annotation лежат на верхнем уровне рядом с class_key."""
    for p in (os.path.join(ann_root, agent, f"{name}_auto.json"), os.path.join(ann_root, f"{name}_auto.json")):
        if os.path.exists(p):
            try:
                with open(p) as f:
                    a = json.load(f)
            except (OSError, ValueError) as e:
                print(f"bad auto-json {p}: {e}", file=sys.stderr)
                return None
            return as_label(a.get("class_key"), a, {"confidence": a.get("confidence"),
                                                    "unresolved": bool(a.get("unresolved"))})
    return None


def load_episodes(root, ann_root=None):
    """Кадры и actions всегда из <root>; метки из meta (ручные) и, если задан ann_root, из gt (auto)."""
    eps, skipped = [], Counter()
    for agent, meta_dir, frames_dir in agent_dirs(root):
        for fn in sorted(os.listdir(meta_dir)):
            if not fn.endswith("_meta.json"):
                continue
            name = fn[: -len("_meta.json")]
            eid = f"{agent}/{name}"
            try:
                with open(os.path.join(meta_dir, fn)) as f:
                    m = json.load(f)
            except (OSError, ValueError) as e:
                print(f"skip {eid}: {e}", file=sys.stderr)
                skipped["bad_json"] += 1
                continue
            if not m.get("actions"):
                skipped["no_actions"] += 1
                continue
            fd = os.path.join(frames_dir, f"{name}_frames")
            n_png = len([p for p in os.listdir(fd) if p.endswith(".png")]) if os.path.isdir(fd) else 0
            if n_png == 0:
                skipped["no_frames"] += 1
                continue
            g = [a[6] for a in m["actions"]]
            eps.append({
                "id": eid, "agent": agent, "task": re.sub(r"_\d+$", "", name),
                "frames_dir": fd, "n_frames": n_png,
                "instruction": m.get("instruction", ""),
                # смены команды гриппера: [кадр, новое значение]; по actions, не по states
                "flips": [[t, g[t]] for t in range(1, len(g)) if g[t] != g[t - 1]],
                "manual": as_label(m.get("class_key"), m.get("annotation")),
                "auto": read_auto(ann_root, agent, name) if ann_root else None,
            })
    if skipped:
        print(f"skipped: {dict(skipped)}", file=sys.stderr)
    return eps


def labeled(eps, src):
    """Эпизоды, у которых есть метка нужного источника, с полями cls/obs/recovery на верхнем уровне."""
    return [dict(e, **e[src]) for e in eps if e[src]]


def agreement_report(eps):
    """Матрица ручной разметки против gt на эпизодах, где есть обе. Задаёт потолок метрики."""
    both = [e for e in eps if e["manual"] and e["auto"]]
    if not both:
        print("\nпересечения ручной и gt разметки нет — сравнить нечем")
        return
    cm = Counter((e["manual"]["cls"], e["auto"]["cls"]) for e in both)
    same = sum(n for (a, b), n in cm.items() if a == b)
    print(f"\nсогласие разметок на {len(both)} эпизодах: {same}/{len(both)} = {same / len(both):.1%}"
          f"  (потолок macro-F1 на ручном held-out, если учить по gt)")
    print(f"{'manual/gt':>10s}" + "".join(f"{c:>6s}" for c in CLASSES))
    for a in CLASSES:
        row = [cm.get((a, b), 0) for b in CLASSES]
        if sum(row):
            print(f"{a:>10s}" + "".join(f"{v:6d}" for v in row))
    bad = [(e["id"], e["manual"]["cls"], e["auto"]["cls"]) for e in both if e["manual"]["cls"] != e["auto"]["cls"]]
    for i, a, b in bad[:10]:
        print(f"  расходятся: {i} manual={a} gt={b}")
    if len(bad) > 10:
        print(f"  ... ещё {len(bad) - 10}")


def _uniform(cands, k, rng, jitter):
    """k элементов из cands: по одному из k равных бинов (центр бина или случайный при jitter)."""
    out = []
    for b in range(k):
        lo = b * len(cands) // k
        hi = (b + 1) * len(cands) // k
        pos = rng.randrange(lo, hi) if jitter else (lo + hi - 1) // 2
        out.append(cands[pos])
    return out


def select_frames(n, flips, strategy="surr2", rng=None, jitter=False):
    """N_FRAMES индексов кадров по возрастанию.

    surr2       : ±KEY_WINDOW вокруг каждой смены команды гриппера + равномерная добивка
    uniform     : равномерно по всему эпизоду
    dense_sparse: DENSE кадров подряд вокруг первого закрытия гриппера + равномерная добивка
    rng=None или jitter=False -> детерминированно (центры бинов). Короткий эпизод -> дубли последнего.
    """
    if n <= N_FRAMES:
        return list(range(n)) + [n - 1] * (N_FRAMES - n)
    if strategy == "uniform":
        return sorted(_uniform(list(range(n)), N_FRAMES, rng, jitter))
    if strategy == "surr2":
        key = set()
        for t, _ in flips:
            for d in range(-KEY_WINDOW, KEY_WINDOW + 1):
                if 0 <= t + d < n:
                    key.add(t + d)
        key = sorted(key)
        if len(key) > N_FRAMES - MIN_UNIFORM:
            # много переключений: прореживаем ключевые, иначе начало/конец эпизода выпадут совсем
            k = N_FRAMES - MIN_UNIFORM
            key = sorted({key[round(i * (len(key) - 1) / (k - 1))] for i in range(k)})
    elif strategy == "dense_sparse":
        closes = [t for t, v in flips if v < 0]
        t = closes[0] if closes else (flips[0][0] if flips else n // 2)
        start = min(max(0, t - DENSE // 2 + 1), n - DENSE)
        key = list(range(start, start + DENSE))
    else:
        raise ValueError(f"unknown frame selection: {strategy}")
    rest = [i for i in range(n) if i not in set(key)]
    return sorted(key + _uniform(rest, N_FRAMES - len(key), rng, jitter))


_PROMPT_CACHE = {}


def make_record(ep, target, strategy, rng=None, jitter=False):
    if target not in _PROMPT_CACHE:
        _PROMPT_CACHE[target] = load_prompt(target)
    idx = select_frames(ep["n_frames"], ep["flips"], strategy, rng, jitter)
    paths = [os.path.join(ep["frames_dir"], f"{i:04d}.png") for i in idx]
    prompt = _PROMPT_CACHE[target].replace("{instruction}", ep["instruction"])
    answer = ep["cls"] if target == "cls" else f"{ep['cls']}|{ep['obs']}"
    # InternVL читает только id/image/conversations; остальное нужно evaluate.py (группировка по
    # агенту, пересбор кадров под другую стратегию) и test_pipeline.py (сверка промпта с эталоном)
    return {"id": ep["id"], "image": paths, "conversations": [
        {"from": "human", "value": prompt}, {"from": "gpt", "value": answer}],
        "agent": ep["agent"], "cls": ep["cls"], "instruction": ep["instruction"], "frames_dir": ep["frames_dir"],
        "n_frames": ep["n_frames"], "flips": ep["flips"], "frame_selection": strategy}


def ep_rng(seed, eid, copy):
    h = hashlib.md5(f"{seed}/{eid}/{copy}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def stratified_take(eps, frac, rng):
    """Отбирает ~frac из каждого класса (минимум 1, если в классе >= 2). -> (taken, rest)"""
    by_cls = defaultdict(list)
    for e in eps:
        by_cls[e["cls"]].append(e)
    taken, rest = [], []
    for c in sorted(by_cls):
        items = sorted(by_cls[c], key=lambda e: e["id"])
        rng.shuffle(items)
        k = max(1, round(len(items) * frac)) if len(items) >= 2 else 0
        taken += items[:k]
        rest += items[k:]
    return taken, rest


def take_quota(items, n, rng):
    """n эпизодов из группы с сохранением пропорций классов; добор и обрезка случайные."""
    if n >= len(items):
        return list(items)
    taken, rest = stratified_take(items, n / len(items), rng)
    if len(taken) > n:                       # stratified_take даёт минимум 1 на класс — может перебрать
        rng.shuffle(taken)
        taken = taken[:n]
    elif len(taken) < n:
        rng.shuffle(rest)
        taken += rest[: n - len(taken)]
    return taken


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("VLA_META_ROOT", "/data/trajectories"),
                    help="корень с <агент>/{meta,frames}; по умолчанию $VLA_META_ROOT")
    ap.add_argument("--ann-root", default=os.environ.get("VLA_ANN_ROOT") or None, metavar="DIR",
                    help="корень с gt-разметкой <агент>/<имя>_auto.json: метки train и val берутся оттуда, "
                         "held-out всегда по ручной разметке из meta; по умолчанию $VLA_ANN_ROOT")
    ap.add_argument("--drop-unresolved", action="store_true",
                    help="убрать из train и val эпизоды с unresolved=true в gt-разметке")
    ap.add_argument("--out", default="data/cls")
    ap.add_argument("--target", choices=("cls", "obs"), default="cls")
    ap.add_argument("--frame-selection", choices=FRAME_SELECTIONS, default="surr2")
    ap.add_argument("--holdout-list", default=None, help="файл с id held-out (по умолчанию <out>/heldout.txt)")
    ap.add_argument("--holdout-frac", type=float, default=0.25, help="доля на held-out, если список ещё не создан")
    ap.add_argument("--holdout-per-group", type=int, default=None, metavar="N",
                    help="вместо доли: ровно N эпизодов на каждую пару (агент, задача), "
                         "пропорции классов внутри группы сохраняются")
    ap.add_argument("--holdout-src", choices=("train", "manual", "auto"), default="train",
                    help="источник меток held-out: train — тот же, что у train (по умолчанию), "
                         "manual — ручная разметка из meta (эталон по ТЗ)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="доля val от оставшегося после held-out")
    ap.add_argument("--balance", action="store_true", help="oversampling редких классов в train, потолок x5")
    ap.add_argument("--drop-recovery", action="store_true", help="убрать recovery=true из train и val")
    ap.add_argument("--copies", type=int, default=1, help="копий train с разным джиттером (по одной на эпоху)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for k in ("root", "ann_root", "out", "holdout_list"):  # "~" в argparse не раскрывается сам
        if getattr(args, k):
            setattr(args, k, os.path.expanduser(getattr(args, k)))
    if not os.path.isdir(args.root):
        sys.exit(f"--root {args.root} не существует (или задайте VLA_META_ROOT)")
    if args.ann_root and not os.path.isdir(args.ann_root):
        sys.exit(f"--ann-root {args.ann_root} не существует")

    os.makedirs(args.out, exist_ok=True)
    eps = load_episodes(args.root, args.ann_root)
    if not eps:
        sys.exit(f"нет эпизодов с кадрами и actions в {args.root}")
    train_src = "auto" if args.ann_root else "manual"
    print(f"episodes: {len(eps)}  agents: {dict(sorted(Counter(e['agent'] for e in eps).items()))}")
    for src in ("manual", "auto"):
        lab = labeled(eps, src)
        if lab or src == train_src:
            print(f"  {src:6s} {len(lab):4d} размечено  {dict(sorted(Counter(e['cls'] for e in lab).items()))}")
    if not labeled(eps, train_src):
        sys.exit(f"нет эпизодов с разметкой {train_src}" + (f" в {args.ann_root}" if args.ann_root else ""))
    hold_src = train_src if args.holdout_src == "train" else args.holdout_src
    if not labeled(eps, hold_src):
        sys.exit(f"нет эпизодов с разметкой {hold_src} для held-out")
    print(f"метки: train/val <- {train_src}, held-out <- {hold_src}")

    # --- held-out: фиксированный список, создаётся один раз ---------------
    hold_path = args.holdout_list or os.path.join(args.out, "heldout.txt")
    if os.path.exists(hold_path):
        with open(hold_path) as f:
            hold_ids = {l.strip() for l in f if l.strip()}
        unknown = hold_ids - {e["id"] for e in eps}
        if unknown:
            print(f"warning: {len(unknown)} held-out id не найдены в данных: {sorted(unknown)[:5]}", file=sys.stderr)
        print(f"held-out list: {hold_path} ({len(hold_ids)} id)")
    else:
        pool = labeled(eps, hold_src)
        rng = random.Random(args.seed)
        if args.holdout_per_group:
            groups = defaultdict(list)
            for e in pool:
                groups[(e["agent"], e["task"])].append(e)
            hold = []
            for g in sorted(groups):
                got = take_quota(sorted(groups[g], key=lambda e: e["id"]), args.holdout_per_group, rng)
                hold += got
                if len(got) < args.holdout_per_group:
                    print(f"warning: группа {g}: {len(got)} эпизодов вместо {args.holdout_per_group}", file=sys.stderr)
        else:
            hold, _ = stratified_take(pool, args.holdout_frac, rng)
        hold_ids = {e["id"] for e in hold}
        with open(hold_path, "w") as f:
            f.write("\n".join(sorted(hold_ids)) + "\n")
        print(f"held-out list created: {hold_path} ({len(hold_ids)} id)")
    heldout = [e for e in labeled(eps, hold_src) if e["id"] in hold_ids]
    missing = hold_ids - {e["id"] for e in heldout}
    if missing:
        print(f"warning: {len(missing)} held-out id без разметки {hold_src}, выпали из held-out: "
              f"{sorted(missing)[:5]}", file=sys.stderr)
    rest = [e for e in labeled(eps, train_src) if e["id"] not in hold_ids]

    if args.drop_recovery:
        n0 = len(rest)
        rest = [e for e in rest if not e["recovery"]]
        print(f"drop-recovery: убрано {n0 - len(rest)} из train/val")
    if args.drop_unresolved:
        n0 = len(rest)
        rest = [e for e in rest if not e.get("unresolved")]
        print(f"drop-unresolved: убрано {n0 - len(rest)} из train/val")

    val, train = stratified_take(rest, args.val_frac, random.Random(args.seed + 1))

    # --- защита от утечки ---------------------------------------------------
    ids = {k: {e["id"] for e in v} for k, v in (("train", train), ("val", val), ("heldout", heldout))}
    for a, b in (("train", "heldout"), ("val", "heldout"), ("train", "val")):
        inter = ids[a] & ids[b]
        assert not inter, f"утечка {a}∩{b}: {sorted(inter)[:5]}"

    # --- oversampling -------------------------------------------------------
    cnt = Counter(e["cls"] for e in train)
    factor = {c: min(5, cnt.most_common(1)[0][1] // n) for c, n in cnt.items()} if args.balance else {}
    fs = args.frame_selection
    train_recs = []
    for e in train:
        for k in range(args.copies * factor.get(e["cls"], 1)):
            train_recs.append(make_record(e, args.target, fs, ep_rng(args.seed, e["id"], k), jitter=True))
    random.Random(args.seed).shuffle(train_recs)
    val_recs = [make_record(e, args.target, fs) for e in sorted(val, key=lambda e: e["id"])]
    hold_recs = [make_record(e, args.target, fs) for e in sorted(heldout, key=lambda e: e["id"])]

    for name, recs in (("train", train_recs), ("val", val_recs), ("heldout", hold_recs)):
        write_jsonl(os.path.join(args.out, f"{name}.jsonl"), recs)
    train_path = os.path.abspath(os.path.join(args.out, "train.jsonl"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"robot_errors": {"root": "", "annotation": train_path, "data_augment": False,
                                    "repeat_time": 1, "length": len(train_recs)}}, f, indent=2)
    with open(os.path.join(args.out, "prepare_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'':8s}" + "".join(f"{c:>6s}" for c in CLASSES) + f"{'total':>8s}")
    for name, recs in (("train", train_recs), ("val", val_recs), ("heldout", hold_recs)):
        c = Counter(r["cls"] for r in recs)
        print(f"{name:8s}" + "".join(f"{c.get(k, 0):6d}" for k in CLASSES) + f"{len(recs):8d}")
    if factor:
        print(f"balance factors: {factor}")
    by_group = Counter((e["agent"], e["task"]) for e in heldout)
    if len(by_group) > 1:
        print("held-out по группам (агент, задача): " + ", ".join(f"{a}/{t}={n}" for (a, t), n in sorted(by_group.items())))
    conf = Counter(e.get("confidence") for e in rest if e.get("confidence"))
    if conf:
        print(f"gt confidence в train/val: {dict(sorted(conf.items()))}")
    if args.ann_root:
        agreement_report(eps)
    print(f"\nframe selection: {fs}; written to {args.out}: train.jsonl val.jsonl heldout.jsonl meta.json")


if __name__ == "__main__":
    main()
