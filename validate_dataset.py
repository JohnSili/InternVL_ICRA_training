#!/usr/bin/env python3
"""Валидатор датасета траекторий (pytest).

    VLA_META_ROOT=/data/trajectories python3 validate_dataset.py        # hard, потом soft
    VLA_META_ROOT=... python3 validate_dataset.py --limit 20 --seed 1    # смоук на случайных 20
    VLA_META_ROOT=... VLA_META_LIMIT=20 VLA_META_SEED=1 pytest validate_dataset.py -m "not soft"

Один тест — одно свойство по всему датасету; в сообщении первые 10 нарушителей.
--limit/--seed работают при запуске через python3 (модуль подключается как плагин);
под голым pytest те же значения передаются через VLA_META_LIMIT / VLA_META_SEED.
Корень: <root>/<agent>/meta/*_meta.json + <root>/<agent>/frames/<name>_frames/NNNN.png.
Также принимается папка одного агента (<root>/meta/) или сама папка meta/.
"""

import json
import os
import random
import re
import sys

import pytest

CLASSES = ("A", "B", "C", "D", "E")
STAGE_ORDER = ("approach", "grasp", "transport", "placement")
OBSERVATIONS = {
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
}
PNG_RE = re.compile(r"^(\d{4})\.png$")
SHOW = 10


def agent_dirs(root):
    """-> [(agent, meta_dir, frames_dir)] для трёх поддерживаемых раскладок."""
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


def load_dataset(root, limit=None, seed=0):
    """Список записей: id, meta (или None + error), список индексов png. limit -> случайные limit штук."""
    recs = []
    for agent, meta_dir, frames_dir in agent_dirs(root):
        for fn in sorted(os.listdir(meta_dir)):
            if not fn.endswith("_meta.json"):
                continue
            name = fn[: -len("_meta.json")]
            rec = {"id": f"{agent}/{name}", "path": os.path.join(meta_dir, fn),
                   "meta": None, "error": None, "png_idx": None}
            try:
                with open(rec["path"]) as f:
                    rec["meta"] = json.load(f)
            except (OSError, ValueError) as e:
                rec["error"] = str(e)
            fd = os.path.join(frames_dir, f"{name}_frames")
            if os.path.isdir(fd):
                idx = []
                for p in os.listdir(fd):
                    m = PNG_RE.match(p)
                    if m:
                        idx.append(int(m.group(1)))
                rec["png_idx"] = sorted(idx)
            recs.append(rec)
    recs.sort(key=lambda r: r["id"])
    if limit and limit < len(recs):
        random.Random(seed).shuffle(recs)
        recs = sorted(recs[:limit], key=lambda r: r["id"])
    return recs


def pytest_addoption(parser):
    # работает только когда модуль подключён как плагин (запуск через python3 validate_dataset.py)
    parser.addoption("--limit", type=int, default=None, help="случайные N эпизодов (смоук)")
    parser.addoption("--seed", type=int, default=None, help="seed для выбора подмножества при --limit")


@pytest.fixture(scope="session")
def dataset(request):
    root = os.environ.get("VLA_META_ROOT", "/data/trajectories")
    assert os.path.isdir(root), f"VLA_META_ROOT не существует: {root}"
    limit = request.config.getoption("--limit", None) or os.environ.get("VLA_META_LIMIT")
    seed = request.config.getoption("--seed", None)
    seed = int(os.environ.get("VLA_META_SEED", 0)) if seed is None else seed
    recs = load_dataset(root, int(limit) if limit else None, seed)
    assert recs, f"в {root} не найдено *_meta.json"
    print(f"\n[dataset] root={root} episodes={len(recs)}" + (f" (limit={limit} seed={seed})" if limit else ""))
    return recs


def good(recs):
    """Только читаемые записи: остальные уже провалили test_json_readable."""
    return [r for r in recs if r["meta"] is not None and isinstance(r["meta"], dict)]


def report(what, bad, total):
    """bad: список (id, деталь). Печатает первые SHOW, падает если непусто."""
    if bad:
        lines = "\n".join(f"  {i}: {d}" for i, d in bad[:SHOW])
        more = f"\n  ... ещё {len(bad) - SHOW}" if len(bad) > SHOW else ""
        pytest.fail(f"{what}: {len(bad)}/{total}\n{lines}{more}", pytrace=False)


def n_frames(r):
    return len(r["png_idx"]) if r["png_idx"] is not None else None


# ----------------------------------------------------------------------------
# hard
# ----------------------------------------------------------------------------

def test_json_readable(dataset):
    bad = [(r["id"], r["error"]) for r in dataset if r["meta"] is None]
    bad += [(r["id"], "не dict") for r in dataset if r["meta"] is not None and not isinstance(r["meta"], dict)]
    for r in good(dataset):
        missing = [k for k in ("annotation", "class_key", "actions", "episode_stats") if k not in r["meta"]]
        if missing:
            bad.append((r["id"], f"нет ключей {missing}"))
        elif r["meta"]["class_key"] not in CLASSES:
            bad.append((r["id"], f"class_key={r['meta']['class_key']!r}"))
    report("битые/неполные json", bad, len(dataset))


def test_actions_shape(dataset):
    bad = []
    for r in good(dataset):
        acts = r["meta"].get("actions")
        if not isinstance(acts, list) or not acts:
            bad.append((r["id"], "actions пусты"))
            continue
        for t, a in enumerate(acts):
            if not isinstance(a, list) or len(a) != 7 or not all(isinstance(x, (int, float)) for x in a):
                bad.append((r["id"], f"t={t}: {a!r}"[:120]))
                break
    report("actions не (T,7)", bad, len(dataset))


def test_gripper_command_binary(dataset):
    bad = []
    for r in good(dataset):
        vals = {a[6] for a in r["meta"].get("actions") or [] if isinstance(a, list) and len(a) == 7}
        extra = vals - {-1.0, 1.0}
        if extra:
            bad.append((r["id"], f"gripper cmd values {sorted(extra)[:5]}"))
    report("команда гриппера не из {-1,1}", bad, len(dataset))


def test_frames_contiguous(dataset):
    bad = []
    for r in dataset:
        idx = r["png_idx"]
        if idx is None:
            bad.append((r["id"], "нет папки frames"))
        elif not idx:
            bad.append((r["id"], "нет png"))
        elif idx != list(range(len(idx))):
            missing = sorted(set(range(idx[-1] + 1)) - set(idx))
            bad.append((r["id"], f"n={len(idx)} max={idx[-1]} пропуски {missing[:5]}"))
    report("кадры не сплошные 0000..N", bad, len(dataset))


def test_frame_offset_consistent(dataset):
    offs = {}
    for r in good(dataset):
        n = n_frames(r)
        acts = r["meta"].get("actions") or []
        if n and acts:
            offs.setdefault(n - len(acts), []).append(r["id"])
    if not offs:
        pytest.skip("нечего сравнивать")
    print(f"\n[offset] len(png)-len(actions): { {k: len(v) for k, v in offs.items()} }")
    major = max(offs, key=lambda k: len(offs[k]))
    bad = [(i, f"offset={k} (ожидался {major})") for k, v in offs.items() if k != major for i in v]
    if major not in (0, 1):
        bad = [(i, f"offset={major}") for i in offs[major]] + bad
    report("офсет png/actions не одинаков или не в {0,1}", bad, len(dataset))


def test_terminal_frame_in_range(dataset):
    bad = []
    for r in good(dataset):
        tf = (r["meta"].get("annotation") or {}).get("terminal_frame")
        total = n_frames(r) or len(r["meta"].get("actions") or [])
        if not isinstance(tf, int) or isinstance(tf, bool) or not (0 <= tf < total):
            bad.append((r["id"], f"terminal_frame={tf!r} total={total}"))
    report("terminal_frame вне [0,total)", bad, len(dataset))


def test_stages_contiguous_cover(dataset):
    bad = []
    for r in good(dataset):
        st = (r["meta"].get("annotation") or {}).get("stages")
        if st is None:
            continue
        n_act = len(r["meta"].get("actions") or [])
        ends = {n_act - 1, (n_frames(r) or n_act) - 1}  # с офсетом 1 конец может быть по кадрам
        why = None
        if not isinstance(st, dict) or not st or set(st) - set(STAGE_ORDER):
            why = f"ключи {list(st) if isinstance(st, dict) else st!r}"
        else:
            prev_end, first = -1, True
            for k in STAGE_ORDER:
                if k not in st:
                    continue
                span = st[k]
                if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)):
                    why = f"{k}={span!r}"
                    break
                s, e = span
                if first and s != 0:
                    why = f"начало {s} != 0"
                    break
                if not first and s != prev_end + 1:
                    why = f"{k} начинается с {s}, предыдущая кончилась на {prev_end}"
                    break
                if e < s:
                    why = f"{k}=[{s},{e}] пустая"
                    break
                prev_end, first = e, False
            if why is None and prev_end not in ends:
                why = f"конец {prev_end}, ожидался {sorted(ends)}"
        if why:
            bad.append((r["id"], why))
    report("stages с разрывами/не покрывают эпизод", bad, len(dataset))


def test_observation_matches_class(dataset):
    bad = []
    for r in good(dataset):
        ann = r["meta"].get("annotation") or {}
        ck, obs = r["meta"]["class_key"], ann.get("observation")
        if ck == "E":
            if obs is not None or ann.get("error_stage") is not None or ann.get("cause") != "none":
                bad.append((r["id"], f"E: obs={obs!r} stage={ann.get('error_stage')!r} cause={ann.get('cause')!r}"))
        elif obs not in OBSERVATIONS:
            bad.append((r["id"], f"{ck}: obs={obs!r}"))
    report("observation не соответствует классу", bad, len(dataset))


def test_recovery_start_after_terminal(dataset):
    bad = []
    for r in good(dataset):
        ann = r["meta"].get("annotation") or {}
        rec, rs, tf = ann.get("recovery"), ann.get("recovery_start_frame"), ann.get("terminal_frame")
        total = n_frames(r) or len(r["meta"].get("actions") or [])
        if rec and rs is None:
            bad.append((r["id"], "recovery=true, recovery_start_frame=null"))
        elif rs is not None and (not isinstance(rs, int) or not isinstance(tf, int) or not (tf < rs < total)):
            bad.append((r["id"], f"recovery_start_frame={rs!r} terminal_frame={tf!r} total={total}"))
    report("recovery_start_frame не позже terminal_frame", bad, len(dataset))


# ----------------------------------------------------------------------------
# soft: метка против episode_stats (final_info игнорируем)
# ----------------------------------------------------------------------------

def _stats(r):
    return r["meta"].get("episode_stats") or {}


@pytest.mark.soft
def test_soft_A_but_moved_correct_obj(dataset):
    bad = [(r["id"], "moved_correct_obj=true") for r in good(dataset)
           if r["meta"]["class_key"] == "A" and _stats(r).get("moved_correct_obj") is True]
    report("класс A, но объект двигался", bad, len(dataset))


@pytest.mark.soft
def test_soft_E_but_not_on_target(dataset):
    bad = [(r["id"], "src_on_target=false") for r in good(dataset)
           if r["meta"]["class_key"] == "E" and _stats(r).get("src_on_target") is False]
    report("класс E, но src_on_target=false", bad, len(dataset))


@pytest.mark.soft
def test_soft_D_but_not_grasped(dataset):
    bad = [(r["id"], "is_src_obj_grasped=false") for r in good(dataset)
           if r["meta"]["class_key"] == "D" and _stats(r).get("is_src_obj_grasped") is False]
    report("класс D, но объект не был захвачен", bad, len(dataset))


@pytest.mark.soft
def test_soft_success_vs_class(dataset):
    bad = [(r["id"], f"success={r['meta'].get('success')!r} class={r['meta']['class_key']}")
           for r in good(dataset) if bool(r["meta"].get("success")) != (r["meta"]["class_key"] == "E")]
    report("success расходится с class_key==E", bad, len(dataset))


# ----------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "soft: метка vs episode_stats, не блокирует")


if __name__ == "__main__":
    # hard-тесты определяют код возврата; soft печатаются отдельно и не блокируют
    me = sys.modules[__name__]
    extra = sys.argv[1:]
    print("=" * 30, "HARD", "=" * 30)
    rc = pytest.main([__file__, "-q", "-rA", "-p", "no:cacheprovider", "-m", "not soft"] + extra, plugins=[me])
    print("=" * 30, "SOFT", "=" * 30)
    pytest.main([__file__, "-q", "-rA", "-p", "no:cacheprovider", "-m", "soft"] + extra, plugins=[me])
    sys.exit(int(rc))
