#!/usr/bin/env python3
"""Предполётные тесты пайплайна дообучения: ловят проблемы до того, как сгорит GPU-час.

    pytest test_pipeline.py -q -m "not gpu and not slow"   # уровни 1-2: артефакты prepare_data + совместимость с тренером
    pytest test_pipeline.py -q                             # + уровень 3: смоук на GPU, один раз перед стартом обучения

Окружение: VLA_META_ROOT (корень meta/frames), VLA_DATA_OUT (папка с train.jsonl/val.jsonl/meta.json),
VLA_MODEL (путь или HF-id модели). Нет артефакта (не запускался prepare_data.py, не скачана модель,
нет CUDA/deepspeed/flash-attn) -> pytest.skip с причиной, а не падение.
Один тест = одно свойство по всем данным; в сообщении об ошибке первые 10 нарушителей с id.
Всё, что пишется (подвыборка, смоук-чекпоинт), идёт во временную папку и удаляется.
"""

import glob
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_data import CLASSES, OBSERVATIONS  # noqa: E402  словарь ответов — тот же, что у генератора

META_ROOT = os.environ.get("VLA_META_ROOT", "/data/trajectories")
DATA_OUT = os.path.abspath(os.environ.get("VLA_DATA_OUT", os.path.join(HERE, "data", "cls")))
MODEL = os.environ.get("VLA_MODEL", "OpenGVLab/InternVL3-2B")
N_FRAMES = int(os.environ.get("VLA_N_FRAMES", 16))  # контракт: столько кадров ждёт тренер и evaluate.py
TRAIN_SH = os.path.join(HERE, "train.sh")
REPO = os.path.abspath(os.environ.get("REPO", os.path.join(HERE, "InternVL", "internvl_chat")))
SEED = int(os.environ.get("VLA_SEED", 0))
SHOW = 10
SAMPLE_PATHS = 200
SAMPLE_FRAMES = 50
SEQ_MARGIN = 0.10  # запас бюджета токенов до max_seq_length
MEM_MARGIN = 0.15  # запас памяти карты после пикового батча


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def report(what, bad, total):
    """bad: [(id, деталь)]. Печатает первые SHOW и падает, если непусто."""
    if bad:
        lines = "\n".join(f"  {i}: {d}" for i, d in bad[:SHOW])
        more = f"\n  ... ещё {len(bad) - SHOW}" if len(bad) > SHOW else ""
        pytest.fail(f"{what}: {len(bad)}/{total}\n{lines}{more}", pytrace=False)


def say(request, text):
    """Отчёт, который виден и под -q (обычный print показывается только у упавших тестов)."""
    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        tr.write_line("\n" + text)
    print(text)


def load_jsonl(path):
    recs, errors = [], []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except ValueError as e:
                errors.append((f"{os.path.basename(path)}:{n}", str(e)[:80]))
    return recs, errors


def human(r):
    return r["conversations"][0]["value"]


def answer(r):
    return r["conversations"][1]["value"]


def frame_index(path):
    m = re.search(r"(\d+)\.png$", path)
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------------
# session-фикстуры: всё грузится один раз
# ----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def data():
    train_path = os.path.join(DATA_OUT, "train.jsonl")
    if not os.path.exists(train_path):
        pytest.skip(f"нет {train_path}: сначала prepare_data.py --out {DATA_OUT} (или VLA_DATA_OUT=...)")
    d = {"errors": [], "splits": {}}
    for split in ("train", "val", "heldout"):
        p = os.path.join(DATA_OUT, f"{split}.jsonl")
        if os.path.exists(p):
            recs, errors = load_jsonl(p)
            d["splits"][split] = recs
            d["errors"] += errors
    d["train"] = d["splits"]["train"]
    d["val"] = d["splits"].get("val", [])
    d["all"] = [(s, r) for s, recs in d["splits"].items() for r in recs]
    hold_txt = os.path.join(DATA_OUT, "heldout.txt")
    if os.path.exists(hold_txt):
        with open(hold_txt) as f:
            d["heldout_ids"] = {l.strip() for l in f if l.strip()}
    elif "heldout" in d["splits"]:
        d["heldout_ids"] = {r.get("id") for r in d["splits"]["heldout"]}
    else:
        d["heldout_ids"] = None
    cfg_path = os.path.join(DATA_OUT, "prepare_config.json")
    d["prepare_config"] = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    d["target"] = d["prepare_config"].get("target") or (
        "obs" if any("|" in str(answer(r)) for r in d["train"] if isinstance(r.get("conversations"), list)
                     and len(r["conversations"]) == 2) else "cls")
    return d


def ok_recs(data):
    """Только структурно валидные записи: остальные уже провалили test_jsonl_format."""
    out = []
    for split, r in data["all"]:
        c = r.get("conversations")
        if (isinstance(r, dict) and isinstance(r.get("id"), str) and isinstance(r.get("image"), list)
                and isinstance(c, list) and len(c) == 2 and isinstance(c[0].get("value"), str)
                and isinstance(c[1].get("value"), str)):
            out.append((split, r))
    return out


def parse_train_sh(path, env):
    """--ключ значение из torchrun-блока train.sh с подстановкой NAME=${NAME:-default} и переменных env."""
    with open(path) as f:
        text = f.read()
    defaults = dict(re.findall(r"^(\w+)=(\S+)$", text, re.M))  # простые присваивания вида TB_DIR=$OUTPUT_DIR/tensorboard
    defaults.update(re.findall(r"^(\w+)=\$\{\1:-([^}]*)\}", text, re.M))
    defaults.update(re.findall(r'^(\w+)=\$\(readlink -f "\$\{\1:-([^}]*)\}"\)', text, re.M))

    def resolve(s, depth=0):
        s = s.strip("\"'")
        if depth > 8 or "$" not in s:
            return s
        s = re.sub(r'\$\(readlink -f "?([^)"]*)"?\)', r"\1", s)
        s = re.sub(r'\$\(basename "?([^)"]*)"?\)', lambda m: os.path.basename(resolve(m.group(1), depth + 1)), s)
        s = re.sub(r"\$\{(\w+)(?::-[^}]*)?\}|\$(\w+)",
                   lambda m: env.get(m.group(1) or m.group(2), defaults.get(m.group(1) or m.group(2), "")), s)
        return resolve(s, depth + 1)

    block = text[text.index("torchrun"): text.index("2>&1 | tee")]
    flags = {}
    for m in re.finditer(r"--(\w+)\s+(\"[^\"]*\"|'[^']*'|\S+)", block):
        flags[m.group(1)] = resolve(m.group(2))
    return flags, {k: resolve(v) for k, v in defaults.items()}


def as_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


@pytest.fixture(scope="session")
def train_args():
    if not os.path.exists(TRAIN_SH):
        pytest.skip(f"нет {TRAIN_SH}")
    env = dict(os.environ)
    env.setdefault("HERE", HERE)
    env.setdefault("DATA", DATA_OUT)  # проверяем train.sh на тех же данных, что тестируем
    flags, vars_ = parse_train_sh(TRAIN_SH, env)
    for k in ("meta_path", "max_dynamic_patch", "max_seq_length", "per_device_train_batch_size",
              "gradient_accumulation_steps", "num_train_epochs", "freeze_backbone", "freeze_llm", "freeze_mlp",
              "use_llm_lora", "force_image_size"):
        assert k in flags, f"в train.sh не найден --{k}"
    flags["_vars"] = vars_
    flags["_gpus"] = int(vars_.get("GPUS", env.get("GPUS", 1)) or 1)
    return flags


@pytest.fixture(scope="session")
def model_config():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    except (OSError, ValueError, ImportError) as e:
        pytest.skip(f"конфиг модели {MODEL} не читается ({type(e).__name__}): скачайте модель или задайте VLA_MODEL")


@pytest.fixture(scope="session")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:  # use_fast=False — как в train.sh (--use_fast_tokenizer False)
        return transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)
    except (OSError, ValueError, ImportError) as e:
        pytest.skip(f"токенизатор {MODEL} не читается ({type(e).__name__})")


def tokens_per_tile(cfg, image_size):
    patch = cfg.vision_config.patch_size
    return int((image_size // patch) ** 2 * (cfg.downsample_ratio ** 2))


def system_message(cfg):
    """Системное сообщение шаблона из remote-кода модели; если не достать — верхняя оценка."""
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        get_conv_template = get_class_from_dynamic_module("conversation.get_conv_template", MODEL)
        return get_conv_template(cfg.template).system_message
    except Exception:
        return "x " * 64


@pytest.fixture(scope="session")
def token_budget(data, tokenizer, model_config, train_args):
    """Аналитический бюджет: самый длинный промпт + N_FRAMES * токенов_на_тайл * max_dynamic_patch."""
    image_size = int(train_args["force_image_size"])
    tpt = tokens_per_tile(model_config, image_size)
    mdp = int(train_args["max_dynamic_patch"])
    sys_msg = system_message(model_config)
    uniq = {(human(r), answer(r)) for _, r in ok_recs(data)}  # промпты различаются только инструкцией
    longest, n_uniq = 0, len(uniq)
    for h, a in uniq:
        query = (f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n"
                 f"{h.replace('<image>', '<img></img>')}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n")
        longest = max(longest, len(tokenizer(query).input_ids))
    return {"text": longest, "image": N_FRAMES * tpt * mdp, "total": longest + N_FRAMES * tpt * mdp,
            "tokens_per_tile": tpt, "max_dynamic_patch": mdp, "n_unique_prompts": n_uniq}


# ============================================================================
# уровень 1: артефакты prepare_data.py (секунды, без GPU)
# ============================================================================

def test_jsonl_format(data):
    bad = list(data["errors"])
    for split, r in data["all"]:
        rid = r.get("id", "?") if isinstance(r, dict) else "?"
        why = None
        if not isinstance(r, dict):
            why = "не dict"
        elif not all(k in r for k in ("id", "image", "conversations")):
            why = f"нет ключей {[k for k in ('id', 'image', 'conversations') if k not in r]}"
        elif not isinstance(r["image"], list) or not all(isinstance(p, str) for p in r["image"]):
            why = "image не список строк"
        else:
            c = r["conversations"]
            if not isinstance(c, list) or len(c) != 2:
                why = f"conversations: {len(c) if isinstance(c, list) else type(c).__name__} ходов, нужно 2"
            elif c[0].get("from") != "human" or c[1].get("from") != "gpt":
                why = f"from = {c[0].get('from')!r}, {c[1].get('from')!r}, нужно human, gpt"
            elif not isinstance(c[0].get("value"), str) or not isinstance(c[1].get("value"), str):
                why = "value не строка"
        if why:
            bad.append((f"{split}/{rid}", why))
    report("битые записи jsonl", bad, len(data["all"]))


def test_image_placeholder_count(data):
    bad = []
    for split, r in ok_recs(data):
        n_img, n_ph = len(r["image"]), human(r).count("<image>")
        if n_img != n_ph:
            bad.append((r["id"], f"{split}: {n_img} путей, {n_ph} x <image>"))
    report("число кадров != число <image> (тренер InternVL упадёт на этом сэмпле)", bad, len(data["all"]))


def test_fixed_n_frames(data):
    bad = [(r["id"], f"{split}: {len(r['image'])} кадров") for split, r in ok_recs(data) if len(r["image"]) != N_FRAMES]
    report(f"не {N_FRAMES} кадров", bad, len(data["all"]))


def test_paths_absolute(data):
    bad = [(r["id"], p) for split, r in ok_recs(data) for p in r["image"] if not os.path.isabs(p)][:SHOW * 5]
    report("относительные пути (root в meta.json пустой)", bad, len(data["all"]))


def _missing_paths(recs):
    return [(r["id"], p) for r in recs for p in r["image"] if not os.path.exists(p)]


def test_paths_exist_sample(data):
    recs = [r for _, r in ok_recs(data)]
    rng = random.Random(SEED)
    sample = rng.sample(recs, min(SAMPLE_PATHS, len(recs)))
    for split, s in data["splits"].items():  # первую и последнюю запись каждого сплита — всегда
        if s:
            sample += [s[0], s[-1]]
    sample = list({id(r): r for r in sample}.values())
    report(f"нет файлов кадров (выборка {len(sample)} записей; полная проверка: -m slow)", _missing_paths(sample), len(sample))


@pytest.mark.slow
def test_paths_exist_all(data):
    recs = [r for _, r in ok_recs(data)]
    report("нет файлов кадров", _missing_paths(recs), len(recs))


def test_frames_sorted_no_dups(data):
    bad = []
    for split, r in ok_recs(data):
        idx = [frame_index(p) for p in r["image"]]
        if any(i is None for i in idx):
            bad.append((r["id"], "имя кадра не NNNN.png"))
            continue
        if len({os.path.dirname(p) for p in r["image"]}) != 1:
            bad.append((r["id"], "кадры из разных папок"))
            continue
        if idx != sorted(idx):
            bad.append((r["id"], f"не по возрастанию: {idx}"))
            continue
        uniq = sorted(set(idx))
        if len(uniq) != len(idx):
            n = r.get("n_frames")
            tail_only = idx[:len(uniq)] == uniq and all(i == uniq[-1] for i in idx[len(uniq):])
            if not (tail_only and n is not None and n < N_FRAMES and uniq[-1] == n - 1):
                bad.append((r["id"], f"дубли не в хвосте короткого эпизода (n_frames={n}): {idx}"))
    report("кадры не отсортированы или дублируются", bad, len(data["all"]))


def test_answers_valid(data):
    letters, symptoms = set(CLASSES), set(OBSERVATIONS) | {"none"}
    bad = []
    for split, r in ok_recs(data):
        a = answer(r)
        if a != a.strip() or "\n" in a or not a:
            bad.append((r["id"], f"{split}: пробелы/перевод строки/пусто: {a!r}"))
        elif data["target"] == "cls":
            if a not in letters:
                bad.append((r["id"], f"{split}: {a!r} не буква из {CLASSES}"))
        else:
            parts = a.split("|")
            if len(parts) != 2 or parts[0] not in letters or parts[1] not in symptoms:
                bad.append((r["id"], f"{split}: {a!r} не <буква>|<симптом из словаря>"))
            elif (parts[0] == "E") != (parts[1] == "none"):
                bad.append((r["id"], f"{split}: {a!r}: none только у E"))
    report(f"невалидные ответы gpt (target={data['target']})", bad, len(data["all"]))


def test_prompt_matches_reference(data):
    ref_path = os.path.join(HERE, f"prompt_{data['target']}.txt")
    assert os.path.exists(ref_path), f"нет эталонного промпта {ref_path}"
    with open(ref_path) as f:
        ref = f.read()
    assert ref.count("<image>") == N_FRAMES, f"{ref_path}: {ref.count('<image>')} x <image>, ожидалось {N_FRAMES}"
    bad = []
    for split, r in ok_recs(data):
        instr = r.get("instruction")
        if instr is None:
            bad.append((r["id"], f"{split}: нет поля instruction — пересоберите jsonl текущим prepare_data.py"))
            continue
        expected = ref.replace("{instruction}", instr)
        got = human(r)
        if got != expected:
            k = next((i for i, (x, y) in enumerate(zip(got, expected)) if x != y), min(len(got), len(expected)))
            bad.append((r["id"], f"{split}: расходится с позиции {k}: jsonl={got[k:k + 40]!r} эталон={expected[k:k + 40]!r}"))
    report(f"промпт в jsonl != {os.path.basename(ref_path)} (сравнение до/после станет нечестным)", bad, len(data["all"]))


def test_trainer_meta_json(data):
    path = os.path.join(DATA_OUT, "meta.json")
    assert os.path.exists(path), f"нет {path}"
    with open(path) as f:
        meta = json.load(f)
    assert isinstance(meta, dict) and meta, "meta.json пустой или не dict"
    train_path = os.path.abspath(os.path.join(DATA_OUT, "train.jsonl"))
    bad, points_to_train = [], False
    for name, ds in meta.items():
        for k in ("root", "annotation", "length", "data_augment", "repeat_time"):  # тренер читает их без .get
            if k not in ds:
                bad.append((name, f"нет ключа {k}"))
        if ds.get("root", None) != "":
            bad.append((name, f"root={ds.get('root')!r}, должен быть пустым"))
        ann = ds.get("annotation", "")
        if not (isinstance(ann, str) and ann.endswith(".jsonl") and os.path.exists(ann)):
            bad.append((name, f"annotation не существует: {ann!r}"))
            continue
        points_to_train |= os.path.abspath(ann) == train_path
        with open(ann) as f:
            n = sum(1 for l in f if l.strip())
        if ds.get("length") != n:
            bad.append((name, f"length={ds.get('length')} а строк {n}"))
    if not points_to_train:
        bad.append(("meta.json", f"ни один annotation не указывает на {train_path}"))
    if bad:
        pytest.fail(f"meta.json тренера ({path}), проблем: {len(bad)}\n" + "\n".join(f"  {n}: {d}" for n, d in bad[:SHOW]), pytrace=False)


def test_no_leakage_train_val(data):
    inter = {r["id"] for _, r in ok_recs(data) if _ == "train"} & {r["id"] for s, r in ok_recs(data) if s == "val"}
    report("id есть и в train, и в val", [(i, "train ∩ val") for i in sorted(inter)], len(data["train"]))


def test_no_leakage_heldout(data):
    if data["heldout_ids"] is None:
        pytest.skip(f"нет {DATA_OUT}/heldout.txt и heldout.jsonl — held-out набор не задан")
    train_ids = {r["id"] for s, r in ok_recs(data) if s == "train"}
    val_ids = {r["id"] for s, r in ok_recs(data) if s == "val"}
    bad = [(i, "held-out ∩ train") for i in sorted(data["heldout_ids"] & train_ids)]
    bad += [(i, "held-out ∩ val") for i in sorted(data["heldout_ids"] & val_ids)]
    report("утечка held-out в train/val (обесценивает эксперимент)", bad, len(data["heldout_ids"]))


def test_class_distribution(data, request):
    dist = {s: Counter(answer(r)[0] for _, r in ok_recs(data) if _ == s) for s in data["splits"]}
    lines = [f"class distribution ({DATA_OUT}):", f"{'':8s}" + "".join(f"{c:>6s}" for c in CLASSES) + f"{'total':>7s}"]
    for s, c in dist.items():
        lines.append(f"{s:8s}" + "".join(f"{c.get(k, 0):6d}" for k in CLASSES) + f"{sum(c.values()):7d}")
    say(request, "\n".join(lines))
    bad = [(s, f"нет классов {[c for c in CLASSES if not dist[s].get(c)]}") for s in ("train", "val")
           if s in dist and any(not dist[s].get(c) for c in CLASSES)]
    report("не все 5 классов представлены", bad, 2)


# ============================================================================
# уровень 2: совместимость с тренером (минуты, без GPU)
# ============================================================================

def test_model_config_reads(model_config, train_args, request):
    image_size = int(train_args["force_image_size"])
    tpt = tokens_per_tile(model_config, image_size)
    say(request, f"model {MODEL}: image_size={image_size} patch={model_config.vision_config.patch_size} "
                 f"downsample={model_config.downsample_ratio} -> {tpt} tokens/tile; llm={model_config.llm_config.architectures}")
    assert getattr(model_config, "force_image_size", image_size) == image_size, "force_image_size в train.sh != конфиг модели"
    assert tpt == 256, f"{tpt} визуальных токенов на тайл, ожидалось 256 (448px, patch 14, downsample 0.5)"


def test_token_budget(token_budget, train_args, request):
    max_len = int(train_args["max_seq_length"])
    b = token_budget
    say(request, f"token budget: text<={b['text']} + images {N_FRAMES}x{b['tokens_per_tile']}x{b['max_dynamic_patch']}"
                 f"={b['image']} -> {b['total']} of max_seq_length={max_len} ({100 * b['total'] / max_len:.0f}%)")
    assert b["total"] <= max_len * (1 - SEQ_MARGIN), (
        f"бюджет {b['total']} токенов > {1 - SEQ_MARGIN:.0%} от max_seq_length={max_len}: "
        f"проверьте --max_dynamic_patch (сейчас {b['max_dynamic_patch']}) — на GPU это был бы OOM или обрезанный ответ")


def test_answer_letters_tokenize_distinct(tokenizer):
    ids = {c: tokenizer.encode(c, add_special_tokens=False) for c in CLASSES}
    bad = [(c, f"{len(t)} токенов: {t}") for c, t in ids.items() if len(t) != 1]
    report("буква класса не один токен", bad, len(CLASSES))
    firsts = [t[0] for t in ids.values()]
    assert len(set(firsts)) == len(CLASSES), f"буквы делят токены: {ids} (argmax по logits сломается молча)"
    unk = tokenizer.unk_token_id
    assert all(t[0] != unk for t in ids.values()), f"буква -> unk: {ids}"
    # формат <буква>|<симптом> должен начинаться с того же токена буквы
    for c in CLASSES:
        first = tokenizer.encode(f"{c}|none", add_special_tokens=False)[0]
        assert first == ids[c][0], f"{c}|none начинается с токена {first}, а буква {c} = {ids[c][0]}"


def test_train_sh_consistent(train_args, token_budget):
    meta = train_args["meta_path"]
    meta = meta if os.path.isabs(meta) else os.path.join(HERE, meta)
    problems = []
    if not os.path.exists(meta):
        problems.append(f"--meta_path {meta} не существует")
    if int(train_args["max_dynamic_patch"]) != 1:
        problems.append(f"--max_dynamic_patch {train_args['max_dynamic_patch']} (нужно 1: 16 кадров x 12 тайлов = 49k токенов)")
    if int(train_args["max_seq_length"]) < token_budget["total"]:
        problems.append(f"--max_seq_length {train_args['max_seq_length']} < бюджет {token_budget['total']}")
    if not as_bool(train_args["freeze_backbone"]):
        problems.append("--freeze_backbone не True")
    freeze_llm, lora = as_bool(train_args["freeze_llm"]), int(train_args["use_llm_lora"])
    if freeze_llm and lora <= 0:
        problems.append("--freeze_llm True без --use_llm_lora: учиться будет нечему кроме projector")
    if not freeze_llm and lora > 0:
        problems.append("--freeze_llm False вместе с --use_llm_lora: полный FT и LoRA одновременно")
    if as_bool(train_args.get("freeze_mlp", "False")) and freeze_llm and lora <= 0:
        problems.append("всё заморожено: ни одного обучаемого параметра")
    assert not problems, "train.sh:\n  " + "\n  ".join(problems)


def test_tensorboard_available(train_args):
    """--report_to tensorboard без пакета tensorboard роняет тренер на старте (TensorBoardCallback)."""
    report_to = train_args.get("report_to", "none").strip("\"'")
    if "tensorboard" not in report_to:
        pytest.skip(f"--report_to {report_to}: tensorboard не используется")
    assert importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"), \
        "train.sh пишет в tensorboard, а пакета нет: pip install tensorboard"
    assert train_args.get("logging_dir"), "нет --logging_dir: watch_val.sh и тренер должны писать в одну папку"
    assert os.path.exists(os.path.join(HERE, "watch_val.sh")), "нет watch_val.sh, который train.sh запускает в фоне"


def test_frames_readable_same_size(data):
    Image = pytest.importorskip("PIL.Image")
    paths = sorted({p for _, r in ok_recs(data) for p in r["image"]})
    sample = random.Random(SEED).sample(paths, min(SAMPLE_FRAMES, len(paths)))
    bad, sizes = [], Counter()
    for p in sample:
        try:
            with Image.open(p) as im:
                im.convert("RGB").load()
                sizes[im.size] += 1
        except Exception as e:
            bad.append((p, f"{type(e).__name__}: {str(e)[:60]}"))
    report("кадры не читаются PIL", bad, len(sample))
    assert len(sizes) == 1, f"разные разрешения кадров: {dict(sizes)}"


def test_report_epoch_estimate(data, train_args, request):
    n = len(data["train"])
    bs, acc, gpus = int(train_args["per_device_train_batch_size"]), int(train_args["gradient_accumulation_steps"]), train_args["_gpus"]
    eff = bs * acc * gpus
    epochs = float(train_args["num_train_epochs"])
    steps = math.ceil(n / eff)
    say(request, f"epoch estimate: {n} samples, effective batch {bs}x{acc}x{gpus}={eff}, "
                 f"{steps} steps/epoch, {epochs:g} epochs -> {math.ceil(steps * epochs)} optimizer steps "
                 f"({n * epochs:.0f} forward/backward of {N_FRAMES} frames each)")
    assert n >= eff, f"train ({n}) меньше effective batch ({eff}): ни одного полного шага"


# ============================================================================
# уровень 3: смоук на GPU (минуты, один раз перед стартом)
# ============================================================================

def cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("нет CUDA")
    return torch


def trainer_modules():
    """Штатные препроцессинг и коллатор тренера InternVL (коллатор — напрямую из файла: internvl.patch тянет flash_attn)."""
    if not os.path.isdir(os.path.join(REPO, "internvl")):
        pytest.skip(f"нет репозитория InternVL в {REPO} (см. README: git clone + checkout 2410d1d)")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        from internvl.train import dataset as ds
    except ImportError as e:
        pytest.skip(f"internvl.train.dataset не импортируется ({e}): uv sync (decord, opencv-python, imageio)")
    spec = importlib.util.spec_from_file_location("pad_data_collator", os.path.join(REPO, "internvl", "patch", "pad_data_collator.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return ds, mod.concat_pad_data_collator


def trainer_sample(rec, tok, ds, num_image_token, image_size, max_dynamic_patch, max_seq_length):
    """Ровно то, что делает LazySupervisedDataset.multi_modal_multi_image_get_item."""
    import torch
    from PIL import Image
    from copy import deepcopy
    tok.model_max_length = max_seq_length
    transform = ds.build_transform(is_train=False, input_size=image_size)
    images, tiles = [], []
    for p in rec["image"]:
        im = ds.dynamic_preprocess(Image.open(p).convert("RGB"), min_num=1, max_num=max(1, max_dynamic_patch // len(rec["image"])),
                                   image_size=image_size, use_thumbnail=True)
        images += im
        tiles.append(len(im))
    pixel_values = torch.stack([transform(i) for i in images])
    ret = ds.preprocess_internvl2_5("internvl2_5", [deepcopy(rec["conversations"])], tok, [num_image_token * t for t in tiles],
                                    group_by_length=True, ds_name="smoke", num_image=len(rec["image"]))
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
    return dict(input_ids=ret["input_ids"][0], labels=ret["labels"][0], attention_mask=ret["attention_mask"][0],
                position_ids=position_ids[0], pixel_values=pixel_values,
                image_flags=torch.tensor([1] * pixel_values.shape[0], dtype=torch.long))


def to_cuda(batch, torch):
    return {k: (v.to("cuda", torch.bfloat16) if k == "pixel_values" else v.to("cuda")) for k, v in batch.items()}


@pytest.mark.gpu
class TestGpuModel:
    """Модель в памяти один раз на класс; после класса освобождается для subprocess-тестов ниже."""

    @pytest.fixture(scope="class")
    def gpu_model(self):
        torch = cuda_or_skip()
        import gc
        from evaluate import load_model
        try:
            tok, model = load_model(MODEL, None, "cuda")
        except (OSError, ValueError) as e:
            pytest.skip(f"модель {MODEL} не грузится: {e}")
        yield tok, model
        # упавшие тесты держат ссылки на модель через traceback, поэтому del не освобождает карту;
        # переезд на CPU освобождает её независимо от ссылок — subprocess-тесты ниже получают всю память
        model.cpu()
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    def test_model_loads_bf16(self, gpu_model, request):
        torch = cuda_or_skip()
        tok, model = gpu_model
        n_params = sum(p.numel() for p in model.parameters())
        expected = n_params * 2  # bf16
        allocated = torch.cuda.memory_allocated()
        say(request, f"model loaded: {n_params / 1e9:.2f}B params, cuda memory_allocated={allocated / 2 ** 30:.2f} GiB "
                     f"(bf16 expected {expected / 2 ** 30:.2f} GiB), total {torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GiB")
        dtypes = {p.dtype for p in model.parameters()}
        assert dtypes == {torch.bfloat16}, f"параметры не bf16: {dtypes}"
        assert abs(allocated - expected) / expected < 0.15, f"память {allocated} вместо ~{expected} (±15%)"

    def test_forward_one_sample_loss_finite(self, gpu_model, data, train_args, request):
        torch = cuda_or_skip()
        tok, model = gpu_model
        ds, collate = trainer_modules()
        rec = data["train"][0]
        sample = trainer_sample(rec, tok, ds, model.num_image_token, int(train_args["force_image_size"]),
                                int(train_args["max_dynamic_patch"]), int(train_args["max_seq_length"]))
        batch = to_cuda(collate([sample]), torch)
        seq, tiles = batch["input_ids"].shape[1], batch["pixel_values"].shape[0]
        total = torch.cuda.get_device_properties(0).total_memory
        try:
            with torch.no_grad():
                loss = float(model(**batch).loss)
        except torch.cuda.OutOfMemoryError as e:
            pytest.fail(f"OOM на одном сэмпле {seq} токенов (карта {total / 2 ** 30:.1f} GiB): на этой карте обучение "
                        f"не пойдёт даже с batch 1. {str(e)[:120]}", pytrace=False)
        finally:
            del batch
            torch.cuda.empty_cache()
        say(request, f"forward {rec['id']}: seq {seq} tokens, {tiles} tiles, loss={loss:.4f}")
        assert math.isfinite(loss), f"loss={loss}"
        assert 0.0 < loss < 20.0, f"loss={loss} вне разумного диапазона"

    def test_peak_memory_full_batch(self, gpu_model, data, train_args, request):
        torch = cuda_or_skip()
        peft = pytest.importorskip("peft")
        tok, model = gpu_model
        ds, collate = trainer_modules()
        bs = int(train_args["per_device_train_batch_size"])
        # самые тяжёлые записи: больше всего кадров, самый длинный текст
        recs = sorted(data["train"], key=lambda r: (len(r["image"]), len(human(r))), reverse=True)[:bs]
        samples = [trainer_sample(r, tok, ds, model.num_image_token, int(train_args["force_image_size"]),
                                  int(train_args["max_dynamic_patch"]), int(train_args["max_seq_length"])) for r in recs]
        batch = to_cuda(collate(samples), torch)
        # те же обучаемые параметры, что в train.sh: ViT и LLM заморожены, LoRA на LLM, projector учится
        r = int(train_args["use_llm_lora"])
        for p in model.vision_model.parameters():
            p.requires_grad_(not as_bool(train_args["freeze_backbone"]))
        for p in model.mlp1.parameters():
            p.requires_grad_(not as_bool(train_args.get("freeze_mlp", "False")))
        for p in model.language_model.parameters():
            p.requires_grad_(not as_bool(train_args["freeze_llm"]))
        if r > 0:
            cfg = peft.LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                  target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                                  "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"])
            model.language_model = peft.get_peft_model(model.language_model, cfg)
        if as_bool(train_args.get("grad_checkpoint", "True")):
            model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        seq = batch["input_ids"].shape[1]
        try:
            out = model(**batch)
            out.loss.backward()
            torch.cuda.synchronize()
            loss = float(out.loss)
        except torch.cuda.OutOfMemoryError as e:
            pytest.fail(f"OOM на батче {bs} x {seq} токенов (карта {total / 2 ** 30:.1f} GiB): "
                        f"снижайте per_device_train_batch_size и поднимайте gradient_accumulation_steps. {str(e)[:120]}", pytrace=False)
        finally:
            model.zero_grad(set_to_none=True)
            model.eval()
            del batch  # логиты и граф не должны пережить тест: их держал бы traceback при падении
            out = None
            torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated()
        say(request, f"peak memory: batch {bs} x {seq} tokens, forward+backward "
                     f"(LoRA r={r}, grad ckpt) -> {peak / 2 ** 30:.2f} GiB of {total / 2 ** 30:.1f} GiB "
                     f"({100 * peak / total:.0f}%), loss={loss:.4f}")
        assert peak <= total * (1 - MEM_MARGIN), (
            f"пик {peak / 2 ** 30:.2f} GiB > {1 - MEM_MARGIN:.0%} карты ({total / 2 ** 30:.1f} GiB): "
            f"снижайте per_device_train_batch_size (сейчас {bs}) и поднимайте gradient_accumulation_steps")


@pytest.mark.gpu
class TestGpuTrainSmoke:
    """Два шага настоящего тренера во временную папку, затем загрузка сохранённого чекпоинта."""

    @pytest.fixture(scope="class")
    def smoke_train(self, tmp_path_factory, data, train_args):
        cuda_or_skip()
        for mod in ("deepspeed", "flash_attn"):
            if importlib.util.find_spec(mod) is None:
                pytest.skip(f"нет {mod}: тренер InternVL без него не стартует (uv sync --extra train --extra flash)")
        if not os.path.isdir(os.path.join(REPO, "internvl")):
            pytest.skip(f"нет репозитория InternVL в {REPO}")
        tmp = tmp_path_factory.mktemp("smoke_train")
        subset = data["train"][:8]
        sub_path = tmp / "train.jsonl"
        with open(sub_path, "w") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta_path = tmp / "meta.json"
        with open(meta_path, "w") as f:
            json.dump({"smoke": {"root": "", "annotation": str(sub_path), "data_augment": False,
                                 "repeat_time": 1, "length": len(subset)}}, f)
        out_dir = tmp / "out"
        # TB_PORT=0 / WATCH_VAL=0: без сервера и фонового наблюдателя; сам --report_to tensorboard остаётся,
        # чтобы смоук проверил и запись событий тренера
        env = dict(os.environ, META_PATH=str(meta_path), VAL_JSONL=str(sub_path), OUTPUT_DIR=str(out_dir),
                   SELECT="0", WATCH_VAL="0", TB_PORT="0", GRADIENT_ACC="1", MODEL=MODEL,
                   EXTRA_ARGS="--max_steps 2 --warmup_ratio 0 --logging_steps 1 --save_strategy no")
        env.setdefault("DATA", DATA_OUT)
        proc = subprocess.run(["bash", TRAIN_SH], env=env, cwd=HERE, capture_output=True, text=True, timeout=1800)
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        assert proc.returncode == 0, f"train.sh --max_steps 2 завершился с кодом {proc.returncode}:\n{tail}"
        weights = sorted(glob.glob(str(out_dir / "*.safetensors")) + glob.glob(str(out_dir / "checkpoint-*" / "*.safetensors")))
        assert weights, f"в {out_dir} не появилось *.safetensors:\n{tail}"
        yield {"out_dir": os.path.dirname(weights[0]), "weights": weights, "subset": subset, "log_tail": tail,
               "tb_dir": str(out_dir / "tensorboard")}
        shutil.rmtree(tmp, ignore_errors=True)  # чекпоинт весит как модель

    def test_tensorboard_events_written(self, smoke_train, request):
        events = glob.glob(os.path.join(smoke_train["tb_dir"], "**", "events.out.tfevents.*"), recursive=True)
        assert events, f"тренер не записал events.out.tfevents.* в {smoke_train['tb_dir']}:\n{smoke_train['log_tail']}"
        ea = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")
        tags = set()
        loss_points = []
        for ev in events:
            acc = ea.EventAccumulator(ev)
            acc.Reload()
            tags |= set(acc.Tags().get("scalars", []))
            for t in acc.Tags().get("scalars", []):
                if t.endswith("loss"):
                    loss_points += [(s.step, s.value) for s in acc.Scalars(t)]
        say(request, f"tensorboard: {len(events)} event files, scalar tags {sorted(tags)}, loss points {sorted(loss_points)}")
        assert any(t.endswith("loss") for t in tags), f"в событиях нет скаляра loss: {sorted(tags)}"
        assert len(loss_points) >= 2, f"ожидалось >=2 точек loss (--max_steps 2, --logging_steps 1), есть {loss_points}"
        assert all(math.isfinite(v) for _, v in loss_points), f"loss не конечен: {loss_points}"

    def test_two_steps_change_lora_weights(self, smoke_train, train_args, request):
        from safetensors.torch import load_file
        n_lora, n_nonzero = 0, 0
        for w in smoke_train["weights"]:
            for k, v in load_file(w).items():
                if "lora_B" in k:
                    n_lora += 1
                    n_nonzero += bool(v.float().abs().sum() > 0)
        say(request, f"smoke train: {smoke_train['out_dir']}: lora_B tensors {n_lora}, non-zero {n_nonzero}")
        assert n_lora > 0, "в чекпоинте нет lora_B (use_llm_lora не применился)"
        assert n_nonzero == n_lora, f"{n_lora - n_nonzero} из {n_lora} lora_B нулевые: шаги оптимизатора не изменили веса"

    def test_checkpoint_loads_and_forward(self, smoke_train, train_args, request):
        torch = cuda_or_skip()
        import gc
        from evaluate import load_model
        ds, collate = trainer_modules()
        tok, model = load_model(MODEL, smoke_train["out_dir"], "cuda")  # тот же путь, что evaluate.py --lora
        try:
            sample = trainer_sample(smoke_train["subset"][0], tok, ds, model.num_image_token, int(train_args["force_image_size"]),
                                    int(train_args["max_dynamic_patch"]), int(train_args["max_seq_length"]))
            with torch.no_grad():
                loss = float(model(**to_cuda(collate([sample]), torch)).loss)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        say(request, f"checkpoint reload: loss={loss:.4f}")
        assert math.isfinite(loss), f"loss={loss}"


@pytest.mark.gpu
def test_inference_path_evaluate(data, tmp_path, request):
    cuda_or_skip()
    src = next((os.path.join(DATA_OUT, f"{s}.jsonl") for s in ("heldout", "val") if os.path.exists(os.path.join(DATA_OUT, f"{s}.jsonl"))), None)
    if src is None:
        pytest.skip("нет heldout.jsonl/val.jsonl")
    out = tmp_path / "eval"
    cmd = [sys.executable, os.path.join(HERE, "evaluate.py"), "--data", src, "--model", MODEL, "--limit", "4",
           "--out-dir", str(out), "--group-by", "agent", "--seed", str(SEED)]
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=1200)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    assert proc.returncode == 0, f"evaluate.py --limit 4 упал с кодом {proc.returncode}:\n{tail}"
    preds, _ = load_jsonl(str(out / "predictions.jsonl"))
    assert len(preds) == 4, f"predictions.jsonl: {len(preds)} строк вместо 4\n{tail}"
    assert all(p["pred"] in CLASSES and abs(sum(p["probs"].values()) - 1) < 1e-3 for p in preds), preds[:2]
    with open(out / "metrics.json") as f:
        m = json.load(f)
    say(request, f"evaluate.py --limit 4 on {os.path.basename(src)}: macro-F1 {m['overall']['metrics']['macro_f1']:.3f}, "
                 f"majority {m['overall']['majority']['metrics']['macro_f1']:.3f}")
    assert "macro_f1" in m["overall"]["metrics"]
    shutil.rmtree(tmp_path, ignore_errors=True)
