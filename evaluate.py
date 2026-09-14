#!/usr/bin/env python3
"""Оценка InternVL3 на jsonl из prepare_data.py: argmax по logits букв A..E.

    python3 evaluate.py --data data/cls/heldout.jsonl                        # zero-shot база
    python3 evaluate.py --data data/cls/heldout.jsonl --lora work_dirs/cls/checkpoint-12
    python3 evaluate.py --data ... --frame-selection uniform --group-by agent  # ablation по кадрам
    python3 evaluate.py --from-predictions data/cls/eval/heldout_base/predictions.jsonl --group-by agent

Промпт берётся из jsonl как есть и оборачивается в шаблон internvl2_5 ровно так же,
как это делает preprocess_internvl2_5 при обучении. Класс = буква с максимальным logit
на позиции после '<|im_start|>assistant\\n'. Ground truth = первая буква ответа gpt.
--lora принимает папку чекпоинта HF Trainer из train.sh (полный state_dict + LoRA-модули).
В --out-dir пишутся predictions.jsonl (id, agent, gt, pred, probs), metrics.json, run_config.json;
--from-predictions пересчитывает метрики из predictions.jsonl без модели.
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import FRAME_SELECTIONS, MAX_FRAMES, N_FRAMES, PAPER_IMAGE, pick_frames  # noqa: E402
import fsr  # noqa: E402

CLASSES = "ABCDE"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIT_CHUNK = 4  # кадров за один проход ViT (см. predict)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # меньше фрагментации на малых картах


# ----------------------------------------------------------------------------
# модель
# ----------------------------------------------------------------------------

def build_transform(size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(name, lora, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
    # torch_dtype, а не dtype: окружение тренера — transformers 4.37.2, там ещё нет параметра dtype
    model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    if lora:
        # HF-код модели не умеет LoRA, поэтому повторяем wrap_llm_lora из репозитория InternVL
        # и грузим state_dict чекпоинта поверх (там полные веса: vit + mlp1 + llm с lora-модулями).
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        with open(os.path.join(lora, "config.json")) as f:
            r = json.load(f).get("use_llm_lora", 0)
        if r:
            cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                             "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"])
            model.language_model = get_peft_model(model.language_model, cfg)
        sd = {}
        for shard in sorted(glob.glob(os.path.join(lora, "*.safetensors"))):
            sd.update(load_file(shard))
        if not sd:
            sys.exit(f"в {lora} нет *.safetensors")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_lora = sum("lora_" in k for k in sd)
        print(f"[lora] r={r} tensors={len(sd)} lora_tensors={n_lora} missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected or (r and n_lora == 0):
            sys.exit(f"ключи чекпоинта не совпали с моделью: unexpected={unexpected[:5]}")
        if missing:
            print(f"[lora] warning: missing keys, first: {missing[:5]}", file=sys.stderr)
        if r:
            model.language_model = model.language_model.merge_and_unload()
    model = model.to(device).eval()
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    use_sdpa(model)
    return tok, model


def use_sdpa(model):
    """eager-внимание на 4.5k токенов ест ~2GB на слой; sdpa даёт те же logits дешевле.

    transformers >= 4.48 выбирают реализацию по config._attn_implementation при каждом forward;
    в 4.37 (окружение тренера) класс внимания фиксируется при создании модели — тогда подменяем класс
    у уже созданных слоёв: Qwen2SdpaAttention отличается от Qwen2Attention только forward, веса те же.
    flash_attention_2 (если стоит flash-attn) не трогаем."""
    try:
        llm = model.language_model
        if llm.config._attn_implementation == "flash_attention_2":
            return
        llm.config._attn_implementation = "sdpa"
        import importlib
        mod = importlib.import_module(type(llm).__module__)
        classes = getattr(mod, "QWEN2_ATTENTION_CLASSES", None) or getattr(mod, "LLAMA_ATTENTION_CLASSES", None)
        if classes and "sdpa" in classes and "eager" in classes:
            for m in llm.modules():
                if type(m) is classes["eager"]:
                    m.__class__ = classes["sdpa"]
    except Exception as e:
        print(f"[sdpa] не включилось, остаётся eager: {type(e).__name__}: {e}", file=sys.stderr)


def build_query(model, human_value):
    img = "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>"
    value = human_value.replace("<image>", img)
    return (f"<|im_start|>system\n{model.system_message}<|im_end|>\n"
            f"<|im_start|>user\n{value}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def predict(model, tok, transform, rec, letter_ids, device):
    import torch
    from PIL import Image
    with torch.inference_mode():
        pix = torch.stack([transform(Image.open(p)) for p in rec["image"]]).to(device, torch.bfloat16)
        ids = tok(build_query(model, rec["conversations"][0]["value"]), return_tensors="pt").input_ids.to(device)
        # Нужен один вектор logits последней позиции, а model.forward считает их для всех 4.5k позиций
        # (в transformers 4.37 ещё и в fp32: 2.7 GB на сэмпл). Поэтому склеиваем эмбеддинги так же, как
        # InternVLChatModel.forward, и берём lm_head только от последнего скрытого состояния.
        emb = model.language_model.get_input_embeddings()(ids).clone()
        # ViT без flash-attn считает наивное внимание сразу на все кадры (16 x 16 голов x 1025^2);
        # по чанкам результат тот же, а пик памяти в 4 раза меньше
        vit = torch.cat([model.extract_feature(pix[i:i + VIT_CHUNK]) for i in range(0, pix.shape[0], VIT_CHUNK)])
        vit = vit.reshape(-1, emb.shape[-1])
        sel = ids[0] == model.img_context_token_id
        assert int(sel.sum()) == vit.shape[0], f"IMG_CONTEXT {int(sel.sum())} != vit tokens {vit.shape[0]}"
        emb[0, sel] = vit.to(emb.dtype)
        hidden = model.language_model.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids)).last_hidden_state
        logits = model.language_model.lm_head(hidden[:, -1])[0, letter_ids].float()
        probs = torch.softmax(logits, dim=0).tolist()
    return CLASSES[int(logits.argmax())], probs


def reselect_frames(rec, strategy, fsr_args):
    """Та же запись, но кадры отобраны другой стратегией. Текст промпта не меняется; в формате статьи
    перед ним пересобирается префикс "<image>\\n" под новое число кадров."""
    for k in ("frames_dir", "n_frames", "flips"):
        if k not in rec:
            sys.exit(f"{rec['id']}: в jsonl нет поля {k} — пересоберите его текущим prepare_data.py")
    idx = pick_frames(rec, strategy, **(fsr_args if strategy in fsr.AUTHOR_STRATEGIES else {}))
    human = rec["conversations"][0]["value"]
    if rec.get("prompt_format") == "paper":
        prefix = PAPER_IMAGE * len(rec["image"])
        assert human.startswith(prefix) and human.count("<image>") == len(rec["image"]), rec["id"]
        human = PAPER_IMAGE * len(idx) + human[len(prefix):]
    elif not (len(idx) == N_FRAMES == human.count("<image>")):
        sys.exit(f"{rec['id']}: промпт записи рассчитан на {N_FRAMES} кадров, а {strategy} дал {len(idx)}; "
                 f"для стратегий авторов нужны данные prepare_data.py --target paper")
    conv = [dict(rec["conversations"][0], value=human)] + rec["conversations"][1:]
    return dict(rec, image=[os.path.join(rec["frames_dir"], f"{i:04d}.png") for i in idx], conversations=conv)


# ----------------------------------------------------------------------------
# метрики
# ----------------------------------------------------------------------------

def metrics(y_true, y_pred):
    labels = [c for c in CLASSES if c in set(y_true)]  # macro по классам, которые есть в GT
    cm = [[0] * len(CLASSES) for _ in CLASSES]
    for t, p in zip(y_true, y_pred):
        cm[CLASSES.index(t)][CLASSES.index(p)] += 1
    rec, f1 = {}, {}
    for c in labels:
        i = CLASSES.index(c)
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(row[i] for row in cm) - tp
        rec[c] = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1[c] = 2 * prec * rec[c] / (prec + rec[c]) if prec + rec[c] else 0.0
    return {
        "n": len(y_true),
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true),
        "balanced_accuracy": sum(rec.values()) / len(labels),
        "macro_f1": sum(f1.values()) / len(labels),
        "recall": rec,
        "f1": f1,
        "labels": labels,
        "confusion": cm,
    }


def with_majority(preds):
    y_true = [p["gt"] for p in preds]
    majority = max(CLASSES, key=y_true.count)
    return {"metrics": metrics(y_true, [p["pred"] for p in preds]),
            "majority": {"class": majority, "metrics": metrics(y_true, [majority] * len(y_true))}}


def summarize(preds, group_by):
    out = {"overall": with_majority(preds), "groups": {}}
    if group_by:
        keys = sorted({p.get(group_by) or "?" for p in preds})
        out["groups"][group_by] = {k: with_majority([p for p in preds if (p.get(group_by) or "?") == k]) for k in keys}
    return out


def print_summary(s, tag):
    m, mj = s["overall"]["metrics"], s["overall"]["majority"]
    rows = [(tag, s["overall"])]
    for key, groups in s["groups"].items():
        rows = [(f"{key}={k}", v) for k, v in groups.items()] + [("all", s["overall"])]
    w = max(12, min(40, max(len(n) for n, _ in rows)))
    print(f"\n{'':<{w}s}{'n':>5s}{'macroF1':>9s}{'bal-acc':>9s}{'acc':>8s} | {'majority':<9s}{'macroF1':>9s}{'bal-acc':>9s}{'acc':>8s}")
    for name, v in rows:
        a, b = v["metrics"], v["majority"]["metrics"]
        print(f"{name[:w]:<{w}s}{a['n']:5d}{a['macro_f1']:9.3f}{a['balanced_accuracy']:9.3f}{a['accuracy']:8.3f}"
              f" | always {v['majority']['class']:<2s}{b['macro_f1']:9.3f}{b['balanced_accuracy']:9.3f}{b['accuracy']:8.3f}")
    print(f"\n{'recall':<{w}s}" + "".join(f"{c:>7s}" for c in CLASSES))
    for name, v in rows:
        r = v["metrics"]["recall"]
        print(f"{name[:w]:<{w}s}" + "".join(f"{r[c]:7.2f}" if c in r else f"{'-':>7s}" for c in CLASSES))
    print(f"\nconfusion, all (rows=gt, cols=pred)      majority = always {mj['class']}")
    print("     " + "".join(f"{c:>5s}" for c in CLASSES))
    for c, row in zip(CLASSES, m["confusion"]):
        print(f"{c:>5s}" + "".join(f"{v:5d}" for v in row))


def log_tensorboard(s, logdir, step, tag):
    """Метрики как скаляры в TensorBoard (тот же logdir, что у тренера -> кривые val рядом с loss)."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        sys.exit("--tensorboard требует пакет tensorboard: pip install tensorboard")
    w = SummaryWriter(logdir)
    m, mj = s["overall"]["metrics"], s["overall"]["majority"]["metrics"]
    for k in ("macro_f1", "balanced_accuracy", "accuracy"):
        w.add_scalar(f"{tag}/{k}", m[k], step)
        w.add_scalar(f"{tag}/majority_{k}", mj[k], step)
    for c, v in m["recall"].items():
        w.add_scalar(f"{tag}/recall_{c}", v, step)
    for key, groups in s["groups"].items():
        for g, v in groups.items():
            w.add_scalar(f"{tag}/{key}={g}/macro_f1", v["metrics"]["macro_f1"], step)
    w.close()
    print(f"tensorboard: {tag}/* at step {step} -> {logdir}")


def default_step(lora):
    m = re.search(r"checkpoint-(\d+)", lora or "")
    return int(m.group(1)) if m else 0


def git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cls/heldout.jsonl")
    ap.add_argument("--model", default="OpenGVLab/InternVL3-2B")
    ap.add_argument("--lora", default=None, help="папка чекпоинта из train.sh; без флага — базовая модель")
    ap.add_argument("--frame-selection", choices=FRAME_SELECTIONS, default=None,
                    help="пересобрать кадры этой стратегией вместо тех, что в jsonl (ablation)")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="бюджет кадров для стратегий авторов")
    ap.add_argument("--num-surr", type=int, default=2, help="n в surr(n) и dense_sparse")
    ap.add_argument("--tail", type=int, default=0, help="n в tail(n)")
    ap.add_argument("--group-by", choices=("agent", "task"), default=None, help="метрики отдельно по группам + общая строка")
    ap.add_argument("--from-predictions", default=None, help="predictions.jsonl: пересчитать метрики без модели")
    ap.add_argument("--out-dir", default=None, help="по умолчанию <dir(data)>/eval/<split>_<base|ckpt>[_<frames>]")
    ap.add_argument("--limit", type=int, default=None, help="первые N записей (смоук)")
    ap.add_argument("--tensorboard", default=None, metavar="DIR",
                    help="писать метрики скалярами в этот logdir (тот же, что у тренера: <out>/tensorboard)")
    ap.add_argument("--step", type=int, default=None, help="global_step для tensorboard (по умолчанию из checkpoint-N)")
    ap.add_argument("--tb-tag", default=None, help="префикс скаляров (по умолчанию имя сплита: val, heldout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    for k in ("data", "lora", "out_dir", "from_predictions", "tensorboard"):  # "~" в argparse не раскрывается сам
        if getattr(args, k):
            setattr(args, k, os.path.expanduser(getattr(args, k)))
    if args.tb_tag:
        tb_tag = args.tb_tag
    elif args.from_predictions:  # имя папки оценки, например heldout_base
        tb_tag = os.path.basename(os.path.dirname(os.path.abspath(args.from_predictions)))
    else:
        tb_tag = os.path.splitext(os.path.basename(args.data))[0]
    tb_step = args.step if args.step is not None else default_step(args.lora)

    if args.from_predictions:
        preds = load_jsonl(args.from_predictions)
        if not preds:
            sys.exit(f"пусто: {args.from_predictions}")
        s = summarize(preds, args.group_by)
        print_summary(s, os.path.basename(os.path.dirname(os.path.abspath(args.from_predictions))))
        out = os.path.join(os.path.dirname(os.path.abspath(args.from_predictions)), "metrics.json")
        with open(out, "w") as f:
            json.dump(dict(s, source=args.from_predictions, n=len(preds), group_by=args.group_by), f, indent=1)
        print(f"\nsaved {out}")
        if args.tensorboard:
            log_tensorboard(s, args.tensorboard, tb_step, tb_tag)
        return

    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    recs = load_jsonl(args.data)
    recs = recs[: args.limit] if args.limit else recs
    if not recs:
        sys.exit(f"пусто: {args.data}")
    if args.frame_selection:
        fsr_args = {"max_frames": args.max_frames, "num_surr": args.num_surr, "tail": args.tail}
        recs = [reselect_frames(r, args.frame_selection, fsr_args) for r in recs]

    # у другой базовой модели своя папка, иначе zero-shot InternVL3-8B перезапишет результаты 2B
    default_model = ap.get_default("model")
    tag = (os.path.basename(os.path.normpath(args.lora)) if args.lora else
           "base" if args.model == default_model else f"base-{os.path.basename(os.path.normpath(args.model))}")
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.data)), "eval",
        f"{os.path.splitext(os.path.basename(args.data))[0]}_{tag}"
        + (f"_{args.frame_selection}" if args.frame_selection else "") + (f"_limit{args.limit}" if args.limit else ""))
    os.makedirs(out_dir, exist_ok=True)
    run_config = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv, "args": vars(args), "device": device, "n_records": len(recs),
        "git": {"project": git_hash(os.path.dirname(os.path.abspath(__file__)))},
        "lora_config": None, "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
    }
    if args.lora and os.path.exists(os.path.join(args.lora, "..", "run_config.json")):
        with open(os.path.join(args.lora, "..", "run_config.json")) as f:
            run_config["lora_config"] = json.load(f)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=1)

    tok, model = load_model(args.model, args.lora, device)
    transform = build_transform(model.config.force_image_size or 448)
    letter_ids = [tok.convert_tokens_to_ids(c) for c in CLASSES]
    assert all(i is not None and i != tok.unk_token_id for i in letter_ids), letter_ids

    preds, failed, t0 = [], 0, time.time()
    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as fout:  # пишем по одной, чтобы падение не теряло прогон
        for k, rec in enumerate(recs):
            gt = rec["conversations"][1]["value"][0]
            try:
                p, probs = predict(model, tok, transform, rec, letter_ids, device)
            except Exception as e:  # один битый эпизод не должен ронять весь прогон
                failed += 1
                print(f"fail {rec['id']}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
                continue
            row = {"id": rec["id"], "agent": rec.get("agent") or rec["id"].split("/")[0],
                   "task": rec.get("task") or re.sub(r"_\d+$", "", rec["id"].split("/")[-1]),
                   "n_frames": len(rec["image"]), "gt": gt, "pred": p,
                   "probs": {c: round(x, 5) for c, x in zip(CLASSES, probs)}}
            preds.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            if (k + 1) % 10 == 0 or k + 1 == len(recs):
                print(f"[{k + 1}/{len(recs)}] {time.time() - t0:.0f}s", file=sys.stderr)
    if failed:
        print(f"failed: {failed}/{len(recs)}", file=sys.stderr)
    if not preds:
        sys.exit("ни одного предсказания")

    s = summarize(preds, args.group_by)
    print_summary(s, f"{tag}" + (f" [{args.frame_selection}]" if args.frame_selection else ""))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(dict(s, data=args.data, model=args.model, lora=args.lora, frame_selection=args.frame_selection,
                       n=len(preds), failed=failed, group_by=args.group_by), f, indent=1)
    print(f"\nsaved {out_dir}/{{predictions.jsonl,metrics.json,run_config.json}}")
    if args.tensorboard:
        log_tensorboard(s, args.tensorboard, tb_step, tb_tag)


if __name__ == "__main__":
    main()
