#!/usr/bin/env python3
"""Проверка способа чтения ответа: буква по максимуму логитов пяти букв (как в evaluate.py) против свободной
жадной генерации с разбором буквы из текста (как делают обычно).

    python3 decode_check.py --data data/paper/heldout.jsonl --model OpenGVLab/InternVL3-8B --limit 60

Генерация идёт по одному токену с кешем и lm_head только на последней позиции: штатный generate в
transformers 4.37 считает логиты по всем позициям промпта и на 20 кадрах не помещается в малую карту.
Печатает, что модель пишет, как часто буквы совпадают и метрики обоих способов; построчно пишет в --out.
"""

import argparse
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import CLASSES, VIT_CHUNK, build_query, build_transform, load_model, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="OpenGVLab/InternVL3-2B")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", default=None, help="по умолчанию <dir(data)>/eval/decode_check_<split>_<model>.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from PIL import Image

    with open(os.path.expanduser(args.data)) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    recs = recs[: args.limit] if args.limit else recs
    tag = os.path.basename(os.path.normpath(args.lora or args.model))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.data)), "eval",
                                   f"decode_check_{os.path.splitext(os.path.basename(args.data))[0]}_{tag}.json")

    tok, model = load_model(args.model, args.lora, args.device)
    transform = build_transform(model.config.force_image_size or 448)
    letter_ids = [tok.convert_tokens_to_ids(c) for c in CLASSES]
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    lm, embed = model.language_model, model.language_model.get_input_embeddings()

    rows, t0 = [], time.time()
    for k, rec in enumerate(recs):
        with torch.inference_mode():
            pix = torch.stack([transform(Image.open(p)) for p in rec["image"]]).to(args.device, torch.bfloat16)
            vit = torch.cat([model.extract_feature(pix[i:i + VIT_CHUNK]) for i in range(0, pix.shape[0], VIT_CHUNK)])
            ids = tok(build_query(model, rec["conversations"][0]["value"]), return_tensors="pt").input_ids.to(args.device)
            emb = embed(ids).clone()
            emb[0, ids[0] == model.img_context_token_id] = vit.reshape(-1, emb.shape[-1]).to(emb.dtype)
            res = lm.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), use_cache=True)
            past, n = res.past_key_values, ids.shape[1]
            logits = lm.lm_head(res.last_hidden_state[:, -1])[0].float()
            constrained = CLASSES[int(logits[letter_ids].argmax())]
            first = int(logits.argmax())
            nxt, gen = first, []
            for _ in range(args.max_new_tokens):
                if nxt == eos:
                    break
                gen.append(nxt)
                n += 1
                res = lm.model(inputs_embeds=embed(torch.tensor([[nxt]], device=args.device)), past_key_values=past,
                               attention_mask=torch.ones((1, n), dtype=torch.long, device=args.device), use_cache=True)
                past = res.past_key_values
                nxt = int(lm.lm_head(res.last_hidden_state[:, -1])[0].float().argmax())
        text = tok.decode(gen)
        m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", text)
        rows.append({"id": rec["id"], "gt": rec["conversations"][1]["value"][0], "constrained": constrained,
                     "text": text, "parsed": m.group(1) if m else None, "first_token_is_letter": first in letter_ids})
        if (k + 1) % 10 == 0 or k + 1 == len(recs):
            print(f"[{k + 1}/{len(recs)}] {time.time() - t0:.0f}s", file=sys.stderr)

    n = len(rows)
    print(f"{args.model}{' + ' + args.lora if args.lora else ''} на {args.data}: {n} эпизодов")
    print("что генерирует (топ-8):", collections.Counter(r["text"] for r in rows).most_common(8))
    print(f"первый токен — буква A–E: {sum(r['first_token_is_letter'] for r in rows)}/{n}; "
          f"буква в тексте найдена: {sum(r['parsed'] is not None for r in rows)}/{n}; "
          f"совпадает с буквой по логитам: {sum(r['parsed'] == r['constrained'] for r in rows)}/{n}")
    y = [r["gt"] for r in rows]
    for name, key in (("по логитам", "constrained"), ("из текста", "parsed")):
        found = [r for r in rows if r[key] is not None]
        mm = metrics([r["gt"] for r in found], [r[key] for r in found]) if found else None
        print(f"{name:11s}: " + (f"macro-F1 {mm['macro_f1']:.3f}, acc {mm['accuracy']:.3f} на {len(found)} с найденной буквой, "
                                 f"ответы {dict(collections.Counter(r[key] for r in found))}" if mm else "букв нет"))
    print("gt:", dict(collections.Counter(y)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
