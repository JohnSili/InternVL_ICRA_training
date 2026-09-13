#!/usr/bin/env bash
# Наблюдатель для train.sh: как только тренер сохранил очередной checkpoint-N, оценивает его на val
# и пишет macro-F1 / bal-acc / acc / recall по классам в TensorBoard рядом с loss тренера.
#
#   bash watch_val.sh <OUTPUT_DIR> <VAL_JSONL> [MODEL]
#
# train.sh запускает его в фоне сам; можно запустить и руками из другого терминала для уже идущего
# прогона. Завершается, когда в OUTPUT_DIR появился файл .train_done и все чекпоинты оценены.
# Готовность чекпоинта: trainer_state.json пишется HF Trainer после весов, поэтому его наличие
# значит, что *.safetensors уже целые.
set -uo pipefail

OUTPUT_DIR=${1:?OUTPUT_DIR}
VAL_JSONL=${2:?VAL_JSONL}
MODEL=${3:-OpenGVLab/InternVL3-2B}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -x "$HERE/.venv/bin/python3" ] && export PATH="$HERE/.venv/bin:$PATH"   # окружение из uv sync
TB_DIR=$OUTPUT_DIR/tensorboard
POLL=${WATCH_POLL:-30}

scan() {
  for ck in "$OUTPUT_DIR"/checkpoint-*; do
    [ -f "$ck/trainer_state.json" ] || continue
    [ -f "$ck/val_eval/metrics.json" ] && continue
    [ -f "$ck/val_eval.failed" ] && continue
    step=${ck##*-}
    echo "[watch_val] $(date +%H:%M:%S) evaluating $ck on val"
    if python3 "$HERE/evaluate.py" --model "$MODEL" --data "$VAL_JSONL" --lora "$ck" --out-dir "$ck/val_eval" \
         --tensorboard "$TB_DIR" --step "$step" --tb-tag val > "$ck/val_eval.txt" 2>&1; then
      python3 -c "import json; m=json.load(open('$ck/val_eval/metrics.json'))['overall']['metrics']; \
print(f\"[watch_val] step $step  val macro-F1 {m['macro_f1']:.3f}  bal-acc {m['balanced_accuracy']:.3f}  acc {m['accuracy']:.3f}\")"
    else
      touch "$ck/val_eval.failed"
      echo "[watch_val] eval failed: $ck (см. $ck/val_eval.txt)"
    fi
  done
}

while [ ! -f "$OUTPUT_DIR/.train_done" ]; do
  scan
  sleep "$POLL"
done
scan
echo "[watch_val] done"
