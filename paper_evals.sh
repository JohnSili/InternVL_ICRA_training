#!/usr/bin/env bash
# Очередь оценок для статьи на test: zero-shot и дообученная модель из train.sh.
# На каждую модель: полные токены на heldout и heldout_human, две другие стратегии кадров авторов на heldout,
# DivPrune с долями RATIOS и случайный
# контроль с теми же долями, Top-K из пула POOL кадров (k <= TOPK). Прогон, у которого уже есть
# metrics.json, пропускается, поэтому после падения скрипт просто запускают ещё раз.
# Дообученная модель берётся из RUN/best_checkpoint.txt. Если обучение ещё идёт, скрипт ждёт его конца
# и отбора чекпоинта и до тех пор ничего не запускает, чтобы не делить карту с тренером.
#
#   bash paper_evals.sh 2>&1 | tee paper_evals.log                    # во время или после обучения data/paper
#   MODELS=base MODEL=OpenGVLab/InternVL3-8B bash paper_evals.sh       # zero-shot 8B, без ожидания
#   ABLATIONS=strategies MODELS=base MODEL=OpenGVLab/InternVL3-8B bash paper_evals.sh   # 8B без прунинга и Top-K
#   LIMIT=5 RATIOS=0.5 MODELS=base bash paper_evals.sh                 # смоук
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export PATH="$HERE/.venv/bin:$PATH"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

DEFAULT_MODEL=OpenGVLab/InternVL3-2B
DATA=${DATA:-data/paper}
RUN=${RUN:-work_dirs/$(basename "$DATA")}
MODEL=${MODEL:-$DEFAULT_MODEL}
MODELS=${MODELS:-base ft}
# пул Top-K собирается той же стратегией, что и данные
FS=${FS:-$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['frame_selection'])" "$DATA/prepare_config.json")}
POOL=${POOL:-30}
TOPK=${TOPK:-20}
RATIOS=${RATIOS:-0.2 0.5 0.7}
ABLATIONS=${ABLATIONS:-strategies prune topk}  # что считать сверх полных токенов на heldout и heldout_human
for a in $ABLATIONS; do
  case "$a" in strategies|prune|topk) ;; *) echo "ABLATIONS: strategies, prune, topk, а не $a"; exit 1 ;; esac
done
has() { [[ " $ABLATIONS " == *" $1 "* ]]; }
LIMIT=${LIMIT:-}
WAIT_HOURS=${WAIT_HOURS:-12}

[ -f "$DATA/heldout.jsonl" ] || { echo "нет $DATA/heldout.jsonl"; exit 1; }

wait_best() {
  # best_checkpoint.txt должен быть от этого обучения, то есть новее run_config.json, который train.sh пишет
  # на старте. Сравнивать с .train_done нельзя: train.sh трогает его ещё раз при выходе (trap EXIT), уже
  # после записи best_checkpoint.txt, и условие никогда бы не выполнилось.
  local best="$RUN/best_checkpoint.txt" done="$RUN/.train_done" waited=0
  until [ -f "$best" ] && [ -f "$done" ] && [ "$best" -nt "$RUN/run_config.json" ]; do
    if [ -f "$done" ] && [ $(( $(date +%s) - $(stat -c %Y "$done") )) -gt 7200 ]; then
      echo "обучение завершилось больше 2 ч назад, а свежего $best нет: см. $RUN/training_log.txt"
      return 1
    fi
    if [ "$waited" -ge $(( WAIT_HOURS * 3600 )) ]; then
      echo "не дождался $best за $WAIT_HOURS ч"
      return 1
    fi
    [ "$waited" -eq 0 ] && echo "$(date +%H:%M) жду конца обучения и отбора чекпоинта: $best"
    sleep 300
    waited=$((waited + 300))
  done
}

failed=()
run() {  # run <out_dir> <аргументы evaluate.py>
  local out="$1"
  shift
  # готов только читаемый metrics.json: при падении сервера во время записи файл остаётся пустым
  if python3 -c "import json, sys; json.load(open(sys.argv[1]))" "$out/metrics.json" 2>/dev/null; then
    echo "готово, пропускаю: $out"
    return
  fi
  echo "=== $(date +%H:%M) $out"
  python3 evaluate.py --model "$MODEL" --out-dir "$out" --group-by agent --seed 0 ${LIMIT:+--limit "$LIMIT"} "$@" \
    || failed+=("$out")
}

# ждём до любых прогонов: zero-shot во время обучения делил бы карту с тренером.
# Если обучение упало, zero-shot всё равно считается, а дообученная модель пропускается.
ck=""
case " $MODELS " in
  *" ft "*)
    if wait_best; then
      ck=$(cat "$RUN/best_checkpoint.txt")
      echo "дообученная модель: $ck"
    else
      failed+=("ft: нет отобранного чекпоинта")
    fi ;;
esac

sfx=${LIMIT:+_limit$LIMIT}
E="$DATA/eval"
for m in $MODELS; do
  case "$m" in
    base)
      model_args=()
      if [ "$MODEL" = "$DEFAULT_MODEL" ]; then tag=base; else tag="base-$(basename "$MODEL")"; fi ;;
    ft)
      [ -n "$ck" ] || continue
      model_args=(--lora "$ck")
      tag=$(basename "$ck") ;;
    *)
      echo "MODELS: base и/или ft, а не $m"
      exit 1 ;;
  esac
  margs=${model_args[@]+"${model_args[@]}"}
  run "$E/heldout_$tag$sfx" --data "$DATA/heldout.jsonl" $margs
  if [ -f "$DATA/heldout_human.jsonl" ]; then
    run "$E/heldout_human_$tag$sfx" --data "$DATA/heldout_human.jsonl" $margs
  fi
  if has strategies; then
    for s in uniform dense_sparse surrounding; do  # таблица стратегий кадров; стратегия данных уже посчитана выше
      [ "$s" = "$FS" ] && continue
      run "$E/heldout_${tag}_$s$sfx" --data "$DATA/heldout.jsonl" --frame-selection "$s" $margs
    done
  fi
  if has prune; then
    for r in $RATIOS; do
      run "$E/heldout_${tag}_tok$r$sfx" --data "$DATA/heldout.jsonl" --token-ratio "$r" $margs
      run "$E/heldout_${tag}_rand$r$sfx" --data "$DATA/heldout.jsonl" --token-ratio "$r" --token-random $margs
    done
  fi
  if has topk; then
    run "$E/heldout_${tag}_${FS}_mf${POOL}_topk$TOPK$sfx" --data "$DATA/heldout.jsonl" \
      --frame-selection "$FS" --max-frames "$POOL" --topk "$TOPK" $margs
  fi
done

echo
echo "=== сводка по $E/heldout*"
for f in "$E"/heldout*/metrics.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
o = m["overall"]["metrics"]
vals = " ".join(f"{k}={v:.3f}" for k, v in o.items() if isinstance(v, float))
extra = f" frames={m['mean_frames']:.1f} tokens={m['mean_visual_tokens']:.0f}" if m.get("mean_frames") else ""
print(f"{os.path.basename(os.path.dirname(sys.argv[1])):45s} n={m.get('n')} {vals}{extra}")
PY
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "не получилось (${#failed[@]}):"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
