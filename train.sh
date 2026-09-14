#!/usr/bin/env bash
# LoRA-дообучение InternVL3-2B на jsonl из prepare_data.py и отбор чекпоинта по macro-F1 на val.
#
# Основа: InternVL/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_2b_dynamic_res_2nd_finetune_full.sh
# + LoRA-флаги из internvl2.5/2nd_finetune/internvl2_5_2b_dynamic_res_2nd_finetune_lora.sh. Свой train-loop не пишем.
#
#   DATA=data/cls bash train.sh                       # -> work_dirs/cls
#   DATA=data/obs OUTPUT_DIR=work_dirs/obs SEED=1 bash train.sh
#
# Окружение: uv sync && uv sync --extra train --extra flash (см. pyproject.toml: transformers 4.37.2, peft 0.10,
# deepspeed, flash-attn — internvl_chat_finetune.py включает flash_attention_2 для LLM безусловно, tensorboard).
# Скрипт сам подхватывает .venv рядом с собой, так что `bash train.sh` и `uv run bash train.sh` равнозначны.
# Одна A100. Если OOM: PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACC=16 bash train.sh. max_dynamic_patch не трогать —
# 16 кадров x 256 токенов = 4096 (20 кадров данных статьи = 5120), любой другой max_dynamic_patch раздует контекст в разы.
#
# Мониторинг: loss/lr тренера пишутся в OUTPUT_DIR/tensorboard (каждый шаг); watch_val.sh в фоне оценивает
# каждый сохранённый checkpoint-N на val и пишет туда же val/macro_f1 и т.д.; tensorboard-сервер
# поднимается на TB_PORT (0 — не поднимать) с logdir на уровень выше OUTPUT_DIR, так что видны все прогоны.
# В OUTPUT_DIR пишется run_config.json (все параметры, git-хеши, дата, конфиг prepare_data).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -x "$HERE/.venv/bin/python3" ] && export PATH="$HERE/.venv/bin:$PATH"   # окружение из uv sync
REPO=${REPO:-$HERE/InternVL/internvl_chat}
DATA=${DATA:-data/cls}
META_PATH=$(readlink -f "${META_PATH:-$DATA/meta.json}")
VAL_JSONL=$(readlink -f "${VAL_JSONL:-$DATA/val.jsonl}")
OUTPUT_DIR=${OUTPUT_DIR:-$HERE/work_dirs/$(basename "$DATA")}
MODEL=${MODEL:-OpenGVLab/InternVL3-2B}

GPUS=${GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=${GRADIENT_ACC:-8}
EPOCHS=${EPOCHS:-2}
LR=${LR:-5e-5}
LORA_R=${LORA_R:-16}
SEED=${SEED:-0}
MASTER_PORT=${MASTER_PORT:-34229}
SELECT=${SELECT:-1}     # 0 — только обучение, без выбора лучшего чекпоинта по val
# Конфиг deepspeed. По умолчанию наш, с "torch_adam": true: иначе deepspeed подставляет свой FusedAdam
# и компилирует его через nvcc на первом шаге обучения, а toolkit нужен далеко не всегда.
# DS_CONFIG= (пусто) — запустить вообще без deepspeed: на одной карте ZeRO-1 ничего не даёт.
DS_CONFIG=${DS_CONFIG-$HERE/zero_stage1_torch_adam.json}
WATCH_VAL=${WATCH_VAL:-1}   # 0 — не оценивать чекпоинты на val по ходу обучения
TB_PORT=${TB_PORT:-6006}    # 0 — не поднимать tensorboard-сервер; TB_BIND_ALL=1 — слушать не только localhost
# EXTRA_ARGS — дописываются в конец команды тренера (последнее вхождение флага побеждает),
# например EXTRA_ARGS="--max_steps 2 --warmup_ratio 0" для смоука из test_pipeline.py

[ -f "$META_PATH" ] || { echo "нет $META_PATH — сначала prepare_data.py"; exit 1; }
[ -d "$REPO/internvl" ] || { echo "нет репозитория InternVL в $REPO (см. README: git clone + checkout 2410d1d)"; exit 1; }
python3 -c "import tensorboard" 2>/dev/null || { echo "нет пакета tensorboard (pip install tensorboard): тренер с --report_to tensorboard упадёт на старте"; exit 1; }
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(readlink -f "$OUTPUT_DIR")
TB_DIR=$OUTPUT_DIR/tensorboard
mkdir -p "$TB_DIR"

# --- run_config.json: пишем до старта, чтобы он был даже если обучение упало; копия в Text-вкладку TB ----
HERE="$HERE" REPO="$REPO" MODEL="$MODEL" DATA="$DATA" META_PATH="$META_PATH" VAL_JSONL="$VAL_JSONL" OUTPUT_DIR="$OUTPUT_DIR" \
GPUS="$GPUS" PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" GRADIENT_ACC="$GRADIENT_ACC" EPOCHS="$EPOCHS" LR="$LR" \
LORA_R="$LORA_R" SEED="$SEED" TB_DIR="$TB_DIR" EXTRA_ARGS="${EXTRA_ARGS:-}" python3 - <<'EOF'
import json, os, subprocess, datetime
e = os.environ
def git(d):
    try:
        return subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
prep = os.path.join(os.path.dirname(e["META_PATH"]), "prepare_config.json")
cfg = {
    "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "script": "train.sh", "model": e["MODEL"], "data": e["DATA"], "meta_path": e["META_PATH"],
    "val_jsonl": e["VAL_JSONL"], "output_dir": e["OUTPUT_DIR"], "tensorboard": e["TB_DIR"],
    "gpus": int(e["GPUS"]), "per_device_train_batch_size": int(e["PER_DEVICE_BATCH_SIZE"]),
    "gradient_accumulation_steps": int(e["GRADIENT_ACC"]), "num_train_epochs": float(e["EPOCHS"]),
    "learning_rate": float(e["LR"]), "use_llm_lora": int(e["LORA_R"]), "seed": int(e["SEED"]),
    "conv_style": "internvl2_5", "force_image_size": 448, "max_dynamic_patch": 1, "max_seq_length": 8192,
    "freeze_backbone": True, "freeze_mlp": False, "freeze_llm": True, "warmup_ratio": 0.03,
    "weight_decay": 0.05, "lr_scheduler_type": "cosine", "bf16": True, "grad_checkpoint": True,
    "extra_args": e["EXTRA_ARGS"],
    "git": {"project": git(e["HERE"]), "InternVL": git(e["REPO"])},
    "prepare_config": json.load(open(prep)) if os.path.exists(prep) else None,
}
text = json.dumps(cfg, indent=2, ensure_ascii=False)
open(os.path.join(e["OUTPUT_DIR"], "run_config.json"), "w").write(text)
print("run_config.json ->", e["OUTPUT_DIR"])
from torch.utils.tensorboard import SummaryWriter
w = SummaryWriter(e["TB_DIR"]); w.add_text("run_config", "```\n" + text + "\n```"); w.close()
EOF

# --- tensorboard-сервер: logdir на уровень выше, чтобы сравнивать прогоны -----------------------------------
LOGDIR_ALL=$(dirname "$OUTPUT_DIR")
if [ "$TB_PORT" != "0" ]; then
  if ! command -v tensorboard >/dev/null 2>&1; then
    echo "tensorboard CLI не найден; запустите вручную: tensorboard --logdir $LOGDIR_ALL --port $TB_PORT"
  elif (exec 3<>"/dev/tcp/127.0.0.1/$TB_PORT") 2>/dev/null; then
    echo "tensorboard: порт $TB_PORT уже занят, считаю что сервер запущен: http://localhost:$TB_PORT"
  else
    nohup tensorboard --logdir "$LOGDIR_ALL" --port "$TB_PORT" ${TB_BIND_ALL:+--bind_all} > "$OUTPUT_DIR/tensorboard.log" 2>&1 &
    echo $! > "$OUTPUT_DIR/tensorboard.pid"
    echo "tensorboard: http://localhost:$TB_PORT  (logdir $LOGDIR_ALL, pid $!, остановить: kill \$(cat $OUTPUT_DIR/tensorboard.pid))"
  fi
  echo "с другой машины: ssh -L $TB_PORT:localhost:$TB_PORT <host>, затем http://localhost:$TB_PORT"
fi

# --- наблюдатель: val-метрики каждого чекпоинта по ходу обучения -> TensorBoard -----------------------------
rm -f "$OUTPUT_DIR/.train_done"
trap 'touch "$OUTPUT_DIR/.train_done"' EXIT   # при любом исходе даём watch_val.sh дооценить и выйти
WATCH_PID=""
if [ "$WATCH_VAL" = "1" ]; then
  bash "$HERE/watch_val.sh" "$OUTPUT_DIR" "$VAL_JSONL" "$MODEL" > "$OUTPUT_DIR/watch_val.log" 2>&1 &
  WATCH_PID=$!
  echo "watch_val.sh запущен (pid $WATCH_PID), лог: $OUTPUT_DIR/watch_val.log"
fi

DS_ARGS=()
if [ -n "$DS_CONFIG" ]; then
  [ -f "$DS_CONFIG" ] || { echo "нет файла deepspeed-конфига: $DS_CONFIG"; exit 1; }
  DS_ARGS=(--deepspeed "$(readlink -f "$DS_CONFIG")")
  echo "deepspeed: $DS_CONFIG"
else
  echo "deepspeed отключён (DS_CONFIG пуст)"
fi

export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

cd "$REPO"
# effective batch = GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACC = 16
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "$MODEL" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 1 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora ${LORA_R} \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 20 \
  --learning_rate ${LR} \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --logging_dir "$TB_DIR" \
  --max_seq_length 8192 \
  --seed ${SEED} \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  ${DS_ARGS[@]+"${DS_ARGS[@]}"} \
  --report_to "tensorboard" \
  ${EXTRA_ARGS:-} \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

cd "$HERE"
touch "$OUTPUT_DIR/.train_done"
if [ -n "$WATCH_PID" ]; then
  echo "жду watch_val.sh: дооценивает последние чекпоинты"
  wait "$WATCH_PID" || echo "watch_val.sh завершился с ошибкой, см. $OUTPUT_DIR/watch_val.log"
fi

[ "$SELECT" = "1" ] || exit 0

# --- отбор чекпоинта: macro-F1 на val, не accuracy и не loss ---------------------------------
# метрики, уже посчитанные watch_val.sh, переиспользуются; остальные чекпоинты оцениваются здесь
best=""
best_f1="-1"
for ck in "$OUTPUT_DIR"/checkpoint-*; do
  [ -d "$ck" ] || continue
  if [ ! -f "$ck/val_eval/metrics.json" ]; then
    if ! python3 evaluate.py --model "$MODEL" --data "$VAL_JSONL" --lora "$ck" --out-dir "$ck/val_eval" --seed "$SEED" \
         --tensorboard "$TB_DIR" --step "${ck##*-}" --tb-tag val > "$ck/val_eval.txt" 2>&1; then
      echo "eval failed: $ck (см. $ck/val_eval.txt)"
      continue
    fi
  fi
  f1=$(python3 -c "import json; print(json.load(open('$ck/val_eval/metrics.json'))['overall']['metrics']['macro_f1'])")
  echo "$ck  val macro-F1 = $f1"
  if python3 -c "import sys; sys.exit(0 if $f1 > $best_f1 else 1)"; then
    best="$ck"; best_f1="$f1"
  fi
done
[ -n "$best" ] || { echo "ни один чекпоинт не оценился"; exit 1; }
echo "$best" > "$OUTPUT_DIR/best_checkpoint.txt"
echo "best checkpoint: $best (val macro-F1 $best_f1) -> $OUTPUT_DIR/best_checkpoint.txt"
echo "held-out: python3 evaluate.py --data $(dirname "$VAL_JSONL")/heldout.jsonl --lora $best --group-by agent --tensorboard $TB_DIR"
