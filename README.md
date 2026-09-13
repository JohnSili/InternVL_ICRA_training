# Дообучение InternVL3-2B: классификация ошибок манипуляции по кадрам траектории

## Окружение (uv)

Одно окружение на всё: подготовка данных, оценка, тесты, обучение. Версии зафиксированы в `pyproject.toml`
и `uv.lock`; `transformers==4.37.2` и `peft==0.10.0` взяты из `InternVL/requirements/internvl_chat.txt`,
потому что тренер InternVL написан под них (новые transformers не принимают его `--evaluation_strategy`).

```bash
uv sync                                   # .venv: torch, transformers, peft, tensorboard, pytest, ... (оценка и тесты)
uv sync --extra train --extra flash       # на машине с GPU и nvcc: deepspeed + flash-attn (обучение)
git clone https://github.com/OpenGVLab/InternVL.git && git -C InternVL checkout -q 2410d1dbf208f0e799459aff9376e5747dbf41a2   # тренировочный код: не ставится, train.sh берёт его через PYTHONPATH; коммит тот, на котором всё проверено
```

Второй `uv sync` отдельным шагом не случайно: `setup.py` у deepspeed и flash-attn импортируют torch, поэтому
они собираются без изоляции уже поверх `.venv` с torch. Обоим нужен CUDA toolkit (`nvcc`, `CUDA_HOME`), без него
сборка падает; flash-attn компилируется из исходников десятки минут (`MAX_JOBS=8` ускоряет). Python берётся из
`.python-version` (3.12), uv скачает его сам.

`uv sync` синхронизирует окружение точно: голый `uv sync` после установки экстр удалит deepspeed и flash-attn,
поэтому на тренировочной машине всегда добавляйте `--extra train --extra flash`. Запуск скриптов:
`source .venv/bin/activate` и обычные `python3` / `pytest`, либо `uv run --no-sync ...`. `train.sh` и
`watch_val.sh` сами подхватывают `.venv` рядом с собой. Если загрузка модели с HF зависает, `HF_HUB_DISABLE_XET=1`.

Проверено на этой машине: `uv sync` + валидатор, быстрый набор тестов, zero-shot оценка и GPU-тесты уровня 3
проходят в `.venv` (transformers 4.37.2 поверх torch 2.14); экстры `train`/`flash` здесь не собираются из-за
отсутствия `nvcc`, их проверяет `pytest test_pipeline.py -q` на машине с CUDA toolkit.

## Запуск на сервере: полный набор команд

Команды идут по порядку, каждый следующий шаг предполагает, что предыдущий отработал. Корень данных на
сервере: `~/Simpler/trajectories` (внутри папки агентов `INTACT-pi0-scratch-bridge`, `openvla-7b`, у каждого
`meta/` и `frames/`); соседняя `~/Simpler/gt` к пайплайну не относится. Неразмеченные эпизоды (без `class_key`
из A–E) валидатор и `prepare_data.py` считают и пропускают, в обучение они не попадают.

**1. Данные на сервер** — только если их там ещё нет (png не хранятся в git):

```bash
rsync -a --info=progress2 ~/Documents/vlmfinetuning/INTACT-pi0-scratch-bridge/ <user>@<host>:~/Simpler/trajectories/INTACT-pi0-scratch-bridge/
```

**2. CUDA toolkit** (на сервере, если `which nvcc` пуст). flash-attn и deepspeed компилируют CUDA-расширения,
драйвера и `nvidia-smi` для этого мало. Мажорная версия toolkit должна совпадать с CUDA у torch из `uv.lock`
(`2.14.0+cu130` → 13.x). Ставить только `cuda-toolkit-*`, не метапакет `cuda`: тот тянет драйвер и в контейнере
ломает GPU.

```bash
. /etc/os-release && echo "ubuntu${VERSION_ID/./}"          # ubuntu2204 / ubuntu2404 → подставить в URL ниже
apt-get update && apt-get install -y wget
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y cuda-toolkit-13-0
export CUDA_HOME=/usr/local/cuda-13.0 && export PATH=$CUDA_HOME/bin:$PATH   # нужны и при обучении: deepspeed собирает ops JIT
nvcc --version | tail -1
```

**3. Код и окружение** (на сервере):

```bash
git clone git@github.com:JohnSili/InternVL_ICRA_training.git vlmfinetuning && cd vlmfinetuning
uv sync                                    # torch, transformers 4.37.2, peft, tensorboard, pytest
MAX_JOBS=8 uv sync --extra train --extra flash   # deepspeed + flash-attn, компиляция flash-attn занимает десятки минут
.venv/bin/python -c "import deepspeed, flash_attn, torch; print(deepspeed.__version__, flash_attn.__version__, torch.version.cuda)"
git clone https://github.com/OpenGVLab/InternVL.git && git -C InternVL checkout -q 2410d1dbf208f0e799459aff9376e5747dbf41a2
source .venv/bin/activate
```

**4. Данные и предполётные проверки:**

```bash
export VLA_META_ROOT=~/Simpler/trajectories
python3 validate_dataset.py                                                        # hard-проверки должны быть зелёные
python3 prepare_data.py --root $VLA_META_ROOT --out data/cls --holdout-list data/cls/heldout.txt
pytest test_pipeline.py -q -m "not gpu and not slow"                               # секунды
pytest test_pipeline.py -q                                                         # с GPU-смоуком: два шага тренера, память, чекпоинт
```

`--holdout-list data/cls/heldout.txt` обязателен: файл лежит в репозитории, и без него held-out нарежется заново.

**5. Точка отсчёта до обучения** (zero-shot, должен совпасть с majority):

```bash
python3 evaluate.py --data data/cls/heldout.jsonl --group-by agent
python3 evaluate.py --data data/cls/val.jsonl
```

**6. Обучение** (под `tmux` или `nohup`, чтобы пережило обрыв ssh):

```bash
DATA=data/cls bash train.sh 2>&1 | tee train_cls.log
```

Сам поднимет TensorBoard на 6006 и наблюдатель по val, в конце запишет `work_dirs/cls/best_checkpoint.txt`.
С рабочей машины: `ssh -L 6006:localhost:6006 <user>@<host>`, затем http://localhost:6006.

**7. Итоговая оценка на held-out:**

```bash
python3 evaluate.py --data data/cls/heldout.jsonl --lora $(cat work_dirs/cls/best_checkpoint.txt) --group-by agent --tensorboard work_dirs/cls/tensorboard
```

Сравнивать строку модели со строкой `majority` в том же выводе. Результат в `data/cls/eval/heldout_checkpoint-N/`.

**Варианты:**

```bash
PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACC=16 DATA=data/cls bash train.sh          # если OOM
python3 prepare_data.py --root $VLA_META_ROOT --out data/obs --target obs --balance --drop-recovery --holdout-list data/cls/heldout.txt
DATA=data/obs bash train.sh                                                  # ответ <класс>|<симптом>, oversampling редких классов
for fs in surr2 uniform dense_sparse; do                                     # ablation по стратегии отбора кадров
  python3 evaluate.py --data data/cls/heldout.jsonl --lora $(cat work_dirs/cls/best_checkpoint.txt) --frame-selection $fs
done
```

После шага 3 голый `uv sync` больше не запускать: он удалит deepspeed и flash-attn. Пересинхронизация только
с `--extra train --extra flash`.

## Скрипты

Четыре скрипта, общаются через файлы:

| шаг | команда | результат |
|---|---|---|
| валидация meta | `VLA_META_ROOT=<root> python3 validate_dataset.py` | hard-проверки → код возврата, soft → отчёт |
| подготовка | `python3 prepare_data.py --root <root> --out data/cls` | `train/val/heldout.jsonl`, `meta.json`, `heldout.txt` |
| zero-shot | `python3 evaluate.py --data data/cls/heldout.jsonl --group-by agent` | `data/cls/eval/heldout_base/` |
| обучение | `DATA=data/cls bash train.sh` | `work_dirs/cls/`, `best_checkpoint.txt` по macro-F1 на val |
| после обучения | `python3 evaluate.py --data data/cls/heldout.jsonl --lora $(cat work_dirs/cls/best_checkpoint.txt)` | сравнение с majority-baseline |

Эталонный промпт лежит в `prompt_cls.txt` / `prompt_obs.txt`; `prepare_data.py` подставляет в него инструкцию,
`evaluate.py` берёт текст из jsonl, поэтому train и eval видят один и тот же промпт.

## Мониторинг обучения (TensorBoard)

`train.sh` пишет в `OUTPUT_DIR/tensorboard`:

- `loss`, `learning_rate`, `epoch` от HF Trainer на каждом шаге (`--logging_steps 1`);
- `val/macro_f1`, `val/balanced_accuracy`, `val/accuracy`, `val/recall_A..E` и `val/majority_*` на каждом
  сохранённом `checkpoint-N`: их считает `watch_val.sh`, который `train.sh` запускает в фоне (лог в
  `OUTPUT_DIR/watch_val.log`), по оси X `global_step` чекпоинта;
- `run_config` во вкладке Text.

Сервер поднимается сам на `TB_PORT` (по умолчанию 6006) с `logdir` на уровень выше `OUTPUT_DIR`, так что в
одном окне видны все прогоны из `work_dirs/`. С удалённой машины: `ssh -L 6006:localhost:6006 <host>`, затем
http://localhost:6006. `TB_PORT=0` не поднимать сервер, `WATCH_VAL=0` не оценивать чекпоинты по ходу,
`TB_BIND_ALL=1` слушать все интерфейсы. Вручную: `tensorboard --logdir work_dirs --port 6006`.

Что считать нормой: `loss` падает от ~2–4 к <0.5 за две эпохи; `val/macro_f1` выше `val/majority_macro_f1`
(иначе модель просто выучила мажоритарный класс); `learning_rate` идёт по warmup и косинусу. Любая оценка
`evaluate.py` дописывается в тот же лог флагом `--tensorboard OUTPUT_DIR/tensorboard --step N`.

## Предполётные тесты (`test_pipeline.py`)

Переменные окружения: `VLA_META_ROOT` (корень meta/frames), `VLA_DATA_OUT` (папка с train.jsonl/val.jsonl/meta.json,
по умолчанию `data/cls`), `VLA_MODEL` (путь или HF-id, по умолчанию `OpenGVLab/InternVL3-2B`).

```bash
pytest test_pipeline.py -q -m "not gpu and not slow"   # уровни 1–2, без GPU, минуты
pytest test_pipeline.py -q                             # всё, включая GPU-смоук
```

Первую команду гонять после каждого изменения `prepare_data.py` (и перед каждым запуском `train.sh`).
Вторую гонять один раз перед длинным прогоном обучения: она грузит модель, меряет пиковую память на
полном батче, делает два шага тренера во временную папку и проверяет загрузку чекпоинта и `evaluate.py`.
