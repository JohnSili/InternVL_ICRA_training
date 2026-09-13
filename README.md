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

Второй `uv sync` отдельным шагом не случайно: `setup.py` у deepspeed импортирует torch, поэтому он собирается
без изоляции уже поверх `.venv` с torch, и ему нужен `CUDA_HOME` из шага 2. flash-attn ставится готовым колесом
под cp312/x86_64/torch2.9/cu12 прямо с GitHub, ничего не компилирует. Ровно поэтому torch зафиксирован на 2.9.1:
под более свежие версии готовых колёс flash-attn нет, а тренер InternVL импортирует его безусловно.
Python берётся из `.python-version` (3.12), uv скачает его сам.

`uv sync` синхронизирует окружение точно: голый `uv sync` после установки экстр удалит deepspeed и flash-attn,
поэтому на тренировочной машине всегда добавляйте `--extra train --extra flash`. Запуск скриптов:
`source .venv/bin/activate` и обычные `python3` / `pytest`, либо `uv run --no-sync ...`. `train.sh` и
`watch_val.sh` сами подхватывают `.venv` рядом с собой. Если загрузка модели с HF зависает, `HF_HUB_DISABLE_XET=1`.

Проверено на этой машине: `uv sync` + валидатор, быстрый набор тестов, zero-shot оценка и GPU-тесты уровня 3
проходят в `.venv` (transformers 4.37.2 поверх torch 2.14); экстры `train`/`flash` здесь не собираются из-за
отсутствия `nvcc`, их проверяет `pytest test_pipeline.py -q` на машине с CUDA toolkit.

## Запуск на сервере: полный набор команд

Команды идут по порядку, каждый следующий шаг предполагает, что предыдущий отработал.

Два корня данных на сервере. `~/Simpler/trajectories` — кадры и `actions`: папки агентов
`INTACT-pi0-scratch-bridge`, `openvla-7b`, у каждого `meta/` и `frames/`; в `meta` лежит **ручная** разметка.
`~/Simpler/gt` — **gt-разметка**, по файлу `<имя>_auto.json` на траекторию, поля `annotation` там на верхнем
уровне рядом с `class_key`. Учимся по gt, а held-out держим на ручной, потому что она эталон. Неразмеченные
эпизоды (нет `class_key` из A–E в выбранном источнике) считаются и пропускаются.

**1. Данные на сервер** — только если их там ещё нет (png не хранятся в git):

```bash
rsync -a --info=progress2 ~/Documents/vlmfinetuning/INTACT-pi0-scratch-bridge/ <user>@<host>:~/Simpler/trajectories/INTACT-pi0-scratch-bridge/
```

**2. CUDA toolkit 12.8** (на сервере, если `which nvcc` пуст). `deepspeed` требует `CUDA_HOME` даже когда
ничего не собирает, поэтому драйвера и `nvidia-smi` мало. Версия должна совпадать с CUDA у torch из `uv.lock`
(`2.9.1+cu128` → 12.8), не с версией драйвера: драйвер 580 показывает «CUDA 13.0», но прекрасно исполняет
бинарники CUDA 12. Ставить только `cuda-toolkit-*`, не метапакет `cuda`: тот тянет драйвер и в контейнере
ломает GPU.

```bash
. /etc/os-release && echo "ubuntu${VERSION_ID/./}"          # ubuntu2204 / ubuntu2404 → подставить в URL ниже
apt-get update && apt-get install -y wget
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y cuda-toolkit-12-8
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH   # нужны и при обучении: deepspeed собирает ops JIT
nvcc --version | tail -1                                     # должно быть release 12.8
```

**3. Код и окружение** (на сервере):

```bash
git clone git@github.com:JohnSili/InternVL_ICRA_training.git vlmfinetuning && cd vlmfinetuning
uv sync                                    # torch, transformers 4.37.2, peft, tensorboard, pytest
uv sync --extra train --extra flash         # flash-attn ставится готовым колесом, компиляции нет
.venv/bin/python -c "import deepspeed, flash_attn, torch; print(deepspeed.__version__, flash_attn.__version__, torch.version.cuda)"
git clone https://github.com/OpenGVLab/InternVL.git && git -C InternVL checkout -q 2410d1dbf208f0e799459aff9376e5747dbf41a2
source .venv/bin/activate
```

**4. Данные и предполётные проверки:**

```bash
source .venv/bin/activate                                                          # в каждой новой ssh-сессии
export VLA_META_ROOT=~/Simpler/trajectories VLA_ANN_ROOT=~/Simpler/gt
python3 validate_dataset.py                                                        # проверяет gt-разметку
VLA_ANN_ROOT= python3 validate_dataset.py                                          # то же по ручной разметке
python3 prepare_data.py --out data/cls --holdout-list data/cls/heldout400.txt --holdout-per-group 50 --val-frac 0.03
pytest test_pipeline.py -q -m "not gpu and not slow"                               # секунды
pytest test_pipeline.py -q                                                         # с GPU-смоуком: два шага тренера, память, чекпоинт
```

Held-out: `--holdout-per-group 50` берёт 50 эпизодов на каждую пару (агент, задача), то есть 400 при двух
агентах и четырёх задачах, с сохранением пропорций классов внутри группы. Список пишется один раз в файл из
`--holdout-list` и дальше только читается, поэтому его надо закоммитить: пересборка с тем же файлом даёт тот
же held-out. Метки held-out по умолчанию из того же источника, что и train (gt); `--holdout-src manual`
переключает на ручную разметку, если нужен эталонный замер по ТЗ.

`--val-frac` задаёт val, который `watch_val.sh` прогоняет на каждом чекпоинте: при 7600 оставшихся эпизодах
0.03 даёт около 230, а 0.15 дало бы 1100 и оценка одного чекпоинта заняла бы десятки минут.
Без `--ann-root` (или `VLA_ANN_ROOT`) метки для train и val берутся из ручной разметки.

`prepare_data.py` напечатает согласие двух разметок на пересечении и первые расхождения. Это число — потолок:
если gt и ручная совпадают, скажем, на 85%, то macro-F1 выше 0.85 на ручном held-out ждать не стоит, как бы
хорошо модель ни выучила gt. Если расхождений много, смотреть надо на них, а не на гиперпараметры.
`--drop-unresolved` выкидывает из train и val эпизоды, которые авторазметчик пометил как нерешённые.

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
python3 prepare_data.py --out data/obs --target obs --balance --drop-recovery --holdout-list data/cls/heldout400.txt
DATA=data/obs bash train.sh                                                  # ответ <класс>|<симптом>, oversampling редких классов
for fs in surr2 uniform dense_sparse; do                                     # ablation по стратегии отбора кадров
  python3 evaluate.py --data data/cls/heldout.jsonl --lora $(cat work_dirs/cls/best_checkpoint.txt) --frame-selection $fs
done
```

После шага 3 голый `uv sync` больше не запускать: он удалит deepspeed и flash-attn. Пересинхронизация только
с `--extra train --extra flash`.

### deepspeed и nvcc

`train.sh` по умолчанию берёт `zero_stage1_torch_adam.json` из этого репозитория. Это копия конфига ZeRO-1 из
InternVL плюс `"torch_adam": true`. Без этого флага deepspeed подменяет AdamW своим FusedAdam и компилирует его
через `nvcc` на первом шаге обучения, то есть падает через минуту после старта, если toolkit неполный:

```
RuntimeError: Error building extension 'fused_adam'
/bin/sh: 1: /usr/local/cuda-12.8/bin/nvcc: not found
```

`DS_CONFIG=` (пустое значение) запускает обучение вообще без deepspeed. На одной карте ZeRO-1 ничего не даёт:
модель 2B с LoRA занимает 26 ГБ из 48, шардить нечего. Сам пакет deepspeed при этом всё равно нужен, его
безусловно импортирует `internvl/dist_utils.py`.

### Отложенные варианты разбивки

Не сделано намеренно, сделать при наличии времени после первого прогона. Held-out при обеих пересборках
не меняется: он зафиксирован списком `data/cls/heldout400.txt`.

**Oversampling под macro-F1.** В gt-разметке класс C редкий (291 из 8000, 3.6%; в train 267, в val 8,
в held-out 16), а macro-F1 усредняет классы поровну, поэтому итог заметно болтается из-за C.
`--balance` при потолке x5 даёт C x5 и D x3, train растёт примерно до 10 тысяч, эпоха длиннее на треть:

```bash
python3 prepare_data.py --out data/cls_bal --holdout-list data/cls/heldout400.txt --holdout-per-group 50 --val-frac 0.03 --balance
DATA=data/cls_bal bash train.sh
```

**Второй замер по ручной разметке.** Held-out на gt-метках отвечает на вопрос «выучила ли модель
разметчика». Чтобы получить согласие с человеком, нужен тот же held-out с ручными метками:

```bash
python3 prepare_data.py --out data/cls_manual --holdout-list data/cls/heldout400.txt --holdout-src manual --val-frac 0.03
python3 evaluate.py --data data/cls_manual/heldout.jsonl --lora $(cat work_dirs/cls/best_checkpoint.txt) --group-by agent
```

Ручная разметка есть только у 141 эпизода, поэтому пересечение с held-out будет небольшим. Согласие двух
разметок на этих 141: 66.7%. По классам оно неравномерно: E совпадает полностью (52 из 52), а из 17 эпизодов,
размеченных человеком как C, gt не назвал C ни одного (12 ушли в B, 4 в E, 1 в D). Это потолок для любого
сравнения с человеческим суждением и причина не ждать высокого macro-F1 на классе C.

## Скрипты

Четыре скрипта, общаются через файлы:

| шаг | команда | результат |
|---|---|---|
| валидация meta | `VLA_META_ROOT=<root> python3 validate_dataset.py` | hard-проверки → код возврата, soft → отчёт |
| подготовка | `python3 prepare_data.py --root <root> [--ann-root <gt>] --out data/cls` | `train/val/heldout.jsonl`, `meta.json`, `heldout.txt` |
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
