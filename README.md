# Violence Video Classifier

Тестовый проект на `PyTorch` для бинарной классификации видео: `violence` и `nonviolence`.

В репозитории есть:
- подготовка `train.csv` и `val.csv` из набора видео;
- обучение модели на базе `torchvision` video backbone;
- сохранение лучшего чекпоинта в `.safetensors`;
- валидация на `val`-манифесте;
- отдельный скрипт для проверки на внешних видео;
- простой `FastAPI`-эндпоинт для демонстрации инференса.

## Что внутри

- `scripts/prepare_dataset.py` - собирает манифесты из `data/raw`;
- `scripts/train.py` - запускает обучение и сохраняет артефакты;
- `scripts/evaluate.py` - считает метрики на валидации;
- `scripts/evaluate_real_videos.py` - прогоняет модель на внешних видео;
- `app/api/main.py` - API для загрузки видео и получения предсказания.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Данные

В репозитории нет самого датасета, потому что видеофайлы большие. Ожидаемая структура:

```text
data/raw/
  train/
    Violence/
    NonViolence/
  val/
    Violence/
    NonViolence/
```

Если в исходном наборе нет готового `train/val`, скрипт сам сделает стратифицированный split.

## Подготовка манифестов

```bash
python3 scripts/prepare_dataset.py \
  --data-dir data/raw \
  --manifests-dir data/manifests \
  --val-ratio 0.2
```

После этого появятся:
- `data/manifests/train.csv`
- `data/manifests/val.csv`

## Обучение

Базовая конфигурация лежит в `configs/baseline.json`.

```bash
python3 scripts/train.py --config configs/baseline.json
```

Основные артефакты:
- `artifacts/run_r3d18/best_model.safetensors`
- `artifacts/run_r3d18/history.json`
- `artifacts/run_r3d18/training_summary.json`
- `artifacts/run_r3d18/used_config.json`

## Валидация

```bash
python3 scripts/evaluate.py \
  --model artifacts/run_r3d18/best_model.safetensors \
  --manifest data/manifests/val.csv
```

Отчет сохраняется в:

```text
artifacts/reports/validation_report.json
```

## Проверка на внешних видео

Дополнительные видео можно положить в `data/external_videos/`.

Если нужно автоматически посчитать accuracy по именам файлов, удобно использовать такой формат:

```text
violence__street_fight_01.mp4
nonviolence__people_walking_01.mp4
```

Запуск:

```bash
python3 scripts/evaluate_real_videos.py \
  --model artifacts/run_r3d18/best_model.safetensors \
  --input-dir data/external_videos
```

Скрипт сохраняет:
- `artifacts/reports/real_videos.json`
- `artifacts/reports/real_videos.csv`
- `artifacts/reports/real_videos_summary.json`

## API

```bash
MODEL_PATH=artifacts/run_r3d18/best_model.safetensors \
uvicorn app.api.main:app --reload
```

Swagger:
- `http://127.0.0.1:8000/docs`

## Почему выбран `r3d_18`

По умолчанию используется `torchvision r3d_18`.

Причины:
- для видео это нормальный стартовый backbone;
- есть pretrained weights;
- модель не слишком тяжелая для базового эксперимента;
- ее проще быстро обучить и потом проверить на inference.

При необходимости можно переключить backbone в конфиге на:
- `mc3_18`
- `r2plus1d_18`

## Примечания

- Экспорт в `.safetensors` оставлен как основной формат чекпоинта.
- Итоговая accuracy зависит от реального запуска обучения на полном датасете.
- Чтобы добрать качество на GPU, в первую очередь есть смысл попробовать больше `epochs`, увеличить `batch_size` и сравнить `r3d_18` с `r2plus1d_18`.
