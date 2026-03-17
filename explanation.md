# Explanation

Сделал clean project под задачу `violence / nonviolence video classification`.

## Что я собрал

- `prepare_dataset.py`
  сканирует raw videos и делает `train.csv` / `val.csv`
- `train.py`
  обучает или fine-tune video model на `PyTorch`
- `evaluate.py`
  считает метрики на validation set
- `evaluate_real_videos.py`
  гоняет 3-5 real videos и делает reports
- `FastAPI` demo
  чтобы можно было быстро показать inference через upload

## Почему такой стек

Я взял `torchvision r3d_18` как default backbone.

Причина simple:
- model already made for video
- легче и чище как baseline
- pretrained weights help to reach target faster
- код получается compact и easy to explain

## Как работает pipeline

1. Raw dataset кладется в `data/raw`
2. Скрипт делает manifests
3. Dataset loader читает видео через `OpenCV`
4. Из каждого видео берется clip фиксированной длины
5. Clip нормализуется и идет в model
6. Во время train сохраняется best checkpoint в `.safetensors`
7. Потом этот же checkpoint идет на validation и на real internet videos

## Почему `.safetensors`

Это было прямое требование задания.

Плюс это nice format because:
- safe loading
- удобный export/import
- просто хранить final artifact

## Что еще важно

- comments в коде оставил короткие и only where they really help
- split dataset сделан аккуратно: если в Kaggle dataset уже есть `train/val`, код их сохранит
- если split нет, будет нормальный stratified split
- внешний evaluation сделан отдельным script, чтобы показать именно real-world check

## Что нужно сделать перед финальным GitHub push

1. Скачать Kaggle dataset в `data/raw`
2. Поставить зависимости
3. Запустить `prepare_dataset.py`
4. Запустить train
5. Проверить `validation_report.json`
6. Добавить 3-5 internet videos в `data/external_videos`
7. Запустить external evaluation
8. После реального train положить финальный `.safetensors` в repo

## Что я уже реально проверил локально

- `pytest`:
  `5 passed`
- syntax smoke:
  `python -m compileall app scripts tests`
- model/runtime smoke:
  создание `ViolenceVideoClassifier`
- synthetic end-to-end smoke:
  `prepare_dataset.py -> train.py -> evaluate.py -> evaluate_real_videos.py`

Важно:
- это был synthetic dataset smoke-check, not Kaggle final benchmark
- итоговые `85%+` можно честно заявлять только после реального train/eval на настоящем датасете

## If very short

Repo already has:
- structure
- train code
- eval code
- export to `.safetensors`
- optional API for demo

What is left for final production-like result:
- real dataset download
- real training run
- final metrics
- final model artifact
