# Violence Video Classifier

PyTorch pipeline for binary video classification: `violence` vs `nonviolence`.

Задача закрывается так:
- train/fine-tune предобученную video model на Kaggle dataset `real-life-violence-situations-dataset`
- сохранить лучший чекпоинт в формате `.safetensors`
- проверить качество на validation videos
- прогнать 3-5 real videos from internet через отдельный evaluation script
- при желании поднять маленький `FastAPI` inference API для demo

## What is inside

- `scripts/prepare_dataset.py`:
  builds `train.csv` / `val.csv` manifests
- `scripts/train.py`:
  train/fine-tune pipeline with `r3d_18` backbone by default
- `scripts/evaluate.py`:
  validation metrics for saved `.safetensors` model
- `scripts/evaluate_real_videos.py`:
  runs inference on 3-5 external videos and saves JSON/CSV report
- `app/api/main.py`:
  small FastAPI layer for upload-based inference

## Recommended setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Dataset structure

Repo does not bundle Kaggle videos because dataset is large.

Expected raw data location:

```text
data/raw/
  train/                  # optional, if dataset already has split
    Violence/
    NonViolence/
  val/                    # optional
    Violence/
    NonViolence/
```

If your downloaded dataset has no `train/val` folders, script will do stratified split automatically.

## 1. Build manifests

```bash
python3 scripts/prepare_dataset.py \
  --data-dir data/raw \
  --manifests-dir data/manifests \
  --val-ratio 0.2
```

Output:
- `data/manifests/train.csv`
- `data/manifests/val.csv`

## 2. Train model

Default config lives in `configs/baseline.json`.

```bash
python3 scripts/train.py --config configs/baseline.json
```

Best model is saved to:

```text
artifacts/run_r3d18/best_model.safetensors
```

Also saved:
- `artifacts/run_r3d18/history.json`
- `artifacts/run_r3d18/training_summary.json`
- `artifacts/run_r3d18/used_config.json`

## 3. Evaluate on validation set

```bash
python3 scripts/evaluate.py \
  --model artifacts/run_r3d18/best_model.safetensors \
  --manifest data/manifests/val.csv
```

Validation report is saved to:

```text
artifacts/reports/validation_report.json
```

## 4. Evaluate on 3-5 real internet videos

Put extra videos here:

```text
data/external_videos/
```

If you want auto-accuracy on those clips, name them like:

```text
violence__street_fight_01.mp4
nonviolence__people_walking_01.mp4
```

Then run:

```bash
python3 scripts/evaluate_real_videos.py \
  --model artifacts/run_r3d18/best_model.safetensors \
  --input-dir data/external_videos
```

Outputs:
- `artifacts/reports/real_videos.json`
- `artifacts/reports/real_videos.csv`
- `artifacts/reports/real_videos_summary.json`

## 5. Optional FastAPI demo

```bash
MODEL_PATH=artifacts/run_r3d18/best_model.safetensors \
uvicorn app.api.main:app --reload
```

Open:
- `http://127.0.0.1:8000/docs`

## Model choice

Default backbone is `torchvision` `r3d_18`:
- lightweight enough for a clean baseline
- pretrained weights are available
- good fit for short violence / non-violence clips

You can swap backbone in config to:
- `mc3_18`
- `r2plus1d_18`

## Notes

- `.safetensors` is used for final checkpoint export, as requested in the task.
- Current repo includes training/evaluation code and report pipeline.
- Final `85%+ accuracy` depends on real training run with dataset, pretrained weights download and available compute.
- For a stronger run on GPU, first try:
  increase `epochs` to `15-20`
- For a stronger run on GPU, second try:
  set `batch_size` to the highest value your VRAM allows
- For a stronger run on GPU, third try:
  compare `r3d_18` vs `r2plus1d_18`
