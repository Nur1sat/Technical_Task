from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference.predictor import Predictor


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/run_r3d18/best_model.safetensors"))
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    return Predictor(
        model_path=model_path,
        device=os.getenv("MODEL_DEVICE", "auto"),
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Violence Detection Inference API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        model_path = Path(os.getenv("MODEL_PATH", "artifacts/run_r3d18/best_model.safetensors"))
        return {
            "status": "ok",
            "model_ready": model_path.exists(),
            "model_path": str(model_path),
        }

    @app.post("/predict")
    async def predict(video: UploadFile = File(...)) -> dict[str, str | float | dict[str, float]]:
        suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"

        try:
            predictor = get_predictor()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            shutil.copyfileobj(video.file, handle)
            temp_path = Path(handle.name)

        try:
            return predictor.predict_video(temp_path).to_dict()
        finally:
            temp_path.unlink(missing_ok=True)

    return app


app = create_app()
