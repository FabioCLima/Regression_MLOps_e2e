import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.middleware import RequestIdMiddleware
from src.api.model_loader import load_artifacts
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.service import predict_batch, predict_one
from src.logging_config import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger(log_file="api.log", level=os.getenv("LOG_LEVEL", "INFO"))
    try:
        app.state.artifacts = load_artifacts()
    except Exception as e:
        logger.error("artifact_load_failed error={}", e)
        app.state.artifacts = None
    yield


app = FastAPI(
    title="Regression MLOps E2E API",
    version="0.1.0",
    description="REST API for house price prediction. Part of the regression-mlops-e2e project.",
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)

cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def _get_artifacts(request: Request):
    return getattr(request.app.state, "artifacts", None)


def _require_artifacts(request: Request):
    artifacts = _get_artifacts(request)
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Check /health for details.",
        )
    return artifacts


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "regression-mlops-e2e-api", "status": "running"}


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    artifacts = _get_artifacts(request)
    if artifacts is None:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="unhealthy",
                model_loaded=False,
                encoders_loaded=False,
                model_version=None,
                n_features_expected=None,
            ).model_dump(),
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        encoders_loaded=artifacts.frequency_encoder is not None
        and artifacts.target_encoder is not None,
        model_version=artifacts.version_string,
        n_features_expected=artifacts.n_features,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(body: PredictionRequest, request: Request) -> PredictionResponse:
    artifacts = _require_artifacts(request)
    return predict_one(body, artifacts)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_endpoint(body: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    artifacts = _require_artifacts(request)
    return predict_batch(body, artifacts)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    artifacts = _require_artifacts(request)
    return ModelInfoResponse(
        version_string=artifacts.version_string,
        source=artifacts.source,
        loaded_at=artifacts.loaded_at,
        n_features_expected=artifacts.n_features,
        train_columns=artifacts.train_columns,
    )
