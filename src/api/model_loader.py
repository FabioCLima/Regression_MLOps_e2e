import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.config import paths
from src.inference.predict import (
    default_model_path,
    load_feature_encoders,
    load_training_feature_columns,
)


@dataclass
class Artifacts:
    model: object
    frequency_encoder: object | None
    target_encoder: object | None
    train_columns: list[str]
    version_string: str
    source: str
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def encoders(self) -> tuple:
        return (self.frequency_encoder, self.target_encoder)

    @property
    def n_features(self) -> int:
        return len(self.train_columns)


def _sha256_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:n]


def _load_from_disk(models_dir: Path, train_features_path: Path, source: str) -> Artifacts:
    model_path = default_model_path() if models_dir == paths.models_dir else (
        next(
            (models_dir / f for f in ["xgboost_tuned_model.pkl", "best_model.pkl"] if (models_dir / f).exists()),
            None,
        )
    )

    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"No model file found in {models_dir}")

    from joblib import load
    model = load(model_path)

    frequency_map, target_encoder = load_feature_encoders(models_dir)
    train_columns = load_training_feature_columns(train_features_path)

    n_expected = len(train_columns)
    n_model = getattr(model, "n_features_in_", None)
    if n_model is not None and n_model != n_expected:
        raise RuntimeError(
            f"Feature mismatch: model expects {n_model} features, "
            f"train schema has {n_expected}. Re-run the training pipeline."
        )

    version_string = f"{model_path.name}@sha256:{_sha256_prefix(model_path)}@{source}"
    logger.info("Artifacts loaded: {}", version_string)

    return Artifacts(
        model=model,
        frequency_encoder=frequency_map,
        target_encoder=target_encoder,
        train_columns=train_columns,
        version_string=version_string,
        source=source,
    )


# s3_key → local filename inside cache_dir
_S3_ARTIFACT_KEYS: dict[str, str] = {
    "models/xgboost_tuned_model.pkl": "xgboost_tuned_model.pkl",
    "models/best_model.pkl": "best_model.pkl",
    "models/zipcode_frequency_encoder.pkl": "zipcode_frequency_encoder.pkl",
    "models/city_full_target_encoder.pkl": "city_full_target_encoder.pkl",
    "data/processed/feature_engineered_train.csv": "feature_engineered_train.csv",
}

# startup fails fast if any of these are missing
_REQUIRED_S3_KEYS: frozenset[str] = frozenset({
    "models/xgboost_tuned_model.pkl",
    "data/processed/feature_engineered_train.csv",
})


def _download_from_s3(bucket: str, cache_dir: Path) -> None:
    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2}),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    failed_required: list[str] = []
    for s3_key, filename in _S3_ARTIFACT_KEYS.items():
        local_path = cache_dir / filename
        if local_path.exists():
            logger.debug("Cache hit: {}", local_path)
            continue
        try:
            logger.info("Downloading s3://{}/{}", bucket, s3_key)
            s3.download_file(bucket, s3_key, str(local_path))
        except Exception as e:
            if s3_key in _REQUIRED_S3_KEYS:
                failed_required.append(s3_key)
                logger.error("Failed to download required artifact {}: {}", s3_key, e)
            else:
                logger.warning("Could not download optional artifact {}: {}", s3_key, e)

    if failed_required:
        raise RuntimeError(
            f"Required S3 artifacts missing: {failed_required}. "
            f"Bucket: s3://{bucket}. Check IAM permissions and S3 keys."
        )


def load_artifacts(
    bucket: str | None = None,
    cache_dir: Path | None = None,
    train_features_path: Path | None = None,
) -> Artifacts:
    """Loads model artifacts from local disk or S3 (when AWS_S3_BUCKET is set)."""
    bucket = bucket or os.getenv("AWS_S3_BUCKET")

    if bucket:
        resolved_cache = cache_dir or Path(os.getenv("ARTIFACT_CACHE_DIR", str(paths.models_dir)))
        _download_from_s3(bucket, resolved_cache)
        resolved_train_path = train_features_path or (resolved_cache / "feature_engineered_train.csv")
        return _load_from_disk(resolved_cache, resolved_train_path, source=f"s3://{bucket}")

    resolved_train_path = train_features_path or (
        paths.processed_data_dir / "feature_engineered_train.csv"
    )
    return _load_from_disk(paths.models_dir, resolved_train_path, source="local")
