import pandas as pd
from loguru import logger

from src.inference.predict import predict
from src.api.model_loader import Artifacts
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)


_OPTIONAL_INPUT_FIELDS: tuple[str, ...] = tuple(
    name
    for name, field in PredictionRequest.model_fields.items()
    if not field.is_required() and name != "price"
)


def _compute_missing_features(request: PredictionRequest) -> list[str]:
    """Returns optional API input fields not provided by the client.

    The model feature schema contains derived columns such as year, lat, lng,
    zipcode_freq, and city_full_encoded. Reporting those as missing would be
    misleading because the inference pipeline creates them internally.
    """
    payload = request.model_dump()
    return [field for field in _OPTIONAL_INPUT_FIELDS if payload.get(field) is None]


def predict_one(request: PredictionRequest, artifacts: Artifacts) -> PredictionResponse:
    row = request.to_raw_dict()
    input_df = pd.DataFrame([row])

    predictions, _ = predict(
        input_df=input_df,
        model=artifacts.model,
        encoders=artifacts.encoders,
        train_columns=artifacts.train_columns,
    )

    predicted_price = float(predictions["predicted_price"].iloc[0])
    missing = _compute_missing_features(request)

    if missing:
        logger.warning("missing_features count={} features={}", len(missing), missing)

    return PredictionResponse(
        predicted_price=predicted_price,
        model_version=artifacts.version_string,
        missing_features=missing,
    )


def predict_batch(request: BatchPredictionRequest, artifacts: Artifacts) -> BatchPredictionResponse:
    # Process each record independently to avoid dedup side-effects from preprocess_dataset
    responses = [predict_one(record, artifacts) for record in request.records]
    return BatchPredictionResponse(rows_predicted=len(responses), predictions=responses)
