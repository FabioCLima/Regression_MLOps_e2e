"""Shared fixtures for the test suite."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import feature_config
from src.api.model_loader import Artifacts


class FakeModel:
    n_features_in_ = 8

    def predict(self, X):
        return [450000.0 + i * 1000 for i in range(len(X))]


class FakeTargetEncoder:
    def transform(self, X):
        return pd.DataFrame(
            {feature_config.target_encode_column: [10.0 + i for i in range(len(X))]}
        )


TRAIN_COLUMNS = [
    "year", "quarter", "month",
    "median_list_price", "lat", "lng",
    "zipcode_freq", "city_full_encoded",
]


def make_raw_inference_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2022-01-01", "2022-02-01"],
            "city_full": ["Atlanta-Sandy Springs-Alpharetta", "Pittsburgh"],
            "city": ["ATL", "PGH"],
            "zipcode": [30301, 15213],
            "median_sale_price": [100000, 110000],
            "median_list_price": [120000, 130000],
            "price": [125000, 135000],
        }
    )


@pytest.fixture
def fake_artifacts() -> Artifacts:
    fake_model = FakeModel()
    return Artifacts(
        model=fake_model,
        frequency_encoder=pd.Series({30301: 5, 15213: 3}),
        target_encoder=FakeTargetEncoder(),
        train_columns=TRAIN_COLUMNS,
        version_string="fake_model@sha256:abcd1234@local",
        source="local",
    )


@pytest.fixture
def api_client(fake_artifacts):
    from src.api.app import app

    app.state.artifacts = fake_artifacts
    yield TestClient(app, raise_server_exceptions=False)
    app.state.artifacts = None


@pytest.fixture
def api_client_no_artifacts():
    from src.api.app import app

    app.state.artifacts = None
    yield TestClient(app, raise_server_exceptions=False)
    app.state.artifacts = None


@pytest.fixture
def minimal_predict_payload() -> dict:
    return {
        "date": "2022-01-01",
        "city_full": "Atlanta-Sandy Springs-Alpharetta",
        "city": "ATL",
        "zipcode": 30301,
    }
