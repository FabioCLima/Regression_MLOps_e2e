import pandas as pd
import pytest
from joblib import dump

from src.config import feature_config, inference_config
from src.inference.predict import (
    align_features_to_training_schema,
    build_predictions_output,
    calculate_inference_metrics,
    predict,
    run_inference,
)


class FakeModel:
    def predict(self, X):
        return [1000.0 + i for i in range(len(X))]


class FakeTargetEncoder:
    def transform(self, X):
        return pd.DataFrame(
            {
                feature_config.target_encode_column: [
                    10.0 + i for i in range(len(X))
                ]
            }
        )


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


def test_align_features_to_training_schema_adds_missing_and_orders_columns():
    dataset = pd.DataFrame({"b": [2], "c": [3]})

    result = align_features_to_training_schema(dataset, ["a", "b"])

    assert result.columns.to_list() == ["a", "b"]
    assert result.loc[0, "a"] == 0
    assert result.loc[0, "b"] == 2


def test_build_predictions_output_adds_actuals_and_errors():
    features = pd.DataFrame({"feature": [1, 2]})
    actuals = pd.Series([900.0, 1100.0])

    result = build_predictions_output(features, [1000.0, 1001.0], actuals)

    assert inference_config.prediction_column in result.columns
    assert inference_config.actual_column in result.columns
    assert inference_config.prediction_error_column in result.columns
    assert result[inference_config.prediction_error_column].to_list() == [100.0, -99.0]


def test_calculate_inference_metrics_returns_none_without_actuals():
    assert calculate_inference_metrics(None, [1.0, 2.0]) is None


def test_predict_runs_full_inference_flow(tmp_path):
    model_path = tmp_path / "model.pkl"
    encoders_dir = tmp_path / "encoders"
    encoders_dir.mkdir()
    train_features_path = tmp_path / "feature_engineered_train.csv"
    dump(FakeModel(), model_path)
    dump(
        pd.Series({30301: 5, 15213: 3}),
        encoders_dir / feature_config.frequency_encoder_filename,
    )
    dump(
        FakeTargetEncoder(),
        encoders_dir / feature_config.target_encoder_filename,
    )
    pd.DataFrame(
        {
            "year": [2022],
            "quarter": [1],
            "month": [1],
            "median_list_price": [120000],
            "lat": [33.7],
            "lng": [-84.3],
            "zipcode_freq": [5],
            "city_full_encoded": [10],
            "price": [125000],
        }
    ).to_csv(train_features_path, index=False)

    predictions, metrics = predict(
        input_df=make_raw_inference_dataset(),
        model_path=model_path,
        encoders_dir=encoders_dir,
        train_features_path=train_features_path,
        metros_path=None,
    )

    assert predictions[inference_config.prediction_column].to_list() == [1000.0, 1001.0]
    assert inference_config.actual_column in predictions.columns
    assert metrics is not None
    assert set(metrics) == {"mae", "rmse", "r2"}


def test_run_inference_requires_input_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Inference input file not found"):
        run_inference(input_path=tmp_path / "missing.csv")
