import pandas as pd
import pytest

from src.config import feature_config
from src.data.preprocess_data import cleaned_split_filename
from src.features.feature_engineering import (
    add_date_features,
    apply_frequency_encoder,
    drop_unused_columns,
    engineer_features,
    feature_engineered_split_filename,
    fit_frequency_encoder,
    load_cleaned_splits,
    register_feature_engineering_artifacts,
    run_feature_engineering,
)


def make_feature_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "city_full": ["atlanta", "atlanta", "miami"],
            "city": ["ATL", "ATL", "MIA"],
            "zipcode": [30301, 30301, 33101],
            "median_sale_price": [100000, 110000, 120000],
            "median_list_price": [120000, 130000, 140000],
            "price": [200000, 220000, 300000],
        }
    )


def test_add_date_features_adds_year_quarter_and_month_without_mutating_input():
    dataset = make_feature_dataset()

    result = add_date_features(dataset)

    assert ["date", "year", "quarter", "month"] == result.columns[:4].to_list()
    assert result["year"].to_list() == [2020, 2020, 2020]
    assert result["quarter"].to_list() == [1, 1, 1]
    assert result["month"].to_list() == [1, 2, 3]
    assert "quarter" not in dataset.columns


def test_add_date_features_requires_date_column():
    dataset = pd.DataFrame({"price": [100000]})

    with pytest.raises(ValueError, match="date"):
        add_date_features(dataset)


def test_frequency_encoder_maps_unknown_categories_to_zero():
    train_df = pd.DataFrame({"zipcode": [1, 1, 2]})
    eval_df = pd.DataFrame({"zipcode": [1, 3]})

    frequency_map = fit_frequency_encoder(train_df, "zipcode")
    result = apply_frequency_encoder(eval_df, "zipcode", frequency_map)

    assert frequency_map.to_dict() == {1: 2, 2: 1}
    assert result["zipcode_freq"].to_list() == [2, 0]


def test_apply_frequency_encoder_skips_missing_column():
    dataset = pd.DataFrame({"price": [100000]})
    frequency_map = pd.Series({1: 2})

    result = apply_frequency_encoder(dataset, "zipcode", frequency_map)

    pd.testing.assert_frame_equal(result, dataset)
    assert result is not dataset


def test_engineer_features_fits_on_train_and_drops_unused_columns():
    train_df = make_feature_dataset()
    eval_df = make_feature_dataset().assign(
        date=["2021-01-01", "2021-02-01", "2021-03-01"],
        zipcode=[30301, 99999, 33101],
        city_full=["atlanta", "unknown", "miami"],
    )
    holdout_df = make_feature_dataset().assign(
        date=["2022-01-01", "2022-02-01", "2022-03-01"],
        zipcode=[99999, 30301, 33101],
        city_full=["unknown", "atlanta", "miami"],
    )

    train_result, eval_result, holdout_result, frequency_map, target_encoder = (
        engineer_features(train_df, eval_df, holdout_df)
    )

    assert frequency_map is not None
    assert target_encoder is not None
    assert eval_result["zipcode_freq"].to_list() == [2, 0, 1]
    assert holdout_result["zipcode_freq"].to_list() == [0, 2, 1]

    for result in (train_result, eval_result, holdout_result):
        assert "zipcode_freq" in result.columns
        assert "city_full_encoded" in result.columns
        for dropped_column in feature_config.drop_columns:
            assert dropped_column not in result.columns


def test_drop_unused_columns_ignores_missing_columns():
    dataset = pd.DataFrame({"price": [100000], "city": ["ATL"]})

    result = drop_unused_columns(dataset)

    assert result.columns.to_list() == ["price"]


def test_load_cleaned_splits_requires_all_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing cleaned split files"):
        load_cleaned_splits(input_dir=tmp_path)


def test_run_feature_engineering_saves_outputs_and_encoders(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    encoders_dir = tmp_path / "models"
    input_dir.mkdir()
    dataset = make_feature_dataset()

    for split in feature_config.splits:
        dataset.to_csv(input_dir / cleaned_split_filename(split), index=False)

    train_df, eval_df, holdout_df, frequency_map, target_encoder = (
        run_feature_engineering(
            input_dir=input_dir,
            output_dir=output_dir,
            encoders_dir=encoders_dir,
            register_artifact=False,
        )
    )

    assert train_df.shape == eval_df.shape == holdout_df.shape
    assert frequency_map is not None
    assert target_encoder is not None

    for split in feature_config.splits:
        assert (output_dir / feature_engineered_split_filename(split)).exists()

    assert (encoders_dir / feature_config.frequency_encoder_filename).exists()
    assert (encoders_dir / feature_config.target_encoder_filename).exists()


def test_register_feature_engineering_artifacts_requires_engineered_files(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="Missing feature-engineered split files",
    ):
        register_feature_engineering_artifacts(
            data_dir=tmp_path,
            models_dir=tmp_path,
        )
