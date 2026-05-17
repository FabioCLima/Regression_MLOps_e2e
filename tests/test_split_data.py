import pandas as pd
import pytest

from src.config import split_config
from src.data.split_data import (
    load_and_split_data,
    register_data_splits,
    split_dataset_by_date,
)


def test_split_dataset_by_date_returns_expected_splits():
    dataset = pd.DataFrame(
        {
            "date": [
                "2022-03-01",
                "2019-12-31",
                "2021-06-01",
                "2020-01-01",
            ],
            "price": [400000, 100000, 300000, 200000],
        }
    )

    train_df, eval_df, holdout_df = split_dataset_by_date(dataset)

    assert train_df["date"].max() < pd.Timestamp("2020-01-01")
    assert eval_df["date"].min() >= pd.Timestamp("2020-01-01")
    assert eval_df["date"].max() < pd.Timestamp("2022-01-01")
    assert holdout_df["date"].min() >= pd.Timestamp("2022-01-01")

    assert train_df.shape == (1, 2)
    assert eval_df.shape == (2, 2)
    assert holdout_df.shape == (1, 2)


def test_split_dataset_by_date_requires_date_column():
    dataset = pd.DataFrame({"price": [100000, 200000]})

    with pytest.raises(ValueError, match="date"):
        split_dataset_by_date(dataset)


def test_load_and_split_data_saves_split_files(tmp_path):
    raw_path = tmp_path / "raw.csv"
    output_dir = tmp_path / "processed"
    dataset = pd.DataFrame(
        {
            "date": ["2019-01-01", "2020-06-01", "2022-02-01"],
            "price": [100000, 200000, 300000],
        }
    )
    dataset.to_csv(raw_path, index=False)

    train_df, eval_df, holdout_df = load_and_split_data(
        raw_path=raw_path,
        output_dir=output_dir,
        register_artifact=False,
    )

    assert train_df.shape == (1, 2)
    assert eval_df.shape == (1, 2)
    assert holdout_df.shape == (1, 2)

    assert (output_dir / split_config.train_filename).exists()
    assert (output_dir / split_config.eval_filename).exists()
    assert (output_dir / split_config.holdout_filename).exists()


def test_register_data_splits_requires_split_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing split files"):
        register_data_splits(output_dir=tmp_path)
