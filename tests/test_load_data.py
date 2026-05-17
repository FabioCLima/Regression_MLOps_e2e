import pandas as pd
import pytest

from src.data.load_data import load_raw_dataset


def test_load_raw_dataset_returns_dataframe(tmp_path):
    data_path = tmp_path / "sample.csv"
    expected_dataset = pd.DataFrame(
        {
            "date": ["2012-03-31", "2012-04-30"],
            "price": [200773.99, 202421.06],
            "city": ["ATL", "ATL"],
        }
    )
    expected_dataset.to_csv(data_path, index=False)

    dataset = load_raw_dataset(data_path)

    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == (2, 3)
    assert dataset.columns.to_list() == ["date", "price", "city"]


def test_load_raw_dataset_raises_error_when_file_does_not_exist(tmp_path):
    missing_data_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        load_raw_dataset(missing_data_path)
