import pandas as pd
import pytest

from src.config import preprocessing_config
from src.data.preprocess_data import (
    clean_and_merge_metros,
    cleaned_split_filename,
    drop_duplicate_records,
    normalize_city,
    normalize_metro_name,
    preprocess_split,
    register_cleaned_splits,
    remove_price_outliers,
)


def test_normalize_city_handles_spaces_case_dashes_and_missing_values():
    assert normalize_city("  Atlanta—Sandy   Springs ") == "atlanta-sandy springs"
    assert pd.isna(normalize_city(pd.NA))


def test_normalize_metro_name_removes_state_suffix():
    assert normalize_metro_name(" Atlanta-Sandy Springs-Roswell, GA ") == (
        "atlanta-sandy springs-roswell"
    )
    assert normalize_metro_name("Washington-Arlington-Alexandria, DC-VA-MD-WV") == (
        "washington-arlington-alexandria"
    )


def test_clean_and_merge_metros_applies_city_mapping_and_merges_lat_lng(tmp_path):
    metros_path = tmp_path / "metros.csv"
    metros = pd.DataFrame(
        {
            "metro_full": [
                "Atlanta-Sandy Springs-Roswell, GA",
                "Atlanta-Sandy Springs-Roswell, GA",
            ],
            "lat": [33.75, 99.99],
            "lng": [-84.39, 99.99],
        }
    )
    metros.to_csv(metros_path, index=False)
    dataset = pd.DataFrame(
        {
            "city_full": ["Atlanta-Sandy Springs-Alpharetta"],
            "median_list_price": [500000],
        }
    )

    result = clean_and_merge_metros(dataset, metros_path=metros_path)

    assert result.shape[0] == 1
    assert result.loc[0, "city_full"] == "atlanta-sandy springs-roswell"
    assert result.loc[0, "lat"] == 33.75
    assert result.loc[0, "lng"] == -84.39


def test_clean_and_merge_metros_skips_when_city_column_is_missing(tmp_path):
    dataset = pd.DataFrame({"price": [100000]})

    result = clean_and_merge_metros(dataset, metros_path=tmp_path / "missing.csv")

    pd.testing.assert_frame_equal(result, dataset)


def test_clean_and_merge_metros_skips_when_lat_lng_already_exist(tmp_path):
    dataset = pd.DataFrame(
        {
            "city_full": ["Atlanta-Sandy Springs-Alpharetta"],
            "lat": [10.0],
            "lng": [20.0],
        }
    )

    result = clean_and_merge_metros(dataset, metros_path=tmp_path / "missing.csv")

    assert result.loc[0, "lat"] == 10.0
    assert result.loc[0, "lng"] == 20.0


def test_clean_and_merge_metros_skips_invalid_metros_file(tmp_path):
    metros_path = tmp_path / "metros.csv"
    pd.DataFrame({"metro_full": ["Atlanta"]}).to_csv(metros_path, index=False)
    dataset = pd.DataFrame({"city_full": ["Atlanta"]})

    result = clean_and_merge_metros(dataset, metros_path=metros_path)

    assert "lat" not in result.columns
    assert "lng" not in result.columns


def test_drop_duplicate_records_keeps_first_duplicate_record():
    dataset = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "year": [2020, 2020, 2020],
            "zipcode": [12345, 12345, 99999],
            "price": [100000, 100000, 300000],
        }
    )

    result = drop_duplicate_records(dataset)

    assert result.shape[0] == 2
    assert result["zipcode"].to_list() == [12345, 99999]


def test_drop_duplicate_records_returns_copy_when_only_ignored_columns_exist():
    dataset = pd.DataFrame({"date": ["2020-01-01"], "year": [2020]})

    result = drop_duplicate_records(dataset)

    assert result is not dataset
    pd.testing.assert_frame_equal(result, dataset)


def test_remove_price_outliers_removes_rows_above_threshold():
    dataset = pd.DataFrame(
        {
            "median_list_price": [
                preprocessing_config.max_median_list_price,
                preprocessing_config.max_median_list_price + 1,
            ]
        }
    )

    result = remove_price_outliers(dataset)

    assert result.shape[0] == 1
    assert result.iloc[0]["median_list_price"] == (
        preprocessing_config.max_median_list_price
    )


def test_remove_price_outliers_handles_non_numeric_values():
    dataset = pd.DataFrame(
        {"median_list_price": ["100000", "invalid", "20000000", None]}
    )

    result = remove_price_outliers(dataset)

    assert result["median_list_price"].iloc[:2].to_list() == ["100000", "invalid"]
    assert pd.isna(result["median_list_price"].iloc[2])


def test_remove_price_outliers_skips_when_column_is_missing():
    dataset = pd.DataFrame({"price": [100000]})

    result = remove_price_outliers(dataset)

    pd.testing.assert_frame_equal(result, dataset)
    assert result is not dataset


def test_preprocess_split_saves_cleaned_file(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "year": [2020],
            "city_full": ["Atlanta-Sandy Springs-Alpharetta"],
            "median_list_price": [500000],
        }
    ).to_csv(input_dir / "train.csv", index=False)

    result = preprocess_split(
        split="train",
        input_dir=input_dir,
        output_dir=output_dir,
        metros_path=None,
    )

    assert result.shape == (1, 4)
    assert (output_dir / cleaned_split_filename("train")).exists()


def test_preprocess_split_requires_input_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        preprocess_split(split="train", input_dir=tmp_path, metros_path=None)


def test_register_cleaned_splits_requires_cleaned_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing cleaned split files"):
        register_cleaned_splits(output_dir=tmp_path)
