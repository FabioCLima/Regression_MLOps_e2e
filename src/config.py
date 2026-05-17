from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectPaths(BaseModel):
    """Centralizes the main project paths used by the ML pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    project_dir: Path = PROJECT_ROOT

    @computed_field
    @property
    def src_dir(self) -> Path:
        return self.project_dir / "src"

    @computed_field
    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @computed_field
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw_data"

    @computed_field
    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field
    @property
    def models_dir(self) -> Path:
        return self.project_dir / "models"

    @computed_field
    @property
    def logs_dir(self) -> Path:
        return self.project_dir / "logs"

    @computed_field
    @property
    def tests_dir(self) -> Path:
        return self.project_dir / "tests"

    def create_runtime_dirs(self) -> None:
        """Creates directories that the pipeline may write to during execution."""
        for directory in (
            self.processed_data_dir,
            self.models_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class WandbConfig(BaseModel):
    """Centralizes W&B project and artifact settings."""

    model_config = ConfigDict(frozen=True)

    project_name: str = "regression-mlops-e2e"
    raw_dataset_artifact_name: str = "house_ts_raw"
    processed_dataset_artifact_name: str = "house_ts_processed"
    cleaned_dataset_artifact_name: str = "house_ts_cleaned"
    feature_engineered_dataset_artifact_name: str = "house_ts_features"
    feature_encoders_artifact_name: str = "house_ts_feature_encoders"
    best_model_artifact_name: str = "house_ts_best_model"
    tuned_model_artifact_name: str = "house_ts_tuned_model"


class SplitConfig(BaseModel):
    """Centralizes the dataset split settings."""

    model_config = ConfigDict(frozen=True)

    date_column: str = "date"
    eval_start_date: str = "2020-01-01"
    holdout_start_date: str = "2022-01-01"
    train_filename: str = "train.csv"
    eval_filename: str = "eval.csv"
    holdout_filename: str = "holdout.csv"


class PreprocessingConfig(BaseModel):
    """Centralizes the dataset preprocessing settings."""

    model_config = ConfigDict(frozen=True)

    splits: tuple[str, ...] = ("train", "eval", "holdout")
    city_column: str = "city_full"
    metro_column: str = "metro_full"
    latitude_column: str = "lat"
    longitude_column: str = "lng"
    outlier_column: str = "median_list_price"
    max_median_list_price: int = 19_000_000
    duplicate_ignore_columns: tuple[str, ...] = ("date", "year")
    cleaned_filename_prefix: str = "cleaning"
    metros_filename: str = "usmetros.csv"
    city_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
            "denver-aurora-lakewood": "denver-aurora-centennial",
            "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
            "austin-round rock-georgetown": "austin-round rock-san marcos",
            "miami-fort lauderdale-pompano beach": (
                "miami-fort lauderdale-west palm beach"
            ),
            "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
            "dc_metro": "washington-arlington-alexandria",
            "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
        }
    )


class FeatureEngineeringConfig(BaseModel):
    """Centralizes feature engineering settings."""

    model_config = ConfigDict(frozen=True)

    splits: tuple[str, ...] = ("train", "eval", "holdout")
    target_column: str = "price"
    date_column: str = "date"
    frequency_encode_column: str = "zipcode"
    target_encode_column: str = "city_full"
    frequency_encoder_filename: str = "zipcode_frequency_encoder.pkl"
    target_encoder_filename: str = "city_full_target_encoder.pkl"
    output_filename_prefix: str = "feature_engineered"
    drop_columns: tuple[str, ...] = (
        "date",
        "city_full",
        "city",
        "zipcode",
        "median_sale_price",
    )


class ModelTrainingConfig(BaseModel):
    """Centralizes model training settings."""

    model_config = ConfigDict(frozen=True)

    target_column: str = "price"
    candidate_models: tuple[str, ...] = (
        "dummy",
        "linear_regression",
        "ridge",
        "random_forest",
        "xgboost",
    )
    primary_metric: str = "rmse"
    random_state: int = 42
    sample_frac: float | None = None
    metrics_filename: str = "metrics.json"
    best_model_filename: str = "best_model.pkl"
    model_filename_template: str = "{model_name}.pkl"
    xgboost_n_estimators: int = 500
    random_forest_n_estimators: int = 200


class TuningConfig(BaseModel):
    """Centralizes hyperparameter tuning settings."""

    model_config = ConfigDict(frozen=True)

    supported_model_name: str = "xgboost"
    n_trials: int = 15
    direction: str = "minimize"
    random_state: int = 42
    sample_frac: float | None = None
    tuned_model_filename: str = "xgboost_tuned_model.pkl"
    tuning_metrics_filename: str = "tuning_metrics.json"
    trials_filename: str = "optuna_trials.csv"


class InferenceConfig(BaseModel):
    """Centralizes inference settings."""

    model_config = ConfigDict(frozen=True)

    predictions_filename: str = "predictions.csv"
    prediction_column: str = "predicted_price"
    actual_column: str = "actual_price"
    prediction_error_column: str = "prediction_error"
    inference_metrics_filename: str = "inference_metrics.json"


paths = ProjectPaths()
wandb_config = WandbConfig()
split_config = SplitConfig()
preprocessing_config = PreprocessingConfig()
feature_config = FeatureEngineeringConfig()
model_config = ModelTrainingConfig()
tuning_config = TuningConfig()
inference_config = InferenceConfig()
