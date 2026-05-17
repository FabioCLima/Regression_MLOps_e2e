from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Raw input for a single prediction. Mirrors the HouseTS.csv schema.

    Required fields are the minimum needed for preprocessing to run.
    Optional fields improve prediction quality — missing ones are filled with 0
    after feature engineering (align_features_to_training_schema).
    """

    model_config = ConfigDict(extra="forbid")

    # Required
    date: str = Field(..., examples=["2023-01-15"])
    city_full: str = Field(..., examples=["austin-round rock-san marcos"])
    city: str = Field(..., examples=["Austin"])
    zipcode: int = Field(..., examples=[78701])

    # Market metrics — optional
    median_sale_price: float | None = None
    median_list_price: float | None = None
    median_ppsf: float | None = None
    median_list_ppsf: float | None = None
    homes_sold: float | None = None
    pending_sales: float | None = None
    new_listings: float | None = None
    inventory: float | None = None
    median_dom: float | None = None
    avg_sale_to_list: float | None = None
    sold_above_list: float | None = None
    off_market_in_two_weeks: float | None = None

    # POI counts — optional
    bank: float | None = None
    bus: float | None = None
    hospital: float | None = None
    mall: float | None = None
    park: float | None = None
    restaurant: float | None = None
    school: float | None = None
    station: float | None = None
    supermarket: float | None = None

    # Demographics — optional (snake_case in API; mapped to original names in to_raw_dict)
    total_population: float | None = None
    median_age: float | None = None
    per_capita_income: float | None = None
    total_families_below_poverty: float | None = None
    total_housing_units: float | None = None
    median_rent: float | None = None
    median_home_value: float | None = None
    total_labor_force: float | None = None
    unemployed_population: float | None = None
    total_school_age_population: float | None = None
    total_school_enrollment: float | None = None
    median_commute_time: float | None = None

    # Target (only used for evaluation — ignored in inference)
    price: float | None = None

    def to_raw_dict(self) -> dict:
        """Converts to a dict using the original column names (with spaces) expected by the pipeline."""
        d = self.model_dump(exclude_none=True)
        alias_map = {
            "total_population": "Total Population",
            "median_age": "Median Age",
            "per_capita_income": "Per Capita Income",
            "total_families_below_poverty": "Total Families Below Poverty",
            "total_housing_units": "Total Housing Units",
            "median_rent": "Median Rent",
            "median_home_value": "Median Home Value",
            "total_labor_force": "Total Labor Force",
            "unemployed_population": "Unemployed Population",
            "total_school_age_population": "Total School Age Population",
            "total_school_enrollment": "Total School Enrollment",
            "median_commute_time": "Median Commute Time",
        }
        for snake, original in alias_map.items():
            if snake in d:
                d[original] = d.pop(snake)
        return d


class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str
    missing_features: list[str] = Field(default_factory=list)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    rows_predicted: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    encoders_loaded: bool
    model_version: str | None
    n_features_expected: int | None


class ModelInfoResponse(BaseModel):
    version_string: str
    source: str
    loaded_at: datetime
    n_features_expected: int
    train_columns: list[str]
