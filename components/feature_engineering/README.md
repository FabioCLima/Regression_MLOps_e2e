# feature_engineering

Applies feature transformations to the cleaned splits. Encoders are fitted
**only on the train split** and then applied to eval and holdout — preventing data leakage.

## What it does

1. Downloads `house_ts_cleaned:latest` from W&B
2. Adds date features: `year`, `quarter`, `month`
3. Fits frequency encoder on train `zipcode`, applies to all splits
4. Fits target encoder on train `city_full`, applies to all splits
5. Drops raw columns that would leak information (`date`, `city_full`, `zipcode`, etc.)
6. Saves 3 engineered CSVs + 2 encoder `.pkl` files as artifacts

## Leakage-safe design

Encoders are fitted exclusively on the train split and saved as artifacts alongside
the datasets. The encoder artifacts are what makes the inference pipeline reproducible —
at serving time, the same encoders that were used at training time are loaded.

## Files

| File | Role |
|---|---|
| `run.py` | Component logic + argparse entry point |
| `MLproject` | MLflow contract: parameters and command |
| `python_env.yaml` | Isolated virtualenv for this component |

## Running locally

```bash
python components/feature_engineering/run.py \
  --input_artifact house_ts_cleaned:latest \
  --output_artifact house_ts_features \
  --output_type feature_dataset \
  --output_description "Feature-engineered splits with fitted encoders" \
  --pipeline_run_id local-test-001
```

## Running via MLflow

```bash
mlflow run components/feature_engineering \
  -P input_artifact=house_ts_cleaned:latest \
  -P output_artifact=house_ts_features \
  -P output_type=feature_dataset \
  -P output_description="Feature-engineered splits with fitted encoders" \
  --env-manager virtualenv
```

## Environment variables required

```bash
export WANDB_API_KEY=<your-key>
export MLFLOW_TRACKING_URI=<uri>   # default: local ./mlruns
```
