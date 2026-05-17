# preprocessing

Applies temporal split and preprocessing transformations to the raw housing dataset.
The split boundaries are key business decisions and are explicitly logged in MLflow.

## What it does

1. Downloads `house_ts_raw:latest` from W&B (contains HouseTS.csv + usmetros.csv)
2. Splits the dataset by date into train / eval / holdout
3. For each split: normalizes city names, merges lat/lng from metros data, removes duplicates and price outliers
4. Saves 3 cleaned CSVs and logs them to W&B and MLflow

## Business decisions captured as MLflow params

| Param | Default | Meaning |
|---|---|---|
| `eval_start_date` | 2020-01-01 | Start of the evaluation period |
| `holdout_start_date` | 2022-01-01 | Start of the holdout (final test) period |

These dates are **trackable** — given any model artifact, you can look up its preprocessing run
and see exactly which temporal boundaries were used.

## Files

| File | Role |
|---|---|
| `run.py` | Component logic + argparse entry point |
| `MLproject` | MLflow contract: parameters and command |
| `python_env.yaml` | Isolated virtualenv for this component |

## Running locally

```bash
python components/preprocessing/run.py \
  --input_artifact house_ts_raw:latest \
  --output_artifact house_ts_cleaned \
  --output_type cleaned_dataset \
  --output_description "Temporally split and cleaned dataset" \
  --eval_start_date 2020-01-01 \
  --holdout_start_date 2022-01-01 \
  --pipeline_run_id local-test-001
```

## Running via MLflow

```bash
mlflow run components/preprocessing \
  -P input_artifact=house_ts_raw:latest \
  -P output_artifact=house_ts_cleaned \
  -P output_type=cleaned_dataset \
  -P output_description="Temporally split and cleaned dataset" \
  -P eval_start_date=2020-01-01 \
  -P holdout_start_date=2022-01-01 \
  --env-manager virtualenv
```

## Environment variables required

```bash
export WANDB_API_KEY=<your-key>
export MLFLOW_TRACKING_URI=<uri>   # default: local ./mlruns
```
