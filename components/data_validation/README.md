# data_validation

Validates the schema and data quality of the raw dataset before any transformation.
Raises an error and stops the pipeline if validation fails.

## What it does

1. Downloads `house_ts_raw:latest` from W&B
2. Checks: expected columns present, no nulls in critical columns, price within valid range, date column parseable
3. Logs pass/fail metrics to MLflow
4. Logs a `validation_report.json` artifact to both W&B and MLflow

## MLflow metrics logged

| Metric | Meaning |
|---|---|
| `schema_valid` | 1 if all expected columns are present |
| `total_nulls` | Total null count across critical columns |
| `price_out_of_range` | Rows with price < 50k or > 50M |
| `date_format_valid` | 1 if date column is parseable |
| `validation_passed` | 1 if all checks pass |

## Files

| File | Role |
|---|---|
| `run.py` | Component logic + argparse entry point |
| `MLproject` | MLflow contract: parameters and command |
| `python_env.yaml` | Isolated virtualenv for this component |

## Running locally

```bash
python components/data_validation/run.py \
  --input_artifact house_ts_raw:latest \
  --output_artifact house_ts_validation \
  --output_type validation_report \
  --output_description "Schema and quality validation report" \
  --pipeline_run_id local-test-001
```

## Running via MLflow

```bash
mlflow run components/data_validation \
  -P input_artifact=house_ts_raw:latest \
  -P output_artifact=house_ts_validation \
  -P output_type=validation_report \
  -P output_description="Schema and quality validation report" \
  --env-manager virtualenv
```

## Environment variables required

```bash
export WANDB_API_KEY=<your-key>
export MLFLOW_TRACKING_URI=<uri>   # default: local ./mlruns
```
