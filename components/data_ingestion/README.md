# data_ingestion

Reads the raw source files and logs them as versioned artifacts in W&B and MLflow.
This is the first component in the pipeline — it has no upstream artifact dependency.

## What it does

1. Reads `HouseTS.csv` (house time series) and `usmetros.csv` (geospatial reference)
2. Logs file shapes and paths as MLflow params
3. Creates a W&B artifact with both files attached
4. Logs both files as MLflow artifacts

## Files

| File | Role |
|---|---|
| `run.py` | Component logic + argparse entry point |
| `MLproject` | MLflow contract: parameters and command |
| `python_env.yaml` | Isolated virtualenv for this component |

## Running locally

```bash
# From the project root
python components/data_ingestion/run.py \
  --raw_data data/raw_data/HouseTS.csv \
  --reference_data data/raw_data/usmetros.csv \
  --output_artifact house_ts_raw \
  --output_type raw_dataset \
  --output_description "Raw house time series + geospatial reference" \
  --pipeline_run_id local-test-001
```

## Running via MLflow

```bash
# From the project root
mlflow run components/data_ingestion \
  -P raw_data=data/raw_data/HouseTS.csv \
  -P reference_data=data/raw_data/usmetros.csv \
  -P output_artifact=house_ts_raw \
  -P output_type=raw_dataset \
  -P output_description="Raw house time series + geospatial reference" \
  --env-manager virtualenv
```

## Environment variables required

```bash
export WANDB_API_KEY=<your-key>
export MLFLOW_TRACKING_URI=<uri>   # default: local ./mlruns
```

## Output artifact

W&B artifact `house_ts_raw` (type: `raw_dataset`) containing:
- `HouseTS.csv`
- `usmetros.csv`

MLflow artifact path: `raw_data/`
