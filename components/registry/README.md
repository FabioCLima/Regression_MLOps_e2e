# registry

Registers the tuned model in the MLflow Model Registry, giving it a versioned,
stage-aware identity. This enables the formal promotion workflow:
`None → Staging → Production → Archived`.

## What it does

1. Downloads `house_ts_tuned_model:latest` from W&B
2. Logs the model to MLflow and registers it under `--model_name`
3. Transitions the new version to `--stage` (default: Staging)
4. Tags the MLflow run with the registered model URI for cross-tool traceability

## Answering "which version is in Production?"

```python
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions("house-price-regressor", stages=["Production"])
# → version.version, version.run_id, version.creation_timestamp
```

From the `run_id`, you can navigate back through the full pipeline lineage.

## Files

| File | Role |
|---|---|
| `run.py` | Component logic + argparse entry point |
| `MLproject` | MLflow contract: parameters and command |
| `python_env.yaml` | Isolated virtualenv for this component |

## Running locally

```bash
python components/registry/run.py \
  --input_model house_ts_tuned_model:latest \
  --model_name house-price-regressor \
  --stage Staging \
  --pipeline_run_id local-test-001
```

## Running via MLflow

```bash
mlflow run components/registry \
  -P input_model=house_ts_tuned_model:latest \
  -P model_name=house-price-regressor \
  -P stage=Staging \
  --env-manager virtualenv
```

## Promoting to Production

After reviewing the Staging model:

```bash
python components/registry/run.py \
  --input_model house_ts_tuned_model:latest \
  --model_name house-price-regressor \
  --stage Production
```

## Environment variables required

```bash
export WANDB_API_KEY=<your-key>
export MLFLOW_TRACKING_URI=<uri>   # default: local ./mlruns
```
