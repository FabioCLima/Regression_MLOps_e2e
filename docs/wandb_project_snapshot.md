# W&B Project Snapshot — `regression-mlops-e2e`

> Gerado em: 2026-05-16
> Run de referência: `wise-fog-26` (tuning) / `devoted-aardvark-24` (training)

---

## Projeto

| Campo | Valor |
|---|---|
| **Nome** | regression-mlops-e2e |
| **Versão** | 0.1.0 |
| **W&B Entity** | fabio_lima07-mlops |
| **W&B Project** | regression-mlops-e2e |
| **Objetivo** | Pipeline MLOps completo para regressão de preço imobiliário |
| **Dataset** | HouseTS.csv — séries temporais de preço de imóveis (EUA) |
| **Target** | `price` (preço do imóvel) |
| **Métrica primária** | RMSE |
| **Total de runs** | 26 (25 finished) |
| **Python** | 3.13.12 |
| **Gerenciador de dependências** | uv (uv.lock com versões pinadas) |

---

## Pipelines

| Pipeline | Entry Point | Responsabilidade |
|---|---|---|
| `data-pipeline` | `pipelines.data_pipeline` | Ingestão → validação → pré-processamento |
| `feature-pipeline` | `pipelines.feature_pipeline` | Feature engineering + encoders |
| `training-pipeline` | `pipelines.training_pipeline` | Treino de candidatos + seleção do melhor |
| `tuning-pipeline` | `pipelines.tuning_pipeline` | Otimização de hiperparâmetros (Optuna) |
| `inference-pipeline` | `pipelines.inference_pipeline` | Inferência em holdout |
| `machine-learning-pipeline` | `pipelines.machine_learning_pipeline` | Orquestrador end-to-end |

---

## Artifacts e Lineage

| Artifact | Versão | Tipo | Tamanho | Produzido por | Consome | Aliases |
|---|---|---|---|---|---|---|
| `house_ts_raw` | v0 | dataset | 272 MB | `data-pipeline` | — (fonte local: `HouseTS.csv`) | latest |
| `house_ts_processed` | v0 | dataset | — | `data-pipeline` | `house_ts_raw:v0` | latest |
| `house_ts_cleaned` | v1 | dataset | — | `data-pipeline` | `house_ts_processed:v0` | latest |
| `house_ts_feature_encoders` | v0 | model | ~200 KB | `feature-pipeline` | `house_ts_cleaned:v1` | latest |
| `house_ts_features` | v1 | dataset | ~256 MB | `feature-pipeline` | `house_ts_cleaned:v1` + `house_ts_feature_encoders:v0` | latest |
| `house_ts_trained_models` | v0 | model | — | `training-pipeline` | `house_ts_features:v0` | latest |
| `house_ts_best_model` | v1 | model | — | `training-pipeline` | `house_ts_features:v1` | latest, best, best-rmse, xgboost |
| `house_ts_tuned_model` | v0 | model | 14,4 MB | `tuning-pipeline` | `house_ts_best_model:v1` + `house_ts_features:v1` | latest, tuned, tuned-rmse, xgboost |

### Lineage

```
HouseTS.csv (local)
  └─► house_ts_raw:v0
        └─► house_ts_processed:v0
              └─► house_ts_cleaned:v1
                    ├─► house_ts_feature_encoders:v0
                    └─► house_ts_features:v1
                          ├─► house_ts_trained_models:v0
                          ├─► house_ts_best_model:v1  (XGBoost selecionado)
                          └─► house_ts_tuned_model:v0  ◄── MELHOR MODELO
```

---

## Melhor Modelo — `house_ts_tuned_model:v0`

### Comparação de candidatos (eval set)

| Modelo | RMSE | R² | MAE |
|---|---|---|---|
| **XGBoost Tuned** ✅ | **69.483** | **0.963** | **31.609** |
| XGBoost Base | 73.824 | 0.958 | 33.034 |
| Random Forest | 83.879 | 0.946 | 34.272 |
| Ridge | 121.638 | 0.886 | 54.078 |
| Linear Regression | 121.635 | 0.886 | 54.078 |
| Dummy | 375.960 | -0.093 | 215.535 |

### Métricas por split

| Split | RMSE | R² | MAE |
|---|---|---|---|
| Train | 8.165 | 0.999 | 5.643 |
| Eval | 69.483 | 0.963 | 31.609 |
| Gap (eval − train) | 61.318 | -0.037 | 25.966 |

### Hiperparâmetros (Optuna, 15 trials)

| Parâmetro | Valor |
|---|---|
| `n_estimators` | 574 |
| `max_depth` | 9 |
| `learning_rate` | 0.1478 |
| `subsample` | 0.9974 |
| `colsample_bytree` | 0.6217 |
| `min_child_weight` | 7 |
| `gamma` | 4.817 |
| `reg_alpha` | 0.2105 |
| `reg_lambda` | ~0.0 |
| `random_state` | 42 |
| `tree_method` | hist |
| `objective` | reg:squarederror |

### Feature Engineering aplicada

| Transformação | Coluna |
|---|---|
| Frequency Encoding | `zipcode` |
| Target Encoding | `city_full` |
| Removidas | `date`, `city_full`, `city`, `zipcode`, `median_sale_price` |
| Target | `price` |

### Split temporal dos dados

| Split | Período |
|---|---|
| Train | até 2019-12-31 |
| Eval | 2020-01-01 → 2021-12-31 |
| Holdout | >= 2022-01-01 |

---

## Dependências críticas (uv.lock)

| Biblioteca | Versão |
|---|---|
| xgboost | 3.2.0 |
| scikit-learn | 1.8.0 |
| optuna | 4.8.0 |
| pandas | 3.0.3 |
| numpy | 2.4.5 |
| category-encoders | 2.9.0 |
| wandb | 0.27.0 |
