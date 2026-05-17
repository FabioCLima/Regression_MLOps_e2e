# Regression MLOps E2E

Projeto educacional de engenharia de machine learning para um problema de
regressão imobiliária. A ideia é construir um fluxo ponta a ponta com boas
práticas de MLOps: dados versionados, pipelines separados, logs, testes,
feature engineering reproduzível, treinamento, tuning, inferência e preparação
para deploy.

Este projeto não é apenas um script de treino. Ele foi organizado para servir
como guia de estudo para um engenheiro de ML que quer entender como um pipeline
evolui de código local para uma arquitetura mais próxima de produção.

## O Que Este Projeto Faz

Fluxo principal:

```text
raw dataset
  -> split temporal
  -> preprocessing
  -> feature engineering
  -> treinamento de modelos candidatos
  -> seleção do melhor modelo
  -> fine tuning
  -> inferência
```

Principais ferramentas:

- `pandas` para manipulação de dados;
- `scikit-learn` para modelos baseline e métricas;
- `xgboost` para modelo gradient boosting;
- `optuna` para fine tuning;
- `wandb` para versionamento de artefatos e métricas;
- `loguru` para logging;
- `pydantic` para configuração;
- `pytest` para testes;
- `ruff` para lint.

## Estrutura Do Projeto

```text
src/
  config.py
  logging_config.py
  data/
    load_data.py
    split_data.py
    preprocess_data.py
  features/
    feature_engineering.py
  models/
    train_model.py
    tune_model.py
  inference/
    predict.py

pipelines/
  data_pipeline.py
  feature_pipeline.py
  training_pipeline.py
  tuning_pipeline.py
  inference_pipeline.py
  machine_learning_pipeline.py

tests/
docs/
data/
models/
logs/
```

Regra mental:

- `src/` contém código reutilizável e testável;
- `pipelines/` contém a ordem de execução;
- `tests/` protege o comportamento esperado;
- `docs/` registra decisões técnicas e de negócio;
- `data/`, `models/` e `logs/` são diretórios de artefatos locais.

## Preparação Do Ambiente

Este projeto usa `uv`.

Instale as dependências:

```bash
uv sync
```

Configure o W&B, se ainda não estiver autenticado:

```bash
uv run wandb login
```

O dataset bruto esperado está em:

```text
data/raw_data/HouseTS.csv
```

## Como Executar Por Etapas

Para estudo, o mais recomendado é executar etapa por etapa. Isso ajuda a
entender o que cada pipeline produz e evita rodar treinamento ou tuning sem
necessidade.

### 1. Pipeline De Dados

```bash
uv run data-pipeline
```

Responsabilidades:

- usa o dataset bruto registrado no W&B;
- faz split temporal em train/eval/holdout;
- aplica preprocessing;
- salva arquivos em `data/processed/`;
- registra artefatos no W&B.

Arquivos esperados:

```text
data/processed/train.csv
data/processed/eval.csv
data/processed/holdout.csv
data/processed/cleaning_train.csv
data/processed/cleaning_eval.csv
data/processed/cleaning_holdout.csv
```

### 2. Pipeline De Features

```bash
uv run feature-pipeline
```

Responsabilidades:

- cria features de data;
- aplica frequency encoding;
- aplica target encoding usando apenas o treino;
- salva encoders em `models/`;
- salva datasets feature-engineered;
- registra artefatos no W&B.

Arquivos esperados:

```text
data/processed/feature_engineered_train.csv
data/processed/feature_engineered_eval.csv
data/processed/feature_engineered_holdout.csv
models/zipcode_frequency_encoder.pkl
models/city_full_target_encoder.pkl
```

### 3. Pipeline De Treinamento

```bash
uv run training-pipeline
```

Responsabilidades:

- treina modelos candidatos;
- calcula métricas em train e eval;
- calcula gap de generalização;
- escolhe o melhor modelo por RMSE no eval;
- salva apenas o melhor modelo;
- registra o melhor modelo no W&B.

Modelos candidatos:

- Dummy Regressor;
- Linear Regression;
- Ridge;
- Random Forest;
- XGBoost.

Arquivos esperados:

```text
models/best_model.pkl
models/metrics.json
```

### 4. Pipeline De Fine Tuning

```bash
uv run tuning-pipeline
```

Responsabilidades:

- lê qual foi o melhor modelo da etapa anterior;
- executa Optuna se o melhor modelo suportado for XGBoost;
- salva modelo tunado;
- salva histórico dos trials;
- registra o modelo tunado no W&B.

Arquivos esperados:

```text
models/xgboost_tuned_model.pkl
models/tuning_metrics.json
models/optuna_trials.csv
```

### 5. Pipeline De Inferência

```bash
uv run inference-pipeline --input data/processed/holdout.csv
```

Responsabilidades:

- recebe um CSV compatível com dados raw;
- aplica preprocessing;
- aplica feature engineering usando encoders salvos;
- alinha colunas com o schema de treino;
- carrega o modelo tunado, se existir;
- caso contrário, usa `models/best_model.pkl`;
- salva predições.

Arquivos esperados:

```text
data/processed/predictions.csv
data/processed/inference_metrics.json
```

O arquivo de métricas só é criado se o input tiver a coluna alvo `price`.

## Como Executar O Pipeline Ponta A Ponta

Use o orquestrador central somente quando quiser rodar o fluxo completo.

```bash
uv run machine-learning-pipeline
```

Por padrão, esse comando executa:

```text
data_pipeline
  -> feature_pipeline
  -> training_pipeline
  -> tuning_pipeline
```

Para incluir inferência no final:

```bash
uv run machine-learning-pipeline --include-inference
```

Para pular etapas já executadas:

```bash
uv run machine-learning-pipeline --skip-data --skip-features
```

Para rodar apenas tuning depois de dados, features e treino já existirem:

```bash
uv run machine-learning-pipeline \
  --skip-data \
  --skip-features \
  --skip-training
```

Importante: o pipeline completo pode treinar modelos, executar tuning e registrar
artefatos no W&B. Para estudar o projeto, prefira rodar etapa por etapa.

## Logging

Todos os pipelines usam Loguru.

Logs são salvos em:

```text
logs/
```

O orquestrador central usa:

```text
logs/machine_learning_pipeline.log
```

Os pipelines individuais usam:

```text
logs/pipeline.log
```

## W&B

O W&B é usado para versionar:

- raw dataset;
- splits processados;
- splits limpos;
- datasets com features;
- encoders;
- melhor modelo baseline;
- modelo tunado.

Antes de executar pipelines que registram artefatos, confirme que você está
logado:

```bash
uv run wandb login
```

## Testes E Qualidade

Rodar testes:

```bash
uv run pytest
```

Rodar lint:

```bash
uv run ruff check pipelines src tests
```

Esses comandos devem ser executados antes de mudanças maiores ou antes de
preparar um deploy.

## Nota Sobre Validação Temporal

Este projeto usa dados com componente temporal. Por isso, validações aleatórias
tradicionais, como `KFold` comum, podem causar vazamento temporal ao permitir
que informações do futuro influenciem direta ou indiretamente o treinamento.

Nesta versão do pipeline, usamos uma separação temporal simples:

- Train: dados antes de `2020-01-01`;
- Eval: dados de `2020-01-01` até antes de `2022-01-01`;
- Holdout: dados a partir de `2022-01-01`.

A comparação inicial de overfitting/underfitting é feita medindo métricas no
train e no eval para cada modelo candidato, além do gap entre esses conjuntos.
Uma estratégia mais robusta de validação temporal, como backtesting ou
`TimeSeriesSplit` adaptado para múltiplas regiões, deve ser tratada antes de uma
decisão final de produção.

## Próximo Passo Para Deploy

O próximo passo natural é transformar a inferência em um serviço.

Sugestão de evolução:

```text
src/inference/predict.py
  -> src/api/app.py
  -> Dockerfile
  -> AWS ECR
  -> AWS ECS/Fargate ou SageMaker Endpoint
```

Antes do deploy, ainda é importante:

- definir contrato de entrada da inferência;
- validar schema dos dados recebidos;
- criar uma API com FastAPI;
- containerizar o projeto;
- configurar CI com lint e testes;
- decidir onde o modelo aprovado será carregado, por exemplo W&B ou S3;
- adicionar logs e monitoramento do serviço em produção.

Para este projeto como guia de estudo, uma boa próxima etapa é:

```text
FastAPI + Docker + AWS ECS/Fargate
```

Depois, o projeto pode evoluir para uma arquitetura mais especializada com
SageMaker.
