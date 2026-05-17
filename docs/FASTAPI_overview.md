---
title: FastAPI Overview para Backend de ML
aliases:
  - FastAPI Overview
  - FastAPI para ML Engineering
  - Backend ML com FastAPI
created: 2026-05-16
type: nota
topic:
  - fastapi
  - backend
  - mlops
  - model-serving
  - deployment
project: regression-mlops-e2e
tags:
  - fastapi
  - mlops
  - backend
  - model-serving
  - machine-learning
  - deployment
status: draft
---

# FastAPI Overview para Backend de ML

> [!summary]
> Esta nota explica, de forma didatica, como pensar e implementar um backend com FastAPI para servir um modelo de regressao em um projeto de [[MLOps]]. O foco e manter tudo simples, funcional e alinhado com boas praticas de engenharia de software e deployment de ML.

## Contexto do projeto

Este projeto ja tem uma base importante de ML Engineering:

- codigo reutilizavel em `src/`;
- pipelines executaveis em `pipelines/`;
- configuracao centralizada em `src/config.py`;
- logging com Loguru;
- versionamento de datasets, features, encoders e modelos no W&B;
- testes unitarios e smoke tests;
- funcao de inferencia em `src/inference/predict.py`.

A lacuna atual e a camada de serving.

Hoje, o projeto consegue rodar inferencia por script/pipeline:

```text
CSV -> run_inference_pipeline -> predictions.csv
```

O backend com FastAPI transforma isso em um servico HTTP:

```text
cliente -> POST /predict -> FastAPI -> predict() -> JSON com predicao
```

> [!important]
> O objetivo da API nao e reimplementar a logica de ML. A API deve ser uma camada fina que valida entrada, chama a inferencia existente e retorna uma resposta clara.

---

## Mapa da nota

- [[#Por que precisamos de uma API]]
- [[#O que e FastAPI]]
- [[#Conceitos minimos]]
- [[#Arquitetura recomendada para este projeto]]
- [[#Contrato da API]]
- [[#Estrutura de arquivos recomendada]]
- [[#Fluxo de uma predicao]]
- [[#Boas praticas de engenharia]]
- [[#Boas praticas de deployment de ML]]
- [[#Plano simples de implementacao]]
- [[#Checklist de pronto]]

---

## Por que precisamos de uma API

Treinar um modelo nao significa que ele esta pronto para uso.

Um modelo salvo em `models/` ou registrado no W&B ainda nao e um produto. Para outro sistema usar esse modelo, ele precisa de uma interface estavel.

Sem API, o uso do modelo fica preso a:

- scripts locais;
- notebooks;
- execucoes manuais;
- arquivos CSV;
- conhecimento de quem implementou o pipeline.

Com API, qualquer cliente que saiba fazer uma chamada HTTP pode usar o modelo:

```text
Streamlit
outro backend
dashboard
notebook
servico externo
curl/Postman
```

Exemplo mental:

```text
Antes:
python pipelines/inference_pipeline.py --input data/processed/holdout.csv

Depois:
POST /predict
{
  "date": "2023-01-01",
  "city_full": "austin-round rock-san marcos",
  "zipcode": "78701",
  ...
}
```

Resposta:

```json
{
  "predicted_price": 485000.0,
  "model_name": "xgboost_tuned_model.pkl",
  "model_version": "local"
}
```

---

## O que e FastAPI

FastAPI e um framework Python para criar APIs HTTP.

Ele e muito usado em projetos de ML porque combina bem com:

- Python;
- Pydantic;
- pandas;
- scikit-learn;
- XGBoost;
- Docker;
- Uvicorn;
- documentacao automatica em `/docs`.

Com FastAPI, uma funcao Python vira um endpoint HTTP.

Exemplo simples:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}
```

Rodando a aplicacao:

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Depois disso:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

> [!tip]
> O endpoint `/docs` e uma das melhores partes do FastAPI para aprender. Ele cria uma documentacao interativa automaticamente a partir dos schemas e endpoints.

---

## Conceitos minimos

| Conceito | O que significa | Por que importa |
| --- | --- | --- |
| `FastAPI()` | Instancia principal da aplicacao. | E onde os endpoints sao registrados. |
| `@app.get()` | Endpoint de leitura. | Usado para raiz, health check e metadata. |
| `@app.post()` | Endpoint que recebe dados. | Usado para predicao. |
| Pydantic | Valida dados de entrada e saida. | Evita input quebrado chegando no modelo. |
| Uvicorn | Servidor ASGI. | Executa a aplicacao FastAPI. |
| Schema | Contrato da API. | Define o que entra e o que sai. |
| Endpoint | Rota HTTP. | Interface consumida por clientes. |
| Health check | Verifica se o servico esta pronto. | Essencial para deploy e monitoramento. |

---

## Arquitetura recomendada para este projeto

Este projeto ja separa bem logica e orquestracao:

```text
src/        -> logica reutilizavel
pipelines/  -> execucao/orquestracao
docs/       -> memoria tecnica
tests/      -> validacao automatizada
```

Para o backend, a recomendacao e adicionar uma camada `src/api/`:

```text
src/
  api/
    __init__.py
    app.py
    schemas.py
    service.py
    errors.py
  inference/
    predict.py
```

Responsabilidades:

| Arquivo | Responsabilidade |
| --- | --- |
| `src/api/app.py` | Criar o app FastAPI e declarar endpoints. |
| `src/api/schemas.py` | Definir contratos Pydantic de request/response. |
| `src/api/service.py` | Conectar request da API com `src.inference.predict.predict()`. |
| `src/api/errors.py` | Centralizar erros HTTP amigaveis. |
| `src/inference/predict.py` | Continuar sendo a logica de inferencia do modelo. |

> [!important]
> A API nao deve conter logica de feature engineering, preprocessing ou modelagem. Ela deve chamar as funcoes existentes. Isso preserva testabilidade e reduz duplicacao.

---

## Contrato da API

O contrato da API define o que o cliente precisa enviar e o que o backend promete devolver.

Para este projeto, a entrada deve representar uma observacao raw-compatible, ou seja, parecida com o dado que a funcao `predict()` ja sabe processar.

Exemplo conceitual de request:

```python
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    date: str = Field(..., examples=["2023-01-01"])
    city_full: str
    metro_full: str | None = None
    zipcode: str
    median_list_price: float | None = None
```

Exemplo conceitual de response:

```python
class PredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    model_version: str
```

Para batch prediction:

```python
class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    rows_predicted: int
```

> [!note]
> O schema exato deve ser derivado das colunas que `preprocess_dataset()` e `predict()` precisam receber. Nao chute o contrato final: confirme com os dados raw-compatible do projeto.

---

## Fluxo de uma predicao

Fluxo recomendado:

```text
1. Cliente envia JSON para POST /predict
2. FastAPI valida o JSON com Pydantic
3. API transforma o request em pandas DataFrame
4. service.py chama src.inference.predict.predict()
5. predict() aplica preprocessing, feature engineering e alinhamento de schema
6. modelo gera predicao
7. API transforma o resultado em response JSON
8. cliente recebe predicted_price e metadata minima
```

Representacao:

```text
JSON request
  -> Pydantic schema
  -> pandas DataFrame
  -> preprocess_dataset()
  -> apply_inference_feature_engineering()
  -> align_features_to_training_schema()
  -> model.predict()
  -> JSON response
```

O ponto mais importante e que o projeto atual ja tem a parte critica:

```python
from src.inference.predict import predict
```

A API deve apenas adaptar HTTP para essa funcao.

---

## Endpoints minimos

### `GET /`

Endpoint simples para confirmar que a API esta viva.

Resposta esperada:

```json
{
  "service": "regression-mlops-e2e-api",
  "status": "running"
}
```

### `GET /health`

Health check tecnico.

Deve informar:

- se a API esta viva;
- se o modelo existe/carregou;
- se os encoders existem;
- se o arquivo de schema de treino existe;
- versao basica da aplicacao.

Resposta exemplo:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "encoders_loaded": true,
  "training_schema_available": true
}
```

> [!important]
> Em deploy, `/health` e usado por Docker, load balancer, ECS, Kubernetes ou monitoramento. Ele nao e enfeite.

### `POST /predict`

Endpoint principal para uma predicao.

Entrada:

```json
{
  "date": "2023-01-01",
  "city_full": "austin-round rock-san marcos",
  "metro_full": "austin-round rock-san marcos",
  "zipcode": "78701",
  "median_list_price": 500000
}
```

Saida:

```json
{
  "predicted_price": 485000.0,
  "model_name": "xgboost_tuned_model.pkl",
  "model_version": "local"
}
```

### `POST /predict/batch`

Endpoint para varias predicoes.

Entrada:

```json
{
  "records": [
    {
      "date": "2023-01-01",
      "city_full": "austin-round rock-san marcos",
      "zipcode": "78701"
    },
    {
      "date": "2023-02-01",
      "city_full": "denver-aurora-centennial",
      "zipcode": "80202"
    }
  ]
}
```

Saida:

```json
{
  "rows_predicted": 2,
  "predictions": [
    {"predicted_price": 485000.0},
    {"predicted_price": 515000.0}
  ]
}
```

### `GET /model-info`

Endpoint util para debug e transparencia.

Pode retornar:

- nome do modelo;
- caminho do modelo;
- metrica principal;
- fonte do artifact;
- data de carga;
- numero de features esperadas.

---

## Como conectar com W&B

O projeto ja usa W&B para versionar datasets, features, encoders e modelos.

Para uma primeira versao simples, a API pode carregar arquivos locais:

```text
models/xgboost_tuned_model.pkl
models/zipcode_frequency_encoder.pkl
models/city_full_target_encoder.pkl
data/processed/feature_engineered_train.csv
```

Depois, em uma versao mais madura, a API pode baixar artifacts do W&B no startup.

Fluxo local simples:

```text
treino/tuning -> salva modelo em models/ -> API carrega de models/
```

Fluxo MLOps mais maduro:

```text
treino/tuning -> registra artifact no W&B -> API baixa artifact versionado -> serve modelo
```

> [!tip]
> Para o primeiro backend, mantenha local e funcional. Depois evolua para carregar artifact versionado do W&B ou S3. Simplicidade primeiro, maturidade depois.

---

## Boas praticas de engenharia

### Mantenha a API fina

Errado:

```text
Endpoint faz preprocessing, feature engineering, carrega modelo, calcula metricas e salva arquivo.
```

Certo:

```text
Endpoint valida request -> chama service -> service chama predict() -> retorna response.
```

### Separe contrato de implementacao

Schemas Pydantic devem ficar em `schemas.py`.

Logica de inferencia deve continuar em `src/inference/predict.py`.

Adaptacao entre API e inferencia pode ficar em `service.py`.

### Use erros claros

Se o modelo nao existir, nao retorne stack trace para o cliente.

Melhor:

```json
{
  "detail": "Model artifact is not available. Run the training or tuning pipeline first."
}
```

### Teste endpoints

Crie testes com `TestClient`.

Exemplos:

- `GET /health` retorna 200;
- `POST /predict` rejeita payload invalido;
- `POST /predict` retorna `predicted_price` quando recebe payload valido;
- API responde erro claro quando modelo nao existe.

### Nao misture batch offline com online serving

O pipeline `pipelines/inference_pipeline.py` continua sendo util para batch.

FastAPI deve servir requests online.

Separacao mental:

| Caso | Ferramenta |
| --- | --- |
| Prever um CSV inteiro offline | `inference_pipeline.py` |
| Responder uma requisicao de usuario | `FastAPI` |
| Rodar pipeline completo | `machine_learning_pipeline.py` |

---

## Boas praticas de deployment de ML

### Carregar o modelo no startup

Evite carregar o modelo a cada request.

Melhor:

```text
API sobe -> carrega modelo uma vez -> reutiliza para todas as predicoes
```

Isso reduz latencia e custo.

### Expor health check real

`/health` deve validar dependencias criticas:

- modelo carregado;
- encoders carregados;
- schema de treino disponivel;
- diretorios esperados existem;
- versao/config basica presente.

### Validar schema antes do modelo

Input ruim deve falhar antes de chegar na logica de ML.

FastAPI + Pydantic resolvem boa parte disso.

### Controlar versao do modelo

Toda resposta de predicao deveria carregar alguma metadata:

```json
{
  "predicted_price": 485000.0,
  "model_version": "house_ts_tuned_model:v3"
}
```

Na primeira versao, pode ser:

```json
{
  "model_version": "local"
}
```

Depois, substitua pela versao real do artifact W&B.

### Registrar logs uteis

Logs uteis:

- request recebido sem dados sensiveis;
- latencia;
- status da predicao;
- erro de validacao;
- erro de carregamento do modelo;
- versao do modelo carregado.

Evite logar payload completo se houver dados sensiveis.

### Pensar em monitoramento

Mesmo simples, a API deve preparar caminho para monitorar:

- taxa de erro;
- latencia;
- quantidade de requests;
- distribuicao das features;
- distribuicao das predicoes;
- possivel data drift.

---

## Plano simples de implementacao

### Fase 1: API local minima

- [ ] Adicionar dependencias `fastapi` e `uvicorn`.
- [ ] Criar `src/api/__init__.py`.
- [ ] Criar `src/api/schemas.py`.
- [ ] Criar `src/api/service.py`.
- [ ] Criar `src/api/app.py`.
- [ ] Implementar `GET /`.
- [ ] Implementar `GET /health`.
- [ ] Implementar `POST /predict`.
- [ ] Rodar local com `uv run uvicorn src.api.app:app --reload`.
- [ ] Testar em `http://localhost:8000/docs`.

### Fase 2: Testes

- [ ] Criar `tests/test_api.py`.
- [ ] Testar `/health`.
- [ ] Testar validacao de request.
- [ ] Testar predicao com payload minimo valido.
- [ ] Testar erro quando modelo/artifacts nao existem.

### Fase 3: Melhorias de serving

- [ ] Carregar modelo no startup em vez de a cada request.
- [ ] Criar endpoint `/model-info`.
- [ ] Adicionar metadata de versao do modelo.
- [ ] Padronizar erros HTTP.
- [ ] Adicionar logs de latencia.

### Fase 4: Containerizacao

- [ ] Criar `Dockerfile`.
- [ ] Expor porta `8000`.
- [ ] Rodar API em container.
- [ ] Garantir que `/health` funciona dentro do container.
- [ ] Garantir que modelo e artifacts estejam disponiveis no container.

### Fase 5: Artefatos remotos

- [ ] Escolher origem de artifacts: W&B ou S3.
- [ ] Baixar modelo/encoders no startup.
- [ ] Falhar claramente se artifact nao existir.
- [ ] Registrar versao do artifact carregado.

---

## Dependencias esperadas

O `pyproject.toml` atual ainda nao tem FastAPI e Uvicorn.

Adicionar:

```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
]
```

Se for baixar artifacts de S3 depois:

```toml
dependencies = [
    "boto3>=1.35",
]
```

Para W&B, o projeto ja possui:

```toml
"wandb>=0.27.0"
```

> [!note]
> Como o projeto ja usa Pydantic diretamente em `src/config.py`, FastAPI vai se encaixar bem no estilo atual.

---

## Exemplo de desenho inicial

### `src/api/schemas.py`

```python
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    date: str = Field(..., examples=["2023-01-01"])
    city_full: str
    metro_full: str | None = None
    zipcode: str
    median_list_price: float | None = None


class PredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    model_version: str = "local"
```

### `src/api/service.py`

```python
import pandas as pd

from src.inference.predict import predict
from src.api.schemas import PredictionRequest, PredictionResponse


def predict_one(request: PredictionRequest) -> PredictionResponse:
    input_df = pd.DataFrame([request.model_dump()])
    predictions, _ = predict(input_df)
    row = predictions.iloc[0]

    return PredictionResponse(
        predicted_price=float(row["predicted_price"]),
        model_name="local",
    )
```

### `src/api/app.py`

```python
from fastapi import FastAPI

from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.service import predict_one

app = FastAPI(title="Regression MLOps E2E API", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "regression-mlops-e2e-api", "status": "running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    return predict_one(request)
```

> [!warning]
> O exemplo acima e um ponto de partida didatico. Antes de implementar de verdade, o schema deve ser ajustado para as colunas reais exigidas pelo preprocessing e pela inferencia.

---

## Checklist de pronto

Uma primeira versao do backend esta boa quando:

- [ ] `uv run uvicorn src.api.app:app --reload` sobe sem erro.
- [ ] `GET /` retorna status simples.
- [ ] `GET /health` verifica modelo/artifacts.
- [ ] `POST /predict` recebe JSON e retorna `predicted_price`.
- [ ] `/docs` mostra os schemas corretamente.
- [ ] Payload invalido retorna erro 422.
- [ ] Modelo ausente retorna erro amigavel.
- [ ] Existem testes para endpoints principais.
- [ ] A API nao duplica logica de preprocessing/modelagem.
- [ ] A resposta inclui alguma versao ou identificacao do modelo.

---

## Perguntas que um ML Engineer senior faria

- O payload da API representa dados raw-compatible ou features ja processadas?
- A API esta usando exatamente o mesmo preprocessing do treino?
- Como evitamos training-serving skew?
- Onde o modelo e carregado?
- O modelo e carregado uma vez ou a cada request?
- Como sei qual versao do modelo respondeu uma predicao?
- O endpoint `/health` detecta problemas reais?
- O que acontece se o cliente manda coluna faltando?
- O que acontece se o modelo nao existe?
- Como vou testar isso sem subir o servidor manualmente?
- Como isso sera empacotado em Docker?
- Como o deploy vai obter modelo, encoders e schema de treino?

---

## Relacao com o projeto anterior

O projeto `Regression_ML_EndtoEnd` ja tinha uma API em:

```text
/home/fabiolima/Desktop/MLOps/MLOps_Projects/Regression_ML_EndtoEnd/src/api/main.py
```

Aprendizados bons de la:

- usar FastAPI para servir o modelo;
- criar `/health`;
- criar `/predict`;
- baixar modelo de S3 quando necessario;
- separar FastAPI e Streamlit;
- pensar em container/deploy.

O que devemos melhorar neste projeto:

- usar schemas Pydantic mais explicitos;
- separar `app.py`, `schemas.py` e `service.py`;
- evitar `print` e usar logging;
- retornar erros HTTP claros;
- conectar melhor com W&B/artifacts;
- testar endpoints desde o inicio.

---

## Related

- [[MLOps]]
- [[Weights & Biases]]
- [[Model Serving]]
- [[Model Registry]]
- [[Deployment]]
- [[FastAPI]]
- [[Pydantic]]
- [[Docker]]
