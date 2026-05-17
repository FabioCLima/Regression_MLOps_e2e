# Testando a API e o Frontend

Este guia documenta como validar o backend FastAPI, o frontend Streamlit e a integração entre os dois antes de fazer novos deploys.

Estado atual do projeto:

```text
Backend/API
  -> FastAPI
  -> Dockerfile.api
  -> Deploy ativo em ECS Fargate
  -> URL pública via ALB

Frontend
  -> Streamlit
  -> app.py
  -> Dockerfile.streamlit
  -> Roda localmente e via Docker
  -> Consome a API pela variável API_BASE_URL
```

URL atual do backend em AWS:

```text
http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com
```

## 1. Pré-requisitos

Ferramentas esperadas:

```bash
uv --version
docker --version
curl --version
gh --version
aws --version
```

Carregue as variáveis locais quando precisar consultar AWS:

```bash
source deploy.env
```

O arquivo `deploy.env` é local e não deve ser versionado.

## 2. Endpoints da API

A API expõe estes endpoints principais:

```text
GET  /
GET  /health
GET  /model-info
POST /predict
POST /predict/batch
```

Responsabilidades:

- `/health`: confirma se a API está viva e se os artifacts foram carregados.
- `/model-info`: mostra versão, origem dos artifacts e schema esperado pelo modelo.
- `/predict`: executa uma predição individual.
- `/predict/batch`: executa predições em lote com até 1000 registros.

## 3. Validar API localmente

Primeiro rode lint e testes:

```bash
uv run ruff check app.py src pipelines tests
uv run pytest -q
```

Suba a API local:

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Em outro terminal:

```bash
curl http://localhost:8000/health
```

Resposta esperada:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "encoders_loaded": true,
  "n_features_expected": 39
}
```

Se `status` vier `unhealthy`, verifique se os artifacts locais existem:

```bash
ls -lh \
  models/xgboost_tuned_model.pkl \
  models/best_model.pkl \
  models/zipcode_frequency_encoder.pkl \
  models/city_full_target_encoder.pkl \
  data/processed/feature_engineered_train.csv
```

## 4. Validar API local com Docker

Build:

```bash
docker build -f Dockerfile.api -t regression-mlops-e2e-api:local .
```

Como a imagem de produção não embute artifacts, para testar o modo AWS/S3 localmente use:

```bash
docker run --rm -p 8000:8000 \
  -e AWS_REGION=${AWS_REGION} \
  -e AWS_S3_BUCKET=${S3_BUCKET} \
  -e ARTIFACT_CACHE_DIR=/tmp/artifacts \
  -v ~/.aws:/root/.aws:ro \
  regression-mlops-e2e-api:local
```

Valide:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

O campo `model_version` deve apontar para S3:

```text
s3://regression-mlops-e2e-artifacts-878919573366
```

## 5. Validar API na AWS

Com `deploy.env` carregado:

```bash
curl http://${ALB_DNS}/health
curl http://${ALB_DNS}/model-info
```

Ou usando a URL diretamente:

```bash
curl http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com/health
```

Validar estado do ECS:

```bash
aws ecs describe-services \
  --cluster ${ECS_CLUSTER} \
  --services ${ECS_SERVICE} \
  --region ${AWS_REGION} \
  --query 'services[0].[desiredCount,runningCount,pendingCount,deployments[0].rolloutState,deployments[0].taskDefinition]' \
  --output table
```

Resultado esperado:

```text
desiredCount = 1
runningCount = 1
pendingCount = 0
rolloutState = COMPLETED
```

Validar Target Group:

```bash
aws elbv2 describe-target-health \
  --target-group-arn ${TG_ARN} \
  --region ${AWS_REGION} \
  --query 'TargetHealthDescriptions[*].[Target.Id,Target.Port,TargetHealth.State,TargetHealth.Reason,TargetHealth.Description]' \
  --output table
```

Esperado:

```text
healthy
```

## 6. Testar predição individual

Payload mínimo:

```bash
curl -X POST http://${ALB_DNS}/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2022-01-01","city_full":"Atlanta-Sandy Springs-Alpharetta","city":"ATL","zipcode":30301}'
```

Resposta esperada:

```json
{
  "predicted_price": 312084.09375,
  "model_version": "xgboost_tuned_model.pkl@sha256:...@s3://...",
  "missing_features": ["median_sale_price", "median_list_price"]
}
```

O valor exato da predição pode mudar se o modelo/artifacts mudarem.

A lista `missing_features` não significa falha. Ela mostra campos opcionais que não foram enviados. A API preenche o que falta com `0` depois da engenharia de features.

Payload mais completo:

```bash
curl -X POST http://${ALB_DNS}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2022-01-01",
    "city_full": "Atlanta-Sandy Springs-Alpharetta",
    "city": "ATL",
    "zipcode": 30301,
    "median_sale_price": 395000,
    "median_list_price": 420000,
    "median_ppsf": 215,
    "median_list_ppsf": 225,
    "homes_sold": 620,
    "pending_sales": 510,
    "new_listings": 780,
    "inventory": 1600,
    "median_dom": 28,
    "avg_sale_to_list": 0.99,
    "sold_above_list": 0.31,
    "off_market_in_two_weeks": 0.42,
    "bank": 12,
    "bus": 35,
    "hospital": 4,
    "mall": 2,
    "park": 18,
    "restaurant": 75,
    "school": 21,
    "station": 6,
    "supermarket": 14,
    "total_population": 498715,
    "median_age": 34.8,
    "per_capita_income": 48200,
    "total_families_below_poverty": 11800,
    "total_housing_units": 235000,
    "median_rent": 1850,
    "median_home_value": 390000,
    "total_labor_force": 276000,
    "unemployed_population": 9400,
    "total_school_age_population": 80500,
    "total_school_enrollment": 72100,
    "median_commute_time": 27
  }'
```

## 7. Testar predição batch

Crie um arquivo CSV de exemplo:

```bash
cat > /tmp/prediction_sample.csv <<'EOFCSV'
date,city_full,city,zipcode,median_list_price,median_sale_price
2022-01-01,Atlanta-Sandy Springs-Alpharetta,ATL,30301,420000,395000
2022-02-01,Atlanta-Sandy Springs-Alpharetta,ATL,30303,455000,430000
EOFCSV
```

Converter para JSON e chamar `/predict/batch`:

```bash
python3 - <<'PY'
import pandas as pd
import requests

base_url = "http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com"
df = pd.read_csv("/tmp/prediction_sample.csv")
payload = {"records": df.where(pd.notnull(df), None).to_dict(orient="records")}
response = requests.post(f"{base_url}/predict/batch", json=payload, timeout=60)
response.raise_for_status()
print(response.json())
PY
```

Resposta esperada:

```json
{
  "rows_predicted": 2,
  "predictions": [
    {"predicted_price": 381436.125, "model_version": "...", "missing_features": [...]}
  ]
}
```

## 8. Como o frontend se integra ao backend

O frontend está em `app.py` e usa Streamlit.

Ele lê a URL da API nesta ordem:

```text
API_BASE_URL
API_URL
DEFAULT_API_BASE_URL
```

Valor padrão atual:

```text
http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com
```

O frontend chama:

```text
GET  {API_BASE_URL}/health
GET  {API_BASE_URL}/model-info
POST {API_BASE_URL}/predict
POST {API_BASE_URL}/predict/batch
```

Se o usuário digitar uma URL terminando em `/predict`, o app normaliza para a base da API.

## 9. Rodar frontend localmente

Instale as dependências, incluindo o grupo frontend:

```bash
uv sync --frozen --all-groups
```

Rode:

```bash
API_BASE_URL=http://${ALB_DNS} \
uv run streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true
```

Ou com a URL direta:

```bash
API_BASE_URL=http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com \
uv run streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true
```

Abra:

```text
http://localhost:8501
```

Valide no app:

1. A sidebar deve mostrar `healthy`.
2. A aba `Single prediction` deve retornar preço previsto.
3. A aba `Batch CSV` deve aceitar o CSV de amostra.
4. A aba `Model` deve mostrar as colunas esperadas pelo modelo.

## 10. Rodar frontend com Docker

Build:

```bash
docker build -f Dockerfile.streamlit -t regression-mlops-e2e-frontend:local .
```

Run:

```bash
docker run --rm -p 8501:8501 \
  -e API_BASE_URL=http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com \
  regression-mlops-e2e-frontend:local
```

Validar health do Streamlit:

```bash
curl http://localhost:8501/_stcore/health
```

Resposta esperada:

```text
ok
```

Abrir:

```text
http://localhost:8501
```

## 11. Validar antes de subir ao GitHub

Antes de commit/push:

```bash
uv run ruff check app.py src pipelines tests
uv run pytest -q
docker build -f Dockerfile.api -t regression-mlops-e2e-api:ci .
docker build -f Dockerfile.streamlit -t regression-mlops-e2e-frontend:ci .
```

Checar arquivos pendentes:

```bash
git status --porcelain=v1
```

Checar se não há credenciais:

```bash
rg -n "AKIA|AWS_SECRET_ACCESS_KEY|secret access key|aws_access_key_id|aws_secret_access_key" -S . \
  -g '!data/**' \
  -g '!artifacts/**' \
  -g '!deploy.env' \
  -g '!.venv/**'
```

## 12. Como o CI/CD valida o projeto

O workflow de CI faz:

```text
uv sync --frozen --all-groups
ruff check app.py src pipelines tests
pytest -q
docker build -f Dockerfile.api
docker build -f Dockerfile.streamlit
```

O workflow de CD atual faz deploy apenas do backend/API no ECS.

Mudanças apenas nestes caminhos não disparam redeploy do backend:

```text
docs/**
README.md
app.py
Dockerfile.streamlit
.streamlit/**
```

Isso evita redeploy da API quando só alteramos documentação ou frontend local.

Para acompanhar runs:

```bash
gh run list --repo FabioCLima/Regression_MLOps_e2e --limit 10
```

Para ver falhas:

```bash
gh run view <RUN_ID> --repo FabioCLima/Regression_MLOps_e2e --log-failed
```

## 13. Troubleshooting

### API retorna 503 em `/health`

Possíveis causas:

- artifacts não foram carregados;
- bucket S3 errado;
- task role sem permissão de `s3:GetObject`;
- arquivo obrigatório ausente no S3;
- modelo incompatível com schema de treino.

Verifique logs:

```bash
aws logs tail ${LOG_GROUP} --region ${AWS_REGION} --since 30m
```

### Docker local não acessa S3

Erro comum:

```text
Unable to locate credentials
```

Solução local:

```bash
-v ~/.aws:/root/.aws:ro
```

No ECS, o acesso vem via `taskRoleArn`, não por access key.

### Frontend mostra API unavailable

Verifique:

```bash
curl http://${ALB_DNS}/health
```

Se funcionar no terminal, confirme a URL na sidebar do Streamlit. Ela deve ser a base da API, sem `/predict` no final.

### Batch CSV falha com coluna obrigatória ausente

O CSV precisa ter pelo menos:

```text
date,city_full,city,zipcode
```

Campos opcionais melhoram a predição, mas não são obrigatórios.

### Streamlit pede email ou onboarding

Este projeto versiona:

```text
.streamlit/config.toml
.streamlit/credentials.toml
```

Eles desativam telemetria e execução interativa de onboarding.

### Porta 8501 ocupada

Use outra porta:

```bash
uv run streamlit run app.py --server.port=8502 --server.address=0.0.0.0 --server.headless=true
```

Ou pare o container/processo existente:

```bash
docker ps
docker stop <CONTAINER_ID>
```

## 14. Próximo passo: deploy do frontend na AWS

O frontend já está pronto para containerizar. Para publicar na AWS, o próximo roteiro deve criar:

```text
ECR repository para frontend
CloudWatch log group do frontend
ECS task definition Streamlit
ECS service Streamlit
Target group na porta 8501
Regra no ALB ou ALB separado
CI/CD específico do frontend
```

Decisão de arquitetura a tomar:

1. Usar o mesmo ALB do backend com path/host diferente.
2. Criar um ALB separado para o frontend.
3. Usar Streamlit em ECS privado atrás de CloudFront/ALB com HTTPS.

Para estudo e simplicidade, um ALB separado é mais fácil. Para produção, o ideal é domínio, HTTPS e regras claras de roteamento.
