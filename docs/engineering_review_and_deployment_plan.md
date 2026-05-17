# Engineering Review, Deployment Plan e Documentação CAR

> Documento gerado após análise do código-fonte real do projeto.
> Revisado do ponto de vista de um ML Engineer Senior.

---

## 1. Avaliação do Projeto (ML Engineer Senior)

### 1.1 O que está bem feito

**Separação de responsabilidades**
A divisão `src/` (lógica) e `pipelines/` (orquestração) é a decisão mais importante de
arquitetura e está correta. Qualquer módulo em `src/` pode ser testado sem precisar
executar o pipeline inteiro. Isso reduz o custo de manutenção a longo prazo.

**Configuração tipada com Pydantic**
O `src/config.py` é um dos pontos mais maduros do projeto. Ter configurações separadas
por domínio (`SplitConfig`, `FeatureEngineeringConfig`, `TuningConfig`, etc.) — todas
imutáveis (`frozen=True`) — evita que parâmetros sejam silenciosamente sobrescritos
durante execução. Isso é o que times de ML em produção precisam.

**Contrato de inferência explícito**
A função `predict()` em `src/inference/predict.py` aceita um `DataFrame` bruto e
aplica o mesmo contrato de transformações do treino (preprocessing → feature engineering
→ alinhamento de schema → predição). Isso é correto. Projetos que não fazem isso
entram em produção com training-serving skew.

**Separação temporal no split**
Usar datas para separar treino/eval/holdout em vez de `train_test_split` aleatório é
a decisão certa para dados de séries temporais. Evita leakage temporal.

**Encoders fitados apenas no treino**
Frequency encoder e target encoder são ajustados no split de treino e aplicados nos
demais. Isso está correto e é um dos erros mais comuns em projetos de iniciante.

**Testes existem**
O projeto tem testes unitários e smoke tests para todos os componentes principais.
Isso é raro em projetos pessoais de ML e mostra maturidade.

---

### 1.2 Lacunas reais (o que falta para produção)

**Nenhuma camada de serving**
O projeto produz um `predictions.csv` a partir de um arquivo CSV de entrada. Isso
não é produção. Em produção, um cliente (web app, outro serviço, dashboard) precisa
enviar dados e receber predições via HTTP. Falta uma API.

**Dados e modelos ainda são locais**
Artefatos estão em `data/`, `models/` e `artifacts/` locais. Em produção, esses
artefatos precisam viver em armazenamento durável e compartilhável (S3). Qualquer
novo container ou instância que subir não terá os arquivos.

**Sem validação de schema na entrada da inferência**
A `predict()` recebe um `DataFrame` e assume que as colunas necessárias existem.
Se um cliente enviar dados com colunas faltando ou com tipos errados, o erro vai
aparecer no meio do pipeline — não numa mensagem clara para o cliente.

**Sem monitoramento pós-deploy**
Depois que o modelo vai para produção, é preciso monitorar:
- se a distribuição de entrada mudou (data drift);
- se as predições estão saindo dentro de um range esperado;
- latência e taxa de erro da API.

**Sem containerização**
O projeto roda com `uv run`, que é ótimo para desenvolvimento. Para produção na AWS,
o ambiente precisa ser empacotado em uma imagem Docker para garantir reprodutibilidade.

**CI/CD ausente**
Não há GitHub Actions nem nenhum pipeline automatizado que rode os testes antes de
qualquer deploy. Mudanças vão para produção sem gate de qualidade.

**Sem health check de dependências**
O `inference_pipeline.py` falha com `FileNotFoundError` se o modelo não existir.
Uma API de produção precisa de um endpoint `/health` que informa se o modelo está
carregado e pronto antes de aceitar requisições.

---

### 1.3 Dívida técnica menor (não bloqueante)

- `config.py` na raiz do projeto (`/config.py`) é diferente de `src/config.py` —
  ambos existem. Manter apenas `src/config.py` evita confusão.
- `artifacts/` local está sendo criado pelo W&B como cache. Esse diretório deve
  entrar no `.gitignore`.
- O `wandb/` local também deve estar no `.gitignore`.

---

## 2. Arquitetura de Deploy na AWS (Free Tier)

### 2.1 Restrições do Free Tier (referência rápida)

| Serviço         | Limite free tier                        | Quando expira     |
|----------------|------------------------------------------|-------------------|
| EC2 t2.micro   | 750 horas/mês                           | 12 meses          |
| S3             | 5 GB armazenamento, 20k GET, 2k PUT     | 12 meses          |
| Lambda         | 1M requests/mês, 400k GB-s de compute  | Permanente        |
| API Gateway    | 1M chamadas REST/mês                    | 12 meses          |
| ECR            | 500 MB de armazenamento privado         | Permanente        |
| CloudWatch     | 10 métricas, 10 alarmes, 5 GB logs      | Permanente        |

### 2.2 Por que não usar Lambda para este projeto

Lambda tem limite de 250 MB de pacote descomprimido. O modelo XGBoost + scikit-learn
+ pandas + numpy ultrapassa facilmente esse limite. A alternativa seria Lambda com
container image (até 10 GB), mas o cold start de uma imagem grande pode chegar a
20-30 segundos — inaceitável para uma API de predição.

**Decisão:** EC2 t2.micro para backend e frontend.

### 2.3 Arquitetura proposta

```
┌─────────────────────────────────────────────────────┐
│                    Internet                         │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/HTTPS
              ┌────────▼────────┐
              │   EC2 t2.micro  │
              │  (Free Tier)    │
              │                 │
              │  ┌───────────┐  │
              │  │  Nginx    │  │  ← reverse proxy
              │  └─────┬─────┘  │
              │        │        │
              │  ┌─────▼─────┐  │
              │  │ FastAPI   │  │  ← porta 8000 (backend)
              │  │ + Uvicorn │  │
              │  └───────────┘  │
              │                 │
              │  ┌───────────┐  │
              │  │ Streamlit │  │  ← porta 8501 (frontend)
              │  └───────────┘  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    S3 Bucket    │  ← modelo .pkl, encoders .pkl
              │  (Free Tier)    │
              └─────────────────┘
                       │
              ┌────────▼────────┐
              │  W&B / MLflow   │  ← experiment tracking (SaaS)
              └─────────────────┘
```

**Fluxo de dados em produção:**

```
Usuário → Streamlit (UI) → FastAPI (POST /predict) → carrega modelo do S3
       ← predição + métricas ← FastAPI ← modelo.predict()
```

---

## 3. Plano de Trabalho Detalhado

### Fase 1 — Preparar o código para produção (local)

**Objetivo:** antes de tocar na AWS, a API precisa funcionar localmente.

**Passo 1.1 — Criar `src/api/schema.py` (contrato da API)**
```python
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    date: str
    city_full: str
    zipcode: str
    # ... colunas necessárias

class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str
```
Isso garante que a API rejeita entradas inválidas com erro 422 antes de chegar
no modelo.

**Passo 1.2 — Criar `src/api/app.py` (FastAPI)**
```python
from fastapi import FastAPI
import pandas as pd
from src.inference.predict import predict
from src.api.schema import PredictionRequest, PredictionResponse

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    df = pd.DataFrame([request.model_dump()])
    predictions, _ = predict(df)
    return PredictionResponse(predicted_price=predictions["predicted_price"].iloc[0])
```

**Passo 1.3 — Criar `frontend/app.py` (Streamlit)**
Interface simples com:
- Formulário de entrada dos dados da casa
- Botão "Prever preço"
- Card com o resultado e intervalo de confiança (se disponível)
- Seção de métricas do modelo (MAE, RMSE, R²) carregadas do JSON

**Passo 1.4 — Adicionar `boto3` e refatorar carregamento de artefatos**
A `predict()` hoje carrega de `paths.models_dir` (local). Adicionar lógica:
```python
def load_model(model_path: Path | str | None = None):
    if os.environ.get("AWS_S3_BUCKET"):
        return load_from_s3(bucket, key)
    return load(local_path)
```
Isso mantém o código funcionando localmente e na AWS via variável de ambiente.

**Passo 1.5 — Containerizar com Docker**
```dockerfile
# Dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen
COPY src/ ./src/
COPY pipelines/ ./pipelines/
CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Passo 1.6 — Testar localmente com Docker Compose**
```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - AWS_S3_BUCKET=seu-bucket
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports: ["8501:8501"]
```

**Critério de aceite:** `curl -X POST localhost:8000/predict` com payload válido
retorna predição. Streamlit carrega sem erro.

---

### Fase 2 — Infraestrutura AWS

**Passo 2.1 — Criar S3 Bucket**
```bash
aws s3 mb s3://regression-mlops-e2e-artifacts --region us-east-1
```
Upload dos artefatos:
```bash
aws s3 cp models/best_model.pkl s3://regression-mlops-e2e-artifacts/models/
aws s3 cp models/xgboost_tuned_model.pkl s3://regression-mlops-e2e-artifacts/models/
aws s3 cp models/zipcode_frequency_encoder.pkl s3://regression-mlops-e2e-artifacts/encoders/
aws s3 cp models/city_full_target_encoder.pkl s3://regression-mlops-e2e-artifacts/encoders/
```

**Passo 2.2 — Criar IAM Role para EC2**
Role com política mínima:
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject"],
  "Resource": "arn:aws:s3:::regression-mlops-e2e-artifacts/*"
}
```
Princípio do menor privilégio: a instância EC2 só precisa ler do bucket, não escrever.

**Passo 2.3 — Lançar EC2 t2.micro**
- AMI: Ubuntu 24.04 LTS (free tier eligible)
- Security Group: abrir porta 80 (Nginx), 8000 (API, apenas para diagnóstico), 22 (SSH)
- Attach a IAM Role criada no passo anterior
- Key pair: criar e baixar `.pem`

**Passo 2.4 — Instalar dependências na EC2**
```bash
# via SSH
sudo apt update && sudo apt install -y docker.io docker-compose nginx
sudo usermod -aG docker ubuntu
```

**Passo 2.5 — Push da imagem Docker para ECR**
```bash
aws ecr create-repository --repository-name regression-mlops-api
docker build -t regression-mlops-api .
docker tag regression-mlops-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/regression-mlops-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/regression-mlops-api:latest
```

**Passo 2.6 — Configurar Nginx como reverse proxy**
```nginx
# /etc/nginx/sites-available/regression-mlops
server {
    listen 80;
    location /api/ {
        proxy_pass http://localhost:8000/;
    }
    location / {
        proxy_pass http://localhost:8501/;
    }
}
```
Isso expõe tudo na porta 80:
- `http://ip-ec2/` → Streamlit
- `http://ip-ec2/api/predict` → FastAPI

**Passo 2.7 — Deploy com Docker Compose na EC2**
```bash
docker-compose pull && docker-compose up -d
```

**Critério de aceite:** acesso pelo IP público da EC2. API responde em `/api/health`.
Streamlit carrega em `/`.

---

### Fase 3 — CI/CD com GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install uv && uv sync
      - run: uv run pytest tests/ -v
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build e push ECR
        run: |
          docker build -t $ECR_REPO:$GITHUB_SHA .
          docker push $ECR_REPO:$GITHUB_SHA
      - name: SSH deploy na EC2
        run: |
          ssh ubuntu@$EC2_IP "docker-compose pull && docker-compose up -d"
```

**Critério de aceite:** push na `main` → testes rodam → imagem nova sobe na EC2
automaticamente, sem intervenção manual.

---

### Fase 4 — Monitoramento mínimo

**Passo 4.1 — Logging estruturado na API**
Cada request de predição deve logar:
```json
{
  "timestamp": "2026-05-16T10:23:00",
  "input_shape": [1, 12],
  "predicted_price": 450000,
  "model_version": "xgboost_tuned_v1",
  "latency_ms": 45
}
```

**Passo 4.2 — CloudWatch para logs da EC2**
Instalar o agente CloudWatch na EC2 para enviar logs do container.
Free tier: 5 GB de log ingestion/mês.

**Passo 4.3 — Alarme de erro na API**
Criar alarme no CloudWatch se a taxa de erro HTTP 5xx ultrapassar 5%
em 5 minutos. Custo: zero (free tier permite 10 alarmes).

---

## 4. Como Documentar o Projeto com o Framework CAR

### O que é CAR

CAR é um framework de comunicação para descrever projetos de forma que faça sentido
para recrutadores, stakeholders não técnicos e revisores de portfólio:

- **C — Context (Contexto):** qual era o problema e por que ele importava?
- **A — Action (Ação):** o que você fez especificamente?
- **R — Result (Resultado):** qual foi o impacto mensurável?

O erro mais comum é começar pela ação ("construí um pipeline de MLOps") sem
estabelecer o contexto nem quantificar o resultado.

---

### Template CAR para este projeto

#### Contexto
> O mercado imobiliário dos EUA gera grandes volumes de dados históricos de preços
> por cidade e código postal. A ausência de um sistema de predição estruturado força
> analistas a dependerem de planilhas manuais, sem rastreabilidade e sem possibilidade
> de atualização contínua do modelo.

#### Ação
> Projetei e implementei um pipeline MLOps end-to-end em Python para previsão de
> preços de imóveis, incluindo:
>
> - **Ingestão e validação:** carregamento do dataset HouseTS (280 MB) com split
>   temporal para evitar leakage (treino < 2020, eval 2020–2022, holdout > 2022);
> - **Feature engineering:** frequency encoding de CEP, target encoding de cidade,
>   features de data — encoders fitados apenas no treino e versionados no W&B;
> - **Treinamento e seleção:** comparação entre Dummy, Linear, Ridge, Random Forest
>   e XGBoost, com seleção automática pelo menor RMSE no split de avaliação;
> - **Tuning:** otimização de hiperparâmetros do XGBoost com Optuna (15 trials);
> - **Serving:** API REST em FastAPI com validação de schema via Pydantic, frontend
>   em Streamlit, containerizado com Docker e deployado em EC2 na AWS;
> - **Rastreabilidade:** todos os artefatos (datasets, encoders, modelos) versionados
>   no Weights & Biases com metadados e métricas associadas;
> - **Qualidade:** testes unitários e smoke tests com pytest cobrindo todos os
>   componentes.

#### Resultado
> *(Preencher após deploy — exemplos de métricas a capturar)*
>
> - RMSE no holdout (2022+): `$X`
> - Melhoria relativa sobre o baseline Dummy: `X%`
> - Latência média da API: `<50ms`
> - Tempo total de re-treinamento do pipeline: `<X minutos`
> - Custo de infraestrutura mensal: `~$0` (AWS free tier)

---

### Onde usar o CAR

| Contexto                  | Formato recomendado                                   |
|--------------------------|-------------------------------------------------------|
| LinkedIn / portfólio      | Parágrafo curto (3–5 linhas), foco no resultado       |
| README do repositório     | Seção "About" com as três partes em subtópicos        |
| Entrevista técnica        | Narrativa verbal: contexto → decisões → aprendizados  |
| Case de entrevista        | Documento de 1 página com métricas em destaque        |

---

### Checklist para completar o CAR após o deploy

- [ ] Registrar RMSE, MAE e R² no holdout após treino final
- [ ] Registrar latência P50 e P95 da API em produção
- [ ] Registrar custo mensal real da AWS (mesmo que zero)
- [ ] Documentar a decisão mais difícil do projeto e o tradeoff escolhido
- [ ] Adicionar link para o W&B project com o histórico de runs
- [ ] Adicionar screenshot ou GIF do frontend Streamlit funcionando

---

## 5. Resumo Executivo

| Dimensão              | Status atual             | Próximo passo               |
|-----------------------|--------------------------|-----------------------------|
| Pipeline de treino    | Completo e funcional     | Containerizar                |
| Inferência (batch)    | Completo e funcional     | Expor via API REST           |
| Serving (API)         | Ausente                  | FastAPI — Fase 1             |
| Frontend              | Ausente                  | Streamlit — Fase 1           |
| Armazenamento cloud   | Ausente (local)          | S3 — Fase 2                  |
| Deploy AWS            | Ausente                  | EC2 t2.micro — Fase 2        |
| CI/CD                 | Ausente                  | GitHub Actions — Fase 3      |
| Monitoramento         | Ausente                  | CloudWatch — Fase 4          |
| Documentação CAR      | Template criado          | Preencher após deploy        |
| Custo estimado        | —                        | ~$0/mês (free tier)          |
