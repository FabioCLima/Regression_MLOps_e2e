# Controle de Custos AWS

Este guia registra os controles de custo configurados para o ambiente AWS do projeto e os comandos para operar a aplicação sem deixar recursos caros rodando sem necessidade.

## Estado atual

Serviços publicados:

```text
Backend API: ECS Fargate + ALB
Frontend Streamlit: ECS Fargate + ALB
Artifacts: S3
Imagens Docker: ECR
Logs: CloudWatch Logs
CI/CD: GitHub Actions + OIDC
```

URLs públicas:

```text
Frontend:
http://reg-mlops-fe-alb-1328758802.us-east-1.elb.amazonaws.com

Backend API:
http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com
```

## O que custa continuamente

Principais recursos com custo enquanto o ambiente fica ativo:

```text
Application Load Balancer do backend
Application Load Balancer do frontend
ECS Fargate task do backend
ECS Fargate task do frontend
CloudWatch Logs
ECR storage
S3 storage
```

Reduzir `desired-count` para `0` para as tasks ECS reduz custo de Fargate, mas os ALBs continuam existindo e cobrando. Para custo mínimo absoluto, é preciso remover ALBs/Target Groups também, ou recriar tudo depois.

## Controles já configurados

### 1. ECR lifecycle policy

Foi aplicada uma lifecycle policy nos dois repositórios ECR:

```text
regression-mlops-e2e-api
regression-mlops-e2e-frontend
```

Regra:

```text
manter somente as últimas 5 imagens
expirar imagens mais antigas
```

Arquivo versionado:

```text
infra/aws/ecr-lifecycle-keep-last-5.json
```

Validar:

```bash
source deploy.env

aws ecr get-lifecycle-policy \
  --repository-name ${ECR_REPO} \
  --region ${AWS_REGION}

aws ecr get-lifecycle-policy \
  --repository-name regression-mlops-e2e-frontend \
  --region ${AWS_REGION}
```

### 2. AWS Budget

Foi criado um budget mensal:

```text
Nome: RegressionMLOpsE2E-Monthly-10USD
Limite: 10 USD/mês
Alerta: 80% do orçamento
Email: lima.fisico@gmail.com
```

Arquivos versionados:

```text
infra/aws/budget-monthly-10usd.json
infra/aws/budget-notification-80pct.json
```

Validar:

```bash
source deploy.env

aws budgets describe-budget \
  --account-id ${ACCOUNT_ID} \
  --budget-name RegressionMLOpsE2E-Monthly-10USD \
  --query 'Budget.[BudgetName,BudgetLimit.Amount,BudgetLimit.Unit,TimeUnit,BudgetType]' \
  --output table
```

A AWS pode pedir confirmação do e-mail do subscriber do budget.

### 3. Parada automática hoje

Foram criados dois schedules no EventBridge Scheduler para desligar as tasks hoje às 23:55 no fuso `America/Recife`.

Schedules:

```text
regression-mlops-stop-backend-2026-05-17
regression-mlops-stop-frontend-2026-05-17
```

Horário:

```text
2026-05-17 23:55 America/Recife
```

Ação:

```text
aws ecs update-service --desired-count 0
```

Role usada:

```text
RegressionMLOpsStopServicesSchedulerRole
```

Arquivos versionados:

```text
infra/aws/scheduler-trust-policy.json
infra/aws/scheduler-stop-services-policy.json
```

Validar schedules:

```bash
aws scheduler get-schedule \
  --name regression-mlops-stop-backend-2026-05-17 \
  --region us-east-1 \
  --query '[Name,ScheduleExpression,ScheduleExpressionTimezone,State,Target.Input]' \
  --output table

aws scheduler get-schedule \
  --name regression-mlops-stop-frontend-2026-05-17 \
  --region us-east-1 \
  --query '[Name,ScheduleExpression,ScheduleExpressionTimezone,State,Target.Input]' \
  --output table
```

Cancelar schedules, se precisar manter rodando depois das 23:55:

```bash
aws scheduler delete-schedule \
  --name regression-mlops-stop-backend-2026-05-17 \
  --region us-east-1

aws scheduler delete-schedule \
  --name regression-mlops-stop-frontend-2026-05-17 \
  --region us-east-1
```

## Scripts operacionais

Foram criados scripts em `scripts/`.

### Ver status

```bash
scripts/aws_status.sh
```

Mostra:

```text
ECS desired/running/pending
health do backend
health do frontend
```

### Parar tasks ECS

```bash
scripts/aws_stop_services.sh
```

Isso coloca os dois services com `desired-count=0`:

```text
regression-mlops-e2e-service
regression-mlops-e2e-frontend-service
```

### Ligar tasks ECS

```bash
scripts/aws_start_services.sh
```

Isso coloca os dois services com `desired-count=1`.

Depois de ligar, aguarde alguns minutos até o ECS puxar as imagens, iniciar os containers e passar nos health checks.

### Consultar custo dos últimos 7 dias

```bash
scripts/aws_cost_last_7_days.sh
```

Se a AWS retornar `DataUnavailableException`, aguarde a ingestão de dados do Cost Explorer e tente novamente mais tarde.

## O que passar para um colega testar

Para um colega testar a aplicação, passe apenas a URL do frontend:

```text
http://reg-mlops-fe-alb-1328758802.us-east-1.elb.amazonaws.com
```

Instrução simples:

```text
Abra a URL, confirme que a sidebar mostra healthy, vá em Single prediction, mantenha Use sample values ligado e clique em Run prediction.
```

Para teste direto da API:

```bash
curl http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com/health
```

Predição via API:

```bash
curl -X POST http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2022-01-01","city_full":"Atlanta-Sandy Springs-Alpharetta","city":"ATL","zipcode":30301}'
```

Não passe:

```text
AWS keys
deploy.env
GitHub secrets
permissões no S3
acesso à conta AWS
```

## Checklist para deixar rodando só hoje

Antes de compartilhar:

```bash
scripts/aws_status.sh
```

Confirme:

```text
backend runningCount = 1
frontend runningCount = 1
backend health = healthy
frontend health = ok
```

Confirme que os schedules estão ativos:

```bash
aws scheduler get-schedule --name regression-mlops-stop-backend-2026-05-17 --region us-east-1 --query '[State,ScheduleExpression,ScheduleExpressionTimezone]' --output table
aws scheduler get-schedule --name regression-mlops-stop-frontend-2026-05-17 --region us-east-1 --query '[State,ScheduleExpression,ScheduleExpressionTimezone]' --output table
```

Depois das 23:55, validar que as tasks pararam:

```bash
scripts/aws_status.sh
```

Esperado:

```text
desiredCount = 0
runningCount = 0
```

## Observação sobre ALBs

Mesmo com ECS services parados, os ALBs continuam cobrando. Para uma pausa longa, considere destruir os ALBs e recriá-los depois pelo roteiro de deploy.

Para uma demo curta de um dia, manter os ALBs e parar as tasks à noite é um bom equilíbrio entre simplicidade e controle de custo.

## Modo portfólio sob demanda

Use este modo quando o projeto não precisa ficar público todos os dias.

### Pausa rápida

Para parar somente as tasks ECS e manter os ALBs prontos:

```bash
scripts/aws_stop_services.sh
```

Esse modo reduz custo de Fargate, mas os ALBs continuam cobrando.

### Hibernação de baixo custo

Para remover os ECS services, ALBs e target groups da demo:

```bash
DEMO_TEARDOWN_CONFIRM=delete-demo-infra scripts/aws_teardown_demo_infra.sh
```

Esse modo mantém:

```text
S3 com artifacts
ECR com imagens Docker
Task definitions
IAM roles
CloudWatch logs
Budget
```

Ele remove os recursos de exposição pública que geram custo fixo parado.

### Recriar demo pública

Quando um recrutador ou avaliador pedir acesso:

```bash
scripts/aws_recreate_demo_infra.sh
scripts/aws_status.sh
```

O script recria backend e frontend, grava os novos ARNs/DNS em `deploy.runtime.env` e imprime as novas URLs públicas.

Depois da demonstração:

```bash
DEMO_TEARDOWN_CONFIRM=delete-demo-infra scripts/aws_teardown_demo_infra.sh
```

Observação: as URLs dos ALBs podem mudar a cada recriação. Use sempre a URL impressa pelo script mais recente.
