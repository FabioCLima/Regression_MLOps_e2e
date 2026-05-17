# Roteiro de Deploy na AWS com ECS Fargate

Este documento descreve como replicar o deploy da API FastAPI deste projeto na AWS usando S3, ECR, IAM, ECS Fargate, Application Load Balancer, CloudWatch Logs e GitHub Actions com OIDC.

A ideia central e simples:

```text
GitHub Actions
  -> lint e testes
  -> build Dockerfile.api
  -> push da imagem no Amazon ECR
  -> nova revision da ECS task definition
  -> update do ECS service

ECS Fargate
  -> puxa a imagem do ECR
  -> baixa artifacts do S3 no startup
  -> expõe a API FastAPI na porta 8000
  -> envia logs para CloudWatch
```

## 1. Antes de tocar na AWS

### 1.1. Pré-requisitos locais

Instale e valide:

```bash
aws --version
docker --version
gh --version
git --version
uv --version
```

Autentique na AWS e no GitHub:

```bash
aws sts get-caller-identity
gh auth status
```

O comando `aws sts get-caller-identity` deve mostrar a conta correta. Neste projeto, a conta usada foi:

```text
878919573366
```

### 1.2. Arquivos que precisam existir localmente

Antes de subir artifacts para o S3, valide que os arquivos runtime existem:

```bash
ls -lh \
  models/xgboost_tuned_model.pkl \
  models/best_model.pkl \
  models/zipcode_frequency_encoder.pkl \
  models/city_full_target_encoder.pkl \
  data/processed/feature_engineered_train.csv
```

A API espera esses caminhos no S3:

```text
models/xgboost_tuned_model.pkl
models/best_model.pkl
models/zipcode_frequency_encoder.pkl
models/city_full_target_encoder.pkl
data/processed/feature_engineered_train.csv
```

O arquivo `holdout.csv` não é artifact runtime da API. Ele pertence à etapa de validação e monitoramento. Se quiser mantê-lo na AWS, use um prefixo separado, por exemplo:

```text
validation/holdout.csv
validation/feature_engineered_holdout.csv
monitoring/baseline/holdout_profile.json
```

### 1.3. Validar o projeto localmente

Rode lint e testes:

```bash
uv run ruff check src pipelines tests
uv run pytest -q
```

Build da imagem:

```bash
docker build -f Dockerfile.api -t regression-mlops-e2e-api:local .
```

Rodar localmente sem S3, usando artifacts locais:

```bash
docker run --rm -p 8000:8000 regression-mlops-e2e-api:local
```

Validar:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2022-01-01","city_full":"Atlanta-Sandy Springs-Alpharetta","city":"ATL","zipcode":30301}'
```

### 1.4. O que não deve ir para o GitHub

Não versionar:

```text
/data/
/artifacts/
/models/
deploy.env
regression_mlops_e2e.egg-info/
```

Credenciais nunca devem ir para o repositório. Se uma access key aparecer em arquivo local, revogue-a na AWS e crie outra.

### 1.5. Variáveis locais de deploy

Crie ou carregue `deploy.env` localmente. Esse arquivo deve ficar no `.gitignore`.

```bash
source deploy.env
```

Exemplo usado neste deploy:

```bash
export AWS_REGION=us-east-1
export PROJECT=regression-mlops-e2e
export ACCOUNT_ID=878919573366
export S3_BUCKET=regression-mlops-e2e-artifacts-878919573366
export ECR_REPO=regression-mlops-e2e-api
export ECR_URI=878919573366.dkr.ecr.us-east-1.amazonaws.com/regression-mlops-e2e-api
export TASK_FAMILY=regression-mlops-e2e-api
export ECS_CLUSTER=regression-mlops-e2e-cluster
export ECS_SERVICE=regression-mlops-e2e-service
export CONTAINER_NAME=api
export CONTAINER_PORT=8000
export LOG_GROUP=/ecs/regression-mlops-e2e-api
export AWS_PAGER=""
```

Por que essas variáveis existem:

- `AWS_REGION`: região onde os recursos serão criados.
- `PROJECT`: prefixo para nomes consistentes.
- `ACCOUNT_ID`: evita hardcode manual em ARNs.
- `S3_BUCKET`: bucket de artifacts do modelo.
- `ECR_REPO` e `ECR_URI`: repositório Docker da API.
- `TASK_FAMILY`: família da task definition ECS.
- `ECS_CLUSTER` e `ECS_SERVICE`: destino do deploy.
- `CONTAINER_NAME` e `CONTAINER_PORT`: usados pelo service e pelo workflow.
- `LOG_GROUP`: destino dos logs no CloudWatch.
- `AWS_PAGER`: evita que a AWS CLI prenda o terminal em saídas longas.

## 2. Arquitetura dos serviços AWS

### 2.1. Amazon S3

Guarda artifacts de ML fora da imagem Docker.

Por quê: artifacts são grandes, mudam de forma independente do código e podem ser versionados/promovidos separadamente.

### 2.2. Amazon ECR

Guarda as imagens Docker da API.

Por quê: ECS Fargate puxa imagens diretamente do ECR com IAM, sem depender de registry externo.

### 2.3. IAM Roles

Há três papéis principais:

```text
ecsTaskExecutionRole
  -> usada pelo ECS/Fargate
  -> puxa imagem do ECR
  -> envia logs para CloudWatch

regressionApiTaskRole
  -> usada pelo código dentro do container
  -> permite boto3 ler artifacts no S3

GitHubActionsRegressionMLOpsDeployRole
  -> assumida pelo GitHub Actions via OIDC
  -> permite push no ECR e update no ECS
```

Essa separação é importante. A role de execução não deve ser usada pelo código da aplicação, e a task role não precisa de permissões para gerenciar deploy.

### 2.4. ECS Fargate

Executa a API sem gerenciar servidores EC2.

Configuração usada:

```text
CPU: 1024
Memory: 4096 MB
Network mode: awsvpc
Porta do container: 8000
Health check: /health
```

Começamos com 4 GB porque a API baixa artifacts e carrega modelo/metadata em memória. Depois de medir uso real, dá para reduzir custo.

### 2.5. Application Load Balancer

Recebe tráfego HTTP público na porta 80 e encaminha para a task na porta 8000.

Por quê: o ALB oferece endpoint estável, health checks e caminho natural para HTTPS depois.

### 2.6. CloudWatch Logs

Recebe logs stdout/stderr do container via driver `awslogs`.

Por quê: é o primeiro ponto de diagnóstico para erro de startup, S3, permissões, health check e exceptions da API.

## 3. Criar S3 para artifacts

```bash
aws s3 mb s3://${S3_BUCKET} --region ${AWS_REGION}
```

Bloquear acesso público:

```bash
aws s3api put-public-access-block \
  --bucket ${S3_BUCKET} \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

Subir artifacts:

```bash
aws s3 cp models/xgboost_tuned_model.pkl s3://${S3_BUCKET}/models/xgboost_tuned_model.pkl
aws s3 cp models/best_model.pkl s3://${S3_BUCKET}/models/best_model.pkl
aws s3 cp models/zipcode_frequency_encoder.pkl s3://${S3_BUCKET}/models/zipcode_frequency_encoder.pkl
aws s3 cp models/city_full_target_encoder.pkl s3://${S3_BUCKET}/models/city_full_target_encoder.pkl
aws s3 cp data/processed/feature_engineered_train.csv s3://${S3_BUCKET}/data/processed/feature_engineered_train.csv
```

Validar:

```bash
aws s3 ls s3://${S3_BUCKET}/ --recursive --human-readable --summarize
```

## 4. Validar localmente o modo S3

Rode o container com `AWS_S3_BUCKET`. Localmente, o container precisa de credenciais AWS. Em produção, isso virá da ECS task role.

```bash
docker run --rm -p 8000:8000 \
  -e AWS_REGION=${AWS_REGION} \
  -e AWS_S3_BUCKET=${S3_BUCKET} \
  -e ARTIFACT_CACHE_DIR=/tmp/artifacts \
  -v ~/.aws:/root/.aws:ro   regression-mlops-e2e-api:local
```

Validar:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

O `model_version` deve conter `s3://<bucket>`.

## 5. Criar ECR e publicar imagem inicial

```bash
aws ecr create-repository \
  --repository-name ${ECR_REPO} \
  --image-scanning-configuration scanOnPush=true \
  --region ${AWS_REGION}
```

Login:

```bash
aws ecr get-login-password --region ${AWS_REGION}   | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

Build, tag e push:

```bash
export IMAGE_TAG=local-s3-tested

docker build -f Dockerfile.api -t regression-mlops-e2e-api:local .
docker tag regression-mlops-e2e-api:local ${ECR_URI}:${IMAGE_TAG}
docker push ${ECR_URI}:${IMAGE_TAG}
```

Validar:

```bash
aws ecr describe-images \
  --repository-name ${ECR_REPO} \
  --region ${AWS_REGION} \
  --query 'imageDetails[*].[imageTags,imageSizeInBytes,imagePushedAt]' \
  --output table
```

## 6. Criar IAM roles para ECS

### 6.1. Trust policy do ECS

Arquivo: `infra/aws/ecs-trust-policy.json`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Criar execution role:

```bash
export ECS_EXECUTION_ROLE=ecsTaskExecutionRole

aws iam create-role \
  --role-name ${ECS_EXECUTION_ROLE} \
  --assume-role-policy-document file://infra/aws/ecs-trust-policy.json

aws iam attach-role-policy \
  --role-name ${ECS_EXECUTION_ROLE} \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

Criar task role:

```bash
export ECS_TASK_ROLE=regressionApiTaskRole

aws iam create-role \
  --role-name ${ECS_TASK_ROLE} \
  --assume-role-policy-document file://infra/aws/ecs-trust-policy.json
```

Policy S3 da aplicação: `infra/aws/task-role-s3-policy.json`.

```bash
aws iam put-role-policy \
  --role-name ${ECS_TASK_ROLE} \
  --policy-name regression-mlops-e2e-read-artifacts-s3 \
  --policy-document file://infra/aws/task-role-s3-policy.json
```

Validar:

```bash
aws iam get-role --role-name ${ECS_EXECUTION_ROLE}
aws iam list-attached-role-policies --role-name ${ECS_EXECUTION_ROLE}
aws iam get-role --role-name ${ECS_TASK_ROLE}
aws iam list-role-policies --role-name ${ECS_TASK_ROLE}
```

## 7. Criar CloudWatch Log Group

```bash
aws logs create-log-group \
  --log-group-name ${LOG_GROUP} \
  --region ${AWS_REGION}

aws logs put-retention-policy \
  --log-group-name ${LOG_GROUP} \
  --retention-in-days 14 \
  --region ${AWS_REGION}
```

Validar:

```bash
aws logs describe-log-groups \
  --log-group-name-prefix ${LOG_GROUP} \
  --region ${AWS_REGION} \
  --query 'logGroups[*].[logGroupName,retentionInDays]' \
  --output table
```

## 8. Criar ECS Cluster

```bash
aws ecs create-cluster \
  --cluster-name ${ECS_CLUSTER} \
  --region ${AWS_REGION}
```

Validar:

```bash
aws ecs describe-clusters \
  --clusters ${ECS_CLUSTER} \
  --region ${AWS_REGION} \
  --query 'clusters[*].[clusterName,status,registeredContainerInstancesCount,runningTasksCount,pendingTasksCount]' \
  --output table
```

Em Fargate, `registeredContainerInstancesCount=0` é normal.

## 9. Rede, subnets e security groups

Para o primeiro deploy, foi usada a VPC default com subnets públicas.

Descobrir VPC:

```bash
aws ec2 describe-vpcs \
  --region ${AWS_REGION} \
  --query 'Vpcs[*].[VpcId,IsDefault,CidrBlock]' \
  --output table
```

Descobrir subnets:

```bash
aws ec2 describe-subnets \
  --region ${AWS_REGION} \
  --query 'Subnets[*].[SubnetId,VpcId,AvailabilityZone,CidrBlock,MapPublicIpOnLaunch]' \
  --output table
```

Exemplo usado:

```bash
export VPC_ID=vpc-05b4755b7a4c6d202
export SUBNET_1=subnet-09c4bb8de5fd8ef03
export SUBNET_2=subnet-08f39db837b9b79ec
```

Criar security group do ALB:

```bash
export ALB_SG_NAME=${PROJECT}-alb-sg

export ALB_SG_ID=$(aws ec2 create-security-group \
  --group-name ${ALB_SG_NAME} \
  --description "Allow HTTP access to regression MLOps ALB" \
  --vpc-id ${VPC_ID} \
  --region ${AWS_REGION} \
  --query 'GroupId' \
  --output text)

aws ec2 authorize-security-group-ingress \
  --group-id ${ALB_SG_ID} \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0 \
  --region ${AWS_REGION}
```

Criar security group da task ECS:

```bash
export ECS_SG_NAME=${PROJECT}-ecs-sg

export ECS_SG_ID=$(aws ec2 create-security-group \
  --group-name ${ECS_SG_NAME} \
  --description "Allow API traffic from ALB to ECS Fargate task" \
  --vpc-id ${VPC_ID} \
  --region ${AWS_REGION} \
  --query 'GroupId' \
  --output text)

aws ec2 authorize-security-group-ingress \
  --group-id ${ECS_SG_ID} \
  --protocol tcp \
  --port 8000 \
  --source-group ${ALB_SG_ID} \
  --region ${AWS_REGION}
```

Por quê: o mundo acessa apenas o ALB na porta 80. A task ECS aceita tráfego na porta 8000 somente vindo do ALB.

## 10. Criar Application Load Balancer e Target Group

```bash
export ALB_NAME=${PROJECT}-alb
export TG_NAME=${PROJECT}-tg
```

Criar ALB:

```bash
export ALB_ARN=$(aws elbv2 create-load-balancer \
  --name ${ALB_NAME} \
  --subnets ${SUBNET_1} ${SUBNET_2} \
  --security-groups ${ALB_SG_ID} \
  --scheme internet-facing \
  --type application \
  --ip-address-type ipv4 \
  --region ${AWS_REGION} \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)
```

Criar Target Group:

```bash
export TG_ARN=$(aws elbv2 create-target-group \
  --name ${TG_NAME} \
  --protocol HTTP \
  --port 8000 \
  --vpc-id ${VPC_ID} \
  --target-type ip \
  --health-check-protocol HTTP \
  --health-check-path /health \
  --health-check-port traffic-port \
  --matcher HttpCode=200 \
  --region ${AWS_REGION} \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)
```

Criar listener HTTP:

```bash
aws elbv2 create-listener \
  --load-balancer-arn ${ALB_ARN} \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=${TG_ARN} \
  --region ${AWS_REGION}
```

Capturar DNS:

```bash
export ALB_DNS=$(aws elbv2 describe-load-balancers \
  --load-balancer-arns ${ALB_ARN} \
  --region ${AWS_REGION} \
  --query 'LoadBalancers[0].DNSName' \
  --output text)

echo ${ALB_DNS}
```

## 11. Registrar ECS Task Definition

Arquivo base: `infra/aws/task-definition.json`.

Pontos importantes:

- `executionRoleArn`: role usada pelo ECS para ECR e logs.
- `taskRoleArn`: role usada pela aplicação para S3.
- `AWS_S3_BUCKET`: ativa download de artifacts no startup.
- `ARTIFACT_CACHE_DIR=/tmp/artifacts`: cache efêmero da task.
- `awslogs`: envia logs para CloudWatch.
- `healthCheck`: valida `/health` dentro do container.

Validar JSON:

```bash
python3 -m json.tool infra/aws/task-definition.json > /tmp/task-definition.validated.json
```

Registrar:

```bash
aws ecs register-task-definition \
  --cli-input-json file://infra/aws/task-definition.json \
  --region ${AWS_REGION}
```

Validar:

```bash
aws ecs describe-task-definition \
  --task-definition ${TASK_FAMILY} \
  --region ${AWS_REGION} \
  --query 'taskDefinition.[family,revision,cpu,memory,containerDefinitions[0].image]' \
  --output table
```

## 12. Criar ECS Service

```bash
aws ecs create-service \
  --cluster ${ECS_CLUSTER} \
  --service-name ${ECS_SERVICE} \
  --task-definition ${TASK_FAMILY} \
  --desired-count 1 \
  --launch-type FARGATE \
  --platform-version LATEST \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_1},${SUBNET_2}],securityGroups=[${ECS_SG_ID}],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=${TG_ARN},containerName=${CONTAINER_NAME},containerPort=${CONTAINER_PORT}" \
  --health-check-grace-period-seconds 180 \
  --region ${AWS_REGION}
```

Por que `assignPublicIp=ENABLED`: nesta primeira versão usamos subnets públicas da VPC default. A task precisa sair para ECR, S3 e CloudWatch sem NAT Gateway. Em produção mais madura, prefira subnets privadas com NAT Gateway ou VPC endpoints.

Aguardar estabilidade:

```bash
aws ecs wait services-stable \
  --cluster ${ECS_CLUSTER} \
  --services ${ECS_SERVICE} \
  --region ${AWS_REGION}
```

Validar service:

```bash
aws ecs describe-services \
  --cluster ${ECS_CLUSTER} \
  --services ${ECS_SERVICE} \
  --region ${AWS_REGION} \
  --query 'services[*].[serviceName,status,desiredCount,runningCount,pendingCount,deployments[0].rolloutState]' \
  --output table
```

Validar Target Group:

```bash
aws elbv2 describe-target-health \
  --target-group-arn ${TG_ARN} \
  --region ${AWS_REGION} \
  --query 'TargetHealthDescriptions[*].[Target.Id,Target.Port,TargetHealth.State,TargetHealth.Reason,TargetHealth.Description]' \
  --output table
```

Testar API pública:

```bash
curl http://${ALB_DNS}/health
curl http://${ALB_DNS}/model-info
curl -X POST http://${ALB_DNS}/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2022-01-01","city_full":"Atlanta-Sandy Springs-Alpharetta","city":"ATL","zipcode":30301}'
```

## 13. Publicar projeto no GitHub

Antes de publicar:

```bash
git status --porcelain=v1
git ls-files | xargs -r du -h 2>/dev/null | sort -hr | sed -n '1,40p'
rg -n "AKIA|AWS_SECRET_ACCESS_KEY|secret access key|aws_access_key_id|aws_secret_access_key" -S . \
  -g '!data/**' \
  -g '!artifacts/**' \
  -g '!deploy.env' \
  -g '!.venv/**'
```

Criar repo privado:

```bash
gh repo create Regression_MLOps_e2e --private --source=. --remote=origin --push
```

Se o primeiro push falhar, tente:

```bash
git push -u origin main
```

Repo usado:

```text
FabioCLima/Regression_MLOps_e2e
```

## 14. Configurar GitHub Actions com OIDC

### 14.1. Criar OIDC provider na AWS

Verificar se já existe:

```bash
aws iam list-open-id-connect-providers \
  --query 'OpenIDConnectProviderList[*].Arn' \
  --output text
```

Criar se não existir:

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com
```

### 14.2. Trust policy do GitHub Actions

Arquivo: `infra/aws/github-oidc-trust-policy.json`.

Ela restringe a role ao repo e branch:

```text
repo:FabioCLima/Regression_MLOps_e2e:ref:refs/heads/main
```

Isso impede que outro repositório assuma a mesma role.

### 14.3. Criar role do GitHub Actions

```bash
export GITHUB_ACTIONS_ROLE=GitHubActionsRegressionMLOpsDeployRole

aws iam create-role \
  --role-name ${GITHUB_ACTIONS_ROLE} \
  --assume-role-policy-document file://infra/aws/github-oidc-trust-policy.json
```

Anexar policy de deploy:

```bash
aws iam put-role-policy \
  --role-name ${GITHUB_ACTIONS_ROLE} \
  --policy-name regression-mlops-e2e-github-actions-deploy \
  --policy-document file://infra/aws/github-actions-deploy-policy.json
```

Validar:

```bash
aws iam get-role \
  --role-name ${GITHUB_ACTIONS_ROLE} \
  --query 'Role.[RoleName,Arn]' \
  --output table
```

Role criada:

```text
arn:aws:iam::878919573366:role/GitHubActionsRegressionMLOpsDeployRole
```

### 14.4. Configurar GitHub repository variables

```bash
gh variable set AWS_REGION --body us-east-1 --repo FabioCLima/Regression_MLOps_e2e
gh variable set AWS_ROLE_TO_ASSUME --body arn:aws:iam::878919573366:role/GitHubActionsRegressionMLOpsDeployRole --repo FabioCLima/Regression_MLOps_e2e
gh variable set ECR_REPOSITORY --body regression-mlops-e2e-api --repo FabioCLima/Regression_MLOps_e2e
gh variable set ECS_CLUSTER --body regression-mlops-e2e-cluster --repo FabioCLima/Regression_MLOps_e2e
gh variable set ECS_SERVICE --body regression-mlops-e2e-service --repo FabioCLima/Regression_MLOps_e2e
```

Validar:

```bash
gh variable list --repo FabioCLima/Regression_MLOps_e2e
```

### 14.5. Workflow CD

Arquivo: `.github/workflows/cd.yml`.

Fluxo:

```text
push na main
  -> checkout
  -> instala uv
  -> instala Python
  -> uv sync
  -> ruff
  -> pytest
  -> assume role AWS via OIDC
  -> login no ECR
  -> build e push Docker com tag github.sha
  -> render ECS task definition
  -> deploy ECS service
  -> espera estabilidade
```

A task definition usada pelo workflow é `infra/aws/task-definition.cd.json`. O action `amazon-ecs-render-task-definition` troca a imagem antiga pela imagem nova com tag do commit.

Validar runs:

```bash
gh run list --repo FabioCLima/Regression_MLOps_e2e --limit 10
```

Ver log de falha:

```bash
gh run view <RUN_ID> --repo FabioCLima/Regression_MLOps_e2e --log-failed
```

## 15. Validação final esperada

CI:

```text
completed success
```

CD:

```text
completed success
```

ECS:

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

API pública:

```bash
curl http://${ALB_DNS}/health
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

## 16. Troubleshooting

### AWS CLI fica presa no caractere `:`

Isso geralmente é o pager da AWS CLI. Saia com `q` e exporte:

```bash
export AWS_PAGER=""
```

### Variável vazia quebra comandos

Erro típico:

```text
aws: error: argument --cluster: expected one argument
```

Causa: variável como `${ECS_CLUSTER}` está vazia.

Solução:

```bash
source deploy.env
echo $ECS_CLUSTER
```

### Docker local não acessa S3

Erro:

```text
Unable to locate credentials
```

Solução local:

```bash
docker run --rm -p 8000:8000 \
  -e AWS_REGION=${AWS_REGION} \
  -e AWS_S3_BUCKET=${S3_BUCKET} \
  -e ARTIFACT_CACHE_DIR=/tmp/artifacts \
  -v ~/.aws:/root/.aws:ro   regression-mlops-e2e-api:local
```

No ECS, não use access keys. Use `taskRoleArn`.

### Porta 8000 ocupada localmente

```bash
docker ps
docker stop <CONTAINER_ID>
```

Ou rode em outra porta local:

```bash
docker run --rm -p 8001:8000 regression-mlops-e2e-api:local
```

### ECS service demora para estabilizar

É normal demorar alguns minutos, porque a task precisa:

1. puxar a imagem do ECR;
2. iniciar o container;
3. baixar artifacts do S3;
4. carregar modelo e encoders;
5. passar no health check do ALB;
6. drenar a task antiga.

Comandos úteis:

```bash
aws ecs describe-services \
  --cluster ${ECS_CLUSTER} \
  --services ${ECS_SERVICE} \
  --region ${AWS_REGION} \
  --query 'services[0].events[0:10].[createdAt,message]' \
  --output table

aws elbv2 describe-target-health \
  --target-group-arn ${TG_ARN} \
  --region ${AWS_REGION} \
  --query 'TargetHealthDescriptions[*].[Target.Id,Target.Port,TargetHealth.State,TargetHealth.Reason,TargetHealth.Description]' \
  --output table
```

### CI falha com `ModuleNotFoundError: No module named src.data`

Causa possível: `.gitignore` ignorando `data/` de forma ampla e bloqueando `src/data`.

Use `/data/` para ignorar apenas a pasta de dados da raiz.

### Docker build no GitHub falha copiando `models/` ou `data/`

A imagem de produção não deve depender de artifacts locais. Remova `COPY models` e `COPY data/...` do Dockerfile. A API deve baixar artifacts do S3 quando `AWS_S3_BUCKET` estiver definido.

## 17. Próximas melhorias

- Configurar HTTPS no ALB com ACM.
- Criar domínio no Route 53.
- Mover ECS tasks para subnets privadas.
- Criar VPC endpoints para S3, ECR e CloudWatch.
- Reduzir CPU/memória após medir uso real.
- Criar lifecycle policy no ECR para limpar imagens antigas.
- Criar versionamento formal de artifacts no S3.
- Adicionar monitoramento de drift usando holdout como baseline.
- Adicionar frontend consumindo o endpoint do ALB.

## 18. Limpeza de recursos

Para evitar custos, remova recursos quando não estiver usando.

Ordem segura:

```bash
aws ecs update-service --cluster ${ECS_CLUSTER} --service ${ECS_SERVICE} --desired-count 0 --region ${AWS_REGION}
aws ecs delete-service --cluster ${ECS_CLUSTER} --service ${ECS_SERVICE} --force --region ${AWS_REGION}
aws ecs delete-cluster --cluster ${ECS_CLUSTER} --region ${AWS_REGION}
```

Também remova ALB, target group, security groups, ECR, log group e bucket S3 se o ambiente for descartável.

Cuidado: apagar bucket S3 remove artifacts do modelo.
