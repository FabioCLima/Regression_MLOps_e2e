---
title: Weights & Biases para ML Engineering
aliases:
  - W&B
  - Weights and Biases
  - WandB
  - Guia W&B
created: 2026-05-16
type: nota
topic:
  - mlops
  - experiment-tracking
  - model-registry
  - artifacts
  - observability
project: regression-mlops-e2e
entity: fabio_lima07-mlops
platform: Weights & Biases
tags:
  - mlops
  - wandb
  - machine-learning
  - experiment-tracking
  - model-registry
  - artifacts
  - reproducibility
status: draft
---

# Weights & Biases para ML Engineering

> [!summary]
> Esta nota explica o minimo essencial para usar o Weights & Biases com visao de [[MLOps]], [[Machine Learning Engineering]] e negocio. A ideia nao e apenas abrir graficos, mas entender como a plataforma ajuda a rastrear experimentos, dados, modelos, metricas e decisoes.

## Links rapidos

- Projeto atual: [regression-mlops-e2e](https://wandb.ai/fabio_lima07-mlops/regression-mlops-e2e)
- Entity: `fabio_lima07-mlops`
- Project: `regression-mlops-e2e`
- Ultimo run analisado: [4akpg0c3](https://wandb.ai/fabio_lima07-mlops/regression-mlops-e2e/runs/4akpg0c3)

## Mapa da nota

- [[#O que e o W&B]]
- [[#Por que isso importa para MLOps]]
- [[#Conceitos essenciais]]
- [[#Como olhar este projeto no W&B]]
- [[#Checklist minimo para um novo projeto]]
- [[#Perguntas de negocio]]
- [[#Mentalidade de ML Engineer]]
- [[#Fontes oficiais]]

---

## O que e o W&B

O [[Weights & Biases]] e uma plataforma de observabilidade, rastreabilidade e governanca para projetos de [[Machine Learning]].

Uma forma pratica de pensar:

```text
W&B = dashboard + historico de experimentos + versionamento de artefatos + lineage
```

Ele registra:

- o que foi executado;
- qual configuracao foi usada;
- qual versao dos dados alimentou o processo;
- quais metricas foram geradas;
- quais arquivos, datasets ou modelos sairam da execucao;
- como cada etapa se conecta no pipeline.

Em vez de depender de memoria, notebooks soltos ou arquivos locais, o W&B cria um historico estruturado dos experimentos.

> [!important]
> O W&B nao deve ser tratado apenas como um dashboard bonito. Ele deve funcionar como uma camada de memoria, rastreabilidade e decisao para projetos de ML.

---

## Por que isso importa para MLOps

Em [[MLOps]], o problema nao e apenas treinar um modelo. O problema real e conseguir responder:

- Qual modelo foi treinado?
- Com quais dados?
- Com quais parametros?
- Qual metrica foi usada para comparar?
- Quem ou qual pipeline gerou esse modelo?
- O resultado pode ser reproduzido daqui a meses?
- Esse modelo pode ser promovido para staging ou producao?

Se o projeto nao responde essas perguntas, ele ainda esta em um nivel experimental fraco.

O W&B ajuda a conectar tres camadas:

1. **Experimento tecnico**: runs, metricas, configs e graficos.
2. **Rastreabilidade de engenharia**: artifacts, lineage, datasets e modelos versionados.
3. **Decisao de negocio**: comparacao, reports, riscos e recomendacao de modelo.

---

## Conceitos essenciais

### Project

Um `project` no W&B e o espaco onde ficam todos os experimentos relacionados a um problema, produto, pipeline ou caso de uso.

No projeto atual:

```text
entity: fabio_lima07-mlops
project: regression-mlops-e2e
```

Esse projeto deveria ajudar a responder:

- Qual foi o melhor modelo para prever preco?
- Quais dados foram usados para treinar o modelo?
- Qual pipeline gerou esse modelo?
- Qual metrica foi otimizada?
- Qual versao pode ser usada em producao?
- O resultado e reproduzivel?

> [!tip]
> Se um projeto W&B nao responde essas perguntas, ele esta apenas armazenando logs. O objetivo e transformar o projeto em uma ferramenta de engenharia, auditoria e decisao.

### Run

Um `run` e uma execucao especifica dentro de um projeto.

Exemplos de runs vistos neste projeto:

```text
wise-fog-26
still-grass-25
devoted-aardvark-24
```

Cada run representa uma tentativa controlada, como:

- preprocessing;
- feature engineering;
- treinamento;
- tuning;
- avaliacao;
- inferencia.

O minimo que deve ser analisado em um run:

- `state`: terminou, falhou ou esta rodando;
- `job_type`: tipo da execucao, como `train_model`, `tune_model` ou `feature_engineering`;
- `config`: parametros usados;
- `summary`: resultados finais;
- `charts`: evolucao das metricas;
- `files`: arquivos gerados;
- `artifacts`: entradas e saidas versionadas.

> [!question]
> Sempre pergunte: esse resultado veio de qual dado, qual codigo e qual configuracao?

### Metrics

Metricas sao a ponte entre experimento tecnico e decisao de negocio.

Em um projeto de regressao, metricas comuns sao:

```text
RMSE
MAE
R2
train/rmse
eval/rmse
tuned/eval/rmse
```

Leitura basica:

| Metrica | Como interpretar |
| --- | --- |
| `RMSE` | Penaliza mais erros grandes. |
| `MAE` | Erro medio mais facil de explicar. |
| `R2` | Mede quanto da variabilidade do alvo e explicada pelo modelo. |

A pergunta principal nao e apenas:

```text
O modelo melhorou?
```

A pergunta madura e:

```text
O erro atual e aceitavel para a decisao de negocio que o modelo precisa apoiar?
```

Em previsao de preco, por exemplo, o erro precisa ser interpretado financeiramente. Um RMSE alto pode ser aceitavel dependendo da escala dos imoveis, mas pode ser ruim se afetar uma decisao automatizada de precificacao.

### Config

Tudo que pode influenciar o resultado deve ser salvo em `config`.

Exemplos:

```text
model_name
learning_rate
n_estimators
max_depth
sample_frac
features_used
target_column
random_state
primary_metric
```

Sem config, fica dificil entender por que um run foi melhor ou pior que outro.

> [!note]
> Um bom run precisa responder: o que mudou nesta execucao?

### Artifacts

Artifacts sao objetos versionados pelo W&B. Eles representam entradas e saidas de runs.

Exemplos deste projeto:

```text
house_ts_raw
house_ts_processed
house_ts_cleaned
house_ts_features
house_ts_best_model
house_ts_tuned_model
```

Artifacts podem representar:

- dataset bruto;
- dataset processado;
- dataset limpo;
- features;
- encoders;
- modelos treinados;
- metricas;
- arquivos de predicao;
- relatorios intermediarios.

Ao olhar um artifact, verificar:

- qual run criou o artifact;
- quais runs consumiram o artifact;
- qual e a versao mais recente;
- se existe metadata explicando o conteudo;
- se o artifact participa de um lineage claro.

> [!important]
> Artifacts sao uma das partes mais importantes do W&B para MLOps, porque permitem responder: qual versao de dado gerou qual versao de modelo?

### Lineage

Lineage e o mapa de dependencias entre dados, transformacoes, runs e modelos.

Fluxo ideal para este projeto:

```text
raw dataset
  -> preprocessing
  -> cleaned dataset
  -> feature engineering
  -> feature dataset
  -> training
  -> best model
  -> tuning
  -> tuned model
```

O valor do lineage e conseguir explicar a origem de um modelo.

Resposta fraca:

```text
Acho que esse modelo veio daquele notebook.
```

Resposta forte:

```text
Esse modelo veio deste run, usando esta versao de features, gerada a partir desta versao de dataset limpo.
```

Isso e essencial para reproducibilidade, auditoria e confianca.

### Tables

Tables permitem registrar dados tabulares no W&B.

Em regressao, elas sao uteis para analisar previsoes linha a linha.

Colunas uteis:

```text
y_true
y_pred
error
absolute_error
city
zipcode
date
model_name
```

Isso ajuda a sair de uma leitura superficial:

```text
O RMSE melhorou.
```

Para uma leitura mais proxima do negocio:

```text
Onde o modelo erra mais?
Em quais cidades?
Em quais faixas de preco?
Em quais periodos?
```

### Sweeps

Sweeps sao execucoes automatizadas para busca de hiperparametros.

Estrategias comuns:

- grid search;
- random search;
- bayesian optimization.

Parametros comuns em sweeps:

```text
learning_rate
max_depth
n_estimators
subsample
colsample_bytree
regularization
```

> [!warning]
> Nao comece um projeto pelo sweep. Primeiro garanta pipeline, metricas, artifacts e comparacao manual de runs. Depois automatize a busca.

### Reports

Reports servem para contar a historia do experimento.

Um bom report deve responder:

- Qual era o objetivo?
- Qual dataset foi usado?
- Quais modelos foram testados?
- Qual foi o melhor resultado?
- Quais trade-offs foram encontrados?
- Qual modelo e recomendado?
- Quais riscos continuam existindo?
- Quais sao os proximos passos?

Reports transformam experimentos em comunicacao tecnica e de negocio.

### Model Registry

O [[Model Registry]] organiza modelos candidatos e modelos promovidos.

Estagios comuns:

```text
candidate
staging
production
archived
```

Um modelo nao deve ser considerado pronto apenas porque teve a melhor metrica. Ele precisa ser:

- versionado;
- rastreavel;
- comparado com alternativas;
- validado em dados adequados;
- aprovado para o contexto de uso;
- monitorado depois da promocao.

---

## Como olhar este projeto no W&B

Use esta sequencia para estudar o projeto atual:

- [ ] Abrir o projeto: [regression-mlops-e2e](https://wandb.ai/fabio_lima07-mlops/regression-mlops-e2e)
- [ ] Entrar em `Runs`.
- [ ] Ordenar pelos runs mais recentes.
- [ ] Identificar os tipos de execucao.

Tipos esperados:

```text
feature_engineering
train_model
tune_model
```

Depois:

- [ ] Comparar os runs de treinamento e tuning.
- [ ] Procurar a metrica primaria.

Metricas provaveis:

```text
eval/rmse
tuned/eval/rmse
```

Artifacts principais:

```text
house_ts_features
house_ts_best_model
house_ts_tuned_model
```

Checklist de leitura:

- [ ] Abrir `house_ts_features`.
- [ ] Ver qual run criou esse artifact.
- [ ] Abrir `house_ts_best_model`.
- [ ] Ver quais dados alimentaram o modelo.
- [ ] Abrir `house_ts_tuned_model`.
- [ ] Comparar modelo base e modelo tunado.
- [ ] Verificar lineage dos artifacts.
- [ ] Criar um report curto com a conclusao.

---

## Checklist minimo para um novo projeto

Ao criar um novo projeto com W&B, garanta pelo menos:

- [ ] `wandb.init` com `entity`.
- [ ] `wandb.init` com `project`.
- [ ] `wandb.init` com `job_type`.
- [ ] `wandb.init` com `config`.
- [ ] Metricas de treino.
- [ ] Metricas de validacao ou teste.
- [ ] Dataset registrado como artifact.
- [ ] Modelo registrado como artifact.
- [ ] Metadata minima dos artifacts.
- [ ] Tags uteis.
- [ ] Comparacao entre runs.
- [ ] Report final ou parcial.

Exemplo basico:

```python
import wandb

with wandb.init(
    entity="sua-entidade",
    project="nome-do-projeto",
    job_type="train",
    config={
        "model": "xgboost",
        "target": "price",
        "primary_metric": "rmse",
        "random_state": 42,
    },
) as run:
    run.log(
        {
            "train/rmse": 12000.0,
            "eval/rmse": 18000.0,
            "eval/mae": 9500.0,
            "eval/r2": 0.82,
        }
    )
```

Tags uteis:

```text
baseline
tuning
production-candidate
feature-engineering
xgboost
regression
```

---

## O que um bom run deve ter

- [ ] Nome ou identificacao clara.
- [ ] `job_type` adequado.
- [ ] Configuracao do experimento.
- [ ] Metricas principais.
- [ ] Metricas secundarias.
- [ ] Logs relevantes.
- [ ] Artifact de entrada quando aplicavel.
- [ ] Artifact de saida quando aplicavel.
- [ ] Tags que facilitem filtragem.
- [ ] Estado final bem definido.

---

## O que um bom artifact deve ter

- [ ] Nome estavel.
- [ ] Tipo correto, como `dataset`, `model` ou `predictions`.
- [ ] Versao.
- [ ] Metadata.
- [ ] Descricao.
- [ ] Relacao clara com o run que o criou.
- [ ] Relacao clara com os runs que o consumiram.

Metadata util para modelo:

```text
model_name
training_date
primary_metric
primary_metric_value
features_version
dataset_version
random_state
framework
```

---

## Perguntas de negocio

O W&B deve ajudar a responder:

- Qual modelo esta performando melhor?
- Essa melhora e relevante para o negocio?
- O modelo melhorou apenas no treino ou tambem na validacao?
- O modelo esta estavel entre diferentes execucoes?
- Quais dados alimentaram o modelo final?
- O modelo final e reproduzivel?
- Qual artifact representa a versao candidata?
- Existe risco de data leakage?
- Existe overfitting?
- Quais segmentos tem maior erro?
- O modelo esta pronto para staging ou ainda e apenas experimental?

---

## Mentalidade de ML Engineer

A pergunta inicial costuma ser:

```text
Qual modelo performou melhor?
```

A pergunta madura e:

```text
Qual modelo performou melhor, usando quais dados, com qual configuracao, em qual versao do pipeline, com quais riscos, e eu consigo reproduzir isso daqui a tres meses?
```

> [!quote]
> Essa e a virada principal: sair de "treinei um modelo" para "tenho uma evidencia rastreavel, comparavel e reproduzivel".

---

## Must know resumido

| Area | O que saber | Por que importa |
| --- | --- | --- |
| Project | Organiza os experimentos de um problema. | Da contexto para runs e artifacts. |
| Run | Representa uma execucao. | Permite comparar tentativas. |
| Config | Guarda parametros. | Explica por que resultados mudaram. |
| Metrics | Medem performance. | Conectam modelo a decisao de negocio. |
| Artifacts | Versionam dados e modelos. | Criam reproducibilidade. |
| Lineage | Mostra dependencias. | Explica de onde veio cada modelo. |
| Tables | Mostram previsoes linha a linha. | Ajudam a entender erros por segmento. |
| Sweeps | Automatizam busca de hiperparametros. | Melhoram experimentacao controlada. |
| Reports | Contam a historia do experimento. | Facilitam comunicacao e decisao. |
| Registry | Organiza modelos promovidos. | Apoia staging, producao e governanca. |

---

## Fontes oficiais

- [W&B Models overview](https://docs.wandb.ai/models)
- [Projects](https://docs.wandb.ai/models/track/project-page)
- [Artifacts](https://docs.wandb.ai/models/artifacts)
- [Sweeps](https://docs.wandb.ai/models/sweeps)

---

## Related

- [[MLOps]]
- [[Experiment Tracking]]
- [[Model Registry]]
- [[Data Versioning]]
- [[Reproducibility]]
- [[Machine Learning Engineering]]
