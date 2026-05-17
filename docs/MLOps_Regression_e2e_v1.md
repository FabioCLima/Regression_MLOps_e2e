# MLOps Regression E2E v1

Este documento registra as principais decisões de negócio, código e engenharia de
machine learning tomadas durante a construção da primeira versão funcional do
projeto. A intenção é servir como memória técnica para manutenção futura e como
material de estudo para quem está aprendendo engenharia de ML aplicada.

## Objetivo Do Projeto

Construir um pipeline de regressão ponta a ponta para previsão de preços no
mercado imobiliário, partindo de dados brutos, passando por preparação de dados,
feature engineering, comparação de modelos, fine tuning e inferência.

O foco desta versão não é apenas treinar um modelo, mas organizar o projeto de
forma profissional:

- componentes reutilizáveis em `src/`;
- pipelines executáveis em `pipelines/`;
- configuração centralizada em `src/config.py`;
- logging padronizado com Loguru;
- versionamento de datasets, features, encoders e modelos no W&B;
- testes unitários e smoke tests para reduzir regressões;
- separação clara entre código de domínio, execução e documentação.

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

## Decisões De Arquitetura

### `src/` Para Código Reutilizável

O pacote `src/` contém funções e classes que representam a lógica real do
projeto: carregar dados, dividir splits, limpar dados, criar features, treinar,
tunar e inferir. Essa separação permite testar cada componente isoladamente e
evita que lógica importante fique presa dentro de scripts.

### `pipelines/` Para Orquestração

Os arquivos em `pipelines/` chamam os componentes de `src/` em uma ordem de
execução. Eles são a camada de workflow do projeto.

Exemplo:

- `data_pipeline.py`: split temporal e preprocessing;
- `feature_pipeline.py`: feature engineering e persistência dos encoders;
- `training_pipeline.py`: comparação entre modelos baseline;
- `tuning_pipeline.py`: fine tuning do melhor modelo suportado;
- `inference_pipeline.py`: predição em dados raw-compatible;
- `machine_learning_pipeline.py`: orquestrador central.

Essa decisão evita que um único `main.py` cresça demais e fique difícil de
manter.

### Configuração Centralizada Com Pydantic

O arquivo `src/config.py` concentra paths e parâmetros do projeto usando modelos
Pydantic. Isso deixa explícito quais configurações existem e reduz constantes
duplicadas espalhadas pelo código.

Foram criadas configurações para:

- paths do projeto;
- nomes de artefatos do W&B;
- split temporal;
- preprocessing;
- feature engineering;
- treinamento;
- fine tuning;
- inferência.

Trade-off: para um projeto muito pequeno, constantes soltas seriam mais simples.
Aqui, como o projeto já tem vários módulos e artefatos, a configuração tipada
ajuda a manter consistência.

### Logging Com Loguru

O logging foi centralizado em `src/logging_config.py`. Cada pipeline pode ser
executado de forma independente com logs ativos. O orquestrador central usa o
mesmo logger, mas chama os pipelines sem reconfigurar o sink a cada etapa.

Logs são gravados em `logs/` com rotação, retenção e saída no terminal.

Decisão importante: prints foram removidos dos componentes principais. Em
projetos profissionais, `print` dificulta rastreabilidade; logs estruturados
permitem acompanhar execuções, erros e sumários.

## Pipeline De Dados

O pipeline de dados é responsável por transformar o dataset bruto em splits
limpos.

Fluxo:

```text
HouseTS.csv
  -> registro do raw dataset no W&B
  -> split temporal train/eval/holdout
  -> preprocessing
  -> registro dos splits limpos no W&B
```

### Decisão De Split Temporal

Como o problema envolve dados com coluna de data, o split não deve ser aleatório.
Foi usado split temporal:

- treino: datas anteriores a `2020-01-01`;
- avaliação: de `2020-01-01` até antes de `2022-01-01`;
- holdout: datas a partir de `2022-01-01`.

Essa decisão reduz vazamento temporal e simula melhor um cenário real, em que o
modelo é treinado no passado para prever o futuro.

### Preprocessing

O preprocessing faz:

- normalização de nomes de cidades;
- mapeamento manual de nomes divergentes;
- merge com `usmetros.csv` para adicionar `lat` e `lng`;
- remoção de duplicatas;
- remoção de outliers extremos em `median_list_price`.

Uma correção importante foi tratar `metro_full` removendo o sufixo de estado
após a vírgula, pois o dataset principal usa nomes de áreas metropolitanas sem
esse sufixo. Isso resolveu o problema de `lat` e `lng` ausentes após o join.

Trade-off: o merge de latitude/longitude foi mantido no preprocessing dos
splits. O ideal seria validar cuidadosamente se a tabela auxiliar é estática e
não traz informação futura. Como `usmetros.csv` contém geografia e não target,
o risco de leakage é baixo.

## Feature Engineering

O pipeline de features recebe os splits limpos e produz datasets prontos para
modelagem.

Transformações:

- criação de features de data: `year`, `quarter`, `month`;
- frequency encoding de `zipcode`;
- target encoding de `city_full`;
- remoção de colunas raw, categóricas ou potencialmente vazadas;
- persistência dos encoders com `joblib`;
- registro dos datasets e encoders no W&B.

Decisão crítica: encoders são ajustados apenas no treino e aplicados em eval e
holdout. Isso evita leakage.

Trade-off: target encoding é poderoso, mas precisa ser usado com cuidado porque
usa a variável alvo para codificar categorias. Nesta versão, o encoder é fitado
somente no train, o que é o mínimo necessário. Em versões futuras, pode-se
avaliar target encoding com validação cruzada temporal para reduzir ainda mais
risco de overfitting.

## Treinamento De Modelos

O treinamento compara múltiplos modelos candidatos:

- `DummyRegressor`;
- regressão linear;
- Ridge;
- Random Forest;
- XGBoost.

O objetivo dessa etapa é estabelecer uma comparação honesta entre baseline
simples, modelos lineares, modelo ensemble clássico e gradient boosting.

Métricas usadas:

- MAE;
- RMSE;
- R2.

A seleção do melhor modelo usa `RMSE` no split de avaliação.

Também são calculadas métricas em treino e avaliação, além do gap entre elas.
Isso ajuda a observar sinais de overfitting ou underfitting.

### Persistência De Modelos

Foi decidido salvar e registrar no W&B apenas o melhor modelo, em vez de salvar
todos os candidatos. Essa decisão foi tomada depois de observar que o Random
Forest podia gerar um `.pkl` muito grande.

Benefícios:

- menor uso de disco;
- menor custo e tempo de upload no W&B;
- menor ruído na árvore de artefatos;
- foco no modelo que realmente será usado adiante.

Trade-off: perde-se a possibilidade de baixar todos os candidatos treinados. As
métricas comparativas continuam preservadas em `metrics.json` e no W&B.

## Fine Tuning

O fine tuning usa Optuna para otimizar hiperparâmetros do XGBoost, mas somente
quando o modelo escolhido na etapa anterior também é XGBoost.

Essa decisão evita tunar cegamente um modelo que não foi selecionado como melhor
baseline.

O MLflow foi intencionalmente adiado. Nesta versão, W&B já cobre rastreabilidade
de artefatos, métricas e metadados. MLflow pode entrar depois, quando o projeto
estiver mais maduro e houver necessidade clara de registry, stages, comparação
mais formal de runs ou integração com serving.

Trade-off: usar apenas W&B simplifica a stack inicial. Usar W&B + MLflow desde o
começo aumentaria complexidade operacional antes de o pipeline estar estável.

## Inferência

Inferência é o uso do modelo treinado para gerar previsões em dados novos.

Fluxo:

```text
raw-compatible CSV
  -> preprocessing
  -> feature engineering com encoders salvos
  -> alinhamento de schema com treino
  -> model.predict
  -> predictions.csv
```

O pipeline de inferência:

- carrega o modelo tunado se existir;
- caso contrário, usa o melhor modelo baseline;
- carrega encoders salvos;
- aplica as mesmas transformações usadas no treino;
- alinha colunas com o schema do treino;
- salva predições;
- calcula métricas se a coluna `price` estiver presente.

Essa etapa é essencial para aproximar o projeto de produção, porque produção não
é apenas treinar modelo: é aplicar o mesmo contrato de transformação em dados
novos.

## Orquestrador Central

Foi criado `pipelines/machine_learning_pipeline.py` como orquestrador central.
Por padrão, ele executa:

```text
data_pipeline
  -> feature_pipeline
  -> training_pipeline
  -> tuning_pipeline
```

A inferência é opcional porque geralmente representa outro momento operacional:
treino/tuning produzem modelo; inferência consome modelo.

Comando principal:

```bash
uv run machine-learning-pipeline
```

Executar incluindo inferência:

```bash
uv run machine-learning-pipeline --include-inference
```

Pular etapas já executadas:

```bash
uv run machine-learning-pipeline --skip-data --skip-features
```

Executar pipelines individuais:

```bash
uv run data-pipeline
uv run feature-pipeline
uv run training-pipeline
uv run tuning-pipeline
uv run inference-pipeline --input data/processed/holdout.csv
```

## W&B

O W&B é usado para versionar e rastrear:

- raw dataset;
- splits processados;
- splits limpos;
- datasets feature-engineered;
- encoders;
- melhor modelo baseline;
- modelo tunado.

Os artefatos de modelo têm descrição, metadados e tags para facilitar busca
futura.

Exemplos de metadados úteis:

- modelo selecionado;
- métrica primária;
- valor da métrica;
- métricas de treino;
- métricas de avaliação;
- gap de generalização;
- dataset de features usado;
- parâmetros do tuning;
- número de trials.

## Testes

Foram criados testes para os componentes principais:

- carregamento de dados;
- split temporal;
- preprocessing;
- feature engineering;
- treinamento;
- tuning;
- inferência;
- orquestrador central.

Tipos de testes usados:

- unitários para funções puras e edge cases;
- smoke test para garantir que o orquestrador chama as etapas esperadas;
- testes com monkeypatch para evitar chamadas reais ao W&B ou treino pesado.

Essa abordagem mantém feedback rápido durante desenvolvimento.

## Decisões De Negócio

O target do projeto é `price`.

O holdout temporal deve ser preservado para avaliação final ou simulação de
dados futuros. Ele não deve ser usado para escolher hiperparâmetros no ciclo
normal de tuning.

O foco inicial é ter uma versão funcional, rastreável e simples o suficiente
para manutenção. Otimizações mais sofisticadas devem entrar depois de o pipeline
estar confiável.

## Riscos E Pontos De Atenção

- O dado é temporal, então validação aleatória pode produzir estimativas
  otimistas.
- O target encoding pode causar overfitting se usado sem cuidado.
- O split de eval está sendo usado para seleção e tuning; no futuro, convém
  preservar holdout para avaliação final.
- Modelos como Random Forest podem gerar artefatos muito grandes.
- O pipeline ainda depende de arquivos locais e W&B; para produção na AWS,
  será necessário definir armazenamento, execução e serving.

## Próximos Passos

Para evoluir rumo a deploy na AWS:

- adicionar validação temporal mais robusta, como backtesting ou
  `TimeSeriesSplit`;
- avaliar drift de dados e qualidade de entrada;
- criar contrato de schema para treino e inferência;
- versionar ambiente de execução;
- containerizar o projeto com Docker;
- definir onde os artefatos ficarão em produção, por exemplo S3 + W&B;
- criar job de treino em AWS, como SageMaker, ECS ou Batch;
- criar endpoint de inferência, possivelmente com FastAPI em ECS, Lambda ou
  SageMaker Endpoint;
- adicionar CI para lint, testes e smoke tests;
- decidir se MLflow será incorporado como model registry;
- criar monitoramento de predições, latência e métricas pós-deploy.

## Regra Mental Do Projeto

Uma boa engenharia de ML separa claramente:

- dado bruto;
- dado preparado;
- features;
- modelo;
- métricas;
- artefatos;
- inferência;
- orquestração.

Quando essas fronteiras estão claras, fica mais fácil testar, debugar,
versionar, treinar novamente e levar o projeto para produção.
