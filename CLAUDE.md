# Regression MLOps — E2E Project

## Objetivo

Pipeline MLOps completo para um problema de regressão, cobrindo todas as fases desde a ingestão de dados até o monitoramento em produção.

## Fases do pipeline (e2e)

1. **Data Ingestion** — coleta e armazenamento dos dados brutos
2. **Data Validation** — checagem de qualidade, schema e distribuição
3. **Data Preprocessing** — limpeza e transformações
4. **Feature Engineering** — criação e seleção de features
5. **Model Training** — treinamento com rastreamento de experimentos
6. **Model Evaluation** — métricas, comparação e aprovação do modelo
7. **Model Registry** — versionamento e promoção do modelo
8. **Model Serving** — exposição do modelo para inferência
9. **Monitoring** — drift de dados e performance em produção

## Decisões em aberto (stack não definida)

- Orquestração do pipeline
- Rastreamento de experimentos
- Armazenamento de dados
- Serving / inferência
- Monitoramento

## Diretrizes para o Claude

- Não sugerir nem assumir nenhuma ferramenta ou biblioteca até a stack ser definida
- Ao propor implementações, apresentar opções com tradeoffs em vez de escolher silenciosamente
- Manter cada fase do pipeline em módulos independentes e desacoplados
- Perguntar antes de adicionar dependências externas
