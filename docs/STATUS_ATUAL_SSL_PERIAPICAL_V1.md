# Status Atual - SSL Periapical v1 (Handoff)

Data de atualização: 2026-04-01

## 1) Objetivo do trabalho

- Continual SSL com DINOv2 em radiografias odontológicas (foco inicial em periapicais).
- Comparar backbone SSL contra baseline histórico do classificador periapical 14 classes.

## 2) Estado do pipeline

- Dataset SSL v1 materializado e curado (`keep`):
  - `num_keep=65487` (`Periapical=65052` + `Interproximal=435`)
  - manifesto: [SSL_PERIAPICAL_V1_MANIFESTO.md](/Users/fabioandrade/RMFM/docs/SSL_PERIAPICAL_V1_MANIFESTO.md)
- Treino SSL implementado e operacional:
  - script: [train_ssl_dinov2.py](/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/scripts/train_ssl_dinov2.py)
  - config base EC2: [ec2_full_ssl_periapical_v1.yaml](/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/configs/ec2_full_ssl_periapical_v1.yaml)
- Run efetivo em execução na EC2:
  - `run_name=ec2_full_ssl_periapical_v1_bs8`
  - output esperado: `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/`

## 3) Snapshot congelado para comparação

- Checkpoint congelado da época 15:
  - `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/checkpoints/epoch_015_snapshot.pt`
- Backbone exportado em formato HF:
  - `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/exports/backbone_epoch_015`

## 4) Avaliação comparativa já executada (dataset 1312/14 classes)

Referência histórica (E0):
- run baseline: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`
- métricas:
  - `test_accuracy=0.7462`
  - `test_macro_f1=0.7396`
  - `best_val_macro_f1=0.7709`

Avaliação do backbone SSL `epoch_015`:
- pasta: `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch015`
- E1 (reuso da cabeça antiga):
  - `test_accuracy=0.0000`
  - `test_macro_f1=0.0000`
  - arquivo: `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch015/e1_reuse_head_summary.json`
- E2 (retreino de nova cabeça sob protocolo idêntico):
  - `test_accuracy=0.2284`
  - `test_macro_f1=0.1489`
  - `best_val_macro_f1=0.1847`
  - arquivo: `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch015/e2_retrain_head/summary.json`

## 5) Interpretação técnica atual

- O snapshot `epoch_015` ficou inadequado para downstream (queda grande em E1/E2).
- Há sinal de anisotropia/colapso parcial do embedding no cache de features do E2:
  - componente principal explicando ~89.8% da variância (baseline ~12.6%).
- Conclusão operacional atual:
  - **não promover `epoch_015` para uso downstream**.
  - continuar análise em checkpoints mais tardios e/ou reiniciar SSL com setup mais conservador.

## 6) Dataset e baseline oficial para equidade

- Dataset comparativo oficial (classificação periapical):
  - imagens: `/Users/fabioandrade/RMFM/Downloads/imgs_class`
  - labels: `/Users/fabioandrade/RMFM/Downloads/periapical_classificacao`
  - total: `1312` imagens, `14` classes
- Run baseline oficial:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`

## 7) Próximas ações recomendadas (resumo)

1. Rodar E2 para `epoch_030`, `epoch_040`, `epoch_050` com o mesmo protocolo.
2. Se não recuperar, abrir novo SSL v2 estável (LR menor + validação periódica downstream).
3. Registrar cada tentativa em tabela única (snapshot, config, E1/E2, decisão).

Plano detalhado:
- [TODO_NEXT_EXPERIMENTS_SSL_PERIAPICAL.md](/Users/fabioandrade/RMFM/docs/TODO_NEXT_EXPERIMENTS_SSL_PERIAPICAL.md)
- [EXPERIMENTO_SSL_PERIAPICAL_V1_RESULTADOS_E_ANALISE.md](/Users/fabioandrade/RMFM/docs/EXPERIMENTO_SSL_PERIAPICAL_V1_RESULTADOS_E_ANALISE.md)
- [PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md](/Users/fabioandrade/RMFM/docs/PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md)
