# TODO - Próximos Experimentos SSL Periapical

Data de atualização: 2026-04-01

Objetivo: permitir continuidade imediata do trabalho por qualquer programador novo, com comandos reproduzíveis e critérios claros de decisão.

## Prioridade atual (v2 corrigido)

A trilha principal a partir desta data passa a seguir o plano corrigido de implementação v2:
- [PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md](/Users/fabioandrade/RMFM/docs/PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md)
- [PLANO_EXECUCAO_SSL_PERIAPICAL_V2_CORRIGIDO.md](/Users/fabioandrade/RMFM/docs/PLANO_EXECUCAO_SSL_PERIAPICAL_V2_CORRIGIDO.md)

Decisão operacional:
1. Não promover snapshots com export/eval apenas de `student_backbone`.
2. Adotar `teacher_backbone` como default oficial de export no v2.
3. Adotar gates downstream periódicos como critério de promoção.

## 1) Pré-condições

- Repositório local: `/Users/fabioandrade/RMFM`
- Venv local: `/Users/fabioandrade/RMFM/.venv`
- Baseline oficial (comparação):
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`
- Dataset oficial do comparativo (1312/14):
  - imagens: `/Users/fabioandrade/RMFM/Downloads/imgs_class`
  - labels: `/Users/fabioandrade/RMFM/Downloads/periapical_classificacao`

## 2) Etapa A - Avaliar múltiplos snapshots do run atual

Hipótese: `epoch_015` degradou; verificar se snapshots mais tardios recuperam qualidade de embedding.

### 2.1 Congelar snapshots na EC2 (sem parar treino)

Exemplo (executar na instância de treino):

```bash
cp /dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/checkpoints/last.pt \
   /dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/checkpoints/epoch_030_snapshot.pt
```

Repetir para `epoch_040` e `epoch_050` no fechamento de cada época.

### 2.2 Exportar cada snapshot para formato HF

```bash
cd /dataset/RMFM
source .venv/bin/activate

python experiments/ssl_periapical_dinov2/scripts/export_backbone_checkpoint.py \
  --checkpoint /dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/checkpoints/epoch_030_snapshot.pt \
  --output-dir /dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/exports/backbone_epoch_030_teacher \
  --backbone-key teacher_backbone
```

Repetir para `040` e `050`.

### 2.3 Trazer exports para o Mac

```bash
rsync -avz --progress \
  -e "ssh -i /Users/fabioandrade/.ssh/fabio-ia3.pem" \
  ubuntu@35.92.136.175:/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/exports/backbone_epoch_030/ \
  /Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/exports/backbone_epoch_030/
```

Repetir para `040` e `050`.

### 2.4 Rodar E1/E2 local para cada snapshot

```bash
cd /Users/fabioandrade/RMFM
source .venv/bin/activate

PYTHONPATH=experiments/periapical_dino_classifier/src \
python experiments/ssl_periapical_dinov2/scripts/run_downstream_periapical_eval.py \
  --backbone-dir /Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/exports/backbone_epoch_030 \
  --images-dir /Users/fabioandrade/RMFM/Downloads/imgs_class \
  --labels-dir /Users/fabioandrade/RMFM/Downloads/periapical_classificacao \
  --baseline-run-dir /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1 \
  --output-dir /Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch030
```

Repetir para `040` e `050`.

### 2.5 Consolidar resultados em tabela

Registrar para cada snapshot:
- `E1 test_accuracy`, `E1 test_macro_f1`
- `E2 best_val_macro_f1`, `E2 test_accuracy`, `E2 test_macro_f1`
- decisão (`promover`, `investigar`, `descartar`)

## 3) Etapa B - Se snapshots continuarem ruins, abrir SSL v2 estável

Objetivo: reduzir risco de colapso/anisotropia no embedding.

Parâmetros iniciais sugeridos (v2):
- `lr`: `1e-5` a `3e-5` (começar em `2e-5`)
- `batch_size`: manter conforme capacidade da GPU
- `out_dim`: considerar `16384`
- manter sem flip no SSL periapical
- manter monitoramento visual + TensorBoard

Critério para seguir:
- ganhos claros em E2 vs `epoch_015`
- e idealmente aproximação do baseline oficial.

## 4) Critérios de decisão objetivos

1. Se `E2 test_macro_f1 >= 0.70`: candidato forte a promoção.
2. Se `0.50 <= E2 test_macro_f1 < 0.70`: investigar ajuste fino antes de promover.
3. Se `E2 test_macro_f1 < 0.50`: descartar snapshot/config e priorizar novo setup.

## 5) Artefatos mínimos por experimento

- `summary.json`
- `history.json`
- `features_cache/cache_meta.json`
- `best_head_only.pt`
- `classification_report_test.json`
- `confusion_matrix_test.csv`

## 6) Referências internas

- Status consolidado:
  - [STATUS_ATUAL_SSL_PERIAPICAL_V1.md](/Users/fabioandrade/RMFM/docs/STATUS_ATUAL_SSL_PERIAPICAL_V1.md)
- Projeto Fase 2:
  - [PROJETO_FT_DINOV2_FASE2.md](/Users/fabioandrade/RMFM/docs/PROJETO_FT_DINOV2_FASE2.md)
- Runbook E2E:
  - [RUNBOOK_SSL_PERIAPICAL_V1_END_TO_END.md](/Users/fabioandrade/RMFM/docs/RUNBOOK_SSL_PERIAPICAL_V1_END_TO_END.md)
