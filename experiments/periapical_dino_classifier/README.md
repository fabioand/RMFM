# Periapical Classifier - DINO Frozen Head

Baseline supervisionado para periapicais usando:
- encoder `facebook/dinov2-small` congelado
- cabeça classificadora treinável (LayerNorm + Dropout + Linear)

## Resultados Principais
- Baseline (`run_cached_head_256_v2_60ep`): `test_accuracy=0.6091`, `test_macro_f1=0.5693`
- Com `flip mirror` (`run_cached_head_256_flipmirror_v1`): `test_accuracy=0.7462`, `test_macro_f1=0.7396`
- Ganho do `flip mirror`: `+0.1371` em accuracy e `+0.1703` em macro F1
- Referência RM API (mesmo conjunto 1312): `accuracy=0.9505`

Consolidado:
- `/Users/fabioandrade/RMFM/docs/RESULTADOS_EXPERIMENTOS_ATUAIS.md`

## Estrutura
- `scripts/train_frozen_head.py`: entrypoint de treino
- `src/dino_periapical_cls/data.py`: leitura de dados e split estratificado
- `src/dino_periapical_cls/model.py`: modelo DINO congelado + head
- `src/dino_periapical_cls/train.py`: treino/val/test e export de métricas
- `outputs/`: artefatos de treino

## Rodar treino (dataset atual 1312)
```bash
cd /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier
source /Users/fabioandrade/RMFM/.venv/bin/activate

PYTHONPATH=src python scripts/train_frozen_head.py \
  --images-dir /Users/fabioandrade/RMFM/Downloads/imgs_class \
  --labels-dir /Users/fabioandrade/RMFM/Downloads/periapical_classificacao \
  --output-dir /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_frozen_head_256 \
  --model-id facebook/dinov2-small \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --shortest-edge 256 \
  --crop-size 256 \
  --val-size 0.15 \
  --test-size 0.15 \
  --offline
```

## Rodar treino com embeddings pré-extraídos (encoder congelado)
```bash
cd /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier
source /Users/fabioandrade/RMFM/.venv/bin/activate

PYTHONPATH=src python scripts/train_frozen_head_cached.py \
  --images-dir /Users/fabioandrade/RMFM/Downloads/imgs_class \
  --labels-dir /Users/fabioandrade/RMFM/Downloads/periapical_classificacao \
  --output-dir /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_v1 \
  --model-id facebook/dinov2-small \
  --epochs 30 \
  --batch-size 128 \
  --feature-batch-size 64 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --shortest-edge 256 \
  --crop-size 256 \
  --val-size 0.15 \
  --test-size 0.15 \
  --offline
```

Observação:
- O script reutiliza automaticamente `features_cache/` quando `modelo + resolução + split + labels` forem idênticos.
- Para forçar reextração: adicione `--force-reextract-features`.
- Para aumentar robustez de lateralidade no treino: adicione `--augment-flip-mirror` (flip com remapeamento de classe espelhada).

## Artefatos gerados
- `summary.json`
- `history.json`
- `classification_report_test.json`
- `confusion_matrix_test.csv`
- `best_head.pt`
- `label_to_idx.json`
- `split_stats.json`
- `splits.json`
- `tb/` (eventos TensorBoard)

## TensorBoard
```bash
source /Users/fabioandrade/RMFM/.venv/bin/activate
tensorboard --logdir /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs --port 6006 --host 127.0.0.1
```

## Inferência em imagens sem GT (ex.: 600 fora das 1312 rotuladas)
Script:
- `scripts/predict_unlabeled_grouped_html.py`

Exemplo:
```bash
cd /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier
source /Users/fabioandrade/RMFM/.venv/bin/activate

PYTHONPATH=src python scripts/predict_unlabeled_grouped_html.py \
  --run-dir /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1 \
  --images-dir /Users/fabioandrade/RMFM/Downloads/periapicais_3000 \
  --exclude-labels-dir /Users/fabioandrade/RMFM/Downloads/periapical_classificacao \
  --num-images 600 \
  --seed 42 \
  --output-dir /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/predict_unlabeled_600
```

Saídas principais:
- `summary_predict_unlabeled.json` (inclui latência média/mediana/p95)
- `predictions_rows.json` (1 linha por imagem, com `inference_ms`)
- `grouped_by_predicted_class.html` (cards por classe com tempo no card)

## Inspeção Visual Recomendada
- Teste agrupado por classe predita:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/eval_test_grouped/grouped_by_predicted_class.html`
- Sem GT (amostra externa):
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/predict_unlabeled_600/grouped_by_predicted_class.html`
