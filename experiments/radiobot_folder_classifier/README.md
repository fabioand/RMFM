# Radiobot Folder Classifier - DINOv2 Frozen Head

Classificador supervisionado por classe de pasta (`folder`) a partir de um JSON de lista de imagens.

Estratégia:
- encoder `facebook/dinov2-small` congelado
- pré-extração de embeddings
- treino apenas da cabeça classificadora (LayerNorm + Dropout + Linear)

## Resultado Do Run Principal (EC2)
Run ampliado:
- `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1`

Métricas:
- num_samples: `25125`
- num_classes: `17`
- device: `cuda`
- best_val_macro_f1: `0.9914`
- test_accuracy: `0.9859`
- test_macro_f1: `0.9845`

Run anterior (amostra menor):
- `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_v1`

Métricas:
- num_samples: `1637`
- num_classes: `17`
- device: `cuda`
- best_val_macro_f1: `0.9648`
- test_accuracy: `0.9756`
- test_macro_f1: `0.9764`

Consolidado:
- `/Users/fabioandrade/RMFM/docs/RESULTADOS_EXPERIMENTOS_ATUAIS.md`

## Estrutura
- `scripts/train_frozen_head_cached_from_list.py`: entrypoint de treino
- `src/dino_folder_cls/data.py`: leitura do JSON e split estratificado
- `src/dino_folder_cls/model.py`: modelo DINO congelado + head
- `src/dino_folder_cls/train_cached.py`: treino/val/test com cache de features
- `outputs/`: artefatos do experimento

## Formato esperado do JSON
Compatível com o gerado por `build_sample_list_from_subfolders.py`, por exemplo:
```json
{
  "root_dir": "/dataminer/radiobot",
  "samples": [
    {"folder": "Periapical", "path": "Periapical/img1.jpg"},
    {"folder": "Panoramica", "path": "Panoramica/img2.jpg"}
  ]
}
```

## Rodar treino
```bash
cd /Users/fabioandrade/RMFM/experiments/radiobot_folder_classifier
source /Users/fabioandrade/RMFM/.venv/bin/activate

PYTHONPATH=src python scripts/train_frozen_head_cached_from_list.py \
  --list-json /dataminer/radiobot/sample_list_n100.json \
  --output-dir /Users/fabioandrade/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_v1 \
  --model-id facebook/dinov2-small \
  --epochs 30 \
  --batch-size 128 \
  --feature-batch-size 64 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --shortest-edge 256 \
  --crop-size 256 \
  --val-size 0.15 \
  --test-size 0.15
```

## Artefatos gerados
- `summary.json`
- `history.json`
- `classification_report_test.json`
- `confusion_matrix_test.csv`
- `best_head_only.pt`
- `label_to_idx.json`
- `split_stats.json`
- `features_cache/*`
- `tb/` (TensorBoard, se habilitado)

## Inspeção Visual (HTML agrupado por classe predita)
- `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_v1/predict_grouped_html_17/grouped_by_predicted_class.html`
- resumo de latência e contagens:
  - `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_v1/predict_grouped_html_17/summary_predict_unlabeled.json`

## Inferência em lote para gerar JSON por imagem
Gera um `*.json` por imagem com:
- classe predita,
- confiança da classe predita,
- `top_classes` (top-k),
- `probs_by_class` (softmax completo em todas as 17 classes).

```bash
cd /dataset/RMFM
source /dataset/RMFM/.venv/bin/activate

python experiments/radiobot_folder_classifier/scripts/predict_list_to_json_dir.py \
  --run-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1 \
  --list-json /dataset/RMFM/experiments/radiobot_folder_classifier/inputs/sample_list_all.json \
  --output-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/predictions_json \
  --batch-size 128 \
  --top-k 17 \
  --progress-every 500
```

Saída adicional:
- `_summary.json` com contagens de classes e throughput.

Notas:
- suporta entrada por `--list-json` ou `--images-dir` (com `--recursive` opcional).
- suporta execução offline (`--offline` + `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`).
- tolera imagem inválida: registra em `_errors.jsonl` e continua.

## Mosaico filtrado por classes preditas (inspeção visual)
Script:
- `scripts/build_filtered_grouped_mosaic_from_predictions.py`

Exemplo (exclui Periapical, Fotografia* e Intra-Oral* e ordena classes menos numerosas primeiro):
```bash
python experiments/radiobot_folder_classifier/scripts/build_filtered_grouped_mosaic_from_predictions.py \
  --predictions-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/predictions_json_periapicais_processed_imgs \
  --output-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/mosaic_non_peri_radiografias
```
