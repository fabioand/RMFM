# Matriz de Scripts do Pipeline

Referência rápida do que cada script faz, com entradas e saídas.

## Radiobot Classifier

| Script | Objetivo | Entradas | Saídas | Status |
|---|---|---|---|---|
| `experiments/radiobot_folder_classifier/scripts/train_frozen_head_cached_from_list.py` | Treinar classificador 17 classes com DINOv2 frozen + head | `--list-json`, `--output-dir`, hiperparâmetros | `summary.json`, `best_head_only.pt`, `features_cache/*`, métricas | Ativo |
| `experiments/radiobot_folder_classifier/scripts/predict_list_to_json_dir.py` | Inferência em lote e JSON por imagem | `--run-dir` + (`--list-json` ou `--images-dir`) | `*.json` por imagem, `_summary.json`, `_errors.jsonl` | Ativo |
| `experiments/radiobot_folder_classifier/scripts/build_filtered_grouped_mosaic_from_predictions.py` | Mosaico HTML por classe predita com filtros | `--predictions-dir`, `--output-dir` | `grouped_filtered_by_class.html`, resumos e symlinks | Ativo |

## DINO v2 Lab

| Script | Objetivo | Entradas | Saídas | Status |
|---|---|---|---|---|
| `experiments/dino_v2/scripts/build_sample_list_from_subfolders.py` | Gerar lista JSON de amostras por pasta | `--root-dir`, `--n-per-folder` | `sample_list_*.json` | Ativo |
| `experiments/dino_v2/scripts/cluster_embeddings_report.py` | Embeddings + cluster + HTML | `--images-dir` ou `--images-list-json` | `cluster_report.html`, `summary.json`, `cluster_rows.json` | Ativo |

## Periapical Classifier

| Script | Objetivo | Entradas | Saídas | Status |
|---|---|---|---|---|
| `experiments/periapical_dino_classifier/scripts/train_frozen_head_cached.py` | Treino periapical com cache (opção flip mirror) | `--images-dir`, `--labels-dir` | checkpoints e métricas do run | Ativo |
| `experiments/periapical_dino_classifier/scripts/eval_test_grouped_html.py` | Avaliação do split test + HTML agrupado | `--run-dir` | `summary_eval.json`, `grouped_by_predicted_class.html` | Ativo |
| `experiments/periapical_dino_classifier/scripts/predict_unlabeled_grouped_html.py` | Predição sem GT + HTML agrupado | `--run-dir`, `--images-dir` | `summary_predict_unlabeled.json`, HTML | Ativo |

## RM API Utilities

| Script | Objetivo | Entradas | Saídas | Status |
|---|---|---|---|---|
| `scripts/rm_ia_eval_periapical_gt.py` | Avaliar RM API contra GT local | endpoint + credenciais + labels | `summary.json`, confusões e linhas detalhadas | Ativo |
| `scripts/rm_ia_classify_images.py` | Chamar endpoint RM IA para imagens | endpoint + credenciais + input-dir | JSON/HTML de predição | Ativo |

## Convenções Operacionais

- Preferir sempre `--output-dir` versionado por run.
- Para modelos já baixados: usar offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `--offline`).
- Em jobs grandes, exigir logs de progresso e `summary.json` final.
- Nunca salvar artefatos pesados no Git.

