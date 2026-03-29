# Manifesto de Artefatos - Experimentos Flip Mirror

## Run Principal
- Diretório base:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`

| Artefato | Descrição |
|---|---|
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/summary.json` | Resumo final do treino (métricas, config e paths). |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/history.json` | Curvas por época (loss e macro F1 de treino/val). |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/best_head_only.pt` | Melhor checkpoint da cabeça classificadora. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/classification_report_test.json` | Métricas por classe no teste. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/confusion_matrix_test.csv` | Matriz de confusão do teste. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/features_cache/cache_meta.json` | Metadados do cache (inclui `augment_flip_mirror` e mapa de espelhamento). |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/features_cache/x_train.npy` | Embeddings de treino (já com duplicação flip no split de treino). |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/features_cache/y_train.npy` | Labels de treino com remapeamento de lado. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/features_cache/x_val.npy` | Embeddings de validação (sem augmentação). |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/features_cache/y_val.npy` | Labels de validação. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/features_cache/x_test.npy` | Embeddings de teste (sem augmentação). |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/features_cache/y_test.npy` | Labels de teste. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/tb` | Eventos TensorBoard do run. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/compare_vs_baseline.md` | Comparação textual contra baseline. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/compare_vs_baseline.json` | Comparação detalhada em JSON contra baseline. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/eval_test_grouped/grouped_by_predicted_class.html` | Mosaico HTML agrupado por classe predita (imagem + metadados). |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/eval_test_grouped/predictions_test_rows.json` | Predições por imagem do split de teste. |
| `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/eval_test_grouped/summary_eval.json` | Resumo da avaliação reproduzível do teste. |

## Run Smoke
- Diretório base:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/smoke_flip_mirror`
- Uso: validação rápida de pipeline (não usar como referência de performance final).
