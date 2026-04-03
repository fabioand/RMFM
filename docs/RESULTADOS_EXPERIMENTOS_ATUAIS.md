# Resultados Atuais Dos Experimentos

Este documento consolida os resultados principais já obtidos no repositório.

## 1) Periapicais (DINOv2 frozen + head)

Dataset:
- 1312 imagens rotuladas
- 14 classes periapicais

### 1.1 Baseline sem flip mirror
- Run: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_v2_60ep`
- Fonte: `summary.json`

Métricas:
- best_val_macro_f1: `0.7136`
- test_accuracy: `0.6091`
- test_macro_f1: `0.5693`

### 1.2 Com flip mirror + remapeamento explícito
- Run: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`
- Fonte: `summary.json`

Métricas:
- best_val_macro_f1: `0.7709`
- test_accuracy: `0.7462`
- test_macro_f1: `0.7396`

### 1.3 Ganho do flip mirror vs baseline
- test_accuracy: `+0.1371` (0.6091 -> 0.7462)
- test_macro_f1: `+0.1703` (0.5693 -> 0.7396)
- best_val_macro_f1: `+0.0573` (0.7136 -> 0.7709)

### 1.4 Comparação com classificador RM API (periapical)
- Avaliação RM API no mesmo conjunto (1312): `/Users/fabioandrade/RMFM/out/rm_api_periapical_eval_1312_workers/summary.json`
- Endpoint: `v1/periapicals/classification`

Métricas RM API:
- accuracy: `0.9505`
- cobertura: `1.0`
- latência média por chamada: `484.65 ms`

Comparativo de acurácia (no mesmo conjunto):
- DINO+head (com flip): `0.7462`
- RM API: `0.9505`
- gap: `0.2043` pontos absolutos de acurácia

## 2) Classificação Geral Radiobot (17 classes)

Setup:
- JSON de lista por pastas
- 17 classes
- DINOv2-small congelado + head

Run A (amostra menor):
- `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_v1/summary.json` (EC2)

Métricas:
- device: `cuda`
- num_samples: `1637`
- num_classes: `17`
- best_val_macro_f1: `0.9648`
- test_accuracy: `0.9756`
- test_macro_f1: `0.9764`

Run B (treino ampliado):
- `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/summary.json` (EC2)

Métricas:
- device: `cuda`
- num_samples: `25125`
- num_classes: `17`
- best_val_macro_f1: `0.9914`
- test_accuracy: `0.9859`
- test_macro_f1: `0.9845`

## 3) Inspeção Visual (HTML)

Periapical:
- Avaliação agrupada por classe predita no teste:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/eval_test_grouped/grouped_by_predicted_class.html`
- Predição em imagens sem GT:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/predict_unlabeled_600/grouped_by_predicted_class.html`

Radiobot (17 classes):
- Mosaico agrupado:
  - `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_v1/predict_grouped_html_17/grouped_by_predicted_class.html`
- Resumo desse mosaico:
  - `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_v1/predict_grouped_html_17/summary_predict_unlabeled.json`

Inferência em lote com JSON por imagem:
- Script: `/dataset/RMFM/experiments/radiobot_folder_classifier/scripts/predict_list_to_json_dir.py`
- Exemplo local (3000 periapicais): `/Users/fabioandrade/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/predictions_json_peri3000_local_offline_mps/_summary.json`
- Resultado local (MPS, offline, 3000 imagens):
  - elapsed_s: `71.11`
  - throughput: `42.19 img/s`
  - predições: `Periapical=2663`, `Não-periapical=337`

Latências medidas no mosaico Radiobot (1700 imagens, CUDA):
- preprocess_mean_ms: `2.207`
- inference_mean_ms: `7.259`
- total_mean_ms: `11.169`

## 4) Observações De Interpretação

- Periapical e Radiobot são tarefas diferentes; não comparar diretamente os percentuais entre elas sem considerar complexidade, distribuição e possíveis atalhos de classe.
- O ganho do `flip mirror` foi relevante para reduzir erros de lateralidade no experimento periapical.
- O resultado Radiobot está muito alto; vale manter validação com splits alternativos e amostras externas para checar generalização.

## 5) Periapical em escala (E2 com `best29_teacher`, 31k+)

Run:
- `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/e2_processed32k_best29_teacher`

Resumo:
- num_samples: `31759`
- num_classes: `14`
- device: `cuda`
- best_epoch: `18`
- best_val_macro_f1: `0.9072`
- test_accuracy: `0.9125`
- test_macro_f1: `0.9105`

Distribuição dos erros no teste:
- total de erros: `417`
- lateralidade (apenas): `10` (`2.40%`)
- adjacência (apenas): `337` (`80.82%`)
- lateralidade + adjacência: `0` (`0.00%`)
- outros: `70` (`16.79%`)

Artefatos:
- histograma: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/e2_processed32k_best29_teacher/error_histogram/error_types_histogram_top.png`
- CSV completo: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/e2_processed32k_best29_teacher/error_histogram/error_types_full.csv`

## 6) Decisão Atual Para O Primeiro SSL (v1)

Com base na classificação em massa de `73,411` imagens:
- `Periapical`: `65,052`
- `Interproximal`: `435`

Decisão de composição inicial do dataset SSL:
- incluir `Periapical + Interproximal` no `keep` da v1.
- conjunto elegível atual: `65,487` imagens.

Racional:
- `Interproximal` é próxima de periapical e entra como fração pequena;
- melhora robustez com baixo risco de diluição do domínio;
- captura casos-limite de periapical que caíram nessa classe.

Validação qualitativa:
- inspeção visual realizada nas classes menos numerosas (mosaico por classe);
- decisão mantida de incluir BW/interproximais marginais na v1.
