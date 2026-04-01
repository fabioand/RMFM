# SSL Periapical v1 - Manifesto Oficial de Dados

Status: **ativo (pré-treino SSL v1)**

## Escopo

- Fonte classificada: `/dataminer/rmdatasets/data/periapicais_processed/imgs`
- Predições: `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/predictions_json_periapicais_processed_imgs`
- Modelo de triagem: `run_cached_head_17cls_all_v1` (Radiobot 17 classes)

## Contagens da triagem

- total classificado: `73,411`
- Periapical: `65,052`
- Interproximal: `435`
- elegível v1 (`Periapical + Interproximal`): `65,487`
- não elegível inicial: `7,924`

## Regra de composição v1

Keep:
- `Periapical`
- `Interproximal`

Drop inicial:
- demais classes preditas pelo classificador 17 classes

Justificativa:
- preservar foco periapical com um conjunto marginal BW/interproximal para robustez;
- manter casos-limite anatômicos na borda entre classes.

## Artefatos obrigatórios (dataset contract)

Devem existir e ser versionados por data/revisão:
- `ssl_periapical_v1_keep.txt`
- `ssl_periapical_v1_drop.txt`
- `ssl_periapical_v1_manifest.json`

Local recomendado (EC2):
- `/dataset/RMFM/experiments/ssl_periapical_v1/`

## Estrutura mínima sugerida para `ssl_periapical_v1_manifest.json`

```json
{
  "dataset_id": "ssl_periapical_v1",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "source_images_dir": "/dataminer/rmdatasets/data/periapicais_processed/imgs",
  "predictions_dir": "/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/predictions_json_periapicais_processed_imgs",
  "classifier_run_dir": "/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1",
  "selection_rule": {
    "keep_labels": ["Periapical", "Interproximal"]
  },
  "counts": {
    "num_total": 73411,
    "num_keep": 65487,
    "num_drop": 7924
  },
  "files": {
    "keep_list": "ssl_periapical_v1_keep.txt",
    "drop_list": "ssl_periapical_v1_drop.txt"
  },
  "notes": "v1 inclui conjunto marginal Interproximal para robustez de domínio."
}
```

## Critérios de qualidade antes do treino SSL

- Conferir aleatoriamente amostras de `keep` e `drop`.
- Confirmar que não há drift de path/arquivo inexistente.
- Confirmar rastreabilidade da decisão (classe predita + confiança).

