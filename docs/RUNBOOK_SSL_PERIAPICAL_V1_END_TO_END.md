# Runbook E2E - SSL Periapical v1

Guia único para executar o fluxo completo do primeiro SSL periapical.

## 1) Pré-requisitos

- Ler antes: `docs/LEIA_PRIMEIRO_AMBIENTES_MAC_EC2.md`
- EC2 de validação: `35.92.136.175`
- Projeto na EC2: `/dataset/RMFM`
- Dados principais: `/dataminer/rmdatasets/data/periapicais_processed/imgs`
- Venv: `/dataset/RMFM/.venv`

## 2) Treinar/atualizar classificador Radiobot (17 classes)

Usar JSON completo de amostragem por pasta:
```bash
cd /dataset/RMFM
source /dataset/RMFM/.venv/bin/activate

python experiments/radiobot_folder_classifier/scripts/train_frozen_head_cached_from_list.py \
  --list-json /dataset/RMFM/experiments/radiobot_folder_classifier/inputs/sample_list_all.json \
  --output-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1 \
  --model-id facebook/dinov2-small \
  --epochs 80 \
  --batch-size 256 \
  --feature-batch-size 128 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --shortest-edge 256 \
  --crop-size 256 \
  --val-size 0.15 \
  --test-size 0.15 \
  --num-workers 8
```

## 3) Classificar base grande (73k)

```bash
cd /dataset/RMFM
source /dataset/RMFM/.venv/bin/activate

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python experiments/radiobot_folder_classifier/scripts/predict_list_to_json_dir.py \
  --run-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1 \
  --images-dir /dataminer/rmdatasets/data/periapicais_processed/imgs \
  --output-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/predictions_json_periapicais_processed_imgs \
  --batch-size 256 \
  --top-k 17 \
  --progress-every 1000 \
  --offline
```

Checar resumo:
```bash
cat /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/predictions_json_periapicais_processed_imgs/_summary.json
```

## 4) Inspeção visual das classes não-periapicais

```bash
cd /dataset/RMFM
source /dataset/RMFM/.venv/bin/activate

python experiments/radiobot_folder_classifier/scripts/build_filtered_grouped_mosaic_from_predictions.py \
  --predictions-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/predictions_json_periapicais_processed_imgs \
  --output-dir /dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1/mosaic_non_peri_radiografias
```

## 5) Regra v1 de composição

Regra atual documentada:
- `keep`: classes `Periapical + Interproximal`
- `drop`: classes restantes (com possibilidade de revisão pontual)

Conjunto elegível atual:
- `Periapical=65,052`
- `Interproximal=435`
- total `keep_v1=65,487`

## 6) Materializar artefatos oficiais v1

Arquivos obrigatórios:
- `ssl_periapical_v1_keep.txt`
- `ssl_periapical_v1_drop.txt`
- `ssl_periapical_v1_manifest.json`

Local recomendado (EC2):
- `/dataset/RMFM/experiments/ssl_periapical_v1/`

## 7) Início do treino SSL v1

Antes de iniciar:
- revisar `docs/LEVANTAMENTO_SSL_DINO_MEDICO_ESTADO_DA_ARTE.md`
- congelar preset inicial (crops + augs + schedule)
- registrar configuração exata do run (yaml/json + seed)

Depois:
- executar treino SSL
- salvar checkpoints + métricas
- avaliar em downstream (classificação/segmentação) para validar ganho.

