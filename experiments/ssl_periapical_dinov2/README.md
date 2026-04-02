# SSL Periapical DINOv2 (Fase A)

Implementacao inicial do treino SSL teacher-student com multicrop para periapicais.

## Estrutura

- `src/ssl_periapical_dinov2/`
  - `trainer.py`: loop de treino SSL (DINO-style)
  - `data.py`: ingestao de imagens + multicrop + augmentacoes
  - `ssl_core.py`: loss, heads e atualizacao EMA
  - `visuals.py`: callback visual estilo Hydra (`manifest.jsonl` + `index.html`)
- `scripts/`
  - `train_ssl_dinov2.py`
  - `preview_multicrop_pipeline.py`
  - `export_backbone_checkpoint.py`
  - `run_downstream_periapical_eval.py`
- `configs/`
  - `preset_ft_dinov2_periapical_v1.yaml`
  - `smoke_mac_peri3000_512.yaml`
  - `ec2_full_ssl_periapical_v1.yaml`

## Setup

Usar o mesmo ambiente do repo (dependencias de `torch`, `torchvision`, `transformers`, etc).

## Preview do pipeline de crops/augs

```bash
cd /Users/fabioandrade/RMFM
PYTHONPATH=experiments/ssl_periapical_dinov2/src \
python3 experiments/ssl_periapical_dinov2/scripts/preview_multicrop_pipeline.py \
  --config experiments/ssl_periapical_dinov2/configs/smoke_mac_peri3000_512.yaml \
  --output-dir experiments/ssl_periapical_dinov2/outputs/preview_multicrop_smoke \
  --num-samples 8
```

## Treino smoke no Mac (subset do periapicais_3000)

```bash
cd /Users/fabioandrade/RMFM
PYTHONPATH=experiments/ssl_periapical_dinov2/src \
python3 experiments/ssl_periapical_dinov2/scripts/train_ssl_dinov2.py \
  --config experiments/ssl_periapical_dinov2/configs/smoke_mac_peri3000_512.yaml
```

Saidas da run:

- `summary.json`, `history.json`
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `tb/` (TensorBoard)
- `train_visuals/index.html` + `manifest.jsonl`

## Export do backbone para downstream

```bash
cd /Users/fabioandrade/RMFM
python3 experiments/ssl_periapical_dinov2/scripts/export_backbone_checkpoint.py \
  --checkpoint experiments/ssl_periapical_dinov2/outputs/smoke_mac_peri3000_512/checkpoints/best.pt \
  --output-dir experiments/ssl_periapical_dinov2/outputs/smoke_mac_peri3000_512/exported_backbone
```

## Avaliacao downstream periapical (E1/E2)

```bash
cd /Users/fabioandrade/RMFM
python3 experiments/ssl_periapical_dinov2/scripts/run_downstream_periapical_eval.py \
  --backbone-dir experiments/ssl_periapical_dinov2/outputs/smoke_mac_peri3000_512/exported_backbone \
  --images-dir /Users/fabioandrade/RMFM/Downloads/imgs_class \
  --labels-dir /Users/fabioandrade/RMFM/Downloads/periapical_classificacao \
  --output-dir experiments/ssl_periapical_dinov2/outputs/smoke_mac_peri3000_512/downstream_eval
```

Observacao importante:
- para fluxo oficial Mac + EC2 (export teacher/student + E1/E2 com paths canonicos), usar:
  - [docs/RUNBOOK_E1_E2_PERIAPICAL_MAC_EC2.md](/Users/fabioandrade/RMFM/docs/RUNBOOK_E1_E2_PERIAPICAL_MAC_EC2.md)
