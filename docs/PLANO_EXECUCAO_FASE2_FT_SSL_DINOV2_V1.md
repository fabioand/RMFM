# Plano de Execução - Fase 2 FT/SSL DINOv2 v1

Status: **pronto para execução (2026-03-31)**

## 1) Escopo e decisões já fechadas

Este plano assume as decisões já registradas em `docs/PROJETO_FT_DINOV2_FASE2.md`:

- backbone v1: `facebook/dinov2-small`;
- dataset FT/SSL v1: `~65k` periapicais + `~400` interproximais;
- preset inicial de multicrop:
  - globais: `2 x 384`, `scale=(0.15, 1.0)`;
  - locais: `6 x 192`, `scale=(0.05, 0.15)`;
  - ratio controlado: `0.90–1.10`;
- augmentações seguras para RX (sem flip livre);
- avaliação comparativa oficial contra benchmark periapical 14 classes (`E0/E1/E2`).

## 2) Objetivo operacional imediato

Implementar o pipeline FT/SSL e monitoramento completo no RMFM, rodar smoke local no Mac com algumas centenas de imagens do `periapicais_3000`, e deixar pronto para promoção para EC2 sem mudar contrato de execução.

## 3) Entregáveis de implementação (código)

Criar estrutura dedicada:

- `experiments/ssl_periapical_dinov2/`
- `experiments/ssl_periapical_dinov2/src/...`
- `experiments/ssl_periapical_dinov2/scripts/...`
- `experiments/ssl_periapical_dinov2/configs/...`

Scripts mínimos:

1. `train_ssl_dinov2.py`
- treino SSL teacher-student (EMA) com multicrop.

2. `preview_multicrop_pipeline.py`
- preview de transforms e distribuição de crops (sanity check visual).

3. `capture_ssl_epoch_visuals.py` (ou callback em módulo)
- monitoramento estilo Hydra:
  - `manifest.jsonl`
  - `index.html` filtrável
  - painéis por época com views globais/locais e atenção ViT.

4. `export_backbone_checkpoint.py`
- export do encoder final para downstream.

5. `run_downstream_periapical_eval.py`
- reexecuta comparação oficial `E1/E2` no benchmark 1312/14 classes.

Arquivos de config:

- `configs/preset_ft_dinov2_periapical_v1.yaml`
- `configs/smoke_mac_peri3000_512.yaml`
- `configs/ec2_full_ssl_periapical_v1.yaml`

## 4) Plano de execução por fase

## Fase A - Implementação técnica

1. Implementar dataloader por lista de paths:
- entrada por `keep.txt` (dataset grande) e por pasta/lista local (smoke).

2. Implementar transforms multicrop v1:
- RGB por replicação de grayscale;
- ratio controlado (`0.90–1.10`);
- augmentações seguras (rotação/brilho-contraste/blur/ruído leves).

3. Implementar loop SSL:
- student + teacher (EMA);
- loss DINO;
- logging por época (loss, lr, throughput, tempo).

4. Implementar monitoramento:
- TensorBoard (escalares);
- HTML viewer estilo Hydra com `manifest.jsonl`.

5. Implementar export do backbone e integração de avaliação downstream.

Saída esperada da fase:
- pipeline executável ponta a ponta com smoke local.

## Fase B - Smoke no Mac (subset do periapicais_3000)

Objetivo:
- validar estabilidade, visualização e artefatos sem custo alto.

Dataset smoke:
- origem: `/Users/fabioandrade/RMFM/Downloads/periapicais_3000`
- subset alvo: `300–800` imagens (recomendado inicial: `512`).

Config smoke recomendada:
- `epochs=3..5`
- `batch-size` conservador para MPS
- `num_workers` baixo/moderado
- `save_visuals_every=1`
- `max_visual_samples=8`

Artefatos obrigatórios:
- `outputs/<run>/summary.json`
- `outputs/<run>/history.json`
- `outputs/<run>/tb/`
- `outputs/<run>/train_visuals/index.html`
- `outputs/<run>/train_visuals/manifest.jsonl`
- `outputs/<run>/checkpoints/last.pt` e `best.pt` (ou equivalente)

Gate de aprovação do smoke:
- treino sem crash por todas as épocas;
- viewer HTML funcional com filtros;
- atenção/crops visualmente coerentes;
- throughput e memória documentados.

## Fase C - Avaliação comparativa local (equidade)

Após smoke aprovado:

1. Exportar backbone FT/SSL.
2. Rodar `E1`: encoder FT/SSL + cabeça antiga.
3. Rodar `E2`: encoder FT/SSL + cabeça nova com mesmo protocolo histórico.
4. Comparar contra `E0` (`run_cached_head_256_flipmirror_v1`).

Métricas oficiais:
- `best_val_macro_f1`
- `test_accuracy`
- `test_macro_f1`

Critério mínimo de promoção:
- `E2` não degradar materialmente o benchmark atual e mostrar tendência de ganho em `macro_f1`.

## Fase D - Promoção para EC2 (dataset grande)

Preparação:

1. Sincronizar código/configs para `/dataset/RMFM`.
2. Validar paths EC2:
- projeto: `/dataset/RMFM`
- dados: `/dataminer/...`
3. Validar venv e dependências.
4. Rodar smoke curto na EC2 de validação.

Execução full:

1. Rodar treino completo com `ec2_full_ssl_periapical_v1.yaml`.
2. Publicar artefatos por run (summary/history/checkpoints/tb/viewer).
3. Exportar backbone final.
4. Rodar benchmark downstream periapical (E1/E2) na própria EC2.

## 5) Padrão de organização de runs

Recomendado:

- `experiments/ssl_periapical_dinov2/outputs/<run_name>/`
  - `config_resolved.yaml`
  - `summary.json`
  - `history.json`
  - `checkpoints/`
  - `tb/`
  - `train_visuals/`
    - `index.html`
    - `manifest.jsonl`
    - `epoch_XXXX/...`

## 6) Checklist de execução rápida (Mac -> EC2)

1. Implementação concluída + lint/smoke local.
2. Smoke Mac `periapicais_3000` (subset 512) aprovado.
3. Benchmark comparativo local (`E1/E2`) executado.
4. Config full congelada para EC2.
5. Smoke EC2 curto aprovado.
6. Treino full EC2 iniciado com monitoramento ativo.

## 7) Riscos e mitigação

1. Instabilidade de treino SSL:
- mitigar com LR conservador, warmup e checkpoint frequente.

2. Visualização pesada por I/O:
- reduzir frequência (`every 5 epochs`) no treino full.

3. Diferença Mac vs EC2:
- manter config base única e apenas overrides de device/batch/workers.

4. Comparação injusta:
- manter split e protocolo downstream exatamente iguais ao benchmark histórico.

