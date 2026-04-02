# Status Atual - SSL Periapical v2

Data de atualização: 2026-04-02

## 1) Contexto

Após os resultados ruins do `epoch_015` no v1 (student e teacher), foi iniciado o ciclo v2 com foco em:
- estabilidade de atualização do backbone,
- preservação da representação,
- critérios de gate por downstream e diagnósticos de colapso.

Implementações aplicadas no v2:
- export com seleção explícita de backbone (`student_backbone`/`teacher_backbone`);
- `trainer.py` com LR por grupo (`backbone` e `head`);
- freeze/unfreeze progressivo via `unfreeze_schedule`;
- snapshots por épocas de gate;
- diagnósticos de colapso por época (PC1/top5/normas);
- manutenção de monitoramento visual SSL (`train_visuals`) e TensorBoard.

## 2) Smoke executado com sucesso (EC2 CUDA)

Comando usado:

```bash
cd /dataset/RMFM
source /dataset/RMFM/.venv/bin/activate

PYTHONPATH=experiments/ssl_periapical_dinov2/src \
python3 experiments/ssl_periapical_dinov2/scripts/train_ssl_dinov2.py \
  --config experiments/ssl_periapical_dinov2/configs/ec2_full_ssl_periapical_v2_stable.yaml \
  --run-name ec2_smoke_v2_256img_2ep_bs20 \
  --max-images 256 \
  --epochs 3 \
  --batch-size 20
```

Resumo observado:
- `device=cuda`;
- `num_images=256`;
- `epochs=3`;
- sem OOM/crash;
- throughput de época chegando a ~`39.73 img/s`;
- artefatos gerados (`summary/history/checkpoints/tb/train_visuals/collapse_diagnostics`).

Run:
- `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_smoke_v2_256img_2ep_bs20`

## 3) Decisão operacional

Com base no smoke acima:
- `batch_size=20` aprovado para iniciar o full v2 na máquina atual.

Diretriz:
1. iniciar full em `bs20`;
2. monitorar primeira hora (estabilidade e VRAM);
3. reduzir para `16` apenas se houver sinal de instabilidade.

## 4) Início do full v2

Comando aprovado para início do treino completo:

```bash
cd /dataset/RMFM
source /dataset/RMFM/.venv/bin/activate

PYTHONPATH=experiments/ssl_periapical_dinov2/src \
python3 experiments/ssl_periapical_dinov2/scripts/train_ssl_dinov2.py \
  --config experiments/ssl_periapical_dinov2/configs/ec2_full_ssl_periapical_v2_stable.yaml \
  --run-name ec2_full_ssl_periapical_v2_stable_bs20 \
  --batch-size 20
```

Pasta esperada:
- `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20`

## 5) Gate downstream já validado (epoch_020 teacher)

Resultado no dataset oficial de comparação (`1312` imagens, `14` classes):

- E1 (reuso da cabeça antiga):
  - `test_accuracy=0.4619`
  - `test_macro_f1=0.3797`
- E2 (retreino MLP, 60 épocas):
  - `test_accuracy=0.8426`
  - `test_macro_f1=0.8404`
  - `best_val_macro_f1=0.8958`

Comparação contra baseline histórico (`run_cached_head_256_flipmirror_v1`):
- baseline `test_macro_f1=0.7396`
- v2 epoch_020 teacher E2 `test_macro_f1=0.8404` (delta `+0.1008`)

Teste adicional (E2 com KNN):
- melhor `k=5`
- `test_accuracy=0.5076`
- `test_macro_f1=0.4842`
- conclusão: para esse cenário, MLP continua superior ao KNN.

## 6) Diagnóstico de embedding (simetria + variância): v1 vs v2

Diagnóstico reproduzido no mesmo espírito do v1:
- v1 `epoch_015` (student/teacher) confirma degeneração severa:
  - `PC1 ~0.90-0.93`,
  - `Top5 ~0.999`,
  - `effective_rank ~1.3-1.5`.
- v2 `epoch_020 teacher` mostra recuperação geométrica:
  - `PC1=0.1422` (próximo ao baseline `0.1262`),
  - `Top5=0.4499`,
  - `effective_rank=39.0`.

Relatório completo:
- [DIAGNOSTICO_EMBEDDINGS_V1_V2_SIMETRIA_VARIANCIA.md](/Users/fabioandrade/RMFM/docs/DIAGNOSTICO_EMBEDDINGS_V1_V2_SIMETRIA_VARIANCIA.md)

## 7) Referências

- [PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md](/Users/fabioandrade/RMFM/docs/PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md)
- [PLANO_EXECUCAO_SSL_PERIAPICAL_V2_CORRIGIDO.md](/Users/fabioandrade/RMFM/docs/PLANO_EXECUCAO_SSL_PERIAPICAL_V2_CORRIGIDO.md)
- [TODO_NEXT_EXPERIMENTS_SSL_PERIAPICAL.md](/Users/fabioandrade/RMFM/docs/TODO_NEXT_EXPERIMENTS_SSL_PERIAPICAL.md)
