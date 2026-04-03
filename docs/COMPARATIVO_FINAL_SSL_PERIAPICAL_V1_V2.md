# Comparativo Final - SSL Periapical (v1 vs v2)

Data: 2026-04-03

## 1) Escopo e protocolo

Comparação consolidada dos modelos/backbones avaliados no experimento SSL periapical.

Protocolo comum de downstream:
- dataset: `1312` imagens, `14` classes,
- split: `train/val/test = 918/197/197`,
- métrica principal: `test_macro_f1` (com `accuracy` de apoio),
- E1: reuso da cabeça antiga,
- E2: retreino da cabeça (MLP, 60 épocas, salvo quando explicitamente indicado).

Baseline de referência:
- `run_cached_head_256_flipmirror_v1`
- `test_macro_f1=0.7396`, `accuracy=0.7462`

## 2) Ranking final (E2 MLP)

| Rank | Modelo avaliado | Test Acc | Test Macro-F1 | Delta vs baseline (F1) |
|---|---|---:|---:|---:|
| 1 | `best29_teacher` (checkpoint `best29.pt`, epoch interno 28) | 0.8629 | 0.8584 | +0.1188 |
| 2 | `epoch_030_teacher` | 0.8477 | 0.8426 | +0.1030 |
| 3 | `epoch_020_teacher` | 0.8426 | 0.8404 | +0.1008 |
| 4 | `epoch_040_teacher` | 0.8477 | 0.8383 | +0.0987 |
| 5 | `epoch_050_teacher` | 0.8376 | 0.8293 | +0.0897 |
| 6 | `epoch_015_student` (v1) | 0.2284 | 0.1489 | -0.5907 |
| 7 | `epoch_015_teacher` (v1) | 0.2234 | 0.1412 | -0.5984 |

Leitura:
- v2 superou o baseline com folga em todos os checkpoints principais (020-050 e `best29`);
- `best29_teacher` foi o melhor candidato global até aqui.

## 3) E1 (reuso da cabeça antiga) - comportamento

| Modelo | E1 Test Acc | E1 Test Macro-F1 |
|---|---:|---:|
| `epoch_015_student` (v1) | 0.0000 | 0.0000 |
| `epoch_020_teacher` (v2) | 0.4619 | 0.3797 |
| `epoch_030_teacher` (v2) | 0.4467 | 0.3703 |
| `epoch_040_teacher` (v2) | 0.4061 | 0.3288 |
| `epoch_050_teacher` (v2) | 0.4365 | 0.3626 |
| `best29_teacher` | 0.4772 | 0.4036 |

Leitura:
- E1 melhora substancialmente do v1 para o v2, mas permanece abaixo do E2;
- no cenário atual, a cabeça retreinada continua essencial para extrair o melhor desempenho.

## 4) Ablações importantes

## 4.1 E2 com 100 épocas (epoch_020 teacher)

- E2 padrão (60 ep): `macro_f1=0.8404`
- E2 100 ep: `macro_f1=0.8348`

Conclusão:
- aumentar para 100 épocas não melhorou; houve leve piora (sinal de overfit/oscilação).
- manter 60 épocas como padrão do E2.

## 4.2 E2 com KNN (epoch_020 teacher)

- melhor `k=5`
- `test_macro_f1=0.4842`, `accuracy=0.5076`

Conclusão:
- KNN ficou muito abaixo do MLP no mesmo embedding;
- o ganho observado vem de fronteira não linear aprendida pela cabeça.

## 5) Diagnóstico geométrico de embeddings (degeneração)

Referência detalhada:
- [DIAGNOSTICO_EMBEDDINGS_V1_V2_SIMETRIA_VARIANCIA.md](/Users/fabioandrade/RMFM/docs/DIAGNOSTICO_EMBEDDINGS_V1_V2_SIMETRIA_VARIANCIA.md)

Resumo objetivo:
- v1 `epoch_015` (student/teacher) mostrou degeneração severa:
  - `PC1 ~0.90-0.93`,
  - `Top5 ~0.999`,
  - `effective_rank ~1.3-1.5`.
- v2 (`epoch_020 teacher`) recuperou geometria saudável:
  - `PC1=0.1422` (próximo ao baseline `0.1262`),
  - `Top5=0.4499`,
  - `effective_rank=39.0`.

Conclusão:
- o problema central do v1 (colapso/anisotropia) foi endereçado no v2.

## 6) Interpretação técnica final do experimento

1. O v1 falhou por instabilidade de atualização e perda de geometria útil para downstream.
2. O v2 acertou o regime de otimização (LR diferencial, unfreeze progressivo, controle de estabilidade).
3. A região de melhor desempenho ficou entre os checkpoints intermediários (não no final tardio):
   - pico observado em `best29_teacher`.

## 7) Recomendação de encerramento deste ciclo

1. Campeão atual: `best29_teacher` (promover como backbone líder do ciclo).
2. Padrão de avaliação downstream:
   - E2 MLP com 60 épocas,
   - manter E1 apenas como diagnóstico secundário.
3. Próximo ciclo:
   - expandir dados anotados com seleção ativa por embedding (diversidade + incerteza),
   - repetir gates com o mesmo protocolo para manter comparabilidade.

## 8) Artefatos principais (caminhos)

- Baseline:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/summary.json`
- E2 v2 epoch020:
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch020_teacher/e2_retrain_head/summary.json`
- E2 v2 epoch030:
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch030_teacher/e2_retrain_head/summary.json`
- E2 v2 epoch040:
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch040_teacher/e2_retrain_head/summary.json`
- E2 v2 epoch050:
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch050_teacher/e2_retrain_head/summary.json`
- E2 campeão (`best29_teacher`):
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_best29_teacher/e2_retrain_head/summary.json`
- E2 KNN (ablação):
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch020_teacher_knn/e2_retrain_knn/summary.json`

## 9) Escala supervisionada (31k+) com backbone campeão

Teste adicional após o ciclo v1/v2:
- protocolo: E2 (encoder congelado + MLP),
- backbone: `best29_teacher`,
- dataset: `periapicais_processed` (`31,759` amostras válidas, `14` classes),
- split: `22,231 / 4,764 / 4,764`.

Resultado:
- `best_val_macro_f1=0.9072`,
- `test_accuracy=0.9125`,
- `test_macro_f1=0.9105`.

Leitura:
- o desempenho sobe de forma consistente ao aumentar o volume supervisionado,
- ainda abaixo da referência CNN (~0.95+), mas com ganho relevante em relação aos cenários menores.

Artefato:
- `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/e2_processed32k_best29_teacher/summary.json`

## 10) Distribuição de erros no teste grande (31k+)

A partir da matriz de confusão do run acima:
- erros totais: `417`,
- agrupamento:
  - lateralidade (apenas): `10` (`2.40%`),
  - adjacência (apenas): `337` (`80.82%`),
  - lateralidade + adjacência: `0` (`0.00%`),
  - outros: `70` (`16.79%`).

Conclusão prática:
- o gargalo dominante está em fronteiras anatômicas adjacentes (molar/pré-molar e vizinhanças),
- lateralidade deixou de ser o principal problema.
