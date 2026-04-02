# Diagnóstico de Embeddings - v1 vs v2 (Simetria + Variância)

Data: 2026-04-02

## Objetivo

Repetir no v2 os mesmos testes que evidenciaram degeneração no v1, para comparação direta:
- anisotropia/variância do embedding (`PC1`, `Top5`, `effective_rank`);
- teste de simetria por pares `original vs flip` no `x_train` cacheado.

## Artefato bruto gerado

- [diagnostics.json](/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/embedding_diagnostics_compare_v1_v2/diagnostics.json)

## Fontes (features cacheadas)

- Baseline histórico:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/features_cache`
- SSL v1 epoch 015 (student):
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch015/e2_retrain_head/features_cache`
- SSL v1 epoch 015 (teacher):
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch015_teacher/e2_retrain_head/features_cache`
- SSL v2 epoch 020 (teacher):
  - `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch020_teacher/e2_retrain_head/features_cache`

## Métricas

- `pc1_explained_variance_ratio`: fração da variância explicada pela primeira componente principal.
- `top5_explained_variance_ratio`: fração acumulada nas 5 primeiras componentes.
- `effective_rank`: rank efetivo por entropia espectral (quanto maior, mais distribuída a variância).
- Simetria em pares `original/flip` no `x_train`:
  - `cosine_mean/std` entre embeddings dos pares,
  - `l2_mean/std` entre embeddings dos pares.

## Resultados

| Run | PC1 | Top5 | Effective Rank | Cosine(orig,flip) | L2(orig,flip) |
|---|---:|---:|---:|---:|---:|
| baseline_flipmirror_v1 | 0.1262 | 0.3960 | 63.60 | 0.8188 ± 0.0760 | 28.53 ± 5.85 |
| ssl_v1_epoch015_student_e2 | 0.8988 | 0.9989 | 1.49 | 0.5142 ± 0.4581 | 52.55 ± 39.96 |
| ssl_v1_epoch015_teacher_e2 | 0.9330 | 0.9992 | 1.35 | 0.5657 ± 0.4214 | 50.55 ± 40.88 |
| ssl_v2_epoch020_teacher_e2 | 0.1422 | 0.4499 | 39.00 | 0.5817 ± 0.1386 | 30.21 ± 5.99 |

## Leitura técnica

1. v1 epoch015 (student e teacher) confirma degeneração severa:
- `PC1` ~0.90-0.93 (quase toda variância numa direção);
- `Top5` ~0.999;
- `effective_rank` ~1.3-1.5.

2. v2 epoch020 (teacher) recupera geometria de embedding:
- `PC1=0.1422` (próximo ao baseline 0.1262),
- `Top5=0.4499` (muito distante da degeneração do v1),
- `effective_rank=39.0` (ainda abaixo do baseline, mas muito acima do v1).

3. No teste de simetria de pares `original/flip`:
- v1 degenerado mostra alta instabilidade (`std` muito alto e L2 muito alto);
- v2 estabiliza bastante os pares (L2 e `std` próximos do baseline), apesar de `cosine_mean` menor que baseline.

## Conclusão

Os testes reproduzidos no v2 mostram que:
- a degeneração geométrica observada no v1 **não** se mantém no v2 (epoch020 teacher),
- o espaço de embeddings do v2 está muito mais saudável/útil para downstream,
- o resultado é consistente com o ganho grande de E2 observado no mesmo checkpoint.

