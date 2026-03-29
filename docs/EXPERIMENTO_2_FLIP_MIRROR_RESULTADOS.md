# Experimentos Flip Mirror - DINOv2-small Congelado + Head

## Objetivo
- Reduzir erros de lateralidade (troca lado direito/esquerdo) observados no baseline.
- Manter o encoder DINO congelado e atuar só na cabeça classificadora.

## Estratégia
- Treino em modo `cached_features_head_only`.
- Nova flag: `--augment-flip-mirror`.
- No split de treino, cada imagem é duplicada com flip horizontal.
- O rótulo da imagem flipada é remapeado explicitamente para a classe espelhada (sem ambiguidade de lado).

## Run Smoke (sanidade de pipeline)
- Run: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/smoke_flip_mirror`
- Amostras: 256
- Resultado esperado: validar fluxo (não usar métrica como conclusão científica).
- Métricas:
  - `best_val_macro_f1`: `0.0742`
  - `test_accuracy`: `0.0769`
  - `test_macro_f1`: `0.0311`

## Run Principal (1.312 imagens)
- Run: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`
- Modelo: `facebook/dinov2-small`
- Preprocess: `256x256`
- Classes: `14`
- Melhor época: `56`
- Métricas:
  - `best_val_macro_f1`: `0.7709`
  - `test_accuracy`: `0.7462`
  - `test_macro_f1`: `0.7396`
  - `test_loss`: `0.7077`

## Comparação Contra Baseline (sem flip mirror)
- Baseline: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_v2_60ep`
- Candidato: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`
- Ganhos globais:
  - `test_accuracy`: `0.6091 -> 0.7462` (`+0.1371`)
  - `test_macro_f1`: `0.5693 -> 0.7396` (`+0.1703`)
  - `best_val_macro_f1`: `0.7136 -> 0.7709` (`+0.0573`)
- Redução de trocas espelhadas (exemplos):
  - `36-37-38 <-> 48-47-46`: `12 -> 6`
  - `33 <-> 43`: `8 -> 2`
  - `15-14 <-> 24-25`: `6 -> 4`

## Avaliação e Visualização
- Avaliação do teste (run flip mirror):
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/eval_test_grouped/summary_eval.json`
- Mosaico HTML agrupado por classe predita:
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1/eval_test_grouped/grouped_by_predicted_class.html`

## Conclusão
- O `flip mirror` com remapeamento explícito de rótulo resolveu uma fração relevante dos erros de lateralidade.
- Mesmo com encoder congelado, o ganho foi grande e consistente no conjunto de teste.
- Próximo passo natural: repetir o mesmo protocolo em `512` para medir ganho de detalhe anatômico vs custo de inferência.
