# Experimento 1 - DINOv2-small Congelado + Head (Features Cacheadas, 256)

## Configuração
- Run: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_v2_60ep`
- Modelo: `facebook/dinov2-small`
- Estratégia: encoder congelado, pré-extração de embeddings, treino apenas da cabeça
- Resolução de preprocess: 256 (shortest_edge=256, crop=256)
- Amostras totais: 1312
- Classes: 14
- Melhor época (val): 40

## Métricas principais
- `best_val_macro_f1`: **0.7136**
- `test_accuracy`: **0.6091**
- `test_macro_f1`: **0.5693**
- `test_loss`: **0.9413**

## Leitura do resultado
- O baseline mostrou sinal forte para um encoder não especializado no domínio odontológico.
- Houve melhora consistente ao longo do treino, com pico de validação em macro F1 > 0.71.
- No teste, o desempenho caiu em relação ao melhor ponto de validação, indicando espaço para robustez adicional (especialmente lateralidade).

## Maiores confusões no teste (top)
| True | Pred | Contagem |
|---|---|---:|
| 23 | 13 | 8 |
| 36-37-38 | 48-47-46 | 7 |
| 43 | 33 | 6 |
| 48-47-46 | 36-37-38 | 5 |
| 48-47-46 | 34-35 | 5 |
| 24-25 | 26-27-28 | 5 |
| 18-17-16 | 26-27-28 | 5 |
| 45-44 | 48-47-46 | 4 |
| 15-14 | 24-25 | 4 |
| 26-27-28 | 24-25 | 3 |
| 13 | 12-11-21-22 | 3 |
| 45-44 | 34-35 | 2 |

## Próxima iteração recomendada
1. Treino com `--augment-flip-mirror` para reduzir erros de troca de lado.
2. Comparar 256 vs 512 mantendo mesmo split e protocolo de avaliação.
3. Avaliar fine-tuning parcial do backbone se persistirem ambiguidades por lateralidade.
