# Experimento SSL Periapical v1 - Resultados, Decisões e Análise

Data: 2026-04-01

## 1) Contexto

Objetivo do v1:
- realizar Continual SSL com DINOv2 em periapicais (com pequena fração de interproximais),
- exportar snapshots do backbone,
- comparar contra baseline histórico do classificador periapical 14 classes.

Backbone base:
- `facebook/dinov2-small`

Dataset SSL v1:
- `ssl_periapical_v1_keep.txt`
- total: `65487` imagens (`Periapical=65052`, `Interproximal=435`)

## 2) Setup efetivo do treino SSL

Run efetivo observado na EC2:
- `ec2_full_ssl_periapical_v1_bs8`

Config principal (v1):
- `epochs=50`
- `batch_size=8`
- `lr=1e-4`
- `min_lr=1e-6`
- `warmup_steps=1000`
- `weight_decay=1e-4`
- `teacher_momentum_base=0.996`
- `teacher_momentum_final=1.0`
- `out_dim=32768`
- multicrop: `2x384 (global) + 6x192 (local)`
- augmentações RX-safe (sem flip): rotação/brilho/contraste/blur/ruído leves

Arquivos de referência:
- [ec2_full_ssl_periapical_v1.yaml](/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/configs/ec2_full_ssl_periapical_v1.yaml)
- [trainer.py](/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/src/ssl_periapical_dinov2/trainer.py)
- [ssl_core.py](/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/src/ssl_periapical_dinov2/ssl_core.py)

## 3) Decisões tomadas durante execução

1. Manter composição `Periapical + Interproximal` no v1.
2. Manter pipeline sem flip no SSL.
3. Congelar snapshot da época 15 sem interromper o treino full.
4. Exportar snapshot para formato HF e rodar comparação oficial E1/E2 no dataset histórico de 1312 imagens.

## 4) Snapshot e export utilizados na comparação

Snapshot congelado:
- `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/checkpoints/epoch_015_snapshot.pt`

Backbone exportado:
- `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v1_bs8/exports/backbone_epoch_015`

## 5) Log de loss observado (épocas 1-16)

```text
epoch=001 loss=9.1932
epoch=002 loss=9.6847
epoch=003 loss=9.6600
epoch=004 loss=9.6986
epoch=005 loss=9.6501
epoch=006 loss=9.6807
epoch=007 loss=9.5320
epoch=008 loss=9.6376
epoch=009 loss=9.6119
epoch=010 loss=9.6075
epoch=011 loss=9.7184
epoch=012 loss=9.7155
epoch=013 loss=9.5642
epoch=014 loss=9.6758
epoch=015 loss=9.4711
epoch=016 loss=9.6972
```

Leitura:
- oscilação em faixa estreita sem tendência clara de melhora sustentada.

## 6) Resultados comparativos (E0/E1/E2)

Baseline oficial (E0), run histórico:
- `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`
- `test_accuracy=0.7462`
- `test_macro_f1=0.7396`
- `best_val_macro_f1=0.7709`

E1 (backbone SSL epoch_015 + reuso da cabeça antiga):
- arquivo: `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch015/e1_reuse_head_summary.json`
- `test_accuracy=0.0000`
- `test_macro_f1=0.0000`

E2 (backbone SSL epoch_015 + nova cabeça, mesmo protocolo histórico):
- arquivo: `/Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch015/e2_retrain_head/summary.json`
- `test_accuracy=0.2284`
- `test_macro_f1=0.1489`
- `best_val_macro_f1=0.1847`

## 7) Verificações de equidade e pipeline

Checklist de consistência verificado:
- mesmo dataset comparativo (1312/14 classes),
- mesmo baseline de referência,
- mesmo protocolo E2 (hiperparâmetros lidos do baseline),
- mesmo caminho de extração no classificador (`pooler_output` disponível em ambos os modelos).

Conclusão de engenharia:
- baixa evidência de bug crítico no pipeline de comparação,
- alta evidência de snapshot ruim para downstream.

## 8) Últimas conclusões e especulações técnicas

Conclusões operacionais:
1. O snapshot `epoch_015` não deve ser promovido.
2. O run v1 atual pode continuar apenas como exploração/snapshot mining.
3. A trilha principal deve migrar para v2 mais estável.

Especulações técnicas plausíveis:
- esquecimento/collapse parcial por atualização agressiva do backbone (full fine-tune + LR único),
- `out_dim` elevado com batch efetivo limitado pode dificultar estabilidade,
- ausência de estratégia explícita de preservação (congelamento parcial, LLRD, unfreeze progressivo).

Sinal quantitativo de instabilidade geométrica (no cache de features E2):
- variância explicada pela 1a componente ~`89.8%` (baseline ~`12.6%`).

## 9) Decisão de continuidade

Decisão atual:
- **não interromper necessariamente o run em curso**, mas tratá-lo como secundário.
- **iniciar SSL v2 em paralelo** com foco em estabilidade e preservação de representação.

Próximo plano:
- [PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md](/Users/fabioandrade/RMFM/docs/PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md)

