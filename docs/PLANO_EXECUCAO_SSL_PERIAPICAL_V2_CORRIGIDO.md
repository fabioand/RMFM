# Plano de Execução - SSL Periapical v2 (Corrigido)

Data: 2026-04-01  
Status: **pronto para implementação**

## 1) Escopo

Este plano operacionaliza as correções do v2 para maximizar sucesso no Continual SSL com DINOv2-small em radiografias periapicais, reduzindo risco de colapso e de regressão downstream.

Referência base:
- [PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md](/Users/fabioandrade/RMFM/docs/PLANO_SSL_PERIAPICAL_V2_ESTABILIZACAO.md)

## 2) Problemas que este plano corrige

1. Export errado para downstream (student por default em vez de teacher).
2. LR único para backbone+head sem estratégia de preservação.
3. Ausência de freeze/unfreeze progressivo.
4. Seleção de melhor checkpoint por loss SSL (sem gate downstream).
5. Falta de métrica explícita de colapso/anisotropia durante treino.

## 3) Plano de implementação por arquivo

## Bloco A - Export e avaliação corretos

Arquivo:
- `experiments/ssl_periapical_dinov2/scripts/export_backbone_checkpoint.py`

Mudanças:
1. Adicionar argumento:
   - `--backbone-key` com choices `teacher_backbone` e `student_backbone`.
2. Default oficial:
   - `teacher_backbone`.
3. Registrar no `export_meta.json`:
   - `backbone_key`.
4. Validar que a chave existe no checkpoint.

Teste:
1. Exportar o mesmo checkpoint com teacher e student.
2. Rodar `run_downstream_periapical_eval.py` em ambos.
3. Comparar `E2 macro_f1`.

Critério de aceite:
- Export teacher funciona e fica documentado no `meta`.

## Bloco B - Otimização em grupos + freeze/unfreeze

Arquivo:
- `experiments/ssl_periapical_dinov2/src/ssl_periapical_dinov2/trainer.py`

Mudanças:
1. Introduzir param groups:
   - `head` (`lr_head`).
   - `backbone` (`lr_backbone`).
2. Introduzir agenda de freeze:
   - congelar blocos rasos no início;
   - liberar gradualmente por época.
3. Introduzir agenda de unfreeze configurável por YAML.
4. Logar a cada época:
   - nº de parâmetros treináveis por grupo;
   - LR efetivo por grupo.

Config necessária:
- nova config v2 estável com campos:
  - `training.lr_backbone`
  - `training.lr_head`
  - `training.freeze_schedule`
  - `training.unfreeze_schedule`

Critério de aceite:
- parâmetros treináveis variam conforme esperado ao longo das épocas.

## Bloco C - Gate por snapshot e promoção por downstream

Arquivo:
- `experiments/ssl_periapical_dinov2/src/ssl_periapical_dinov2/trainer.py`

Mudanças:
1. Salvar snapshots explícitos nas épocas de gate (`10/20/30/40/50`).
2. Manter `last.pt`; não promover run por `best_train_loss`.
3. Salvar manifesto de snapshots.

Arquivos auxiliares:
- `experiments/ssl_periapical_dinov2/scripts/run_downstream_periapical_eval.py`

Mudanças:
1. Adicionar no manifesto final o `backbone_key` usado no export.
2. Facilitar tabela consolidada por snapshot.

Critério de aceite:
- cada gate gera snapshot + resultado E1/E2 rastreável.

## Bloco D - Diagnóstico de colapso

Arquivo:
- `experiments/ssl_periapical_dinov2/src/ssl_periapical_dinov2/trainer.py`

Mudanças:
1. Em subset fixo por run, extrair embeddings por gate e calcular:
   - `pc1_explained_variance_ratio`
   - `top5_explained_variance_ratio`
   - `embedding_norm_mean/std`
2. Salvar em:
   - `outputs/<run>/collapse_diagnostics/epoch_XXX.json`

Critério de aceite:
- diagnósticos gerados nos mesmos gates dos snapshots.

## 4) Ordem de execução recomendada

1. Bloco A (export teacher default) e contraprova do `epoch_015`.
2. Bloco B (param groups + freeze/unfreeze).
3. Bloco D (diagnóstico de colapso).
4. Bloco C (gates/snapshots e promoção por downstream).
5. Rodar smoke v2.
6. Rodar full v2.

## 5) Config v2 inicial recomendada

Arquivo alvo:
- `experiments/ssl_periapical_dinov2/configs/ec2_full_ssl_periapical_v2_stable.yaml`

Preset inicial:
- `epochs: 50`
- `batch_size: 8` (ou 16 se estabilidade permitir)
- `lr_backbone: 2e-5`
- `lr_head: 1e-4`
- `min_lr: 1e-6`
- `warmup_steps: 1500`
- `weight_decay: 1e-4`
- `out_dim: 16384`
- `teacher_temp_warmup_epochs: 30`
- multicrop RX-safe mantido

## 6) Protocolo oficial de validação

Para cada gate (`10/20/30/40/50`):

1. Exportar teacher.
2. Rodar E1.
3. Rodar E2.
4. Rodar diagnóstico de colapso.
5. Atualizar tabela única:
   - snapshot
   - backbone_key
   - E1 acc/f1
   - E2 best_val_f1, test_acc, test_macro_f1
   - PC1%
   - decisão (`promover`, `investigar`, `descartar`)

## 7) Critérios de promoção

Obrigatórios:
1. `E2 test_macro_f1 >= 0.70`.
2. sem sinal forte de colapso (`PC1%` controlado e sem piora abrupta).

Desejáveis:
1. aproximação do benchmark histórico (`0.7396`) ou superação.
2. estabilidade entre gates consecutivos.

## 8) Critérios de abortar/reiniciar run

Abortar ou pivotar configuração se:
1. `E2 test_macro_f1 < 0.50` em 2 gates consecutivos.
2. `PC1%` persistentemente alto com degradação de E2.
3. sinais de drift irreversível após ajustes conservadores.

## 9) Backlog técnico de melhoria (pós-v2 inicial)

1. LLRD completo por profundidade de bloco.
2. Distillation regularizer para preservar features do backbone original.
3. Replay leve de subset de referência para reduzir forgetting.
4. Avaliar adapter/LoRA como alternativa a full backbone update.

## 10) Entregáveis finais esperados

1. Código v2 corrigido e reproduzível.
2. Config v2 estável versionada.
3. Execução full com gates completos.
4. Relatório consolidado de snapshots e decisão de promoção.
