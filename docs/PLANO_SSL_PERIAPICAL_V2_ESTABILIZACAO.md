# Plano SSL Periapical v2 - Estabilização e Preservação de Representação

Data: 2026-04-01  
Status: **ativo e revisado com diagnóstico técnico do v1**

## 1) Objetivo

Executar um v2 de Continual SSL com DINOv2-small que:
- preserve a geometria útil do backbone fundacional;
- evite colapso/anisotropia no embedding;
- maximize chance de ganho real no downstream periapical (1312/14 classes);
- mantenha rastreabilidade completa e comparabilidade com o benchmark histórico.

## 2) Diagnóstico consolidado do v1

Com base nos resultados de `epoch_015` e inspeção do código atual:

1. Resultado downstream muito degradado (`E1=0`, `E2 macro_f1~0.149`).
2. Sinal de anisotropia forte (PC1 ~`89.8%` da variância).
3. Loss SSL por época oscilou sem tendência robusta de melhora entre épocas 1-16.
4. Checkpoint foi promovido por menor loss de treino SSL, sem gate downstream periódico.

### Causas prováveis priorizadas

1. Export/eval do `student_backbone` no script de export, enquanto em DINO o backbone EMA (`teacher`) tende a ser o mais estável para downstream.
2. Atualização agressiva do backbone (full fine-tune com LR único para tudo).
3. Ausência de freeze/unfreeze progressivo e ausência de param groups (`head` vs `backbone`).
4. Seleção de checkpoint guiada por loss SSL em vez de métrica downstream alvo.

## 3) Princípios mandatórios do v2

1. **Preservação primeiro, especialização depois** (atualização conservadora).
2. **Teacher-first** para export e avaliação oficial.
3. **Gate periódico downstream** durante o treino (não apenas no final).
4. **Decisão por métrica de negócio técnico**: `E2 test_macro_f1`.
5. **Diagnóstico contínuo de colapso** (não depender só de loss SSL).

## 4) Estratégia de treino (fases)

## Fase 1 - Acomodação estável (épocas 1-5)

- Treinar `student_head` + últimos blocos do backbone somente.
- Blocos iniciais congelados.
- LR backbone baixo, LR head mais alto.
- Objetivo: minimizar drift precoce da representação.

## Fase 2 - Unfreeze progressivo (épocas 6-20)

- Liberar blocos intermediários gradualmente.
- Manter patch embed + blocos mais rasos congelados ou com LR mínimo.
- Rodar gates em `epoch_10` e `epoch_20`.

## Fase 3 - Refino conservador (épocas 21-50)

- Liberar mais camadas apenas se os gates forem positivos.
- Manter LLRD opcional (camadas rasas com LR menor).
- Gates em `epoch_30`, `epoch_40`, `epoch_50`.

## 5) Hiperparâmetros iniciais recomendados (v2)

Treino:
- `epochs=50`
- `batch_size=8` ou `16` (conforme estabilidade)
- `amp=true`
- `grad_clip_norm=1.0`

Otimização:
- LR backbone: `2e-5` (faixa `1e-5..3e-5`)
- LR head: `1e-4`
- `min_lr=1e-6`
- `warmup_steps=1500..2000`
- `weight_decay=1e-4` (testar `5e-4` se necessário)

Head:
- `out_dim=16384` no primeiro v2.

Teacher/loss:
- manter inicialmente `teacher_momentum_base=0.996`, `teacher_momentum_final=1.0`
- `student_temp=0.10`, `teacher_temp_warmup=0.04`, `teacher_temp=0.07`

Augmentações:
- manter preset RX-safe atual (sem flip livre no SSL).

## 6) Mudanças obrigatórias no código

## 6.1 Exportar backbone correto para downstream

Arquivo: `experiments/ssl_periapical_dinov2/scripts/export_backbone_checkpoint.py`

- Adicionar flag `--backbone-key` com opções:
  - `teacher_backbone` (default v2 oficial)
  - `student_backbone` (apenas ablação)
- Persistir no `export_meta.json` qual backbone foi exportado.

## 6.2 Param groups e freeze/unfreeze no treino

Arquivo: `experiments/ssl_periapical_dinov2/src/ssl_periapical_dinov2/trainer.py`

- Separar grupos de parâmetros:
  - grupo `head` com LR alto;
  - grupo `backbone` com LR baixo.
- Implementar congelamento por profundidade/bloco no início.
- Implementar agenda de unfreeze por época.
- Logar quantos parâmetros treináveis por grupo a cada época.

## 6.3 Critério de checkpoint

Arquivo: `trainer.py`

- Manter `last.pt`.
- Deixar de usar só `best_train_loss` como critério de promoção.
- Criar snapshot explícito por época de gate (`epoch_10/20/30/40/50`).
- Promoção oficial deve vir do resultado downstream (`E2`), não do loss SSL.

## 6.4 Diagnóstico de colapso no treino

Arquivo: `trainer.py` (ou util dedicado)

- Em subset fixo, calcular por gate:
  - variância explicada da PC1;
  - participação média dos autovalores top-k;
  - norma média de embedding.
- Salvar em JSON por época para auditoria.

## 7) Plano de avaliação (gates oficiais)

Snapshots obrigatórios: `10`, `20`, `30`, `40`, `50`.

Para cada snapshot:
1. Exportar **teacher_backbone**.
2. Rodar `E1/E2` no dataset oficial (1312/14), com protocolo fixo.
3. Registrar resultados em tabela única.

Métricas foco:
- `E2 test_macro_f1` (principal)
- `E2 test_accuracy`
- `E2 best_val_macro_f1`
- diagnóstico de anisotropia (PC1%)

Critérios de decisão:
1. `E2 macro_f1 >= 0.70`: candidato forte.
2. `0.50 <= E2 macro_f1 < 0.70`: refinar e repetir gate.
3. `E2 macro_f1 < 0.50`: reprovar setup/snapshot.

Critério de segurança adicional:
- Se `PC1% > 60%` com queda de `E2`, tratar como colapso e interromper promoção.

## 8) Sequência operacional recomendada

1. Corrigir export (`teacher` default) e rerodar E1/E2 do `epoch_015` como contraprova.
2. Implementar param groups + freeze/unfreeze + logs diagnósticos.
3. Criar config `ec2_full_ssl_periapical_v2_stable.yaml`.
4. Rodar smoke curto (subset) e validar:
   - sem crash;
   - logs completos;
   - métricas de colapso geradas.
5. Rodar full v2 na EC2.
6. Executar gates periódicos e atualizar tabela consolidada.
7. Promover somente snapshot aprovado no gate downstream.

## 9) Entregáveis mínimos da v2

Código:
- config v2 estável;
- export com escolha explícita de backbone;
- freeze/unfreeze e param groups funcionais;
- logs de colapso por gate.

Artefatos por run:
- `summary.json`, `history.json`, `config_resolved.json`
- checkpoints por gate
- exports HF por snapshot
- resultados E1/E2 por snapshot
- tabela consolidada de decisão

## 10) Referências internas

- análise do v1:
  - [EXPERIMENTO_SSL_PERIAPICAL_V1_RESULTADOS_E_ANALISE.md](/Users/fabioandrade/RMFM/docs/EXPERIMENTO_SSL_PERIAPICAL_V1_RESULTADOS_E_ANALISE.md)
- status consolidado:
  - [STATUS_ATUAL_SSL_PERIAPICAL_V1.md](/Users/fabioandrade/RMFM/docs/STATUS_ATUAL_SSL_PERIAPICAL_V1.md)
- plano de execução detalhado (implementação):
  - [PLANO_EXECUCAO_SSL_PERIAPICAL_V2_CORRIGIDO.md](/Users/fabioandrade/RMFM/docs/PLANO_EXECUCAO_SSL_PERIAPICAL_V2_CORRIGIDO.md)
