# Blueprint Executável: Especialização DINO para Imagens Odontológicas

## 1) Objetivo
Construir um **foundation model odontológico** a partir de DINO pré-treinado (DINOv2/DINOv3), especializado com seu acervo de milhões de exames de ~2000 clínicas no Brasil, com foco em generalização clínica real e governança LGPD.

## 2) Premissas
- Dados já estão na AWS e próximos das GPUs.
- Há grande diversidade de aparelhos, protocolos e populações.
- Mais de 50% dos exames possuem laudo descritivo.
- Existem algumas imagens com medidas/overlays (subconjunto de maior valor supervisionado).
- Modalidades disponíveis: telerradiografia frontal/lateral, periapical, panorâmica, oclusal, interproximal, fotos intra/extraorais e CBCT.

## 3) Estratégia de modelagem (visão macro)
1. **Fase A - Continual Pretraining SSL** (sem rótulo) em todo acervo (ou quase todo).
2. **Fase B - Weak supervision com laudos** para criar heads multi-tarefa por modalidade.
3. **Fase C - Fine-tuning supervisionado** com subsets mais confiáveis (incluindo casos com medidas/overlays).
4. **Fase D - Avaliação externa por clínica/aparelho/região** e calibração clínica.

## 4) Arquitetura recomendada
### 4.1 Backbones
- 2D RX (periapical, pano, interproximal, oclusal, telerradio): `DINOv2 ViT-B/14` como baseline; `ViT-L/14` para versão de maior desempenho.
- Fotos intra/extraorais: backbone separado (domínio visual muito distinto de RX).
- CBCT: pipeline dedicado 3D/2.5D (não misturar diretamente com 2D no início).

### 4.2 Heads por tarefa
- Classificação multilabel de achados.
- Localização (quando houver sinal suficiente): detector/segmentador opcional.
- Regressão para medidas (quando houver ground truth confiável).
- Saída com incerteza/calibração para uso assistivo.

### 4.3 Roteamento por modalidade
- Primeiro classificador de modalidade (ou regra de metadados DICOM).
- Encaminhamento para backbone/head específico.

## 5) Dados e engenharia de dataset
### 5.1 Governança e privacidade
- De-identificação robusta (DICOM tags + OCR de burn-ins quando aplicável).
- Pseudonimização estável por paciente e clínica.
- Controle de acesso mínimo necessário e trilha de auditoria.
- Catálogo de datasets versionados (manifestos com hash).

### 5.2 Qualidade e limpeza
- Deduplicação por perceptual hash + embedding similarity.
- Filtro de baixa qualidade (motion blur, corte severo, contraste inviável).
- Remoção/normalização de overlays para não causar leakage.
- Curadoria mínima de protocolos por modalidade.

### 5.3 Splits corretos (obrigatório)
- Split por **paciente** (nunca cruzar treino/val/teste).
- Split por **clínica** e **aparelho** para teste externo real.
- Recomendação:
  - `Train`: 70-80% clínicas
  - `Val`: 10-15% clínicas
  - `Test interno`: 10-15% clínicas
  - `Test externo`: clínicas totalmente inéditas + distribuição regional balanceada

## 6) NLP dos laudos (weak labels)
### 6.1 Ontologia inicial (MVP)
- Cárie, perda óssea periodontal, lesão periapical, reabsorção, fratura, dente incluso, implante, tratamento endodôntico, ausência dentária etc.
- Atributos: dente/quadrante, lateralidade, severidade, cronicidade quando disponível.

### 6.2 Pipeline de extração
1. Normalização textual (acentos, abreviações, sinônimos).
2. Regras + modelo clínico para NER/extração de achados.
3. Score de confiança por label.
4. Revisão humana amostral por estratos (modalidade/região/clínica).

### 6.3 Política de uso do rótulo fraco
- Treino com pesos por confiança.
- Loss robusta a ruído (focal/asymmetric + label smoothing moderado).
- Curriculum: começar por labels de alta confiança.

## 7) Plano de treino (execução)
## 7.1 Fase A - SSL odontológico
- Inicializar com DINO pré-treinado.
- Continual pretraining em 2D RX + fotos separadamente.
- Para 1M de imagens por stream:
  - 30-50 épocas para baseline forte.
  - 80-120 épocas para versão premium.
- Monitorar:
  - linear probe por modalidade,
  - vizinhança semântica de embeddings,
  - estabilidade por fabricante/aparelho.

## 7.2 Fase B - Heads com laudos
- Congelar backbone (linear probe) 5-10 épocas.
- Descongelar parcial (últimos blocos) 10-20 épocas.
- Full fine-tuning com LR discriminativo 10-30 épocas.

## 7.3 Fase C - Refino com gold subset
- Usar subconjunto com medidas/overlays validados para tarefas de localização/medição.
- Ajuste fino de calibração e thresholds por tarefa clínica.

## 7.4 Hiperparâmetros iniciais (ponto de partida)
- Otimizador: AdamW.
- LR backbone: 1e-5 a 5e-5.
- LR head: 1e-4 a 5e-4.
- Weight decay: 0.04 a 0.2 (grid curto).
- Batch efetivo alto via gradient accumulation.
- Mixed precision (`bf16`/`fp16`) + checkpointing.

## 8) Infra AWS e custo estimado
Base de referência: instância 8 GPUs ~US$55/h.

### 8.1 Continual pretraining (1M imagens, 2D stream)
- 30-50 épocas: **~80-180 h** => **~US$4.4k-9.9k**.
- 80-120 épocas: **~180-420 h** => **~US$9.9k-23.1k**.

### 8.2 Fine-tuning supervisionado (heads + full FT)
- **~20-80 h** por modalidade principal, conforme tamanho/ruído.

### 8.3 Custos indiretos (não ignorar)
- Armazenamento/versionamento de checkpoints.
- ETL, validação clínica, monitoramento e reprocessamentos.
- Rodadas extras para ablação e calibração.

## 9) Métricas de sucesso
### 9.1 Técnicas
- AUC-ROC/PR por achado e modalidade.
- Sensibilidade em operating points clínicos definidos com especialistas.
- ECE/Brier para calibração.
- Robustez por clínica, aparelho e região.

### 9.2 Clínicas
- Taxa de ganho de detecção em achados críticos.
- Redução de falso negativo em cenários prioritários.
- Concordância com especialistas (kappa/consenso).

## 10) Riscos principais e mitigação
- Leakage entre treino/teste por paciente/série: bloqueio por chave de paciente e estudo.
- Viés por clínica/aparelho: amostragem estratificada + avaliação externa obrigatória.
- Ruído de laudo: treinamento com confiança e revisão humana direcionada.
- Drift operacional: monitoramento contínuo pós-deploy por modalidade/região.

## 11) Roadmap sugerido (10 semanas)
1. Semana 1-2: governança LGPD, catálogo, deduplicação, splits e baseline linear probe.
2. Semana 3-5: SSL dental (30-50 épocas) + avaliação de embeddings.
3. Semana 6-7: weak supervision com laudos e heads multi-tarefa.
4. Semana 8: fine-tuning full + calibração.
5. Semana 9: validação externa multi-clínica e análise de subgrupos.
6. Semana 10: pacote de decisão (go/no-go) para piloto clínico assistivo.

## 12) Entregáveis mínimos do MVP
- Dataset versionado com splits auditáveis.
- Backbone odontológico SSL + model card interno.
- Heads por modalidade com métricas por subgrupo.
- Relatório de validação externa e calibração.
- Plano de monitoramento de drift e re-treino trimestral.

## 13) Próximas decisões (curtas e objetivas)
1. Definir 3 tarefas clínicas prioritárias para o MVP (ex.: lesão periapical, perda óssea periodontal, cárie interproximal).
2. Escolher baseline de backbone (`ViT-B/14` vs `ViT-L/14`) para primeira rodada.
3. Congelar escopo inicial de modalidades (recomendação: começar por RX 2D antes de CBCT).
