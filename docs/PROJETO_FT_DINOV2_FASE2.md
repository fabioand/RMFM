# Projeto Fase 2 - Fine-Tuning DINOv2 em Radiografias Odontológicas

Status: **ativo (revisado em 2026-03-31)**

## Objetivo da Fase 2

Executar o primeiro ciclo de fine-tuning do DINOv2 com imagens periapicais já curadas, buscando ganho de representação de domínio odontológico para melhorar tarefas downstream (classificação, segmentação e retrieval).

## Escopo inicial

- Backbone base: `facebook/dinov2-small`
- Dados iniciais: conjunto curado SSL v1 (`Periapical + Interproximal`)
- Estratégia de treino: começar com setup estável e conservador, depois abrir ablações

Decisão de backbone para o ciclo atual:
- executar v1 somente com `dinov2-small`;
- deixar `dinov2-base` para fase posterior de upgrade;
- após estabilizar a fase atual, iniciar comparativos com DINOv3.

## Decisão oficial de dados (v1)

Para o primeiro ciclo de FT/SSL DINOv2:

- usar todas as imagens elegíveis do contrato `ssl_periapical_v1_keep.txt`;
- composição: cerca de `65k` periapicais + cerca de `~400` interproximais (BW);
- total alvo: cerca de `65.5k` imagens.

Racional:
- manter foco no domínio periapical;
- incluir fração pequena de interproximais para robustez em casos-limite de aquisição/transição.

## Primeira discussão: processamento de imagem para treino

Como as imagens já estão curadas, o principal risco agora é aplicar processamento que degrade sinal anatômico fino. O processamento deve preservar estrutura dentária e padrão radiográfico.

### 1) Pipeline de entrada (pré-processamento)

Recomendado para v1:

- converter para 3 canais (replicando grayscale quando necessário)
- normalizar para range esperado pelo processor do DINOv2
- `RandomResizedCrop` com controle de escala anatômica
- manter proporção clínica útil e evitar recortes extremos

Diretriz:
- não aplicar equalizações agressivas no v1 (CLAHE/hist-match) dentro do pipeline principal;
- se necessário, testar como ablação isolada.

Nota técnica:
- o DINOv2 foi pré-treinado em entrada RGB (3 canais), então replicar grayscale em 3 canais mantém compatibilidade direta com os pesos sem alterar o `patch_embed` neste primeiro ciclo.

### 2) Multi-crop (para SSL/continual pretraining)

Preset inicial sugerido:

- `2` crops globais
- `6` crops locais (subir para `8` após estabilidade)
- tamanho global: `384` (avaliar `512` em GPU com folga)
- tamanho local: `192` (faixa alvo `160–224`)
- `global_scale=(0.15, 1.0)`
- `local_scale=(0.05, 0.15)`
- `ratio=(0.90, 1.10)` para evitar crops muito alongados

Racional:
- preservar detalhes finos de ápice, trabeculado e contornos dentários;
- evitar locals pequenos demais para RX odontológico.
- reduzir deformação geométrica por aspect ratio extremo.

Regra de recorte:
- os crops são extraídos da imagem original (resolução nativa) e só depois redimensionados para os tamanhos alvo (global/local).

### 3) Augmentações seguras para RX odontológico

Manter no v1:

- rotação leve: `±5°`
- brilho/contraste leve (janela estreita)
- blur gaussiano leve (baixa probabilidade)
- ruído gaussiano leve (baixa probabilidade)

Evitar no v1:

- hue/saturation (sem valor clínico em RX)
- solarization/posterization
- warps geométricos fortes
- flip horizontal livre sem regra anatômica explícita

### 4) Política de resolução

Estratégia prática:

- iniciar em `384` para ciclo rápido e estável
- validar memória/throughput
- abrir run de `512` como ablação de detalhe

Regra:
- manter resolução e crop constantes entre runs comparativos.

### 5) Checklist de qualidade do processamento (antes de treinar)

1. Gerar mosaico de amostras por batch já transformadas (global/local).
2. Validar visualmente preservação de estruturas odontológicas finas.
3. Confirmar ausência de artefatos sintéticos fortes.
4. Medir distribuição de tamanho efetivo dos crops.
5. Medir distribuição de aspect ratio dos crops e confirmar concentração próxima de quadrado.
6. Congelar preset em arquivo de config versionado (yaml/json).

## Proposta de preset v1 (congelar para o primeiro run)

- input base: `ssl_periapical_v1_keep.txt` (cerca de `65k` periapicais + `~400` interproximais)
- global crops: `2 x 384`, `scale=(0.15, 1.0)`
- local crops: `6 x 192`, `scale=(0.05, 0.15)`
- crop ratio: `0.90–1.10`
- augmentações: rotação leve + brilho/contraste leve + blur/ruído leves
- sem flip livre
- seed fixa + logging completo de config

## Plano imediato (próximos passos)

1. Implementar script de visualização do pipeline de transforms (amostras globais/locais).
2. Rodar smoke de processamento em subset pequeno (ex.: 1k imagens).
3. Ajustar ranges caso haja perda de detalhe clínico.
4. Portar o monitoramento visual do Hydra (manifest + HTML + filtros) para treino ViT.
5. Congelar `preset_ft_dinov2_periapical_v1.yaml`.
6. Iniciar treino FT/SSL com monitoramento de estabilidade.

## Monitoramento visual no treino (decisão de abordagem)

Será reaproveitado o padrão do repositório Hydra:

- captura periódica por época;
- artefatos em `manifest.jsonl`;
- viewer `index.html` filtrável por época/amostra/grupo.

Adaptação para ViT/SSL:

- substituir painéis de máscara (GT) por painéis de views SSL (globais/locais);
- incluir metadados de crop (`scale`, `ratio`, posição);
- incluir atenção do ViT (ex.: CLS->patch/rollout) por camada selecionada.

## Métricas de referência já registradas (benchmark atual)

Tarefa de referência para comparação:
- classificação periapical com encoder congelado + head treinável;
- `1312` imagens, `14` classes;
- backbone histórico: `facebook/dinov2-small`.

Run baseline (sem flip mirror):
- run: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_v2_60ep`
- `best_val_macro_f1 = 0.7136`
- `test_accuracy = 0.6091`
- `test_macro_f1 = 0.5693`

Run melhor (com `--augment-flip-mirror`):
- run: `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`
- `best_val_macro_f1 = 0.7709`
- `test_accuracy = 0.7462`
- `test_macro_f1 = 0.7396`

Observação:
- as contagens por classe no treino não são exatamente uniformes; o range observado no split de treino foi de `44` a `109` por classe.

## Plano de avaliação comparativa (equidade dos testes)

Objetivo:
- comparar o encoder DINOv2 após FT/SSL contra o benchmark atual em cenário estritamente pareado.

### Protocolo oficial de comparação

1. Manter exatamente o mesmo dataset rotulado de `1312` e as mesmas `14` classes.
2. Reutilizar o mesmo split (`train/val/test`) já materializado no experimento periapical.
3. Manter o mesmo script de treino da cabeça e os mesmos hiperparâmetros de head.
4. Medir e reportar as mesmas métricas finais:
- `best_val_macro_f1`
- `test_accuracy`
- `test_macro_f1`
5. Salvar também:
- `classification_report_test.json`
- `confusion_matrix_test.csv`
- HTML de inspeção do teste agrupado por classe predita.

### Experimentos comparativos mínimos

1. `E0` (referência histórica): `run_cached_head_256_flipmirror_v1`.
2. `E1` (teste de compatibilidade): encoder FT/SSL + cabeça antiga (sem retreinar).
3. `E2` (comparação principal): encoder FT/SSL + cabeça nova treinada no mesmo protocolo do benchmark.

Leitura esperada:
- `E1` pode piorar por mudança do espaço de features;
- `E2` é o comparativo válido para medir ganho real do FT/SSL.

### Critério de sucesso da Fase 2 (comparativo principal)

Comparar `E2` contra `E0`:
- alvo principal: aumentar `test_macro_f1` acima de `0.7396`;
- alvo secundário: manter/elevar `test_accuracy` acima de `0.7462`;
- alvo de estabilidade: não regredir `best_val_macro_f1` abaixo de `0.7709`.

### Regras de rastreabilidade

- cada run comparativo deve ter `summary.json` e manifesto de config;
- registrar checkpoint do encoder FT/SSL usado em cada avaliação;
- registrar seed e assinatura do split para permitir reprodução exata.

## Decisões pendentes para a Fase 2

- para a implementação v1, não há bloqueios decisórios abertos;
- pendências futuras ficam para ciclos pós-v1 (ex.: ampliar para `dinov2-base`, comparar com DINOv3 e ablações extras de augment/crop).
