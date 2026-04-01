# Projeto Fase 1 - SSL DINO para Periapicais

## Objetivo
Construir um primeiro dataset grande e confiável de periapicais para SSL (continual pretraining do DINOv2), combinando:
- classificador odontológico geral (17 classes) treinado no projeto,
- clusterização por embeddings DINO,
- revisão visual por mosaico amostrado.

Meta prática inicial:
- partir de ~75k imagens candidatas,
- remover não-periapicais com alta precisão,
- preservar periapicais difíceis (casos de borda), evitando limpeza agressiva.

## Andamento Atual (2026-03-31)

Concluído:
- Treino do classificador Radiobot 17 classes em escala maior (EC2):
  - run: `/dataset/RMFM/experiments/radiobot_folder_classifier/outputs/run_cached_head_17cls_all_v1`
  - `num_samples=25125`
  - `test_accuracy=0.9859`
  - `test_macro_f1=0.9845`
- Pipeline de inferência em lote com JSON por imagem implementado:
  - script: `/dataset/RMFM/experiments/radiobot_folder_classifier/scripts/predict_list_to_json_dir.py`
  - saída por imagem com `pred_label`, `pred_confidence`, `top_classes`, `probs_by_class`
  - modo offline suportado
  - robustez a imagem inválida (continua execução + `_errors.jsonl`)
- Smoke local em 3000 periapicais concluído:
  - `Periapical=2663`, `Não-periapical=337`
  - MPS offline: `71.11s` (~`42.19 img/s`)
- Classificação em massa concluída no conjunto grande:
  - input: `/dataminer/rmdatasets/data/periapicais_processed/imgs`
  - total: `73,411`
  - `Periapical=65,052` (88.6%)
  - `Interproximal=435`
- Inspeção visual de classes menos numerosas concluída (mosaico filtrado por classe):
  - confirmação de presença de casos-limite e transição entre BW/periapical.

Em andamento:
- pesquisa, discussões e planejamento do desenho do treino SSL (fase 2):
  - estratégias de augmentação segura para RX odontológico,
  - definição de multi-crop/global-local,
  - plano de experimentos e ablações.

Próximos marcos:
- materializar artefatos finais do dataset:
  - `ssl_periapical_v1_keep.txt`
  - `ssl_periapical_v1_drop.txt`
  - `ssl_periapical_v1_manifest.json`
- fechar preset do primeiro treino Continual SSL (crops + augs + schedule).
- iniciar treino SSL v1 e monitorar métricas de estabilidade.

## Decisão Atual de Composição (v1)

Decisão prática adotada:
- incluir todas as imagens preditas como `Periapical`;
- incluir também as imagens preditas como `Interproximal` (~400 no lote atual).

Motivação:
- `Interproximal` é modalidade radiográfica próxima de periapical em estrutura/sinal;
- fração pequena no total (~400 frente a ~65k) não deve diluir o domínio;
- ajuda robustez e absorve casos-limite (incluindo periapicais que caíram nessa classe).

Diretriz operacional:
- `ssl_periapical_v1_keep.txt` = `Periapical + Interproximal`
- demais classes seguem para exclusão inicial ou revisão específica por mosaico/cluster.

Conjunto elegível atual para o primeiro SSL:
- `Periapical + Interproximal = 65,487` imagens.

Observação:
- registrar essa escolha explicitamente no `ssl_periapical_v1_manifest.json`;
- planejar ablação posterior (com vs sem `Interproximal`) em tarefa downstream para medir impacto.

## Hipótese de trabalho
Sim: o classificador de 17 classes deve ajudar muito na triagem de não-periapicais.

Por que:
- no run atual (17 classes), o modelo ficou forte (`test_acc ~0.9756`, `macro_f1 ~0.9764`);
- ele já separa modalidades/visões diferentes de forma consistente.

Mas:
- para dataset clínico grande, ainda é recomendável etapa de verificação por cluster/mosaico para reduzir risco de falso negativo (periapical útil removida).

## Estratégia recomendada (híbrida)

### Etapa A - Filtragem supervisionada inicial (17 classes)
1. Rodar inferência no conjunto grande (75k).
2. Manter:
- imagens preditas como `Periapical` com confiança alta (ex.: >= 0.90, calibrável).
3. Separar para revisão:
- predição `Periapical` com confiança intermediária (ex.: 0.60-0.90),
- qualquer classe não-periapical, mas com baixa margem de decisão.

Saídas:
- `kept_periapical_high_conf.txt`
- `review_bucket.txt`
- `dropped_non_periapical_high_conf.txt`

### Etapa B - Clusterização no conjunto mantido/revisão
1. Extrair embeddings DINOv2 em batch.
2. Clusterizar em `k=3..8` (começando com k=3/4 para limpeza grossa).
3. Gerar mosaico amostrado por cluster (não o cluster inteiro).
4. Revisar clusters do menor para o maior.

Saídas:
- `cluster_assignments.csv` (`stem`, `cluster_k3`, `cluster_k4`, ...)
- `cluster_review_decisions.yaml` (manter/excluir/revisar por cluster)
- `stems_to_drop_manual.txt`

### Etapa C - Consolidação do dataset SSL
1. Unir regras:
- exclusões do classificador (alta confiança não-periapical),
- exclusões manuais da revisão por cluster.
2. Gerar:
- `ssl_periapical_v1_keep.txt`
- `ssl_periapical_v1_drop.txt`
- `ssl_periapical_v1_manifest.json`

## Princípios de qualidade
- Evitar over-cleaning: remover só o claramente não-periapical.
- Preservar diversidade: artefatos, angulações incomuns e qualidade variável ajudam robustez.
- Rastreabilidade total: toda exclusão precisa ser reproduzível por arquivo de decisão.

## Formato de armazenamento de embeddings (escala 75k)
Evitar `stem.npy` individual por imagem.

Preferir:
- `embeddings.npy` (N x D),
- `stems.json` (ordem alinhada ao `embeddings.npy`),
- `meta.json` (modelo, resolução, data, hash de entrada).

Opcional (escala maior):
- shards (`embeddings_000.npy`, `embeddings_001.npy`, ...).

## Processo de decisão sugerido
1. Rodada 1:
- classificador 17 classes + k=3/4,
- limpar apenas clusters claramente não-periapicais.
2. Rodada 2:
- reclusterizar restante com k maior (6/8),
- revisar só clusters ambíguos.
3. Fechar `ssl_periapical_v1`.

## Entregáveis da Fase 1
- Dataset filtrado para SSL (`ssl_periapical_v1_keep.txt`).
- Relatório de limpeza com contagens por etapa.
- Mosaicos de revisão por cluster.
- Manifesto de versionamento (`ssl_periapical_v1_manifest.json`).

## Próximo passo técnico
Implementar pipeline em scripts:
1. `embed_large.py`:
- extração batched de embeddings + persistência em matriz.
2. `cluster_sample_report.py`:
- clusterização multi-k + mosaico amostrado + export de decisão manual.
3. `apply_filter_from_decisions.py`:
- aplica decisões e gera manifests finais para SSL.
