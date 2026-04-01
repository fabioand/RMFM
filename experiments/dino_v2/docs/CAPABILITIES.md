# Capacidades Iniciais do DINOv3 no Lab

Nota: no Hugging Face, os checkpoints oficiais `facebook/dinov3-*` estão marcados como gated e exigem token de acesso.

## 1) Sem fine-tuning (DINO puro)
- Feature extraction (embeddings globais e/ou por patch).
- Similaridade semântica de imagem (image retrieval).
- Clustering de exames por padrão visual.
- Anomaly/outlier detection no espaço de embeddings.

## 2) Com fine-tuning
- Classificação multi-classe e multilabel por modalidade.
- Segmentação/detecção com heads apropriadas.
- Tarefas de regressão (ex.: medidas) com dataset supervisionado.

## 3) Checkpoints e heads
- Checkpoints `*-pretrain-*`: tipicamente backbone/encoder.
- Checkpoints como `*-dpt-head`: incluem head específica (ex.: depth estimation).

## 4) Uso recomendado para começar em odontologia
1. Rodar smoke test e extração de embedding.
2. Montar índice de similaridade com subset representativo.
3. Validar visualmente vizinhos por modalidade/achado.
4. Definir baseline de task supervisionada para próximo ciclo.
