# DINOv3 Lab (Hugging Face)

Laboratório inicial para testar **DINOv3 puro** com imagens odontológicas: extração de embeddings, inspeção de capacidades e retrieval por similaridade.

## Estrutura
- `src/dino_v3_lab/`: utilitários de inferência
- `scripts/`: scripts executáveis de teste/experimentos
- `outputs/`: artefatos de execução (JSON, embeddings, índices)
- `docs/`: documentação de uso e próximos passos
- `assets/`: imagens de exemplo (opcional)

## Setup rápido
```bash
cd /Users/fabioandrade/RMFM/experiments/dino_v3
pip install -r requirements.txt
```

## Acesso ao DINOv3 no Hugging Face (gated)
Os checkpoints `facebook/dinov3-*` exigem aceite de licença + autenticação.

```bash
source .venv/bin/activate
huggingface-cli login
export HF_TOKEN=seu_token
```

## 1) Listar modelos DINOv3 disponíveis
```bash
PYTHONPATH=src python scripts/list_dinov3_models.py
```

## 2) Inspecionar capacidades do checkpoint
```bash
PYTHONPATH=src python scripts/inspect_capabilities.py \
  --model-id facebook/dinov3-vits16-pretrain-lvd1689m \
  --output outputs/capabilities.json
```

## 3) Smoke test (encoder + embedding)
```bash
PYTHONPATH=src python scripts/smoke_test_dinov3.py \
  --model-id facebook/dinov3-vits16-pretrain-lvd1689m \
  --output outputs/smoke_test_result.json
```

## 3b) Smoke test sem gating (sanity check de pipeline)
```bash
PYTHONPATH=src python scripts/smoke_test_dinov3.py \
  --model-id facebook/dinov2-small \
  --output outputs/smoke_test_result_dinov2.json \
  --offline
```

## 4) Construir índice de embeddings para retrieval
```bash
PYTHONPATH=src python scripts/build_embedding_index.py \
  --images-dir /caminho/para/imagens \
  --output-dir outputs/index \
  --offline
```

## 5) Consultar similares
```bash
PYTHONPATH=src python scripts/query_embedding_index.py \
  --query-image /caminho/para/query.png \
  --index-dir outputs/index \
  --top-k 5 \
  --offline
```

## 6) Clusterizar e gerar mosaico HTML (primeiro teste odontológico)
```bash
PYTHONPATH=src python scripts/cluster_embeddings_report.py \
  --model-id facebook/dinov2-small \
  --images-dir /Users/fabioandrade/RMFM/Downloads/periapicais_100 \
  --output-dir outputs/cluster_periapicais_100 \
  --n-clusters 3
```

## O que o DINOv3 puro já entrega
- Encoder/backbone para embeddings visuais de alta qualidade.
- Base para retrieval, clustering e detecção de anomalia.
- Ponto de partida para fine-tuning em classificação, detecção e segmentação.
- Alguns checkpoints específicos podem trazer head pronta (ex.: DPT para depth).

Veja detalhes em `docs/CAPABILITIES.md`.
