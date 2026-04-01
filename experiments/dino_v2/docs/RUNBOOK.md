# Runbook Operacional

## Pré-requisitos
- Python 3.9+
- Acesso ao Hugging Face Hub para baixar checkpoints
- Espaço em disco para pesos/cache
- Para DINOv3: acesso liberado ao repo gated + token (`HF_TOKEN`)

## Primeiro uso
1. Ativar ambiente:
```bash
cd /Users/fabioandrade/RMFM/experiments/dino_v3
source .venv/bin/activate
```
2. Rodar smoke test:
```bash
PYTHONPATH=src python scripts/smoke_test_dinov3.py
```
3. Conferir artefato:
- `outputs/smoke_test_result.json`

## Troubleshooting
- Erro de rede/DNS no Hub: validar conectividade e permissões do ambiente.
- Erro de `gated repo`: aceitar termos no checkpoint DINOv3 e autenticar com `huggingface-cli login`.
- OOM/GPU: usar `--cpu` ou reduzir lote/tamanho de imagem no script de produção.
- Latência alta no 1º run: normal devido ao download inicial do modelo.

## Próximo passo sugerido
- Criar `assets/samples/` com subset curado por modalidade e rodar índice de embeddings.
