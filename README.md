# RMFM - Pesquisa em IA Odontológica com DINO

Repositório de experimentos para visão computacional odontológica, com foco em:
- embeddings visuais com DINO (v2/v3),
- clusterização e mapas de atenção,
- classificação de periapicais com encoder congelado + cabeça treinável,
- preparação de pipeline para treino local (Mac) e escala em EC2.

## Status
- Projeto em desenvolvimento ativo.
- Código e documentação organizados para execução reproduzível.
- Artefatos pesados (datasets, outputs, pesos) ficam fora do Git.

## Estrutura
- `experiments/dino_v3/`: laboratório de embeddings, retrieval e clusterização.
- `experiments/periapical_dino_classifier/`: classificação supervisionada de periapicais.
- `scripts/`: utilitários gerais (ex.: integração com API RM IA).
- `docs/`: inventários, relatórios e documentação de experimentos.

## Quickstart
```bash
cd /Users/fabioandrade/RMFM
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r experiments/dino_v3/requirements.txt
```

## Rodar DINO (cluster + HTML)
```bash
cd /Users/fabioandrade/RMFM/experiments/dino_v3
source /Users/fabioandrade/RMFM/.venv/bin/activate

PYTHONPATH=src python scripts/cluster_embeddings_report.py \
  --model-id facebook/dinov2-small \
  --images-dir /caminho/para/imagens \
  --output-dir outputs/cluster_run \
  --n-clusters 10 \
  --save-attention-maps
```

Saídas típicas:
- `cluster_report.html`
- `cluster_rows.json`
- `summary.json`
- `timing.json`

## Experimento de Classificação Periapical
Veja o guia em:
- `experiments/periapical_dino_classifier/README.md`

Inclui:
- treino com encoder congelado,
- treino com features cacheadas,
- variante `flip mirror` para reduzir erros de lateralidade,
- avaliação e HTML agrupado por classe predita.

## Documentação Principal
- `docs/EXPERIMENTO_DINO_PERIAPICAL_FROZEN_HEAD.md`
- `docs/EXPERIMENTO_1_RESULTADOS_DINO_FROZEN_CACHED_256.md`
- `docs/EXPERIMENTO_2_FLIP_MIRROR_RESULTADOS.md`
- `docs/GUIA_LOCAL_EC2_TREINO_ML.md`

## Observações Importantes
- Este repositório ignora artefatos pesados via `.gitignore` (`outputs`, `Downloads`, caches, pesos e imagens).
- Para execução em EC2, preferir ambientes separados por máquina (`.venv-cpu` e `.venv-gpu`).
- Dados clínicos reais devem seguir políticas de segurança, anonimização e conformidade regulatória.
