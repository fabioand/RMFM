# RMFM - Pesquisa em IA Odontológica com DINO

Repositório de experimentos para visão computacional odontológica, com foco em:
- embeddings visuais com DINO (v2/v3),
- clusterização e mapas de atenção,
- classificação de periapicais com encoder congelado + cabeça treinável,
- classificação geral multi-classe por pastas (Radiobot),
- preparação de pipeline para treino local (Mac) e escala em EC2.

## Status
- Projeto em desenvolvimento ativo.
- Código e documentação organizados para execução reproduzível.
- Artefatos pesados (datasets, outputs, pesos) ficam fora do Git.

## Estrutura
- `experiments/dino_v2/`: laboratório de embeddings, retrieval e clusterização.
- `experiments/periapical_dino_classifier/`: classificação supervisionada de periapicais.
- `experiments/radiobot_folder_classifier/`: classificação supervisionada por classes de pasta (17 classes).
- `scripts/`: utilitários gerais (ex.: integração com API RM IA).
- `docs/`: inventários, relatórios e documentação de experimentos.

## Quickstart
```bash
cd /Users/fabioandrade/RMFM
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r experiments/dino_v2/requirements.txt
```

## Rodar DINO (cluster + HTML)
```bash
cd /Users/fabioandrade/RMFM/experiments/dino_v2
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
- `docs/LEIA_PRIMEIRO_AMBIENTES_MAC_EC2.md`  <- referência inicial para qualquer nova thread/agente
- `docs/RUNBOOK_SSL_PERIAPICAL_V1_END_TO_END.md`
- `docs/SSL_PERIAPICAL_V1_MANIFESTO.md`
- `docs/MATRIZ_SCRIPTS_PIPELINE.md`
- `docs/LEVANTAMENTO_SSL_DINO_MEDICO_ESTADO_DA_ARTE.md`
- `docs/EXPERIMENTO_DINO_PERIAPICAL_FROZEN_HEAD.md`
- `docs/EXPERIMENTO_1_RESULTADOS_DINO_FROZEN_CACHED_256.md`
- `docs/EXPERIMENTO_2_FLIP_MIRROR_RESULTADOS.md`
- `docs/RESULTADOS_EXPERIMENTOS_ATUAIS.md`
- `docs/GUIA_LOCAL_EC2_TREINO_ML.md`

## Snapshot De Resultados
Ver consolidado em:
- `docs/RESULTADOS_EXPERIMENTOS_ATUAIS.md`

Resumo rápido:
- Periapical baseline (sem flip mirror): `acc=0.6091`, `macro_f1=0.5693`
- Periapical com flip mirror: `acc=0.7462`, `macro_f1=0.7396`
- RM API (periapical, mesmo conjunto 1312): `acc=0.9505`
- Radiobot 17 classes (run ampliado, 25125 imagens): `test_acc=0.9859`, `test_macro_f1=0.9845`

## Observações Importantes
- Este repositório ignora artefatos pesados via `.gitignore` (`outputs`, `Downloads`, caches, pesos e imagens).
- Para execução em EC2, preferir ambientes separados por máquina (`.venv-cpu` e `.venv-gpu`).
- Dados clínicos reais devem seguir políticas de segurança, anonimização e conformidade regulatória.
