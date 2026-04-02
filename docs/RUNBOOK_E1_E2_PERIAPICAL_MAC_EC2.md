# Runbook Oficial - E1/E2 Periapical (Mac + EC2)

Data: 2026-04-02

Objetivo: permitir que qualquer pessoa execute o fluxo de comparacao downstream (E1/E2) de forma reproduzivel, sem adivinhar caminhos.

## 1) Escopo deste runbook

Este guia cobre:
- export de backbone SSL na EC2 (teacher ou student),
- sincronizacao do export para o Mac,
- execucao de E1/E2 localmente no Mac com os mesmos parametros historicos.

## 2) Fonte da verdade de paths

## EC2 (treino/export)

- Projeto: `/dataset/RMFM`
- Venv: `/dataset/RMFM/.venv`
- Run full v2 atual:
  - `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20`
- Checkpoints:
  - `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/checkpoints`
- Exports:
  - `/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports`

## Mac (avaliacao E1/E2)

- Projeto: `/Users/fabioandrade/RMFM`
- Venv: `/Users/fabioandrade/RMFM/.venv`
- Imagens de classificacao:
  - `/Users/fabioandrade/RMFM/Downloads/imgs_class`
- Labels (JSON) de classificacao:
  - `/Users/fabioandrade/RMFM/Downloads/periapical_classificacao`
- Baseline da cabeca (referencia historica):
  - `/Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1`

## 3) Pre-check obrigatorio (antes de rodar)

No Mac, valide:

```bash
ls -ld /Users/fabioandrade/RMFM/Downloads/imgs_class
ls -ld /Users/fabioandrade/RMFM/Downloads/periapical_classificacao
ls -ld /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1
```

Se qualquer um falhar, nao rode E1/E2.

## 4) Export na EC2 (epoch alvo)

Exemplo oficial para `epoch_020`:

```bash
cd /dataset/RMFM
source /dataset/RMFM/.venv/bin/activate

python3 /dataset/RMFM/experiments/ssl_periapical_dinov2/scripts/export_backbone_checkpoint.py \
  --checkpoint /dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/checkpoints/epoch_020_snapshot.pt \
  --output-dir /dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports/backbone_epoch_020_teacher \
  --backbone-key teacher_backbone
```

Opcional (comparativo student):

```bash
python3 /dataset/RMFM/experiments/ssl_periapical_dinov2/scripts/export_backbone_checkpoint.py \
  --checkpoint /dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/checkpoints/epoch_020_snapshot.pt \
  --output-dir /dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports/backbone_epoch_020_student \
  --backbone-key student_backbone
```

## 5) Trazer export para o Mac

Criar destino local:

```bash
mkdir -p /Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports
```

Trazer teacher:

```bash
rsync -avz -e "ssh -i ~/.ssh/fabio.pem" \
  ubuntu@35.92.136.175:/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports/backbone_epoch_020_teacher \
  /Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports/
```

Trazer student (se aplicavel):

```bash
rsync -avz -e "ssh -i ~/.ssh/fabio.pem" \
  ubuntu@35.92.136.175:/dataset/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports/backbone_epoch_020_student \
  /Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports/
```

## 6) Rodar E1/E2 no Mac (teacher)

```bash
cd /Users/fabioandrade/RMFM
source /Users/fabioandrade/RMFM/.venv/bin/activate

python3 experiments/ssl_periapical_dinov2/scripts/run_downstream_periapical_eval.py \
  --backbone-dir /Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/ec2_full_ssl_periapical_v2_stable_bs20/exports/backbone_epoch_020_teacher \
  --images-dir /Users/fabioandrade/RMFM/Downloads/imgs_class \
  --labels-dir /Users/fabioandrade/RMFM/Downloads/periapical_classificacao \
  --baseline-run-dir /Users/fabioandrade/RMFM/experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1 \
  --output-dir /Users/fabioandrade/RMFM/experiments/ssl_periapical_dinov2/outputs/downstream_eval_epoch020_teacher \
  --offline
```

Resultado esperado:
- E1: `.../downstream_eval_epoch020_teacher/e1_reuse_head_summary.json`
- E2: `.../downstream_eval_epoch020_teacher/e2_retrain_head/summary.json`
- Manifesto: `.../downstream_eval_epoch020_teacher/downstream_eval_manifest.json`

## 7) Convencao de nomes

Para manter rastreabilidade:
- export teacher: `backbone_epoch_XXX_teacher`
- export student: `backbone_epoch_XXX_student`
- downstream teacher: `downstream_eval_epochXXX_teacher`
- downstream student: `downstream_eval_epochXXX_student`

`XXX` sempre com 3 digitos (`010`, `020`, `030`...).

## 8) Erros comuns e causa raiz

1. `n_samples=0` no split:
   - causa: `--labels-dir` errado ou inexistente.
   - no ambiente atual, correto e:
     - `/Users/fabioandrade/RMFM/Downloads/periapical_classificacao`

2. `rsync ... No such file or directory` no destino:
   - causa: pasta pai local nao existe.
   - solucao: `mkdir -p` no destino antes do `rsync`.

3. Linha com `\ ` (barra com espaco no final):
   - causa: shell quebra a continuacao da linha.
   - solucao: nao deixar espaco depois de `\`.

## 9) Politica offline (obrigatoria)

Quando o modelo ja estiver em cache local:
- usar `--offline` nos scripts que suportam.
- evitar depender de download em tempo de execucao.

## 10) Documento de entrada obrigatoria

Antes de qualquer operacao, ler:
- [LEIA_PRIMEIRO_AMBIENTES_MAC_EC2.md](/Users/fabioandrade/RMFM/docs/LEIA_PRIMEIRO_AMBIENTES_MAC_EC2.md)

