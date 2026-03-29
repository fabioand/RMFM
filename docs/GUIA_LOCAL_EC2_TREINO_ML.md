# Guia de Versatilidade Local (Mac) + EC2 (AWS) para Treino ML

Objetivo: garantir que o mesmo projeto rode com mínimo atrito em duas realidades:
- Testes rápidos no Mac (MPS)
- Treinos grandes na EC2 (CUDA)

## 1) Princípio central
Separar **código** de **configuração**.
- Código: lógica de treino/inferência, sem paths hardcoded.
- Configuração: arquivos YAML por ambiente.

## 2) Estrutura recomendada
```text
project/
  configs/
    base.yaml
    local.yaml
    ec2.yaml
  scripts/
    train.py
    eval.py
    run_local.sh
    run_ec2.sh
  src/
    ...
  outputs/
  logs/
```

## 3) Estratégia de configuração (YAML)
- `base.yaml`: hiperparâmetros comuns e defaults.
- `local.yaml`: overrides para Mac (MPS, caminhos locais, batch menor).
- `ec2.yaml`: overrides para AWS (CUDA, batch maior, paths EBS/S3 staging).

### 3.1 Exemplo `configs/base.yaml`
```yaml
experiment:
  name: dino_ssl_panorama
  seed: 42

model:
  backbone: facebook/dinov2-small

train:
  epochs: 30
  lr: 1e-4
  weight_decay: 0.05
  num_workers: 4
  amp: true

data:
  image_size: 518
  dataset_name: panoramicas

runtime:
  device: auto
  output_dir: ./outputs
  log_dir: ./logs
```

### 3.2 Exemplo `configs/local.yaml`
```yaml
inherits: configs/base.yaml

runtime:
  profile: local
  device: mps
  output_dir: /Users/fabioandrade/RMFM/outputs/local

data:
  root: /Users/fabioandrade/datasets/panoramicas_small

train:
  batch_size: 8
  num_workers: 2
```

### 3.3 Exemplo `configs/ec2.yaml`
```yaml
inherits: configs/base.yaml

runtime:
  profile: ec2
  device: cuda
  output_dir: /mnt/efs-or-ebs/outputs/ec2

data:
  root: /mnt/efs-or-ebs/datasets/panoramicas_full

train:
  batch_size: 64
  num_workers: 16
```

## 4) Regras obrigatórias no código
1. Nunca hardcodar path em `src/` e `scripts/`.
2. Ler tudo de `cfg` (arquivo YAML + env vars opcionais).
3. Device com prioridade:
   - valor explícito em config
   - senão auto: `cuda > mps > cpu`
4. Salvar a config final usada junto do checkpoint.
5. Nomear saída com timestamp + nome do experimento.

## 5) Env vars (opcional, mas útil)
Use env vars para segredos e caminhos dinâmicos.

Exemplo:
- `HF_TOKEN`
- `DATA_ROOT`
- `OUTPUT_ROOT`
- `WANDB_API_KEY` (se usar)

No código:
- config define default
- env var pode sobrescrever

## 6) Comandos padronizados
Padronize um único comando por ambiente.

### Local
```bash
python scripts/train.py --config configs/local.yaml
```

### EC2
```bash
python scripts/train.py --config configs/ec2.yaml
```

## 7) Sincronização sem atrito
### 7.1 Código
- versionar no Git
- branch de experimento por feature

### 7.2 Dados
- não versionar datasets no Git
- manter datasets em storage apropriado (EBS/EFS/S3)
- manter um manifesto (CSV/JSON) versionado com:
  - origem
  - hash
  - split

### 7.3 Artefatos
- checkpoints e logs em pasta de saída por execução
- sync para S3 ao final de cada treino (ou por epoch)

## 8) Reprodutibilidade mínima
1. `seed` fixa em config.
2. `requirements.txt` + lock de versões.
3. registrar commit hash no log da execução.
4. salvar:
   - config final
   - métricas
   - melhor checkpoint
   - dataset manifest usado

## 9) Diferenças Local vs EC2 (e como tratar)
1. Device (`mps` vs `cuda`): controlado por config.
2. Batch size: perfil específico.
3. Num workers: perfil específico.
4. Paths: perfil específico.
5. Performance/precision: `amp` por perfil se necessário.

## 10) Automação recomendada
### 10.1 `run_local.sh`
- ativa venv local
- valida config
- roda treino local

### 10.2 `run_ec2.sh`
- ativa venv/conda da EC2
- valida acesso a dataset
- roda treino com `configs/ec2.yaml`
- sincroniza artefatos para S3

## 11) Checklist operacional
### Antes de rodar local
1. Config local aponta para dataset pequeno.
2. Device = `mps` (ou `auto`).
3. Batch reduzido.

### Antes de rodar EC2
1. Config EC2 aponta para dataset full.
2. Device = `cuda`.
3. Storage de saída com espaço suficiente.
4. Token/credenciais carregados.

### Depois de rodar
1. Verificar métricas.
2. Verificar se config final foi salva.
3. Sincronizar artifacts.
4. Registrar observações em changelog de experimento.

## 12) Recomendação prática para começar
1. Implementar parser de config com herança (`base + profile`).
2. Adaptar script de treino atual para ler somente `cfg`.
3. Criar `local.yaml` e `ec2.yaml` agora.
4. Rodar smoke test local e depois um mini-run na EC2 com subset.
5. Só então subir treino grande.

---

Esse padrão evita retrabalho e permite iterar rápido no Mac sem quebrar o fluxo de treino grande na AWS.
