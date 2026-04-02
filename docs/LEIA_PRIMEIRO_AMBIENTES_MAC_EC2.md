# LEIA PRIMEIRO - Ambientes Mac e EC2

Guia operacional para qualquer pessoa (ou agente) conseguir trabalhar no projeto sem atrito entre:
- máquina local (Mac),
- EC2 de trabalho,
- volumes de dados principais.

## 1) Mapeamento de máquinas

### Mac (local)
- Host: iMac/MacBook local de desenvolvimento.
- Papel principal:
  - prototipagem rápida,
  - smoke tests,
  - inspeção visual (HTML/mosaicos),
  - edição de código e documentação.

### EC2 (AWS) - fluxo em duas máquinas

#### EC2 de preparação/validação (sempre ligada)
- Host: `35.92.136.175` (usuário `ubuntu`).
- Papel principal:
  - preparar scripts/configs,
  - smoke tests e validações,
  - inspeção de resultados com amostras pequenas,
  - staging do pipeline antes de escalar.
- Acesso SSH típico:
  - chave: `~/.ssh/fabio.pem`
  - comando: `ssh -i ~/.ssh/fabio.pem ubuntu@35.92.136.175`

#### EC2 de execução pesada (g4dn.2xlarge)
- Papel principal:
  - rodar jobs longos/pesados (treino e inferência em massa).
- Fluxo operacional:
  - prepara e valida no `35.92.136.175`,
  - quando está ok, liga a g4dn e executa o workload pesado.

## 2) Mapa de paths (Mac -> EC2)

### Repositórios
- RMFM:
  - Mac: `/Users/fabioandrade/RMFM`
  - EC2: `/dataset/RMFM`
- Hydra:
  - Mac: `/Users/fabioandrade/hydra` (ou `~/hydra`)
  - EC2: validar no host (não assumir sem checar com `ls`).

### Ambiente Python
- Mac:
  - venv principal: `/Users/fabioandrade/RMFM/.venv`
- EC2:
  - venv principal do projeto: `/dataset/RMFM/.venv`

### Dados (volume grande)
- EC2 (fonte principal, compartilhada entre as duas EC2):
  - `/dataminer/rmdatasets/data` (datasets odontológicos grandes)
  - Exemplo usado com frequência:
    - `/dataminer/rmdatasets/data/periapicais_processed/imgs`
  - Radiobot:
    - `/dataminer/radiobot`
- Observação operacional importante:
  - os mounts `/dataset` e `/dataminer` existem na EC2 de validação (`35.92.136.175`) e também na EC2 g4dn.2xlarge.
  - isso permite preparar/testar em uma e executar pesado na outra sem trocar caminhos.
- Mac (amostras/espelhos locais):
  - `/Users/fabioandrade/RMFM/Downloads/...`

## 3) Regra de ouro para caminhos

- Nunca hardcodar caminho só do Mac em script que será usado na EC2.
- Preferir parâmetros CLI (`--images-dir`, `--list-json`, `--output-dir`, `--run-dir`).
- Manter separação:
  - código e docs em `RMFM`,
  - artefatos grandes em `outputs/` e datasets fora do Git.
- Padronizar caminhos EC2 para portabilidade entre as duas máquinas:
  - projeto em `/dataset/RMFM`
  - dados em `/dataminer/...`

## 4) Sincronização Mac <-> EC2

Padrão recomendado:
- usar `rsync -av -e "ssh -i ~/.ssh/fabio.pem" ...`
- sincronizar apenas scripts/docs necessários,
- evitar copiar `Downloads`, `outputs` grandes e `.venv`.

Exemplo (script específico):
```bash
rsync -av \
  /Users/fabioandrade/RMFM/experiments/radiobot_folder_classifier/scripts/predict_list_to_json_dir.py \
  -e "ssh -i ~/.ssh/fabio.pem" \
  ubuntu@35.92.136.175:/dataset/RMFM/experiments/radiobot_folder_classifier/scripts/
```

## 5) Segurança de dados e da máquina

Dados clínicos:
- tratar todo dataset como sensível (paciente real),
- não exportar dados desnecessários para fora da infraestrutura,
- trabalhar com princípio de menor privilégio,
- manter trilha de auditoria (scripts, listas de input/output, manifests).

Chaves e credenciais:
- não comitar `.env`, chaves SSH, tokens e segredos,
- usar variáveis de ambiente para autenticação,
- revisar permissões de chave:
  - `chmod 600 ~/.ssh/fabio.pem`

Operação segura:
- evitar comandos destrutivos em `/dataminer`,
- inventários em modo read-only sempre que possível,
- para arquivos temporários na EC2, usar `/dataset/RMFM/...` (área do projeto).

## 6) Boas práticas para modelos (neste projeto)

- Preferir execução offline quando o modelo já foi baixado:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - `--offline` nos scripts.
- Em produção de inferência em massa:
  - registrar `summary.json`,
  - registrar erros em `_errors.jsonl`,
  - manter outputs reprodutíveis por run-dir.

## 7) Checklist rápido antes de rodar qualquer job

1. Confirmar máquina alvo (Mac ou EC2).
2. Confirmar venv correto.
3. Confirmar paths de input/output.
4. Confirmar modo offline (quando aplicável).
5. Confirmar que output não sobrescreve run importante.
6. Confirmar que credenciais/segredos não estão no comando/documento.

## 8) Documento de referência para novas threads

Quando iniciar uma nova thread/agente no projeto, pedir para ler primeiro:
- `docs/LEIA_PRIMEIRO_AMBIENTES_MAC_EC2.md`

Para execucao de export + E1/E2 periapical (comandos e paths oficiais):
- [RUNBOOK_E1_E2_PERIAPICAL_MAC_EC2.md](/Users/fabioandrade/RMFM/docs/RUNBOOK_E1_E2_PERIAPICAL_MAC_EC2.md)
