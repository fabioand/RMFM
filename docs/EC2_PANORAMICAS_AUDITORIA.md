# Auditoria Especifica - Pastas de Panoramicas (EC2)

Host: `35.92.136.175`  
Path: `/dataminer/rmdatasets/data`  
Modo: somente leitura

## 1) Contagem por extensao (por pasta)
- `panoramicas`: 159.260 arquivos = 32.518 `.jpg` + 126.742 `.json`
- `panoramicas_anonymized`: 162.719 = 32.518 `.jpg` + 130.201 `.json`
- `panoramics_anonymized`: 130.545 = 32.518 `.jpg` + 98.027 `.json`
- `panoramicas_anonymized.delete`: 77.365 = 16.422 `.jpg` + 60.943 `.json`
- `panoramics_anonymized.delete`: 75.894 = 16.422 `.jpg` + 59.472 `.json`
- `panoramics_mixed`: 31.749 = 10.581 `.jpg` + 21.163 `.json` (+ 5 `.bak`)
- `panoramics_mixed_processed`: 31.744 = 10.581 `.jpg` + 21.163 `.json`
- `panoramics_mixed_processed_anonymized`: 31.744 = 10.581 `.jpg` + 21.163 `.json`
- `panoramicas_sample`: 448 = 141 `.jpg` + 307 `.json`
- `hyperpans`: 1.234 = 615 `.jpg` + 617 `.json` (+ 1 `.py`, 1 `.zip`)

## 2) Sobreposicao de JPGs (nomes de arquivo)
- `panoramicas_anonymized` e `panoramics_anonymized`: intersecao de 32.518 (aparentam ser o mesmo conjunto)
- `panoramicas_anonymized.delete` e `panoramics_anonymized.delete`: intersecao de 16.422 (mesmo conjunto)
- `panoramics_mixed` e `panoramics_mixed_processed`: intersecao de 10.581 (mesmo conjunto)
- `panoramicas` tem intersecao parcial de 1.368 com `panoramics_mixed`

## 3) Conclusao pratica
- Voce estava certo: grande parte dos arquivos nessas pastas de panoramicas sao anotacoes `.json`.
- Existem conjuntos de JPG duplicados entre pastas com nomes diferentes.
- Para experimento com DINO, e melhor escolher **uma fonte canonica de JPG** (ex.: `panoramicas`) e ignorar as pastas derivadas/duplicadas.
