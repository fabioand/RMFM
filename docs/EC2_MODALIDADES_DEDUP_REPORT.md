# Modalidades Odontologicas na EC2 (com deduplicacao)

- Host: `35.92.136.175`
- Path auditado: `/dataminer/rmdatasets/data`
- Metodo: somente leitura, contagem de `.jpg` por pasta e deduplicacao por nome de arquivo dentro de cada modalidade
- Coleta: 2026-03-27 22:35:24

## Resumo por Modalidade

| Modalidade | Bruto (JPG) | Deduplicado | Redundancia | Repositorios |
|---|---:|---:|---:|---:|
| Outros/Ambiguos | 505.979 | 228.858 | 277.121 | 22 |
| Periapicais | 185.268 | 185.268 | 0 | 3 |
| Panoramicas | 162.897 | 84.077 | 78.820 | 10 |
| Teleradiografias | 20.734 | 20.734 | 0 | 1 |
| Tomografia/CBCT | 8.419 | 1.685 | 6.734 | 6 |
| Interproximais/Bitewing | 5.906 | 3.658 | 2.248 | 2 |

## Repositorios por Tipo

### Outros/Ambiguos
- `anno_00`: 113.864 JPG (base/indefinido)
- `procedures_pre`: 100.670 JPG (base/indefinido)
- `denticao`: 46.755 JPG (base/indefinido)
- `denticao_imgs.bak`: 46.755 JPG (derivado)
- `desdentados.bak`: 39.480 JPG (derivado)
- `desdentados`: 33.317 JPG (base/indefinido)
- `desndentadis_imgs.bak`: 33.317 JPG (derivado)
- `anomalias_00`: 18.977 JPG (base/indefinido)
- `longoeixo`: 16.422 JPG (base/indefinido)
- `longoeixo.missmatch`: 10.645 JPG (derivado)
- `procedimentos`: 9.889 JPG (base/indefinido)
- `mista`: 7.654 JPG (base/indefinido)
- `mista_imgs.bak`: 6.163 JPG (derivado)
- `celto_curves`: 5.691 JPG (base/indefinido)
- `procedures_pre_test`: 4.832 JPG (base/indefinido)
- `procedures_pre_ver0`: 4.832 JPG (base/indefinido)
- `celto_curves.bak`: 3.418 JPG (derivado)
- `metais`: 2.000 JPG (base/indefinido)
- `celto_points`: 582 JPG (base/indefinido)
- `anomalias-500`: 496 JPG (base/indefinido)
- `longoeixo_sample`: 180 JPG (derivado)
- `fabio-teeth-segmentation-original-dataset`: 40 JPG (base/indefinido)

### Periapicais
- `periapicais_processed_anonymized`: 79.266 JPG (derivado)
- `periapicais_processed`: 73.411 JPG (derivado)
- `periapicais`: 32.591 JPG (base/indefinido)

### Panoramicas
- `panoramicas`: 32.518 JPG (base/indefinido)
- `panoramicas_anonymized`: 32.518 JPG (derivado)
- `panoramics_anonymized`: 32.518 JPG (derivado)
- `panoramicas_anonymized.delete`: 16.422 JPG (derivado)
- `panoramics_anonymized.delete`: 16.422 JPG (derivado)
- `panoramics_mixed`: 10.581 JPG (base/indefinido)
- `panoramics_mixed_processed`: 10.581 JPG (derivado)
- `panoramics_mixed_processed_anonymized`: 10.581 JPG (derivado)
- `hyperpans`: 615 JPG (base/indefinido)
- `panoramicas_sample`: 141 JPG (derivado)

### Teleradiografias
- `teles`: 20.734 JPG (base/indefinido)

### Tomografia/CBCT
- `tomo_views`: 4.814 JPG (base/indefinido)
- `tomos`: 1.658 JPG (base/indefinido)
- `tomoscoronal`: 555 JPG (base/indefinido)
- `tomobox`: 495 JPG (base/indefinido)
- `tomosuperior.old`: 458 JPG (derivado)
- `tomosuperior`: 439 JPG (base/indefinido)

### Interproximais/Bitewing
- `interproximais`: 3.658 JPG (base/indefinido)
- `interproximais.bak`: 2.248 JPG (derivado)

## Nota Importante
- A deduplicacao foi feita por nome de arquivo JPG (basename) dentro de cada modalidade.
- Se existirem arquivos diferentes com mesmo nome em pastas distintas, a deduplicacao pode subestimar levemente o total unico real.
- Para deduplicacao perfeita, o proximo passo e comparar hash (ex.: SHA1) dos arquivos.