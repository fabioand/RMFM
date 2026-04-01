# Levantamento SSL DINO Médico - Estado da Arte (Fase 2)

Documento de referência para o início do Continual SSL do DINOv2 em radiologia odontológica, com foco prático em:
- periapicais em escala,
- escolhas de crops/augmentations,
- cuidados de domínio médico.

## 1) O que a literatura/prática mostra

### 1.1 DINO clássico (teacher-student + multi-crop)
- O método base usa:
  - 2 views globais,
  - múltiplas views locais,
  - matching student/teacher entre views da mesma imagem.
- No código oficial DINO, defaults conhecidos:
  - global crop size: `224`
  - local crop size: `96`
  - `global_crops_scale=(0.4, 1.0)`
  - `local_crops_scale=(0.05, 0.4)`
  - `local_crops_number=8`

Implicação para odonto:
- o framework multi-crop é sólido, mas os crops padrões podem ser pequenos demais para detalhes dentários finos.

### 1.2 Adaptações para X-ray (insight útil para odonto)
- Em RayDINO (SSL para CXR em larga escala), foi reportado ganho ao preservar mais detalhe:
  - large crops: `512` (vs 224)
  - small crops: `224` (vs 96)
  - large cobrindo pelo menos ~`15%` da imagem
  - small cobrindo até ~`15%`

Implicação:
- para radiologia odontológica (estruturas pequenas), aumentar resolução e ajustar escalas de crop tende a ajudar.

### 1.3 SSL médico costuma usar augmentations “anatômicas”
- Trabalhos de SSL médico (ex.: Swin UNETR) usam combinações de:
  - crop aleatório,
  - rotação controlada,
  - masking/cutout/inpainting,
  - objetivos complementares.
- Revisões de data augmentation médica reforçam:
  - transformações devem preservar plausibilidade anatômica,
  - flips/rotações fortes podem violar semântica clínica dependendo da tarefa.

### 1.4 Evidência em dental sobre flip seletivo
- Há trabalhos odontológicos recentes usando flip horizontal de forma seletiva (apenas em estruturas simétricas).

Implicação:
- em odonto RX, evitar flip “cego”; usar somente quando houver justificativa anatômica/semântica.

## 2) Estratégia recomendada para o primeiro Continual SSL (periapicais)

### 2.1 Escopo do dado
- Fase inicial: dataset filtrado de periapicais (sem mistura agressiva de modalidades).
- Objetivo: especializar embedding no domínio periapical antes de fase multimodal.

### 2.2 Backbone inicial
- Começar com `dinov2-small` (ciclos rápidos e custo menor).
- Avaliar `dinov2-base` após estabilização do pipeline.

### 2.3 Preset inicial de views/crops
- Views:
  - `2` globais
  - `6–8` locais
- Tamanhos:
  - global: `384` ou `512` (conforme memória)
  - local: `160–224`
- Escalas sugeridas:
  - `global_scale=(0.15, 1.0)`
  - `local_scale=(0.05, 0.15)` (ou até `0.2` na exploração inicial)

### 2.4 Augmentations seguras (fase 1)
Manter:
- random resized crop
- rotação pequena (ex.: ±5 graus)
- blur leve
- ruído gaussiano leve
- brilho/contraste em faixa estreita

Evitar (fase inicial):
- color jitter forte (especialmente hue/saturation)
- solarization
- flips livres (sem regra anatômica)
- deformações geométricas agressivas

## 3) Hipótese operacional para o projeto

Com ~70k periapicais filtradas:
- é plausível obter ganho real de representação de domínio,
- principalmente em variações de aquisição, artefatos e padrões anatômicos odontológicos.

Depois da fase periapical:
- fase 2b multimodal (pan, bw, oclusal, tele etc.) com mistura controlada, para robustez geral.

## 4) Cuidados críticos

### 4.1 Segurança e conformidade
- Dados clínicos reais: tratar como sensíveis em todo o pipeline.
- Evitar cópia desnecessária para ambientes fora da infraestrutura definida.
- Manter manifests de entrada/saída e rastreabilidade de filtragem.

### 4.2 Reprodutibilidade
- Salvar config completa por run (yaml/json).
- Versionar splits/listas e decisões de filtro.
- Persistir checkpoints teacher por intervalo fixo.

### 4.3 Risco de over-cleaning
- Não remover “periapicais difíceis” cedo demais.
- Limpar primeiro o claramente não-periapical; depois iterar.

## 5) Próximo passo prático no RMFM

Status atual:
- classificação em massa já concluída (`73,411` imagens);
- conjunto elegível preliminar definido: `Periapical + Interproximal = 65,487`.

Próximos passos imediatos (fase de pesquisa e planejamento):
1. consolidar decisão final de augmentações seguras para RX odontológico;
2. fechar preset de multi-crop (global/local) para v1;
3. definir plano de ablação mínimo (ex.: com/sem Interproximal, crop ranges);
4. fechar `ssl_periapical_v1_keep.txt`, `ssl_periapical_v1_drop.txt` e manifesto;
5. iniciar Continual SSL com monitoramento de estabilidade e qualidade de representação.

## 6) Fontes principais

- DINO (paper): https://arxiv.org/abs/2104.14294
- DINO código oficial (`main_dino.py`): https://raw.githubusercontent.com/facebookresearch/dino/main/main_dino.py
- DINOv2 (paper): https://arxiv.org/abs/2304.07193
- DINOv2 (repo): https://github.com/facebookresearch/dinov2
- RayDINO (CXR SSL): https://ar5iv.labs.arxiv.org/html/2405.01469
- Swin UNETR SSL 3D: https://ar5iv.labs.arxiv.org/html/2111.14791
- Revisão de augmentation médica: https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/
- Exemplo dental (flip seletivo por simetria): https://link.springer.com/article/10.1186/s12903-025-07138-0

## 7) Nota de licença/uso

- O ecossistema DINOv2 inclui materiais recentes para X-ray com licença específica de pesquisa.
- Revisar termos de licença antes de qualquer uso além de pesquisa e antes de integrar em fluxo clínico/produto.
