# Experimento: DINO Congelado + Cabeça Classificadora (Periapicais)

## Visão
Construir uma linha de base robusta para classificação de periapicais usando um encoder visual fundacional (DINOv2) congelado, medindo o quanto a representação pré-treinada já separa classes odontológicas sem ajuste do backbone.

## Objetivos
- Validar se `DINOv2-small` congelado + head linear/MLP atinge desempenho competitivo no dataset atual.
- Estabelecer baseline reproduzível para comparar com:
  - CNNs atuais (ex.: ResNet/U-Net encoder)
  - versões com fine-tuning parcial/total do DINO
  - versões com aumento de dados/pseudo-labels
- Medir gargalos de treinamento e inferência para orientar próximos passos.

## Abordagem
1. Dataset supervisionado por JSON de classificação periapical (`labels`).
2. Interseção imagem+json para garantir consistência de amostras.
3. Split estratificado `train/val/test` por classe.
4. Encoder DINO congelado (`facebook/dinov2-small`).
5. Treino de cabeça classificadora com cross-entropy.
6. Seleção de melhor checkpoint por `val_macro_f1`.
7. Avaliação final em teste com:
   - accuracy
   - macro F1
   - relatório por classe
   - matriz de confusão

## Expectativas
- O baseline deve aprender sinal útil mesmo sem fine-tuning do encoder.
- Classes com menor suporte devem ter maior variância/erro.
- Em geral, espera-se:
  - convergência rápida (head pequena)
  - custo baixo de treino
  - bom ponto de partida para decidir se vale fine-tuning do backbone

## Critérios de sucesso
- Pipeline reprodutível ponta a ponta.
- Métricas por classe exportadas para análise de erros.
- Baseline forte o suficiente para justificar etapa seguinte:
  1. fine-tuning parcial do DINO, ou
  2. expansão de dados/pseudo-labels antes do fine-tuning.

## Estrutura implementada
- `experiments/periapical_dino_classifier/scripts/train_frozen_head.py`
- `experiments/periapical_dino_classifier/src/dino_periapical_cls/data.py`
- `experiments/periapical_dino_classifier/src/dino_periapical_cls/model.py`
- `experiments/periapical_dino_classifier/src/dino_periapical_cls/train.py`
- `experiments/periapical_dino_classifier/README.md`

## Próximos passos sugeridos
1. Rodar baseline congelado com resolução 256 e 512.
2. Comparar F1 macro e por classe.
3. Se houver teto de desempenho, liberar fine-tuning parcial (últimos blocos) mantendo custo controlado.
