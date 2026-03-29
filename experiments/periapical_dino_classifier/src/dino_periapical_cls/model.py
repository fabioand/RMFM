from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class FrozenDinoClassifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        hf_token: str | None = None,
        local_files_only: bool = False,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_id,
            token=hf_token,
            local_files_only=local_files_only,
        )
        self.freeze_backbone = freeze_backbone

        hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def _extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            feats = outputs.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("Saída do backbone não contém pooler_output nem last_hidden_state.")
        return feats

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self._extract_features(pixel_values)
        else:
            feats = self._extract_features(pixel_values)
        logits = self.head(feats)
        return logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
