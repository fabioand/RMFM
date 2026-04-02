from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


def extract_feature(outputs) -> torch.Tensor:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0]
    raise RuntimeError("Saida do backbone sem pooler_output nem last_hidden_state")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, bottleneck_dim: int = 256, out_dim: int = 65536) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last.weight_g.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        teacher_temp_warmup: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_epochs: int = 30,
        total_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = self._build_teacher_temp_schedule(
            teacher_temp_warmup, teacher_temp, warmup_epochs, total_epochs
        )

    @staticmethod
    def _build_teacher_temp_schedule(
        warmup: float, target: float, warmup_epochs: int, total_epochs: int
    ) -> list[float]:
        warmup_epochs = max(1, int(warmup_epochs))
        total_epochs = max(warmup_epochs, int(total_epochs))
        warm = [warmup + (target - warmup) * (i / max(1, warmup_epochs - 1)) for i in range(warmup_epochs)]
        if total_epochs == warmup_epochs:
            return warm
        return warm + [target] * (total_epochs - warmup_epochs)

    def forward(self, student_out: list[torch.Tensor], teacher_out: list[torch.Tensor], epoch: int) -> torch.Tensor:
        eidx = min(max(0, int(epoch) - 1), len(self.teacher_temp_schedule) - 1)
        t_temp = self.teacher_temp_schedule[eidx]

        student_log = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out]
        teacher_prob = [F.softmax((t - self.center) / t_temp, dim=-1).detach() for t in teacher_out]

        n_terms = 0
        total = 0.0
        for iq, q in enumerate(teacher_prob):
            for iv, s in enumerate(student_log):
                if iv == iq:
                    continue
                total = total + torch.sum(-q * s, dim=-1).mean()
                n_terms += 1
        if n_terms == 0:
            raise RuntimeError("DINOLoss sem pares validos")
        loss = total / n_terms
        self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out: list[torch.Tensor]) -> None:
        batch_center = torch.cat(teacher_out, dim=0).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)


@dataclass
class SSLBundle:
    student_backbone: nn.Module
    teacher_backbone: nn.Module
    student_head: ProjectionHead
    teacher_head: ProjectionHead
    dino_loss: DINOLoss


def build_ssl_bundle(
    model_id: str,
    out_dim: int,
    device: str,
    local_files_only: bool = False,
    hf_token: str | None = None,
    head_hidden_dim: int = 2048,
    head_bottleneck_dim: int = 256,
    student_temp: float = 0.1,
    center_momentum: float = 0.9,
    teacher_temp_warmup: float = 0.04,
    teacher_temp: float = 0.07,
    teacher_temp_warmup_epochs: int = 30,
    total_epochs: int = 100,
) -> SSLBundle:
    student_backbone = AutoModel.from_pretrained(model_id, token=hf_token, local_files_only=local_files_only)
    teacher_backbone = AutoModel.from_pretrained(model_id, token=hf_token, local_files_only=local_files_only)
    teacher_backbone.load_state_dict(student_backbone.state_dict(), strict=True)

    hidden_size = int(getattr(student_backbone.config, "hidden_size"))
    student_head = ProjectionHead(
        in_dim=hidden_size,
        hidden_dim=int(head_hidden_dim),
        bottleneck_dim=int(head_bottleneck_dim),
        out_dim=int(out_dim),
    )
    teacher_head = ProjectionHead(
        in_dim=hidden_size,
        hidden_dim=int(head_hidden_dim),
        bottleneck_dim=int(head_bottleneck_dim),
        out_dim=int(out_dim),
    )
    teacher_head.load_state_dict(student_head.state_dict(), strict=True)

    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    student_backbone.to(device)
    teacher_backbone.to(device)
    student_head.to(device)
    teacher_head.to(device)

    teacher_backbone.eval()
    teacher_head.eval()

    dino_loss = DINOLoss(
        out_dim=int(out_dim),
        student_temp=float(student_temp),
        center_momentum=float(center_momentum),
        teacher_temp_warmup=float(teacher_temp_warmup),
        teacher_temp=float(teacher_temp),
        warmup_epochs=int(teacher_temp_warmup_epochs),
        total_epochs=int(total_epochs),
    ).to(device)

    return SSLBundle(
        student_backbone=student_backbone,
        teacher_backbone=teacher_backbone,
        student_head=student_head,
        teacher_head=teacher_head,
        dino_loss=dino_loss,
    )


@torch.no_grad()
def update_teacher_ema(
    student_backbone: nn.Module,
    teacher_backbone: nn.Module,
    student_head: nn.Module,
    teacher_head: nn.Module,
    momentum: float,
) -> None:
    for ps, pt in zip(student_backbone.parameters(), teacher_backbone.parameters()):
        pt.data.mul_(momentum).add_((1.0 - momentum) * ps.data)
    for ps, pt in zip(student_head.parameters(), teacher_head.parameters()):
        pt.data.mul_(momentum).add_((1.0 - momentum) * ps.data)


def cosine_momentum(step: int, total_steps: int, base_m: float = 0.996, final_m: float = 1.0) -> float:
    if total_steps <= 1:
        return final_m
    ratio = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return final_m - (final_m - base_m) * (0.5 * (1.0 + math.cos(math.pi * ratio)))


def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int = 0) -> float:
    if step < warmup_steps and warmup_steps > 0:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps + 1:
        return min_lr
    t = (step - warmup_steps) / max(1, (total_steps - warmup_steps - 1))
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

