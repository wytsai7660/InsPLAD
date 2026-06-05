from itertools import chain
from typing import final, override

import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import classification

from src.helper import Comp


class MyLR(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        annealing_epochs: int = 95,
        start_factor: float = 0.1,
    ):
        super().__init__(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    total_iters=warmup_epochs,
                ),
                CosineAnnealingLR(optimizer, T_max=annealing_epochs),
            ],
            milestones=[warmup_epochs],
        )


@final
class MLP(L.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, mlp_ratio: float = 0.5):
        super().__init__()
        hidden_dim = int(input_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.skip = nn.Linear(input_dim, hidden_dim, bias=False)
        self.head = nn.Linear(hidden_dim, output_dim)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.skip(x) + self.alpha * self.net(x))


@final
class MultiHeadGoodBackbone(L.LightningModule):
    def __init__(
        self,
        name: str = "vit_large_patch16_dinov3.lvd1689m",
        img_size: int = "data.img_size",  # NOTE: actual default: `cli.config["data.img_size"]` (set in `datamodule.py`)
        comp_weight: float = 0.3,
        optimizer: OptimizerCallable = AdamW,
        scheduler: LRSchedulerCallable = MyLR,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        self.optimizer = optimizer
        self.scheduler = scheduler

        example_batch_size = 67
        self.example_input_array = torch.zeros(
            example_batch_size, 3, img_size, img_size
        )

        self.backbone = timm.create_model(
            name, pretrained=True, num_classes=0, dynamic_img_size=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = int(self.backbone.num_features)
        self.comp_head = MLP(embed_dim, len(Comp.name), mlp_ratio=0.5)
        fused_dim = embed_dim + len(Comp.name)
        self.stat_head = MLP(fused_dim, 1, mlp_ratio=0.5)

        self.val_metrics = {
            c: torchmetrics.MetricCollection(
                {
                    # "recall@p92": classification.BinaryRecallAtFixedPrecision(0.92),
                    "AP": classification.BinaryAveragePrecision(),
                },
                postfix=f" ({Comp.to_name(c)})",
            )
            for c in Comp.id.values()
        }

    @override
    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.backbone(img)
        comp_logit = self.comp_head(feature)
        comp_prob = torch.softmax(comp_logit, dim=-1)
        fused_feature = torch.cat([feature, comp_prob], dim=-1)
        stat_logit = self.stat_head(fused_feature).squeeze(-1)
        return comp_logit, stat_logit

    @override
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        img, comp, stat = batch
        comp_pred, stat_pred = self(img)

        # TODO: imbalance-aware loss
        comp_loss = F.cross_entropy(comp_pred, comp)
        stat_loss = F.binary_cross_entropy_with_logits(stat_pred, stat.float())
        loss = stat_loss + self.hparams["comp_weight"] * comp_loss

        self.log("train_loss", loss, prog_bar=True, logger=False)
        return loss

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        img, comp, stat = batch
        comp_pred, stat_pred = self(img)

        # TODO: imbalance-aware loss
        comp_loss = F.cross_entropy(comp_pred, comp)
        stat_loss = F.binary_cross_entropy_with_logits(stat_pred, stat.float())
        loss = stat_loss + self.hparams["comp_weight"] * comp_loss

        batch_size = stat.shape[0]
        self.log("val_loss_comp", comp_loss, batch_size=batch_size, sync_dist=True)
        self.log("val_loss_stat", stat_loss, batch_size=batch_size, sync_dist=True)
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)

        for c, m in self.val_metrics.items():
            mask = comp == c
            m.update(stat_pred[mask], stat[mask])

    @override
    def on_validation_epoch_end(self):
        for m in self.val_metrics.values():
            self.log_dict(m.compute())
            m.reset()

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        # Train only the head
        optimizer = self.optimizer(
            chain(self.comp_head.parameters(), self.stat_head.parameters())
        )
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
