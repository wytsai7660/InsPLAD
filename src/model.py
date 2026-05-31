from typing import final, override

import lightning as L
import timm
import timm.data
import torch
import torch.nn as nn
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
class MultiHeadGoodBackbone(L.LightningModule):
    def __init__(
        self,
        dino: str = "vit_large_patch16_dinov3_qkvb.lvd1689m",
        img_size: int = 512,
        optimizer: OptimizerCallable = AdamW,
        scheduler: LRSchedulerCallable = MyLR,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        self.optimizer = optimizer
        self.scheduler = scheduler

        example_batch_size = 67
        self.example_input_array = (
            torch.zeros(example_batch_size, 3, img_size, img_size),
            torch.zeros(example_batch_size, dtype=torch.int64),
        )
        # 建議使用 pos_weight，你可以先設個通用值，或針對 5 個器材各給一個權重
        self.criterion = nn.BCEWithLogitsLoss()
        self.backbone = timm.create_model(
            dino, pretrained=True, num_classes=0, dynamic_img_size=True
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = self.backbone.num_features
        self.head = nn.Linear(embed_dim, len(Comp.name))

        data_config = timm.data.resolve_model_data_config(self.backbone)
        data_config["input_size"] = (
            3,
            self.hparams["img_size"],
            self.hparams["img_size"],
        )
        self.train_transforms = timm.data.create_transform(
            **data_config, is_training=True
        )
        self.test_transforms = timm.data.create_transform(
            **data_config, is_training=False
        )

        self.val_metrics = {
            c: torchmetrics.MetricCollection(
                {
                    # "recall@p92": classification.BinaryRecallAtFixedPrecision(0.92),
                    "ap": classification.BinaryAveragePrecision(),
                },
                prefix=f"{Comp.to_name(c)}_",
            )
            for c in Comp.id.values()
        }

    @override
    def forward(self, img: torch.Tensor, comp: torch.Tensor):
        feature = self.backbone(img)
        all_logits = self.head(feature)

        batch_size = img.size(0)
        batch_indices = torch.arange(batch_size)

        selected_logits = all_logits[batch_indices, comp]
        return selected_logits

    @override
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        img, comp, stat = batch
        pred = self(img, comp)

        loss = self.criterion(pred, stat.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        img, comp, stat = batch
        pred = self(img, comp)

        loss = self.criterion(pred, stat.float())
        self.log("val_loss", loss, batch_size=stat.shape[0], sync_dist=True)
        for c, m in self.val_metrics.items():
            mask = comp == c
            m.update(pred[mask], stat[mask])

    @override
    def on_validation_epoch_end(self):
        for m in self.val_metrics.values():
            self.log_dict(m.compute())
            m.reset()

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self.optimizer(self.head.parameters())  # Train only the head
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
