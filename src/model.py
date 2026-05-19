from typing import final, override

import lightning as L
import timm
import timm.data
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import classification

from src.helper import Comp


@final
class MultiHeadGoodBackbone(L.LightningModule):
    def __init__(
        self,
        dino: str = "vit_large_patch16_dinov3_qkvb.lvd1689m",
        img_size: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()
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
