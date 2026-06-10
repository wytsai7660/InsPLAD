import multiprocessing as mp
import os
import resource
from collections.abc import Sequence
from typing import Any, final, override

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    BasePredictionWriter,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from src.datamodule import InspladDataModule
from src.model import MultiHeadGoodBackbone


@final
class CSVWriter(BasePredictionWriter):
    def __init__(self, write_interval: str = "epoch"):
        super().__init__(write_interval)

    @override
    def write_on_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: Sequence[tuple[Sequence[str], torch.Tensor]],
        batch_indices: Sequence[Any],
    ):
        all_paths = []
        all_is_bad = []

        for batch in predictions:
            path, is_bad = batch
            all_paths.extend(path)
            all_is_bad.append(is_bad)

        all_is_bad_cat = torch.cat(all_is_bad, dim=0).int()

        df = pd.DataFrame(
            {
                "id": [int(os.path.splitext(p)[0]) for p in all_paths],
                "filename": all_paths,
                "pred_label": all_is_bad_cat,
            }
        )
        df = df.sort_values(by="id")
        df = df.drop(columns=["id"])

        out_file = "submission.csv"
        df.to_csv(out_file, index=False)


@final
class MyCLI(LightningCLI):
    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("seed_everything", "data.kfold_seed")
        parser.link_arguments("data.img_size", "model.img_size")
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(CSVWriter, "csv_writer")
        wandblogger = {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": {
                "project": "InsPLAD",
            },
        }
        parser.set_defaults(
            {
                "data.num_workers": min(mp.cpu_count(), 8),
                # "optimizer.lr": 1e-6,
                "early_stopping.monitor": "val_loss",
                "early_stopping.mode": "min",
                "early_stopping.patience": 5,
                "model_checkpoint.monitor": "val_loss",
                "model_checkpoint.mode": "min",
                "trainer.precision": "bf16-mixed",
                "trainer.max_epochs": 100,
                "trainer.logger": wandblogger,
                "trainer.num_sanity_val_steps": 0,
            }
        )


def cli_main():
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    cli = MyCLI(
        MultiHeadGoodBackbone,
        InspladDataModule,
        auto_configure_optimizers=False,
        seed_everything_default=67,
        save_config_callback=None,
        # run=False,
    )


if __name__ == "__main__":
    cli_main()
