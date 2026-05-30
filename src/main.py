import multiprocessing as mp
import resource
from typing import final, override

from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.datamodule import InspladDataModule
from src.model import MultiHeadGoodBackbone


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
class MyCLI(LightningCLI):
    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("seed_everything", "data.kfold_seed")

        parser.add_optimizer_args(AdamW)
        parser.add_lr_scheduler_args(MyLR)
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        wandblogger = {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": {
                "project": "InsPLAD",
            },
        }
        parser.set_defaults(
            {
                "data.num_workers": mp.cpu_count(),
                # "optimizer.lr": 1e-6,
                "early_stopping.monitor": "val_loss",
                "early_stopping.mode": "min",
                "early_stopping.patience": 5,
                "trainer.logger": wandblogger,
            }
        )


def cli_main():
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    cli = MyCLI(
        MultiHeadGoodBackbone,
        InspladDataModule,
        trainer_defaults={"max_epochs": 100},
        seed_everything_default=67,
        # run=False,
    )


if __name__ == "__main__":
    cli_main()
