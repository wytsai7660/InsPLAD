import multiprocessing as mp
import resource
from typing import final, override

from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from src.datamodule import InspladDataModule
from src.model import MultiHeadGoodBackbone


@final
class MyCLI(LightningCLI):
    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("seed_everything", "data.kfold_seed")
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
                "trainer.num_sanity_val_steps": 0,
            }
        )


def cli_main():
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    cli = MyCLI(
        MultiHeadGoodBackbone,
        InspladDataModule,
        trainer_defaults={"max_epochs": 100},
        auto_configure_optimizers=False,
        seed_everything_default=67,
        save_config_callback=None,
        # run=False,
    )


if __name__ == "__main__":
    cli_main()
