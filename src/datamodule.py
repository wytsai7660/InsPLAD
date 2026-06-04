from pathlib import Path
from typing import Callable, final, override

import gdown
import lightning as L
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from src.helper import Comp, Stat


@final
class TrainingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable):
        super().__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, bool]:
        row = self.df.iloc[index]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        return img, row["comp"], row["stat"]


@final
class InspladDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_folds: int = 5,
        fold_idx: int = 0,
        kfold_seed: int = "seed_everything",  # NOTE: actual default: `cli.config["seed_everything"]` (set in `MyCLI`)
        num_workers: int = "cpu_count",  # NOTE: actual default: `mp.cpu_count()` (set in `MyCLI`)
        batch_size: int = 32,
        test_batch_size: int = 128,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        data_dir: str = "data",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_dir"])
        self.train_dir = Path(data_dir) / "train_dataset"
        self.test_dir = Path(data_dir) / "test_dataset"

        self.train_ds: TrainingDataset
        self.val_ds: TrainingDataset

    @override
    def prepare_data(self):
        train_file_id = r"14B3Jsj4DzoCrMC4YXlXEp0ej_reMgWkq"
        train_checksum = r"md5:449e4617aefa0d9c9d059e21c38b32f5"
        test_file_id = r"1RhPBNwWxRYK0M8UvQcBBnVSpPzsEF3xn"
        test_checksum = r"md5:5edc01fa26e9563449aa7e7885242e71"

        gdown.cached_download(
            id=train_file_id,
            path=f"{self.train_dir}.zip",
            hash=train_checksum,
        )
        gdown.cached_download(
            id=test_file_id,
            path=f"{self.test_dir}.zip",
            hash=test_checksum,
        )
        gdown.extractall(f"{self.train_dir}.zip")
        gdown.extractall(f"{self.test_dir}.zip")

    @override
    def setup(self, stage: str):
        train_transforms = self.trainer.lightning_module.train_transforms  # pyright: ignore[reportOptionalMemberAccess]
        test_transforms = self.trainer.lightning_module.test_transforms  # pyright: ignore[reportOptionalMemberAccess]
        if stage == "fit":
            # Unlovely implementation, but I believe this is the only way to do cross validations with DataModule
            full_df = pd.DataFrame(
                [
                    (p, Comp.to_id(p.parents[1].stem), Stat.to_id(p.parent.stem))
                    for p in sorted(self.train_dir.rglob("*.jpg"))
                ],
                columns=["path", "comp", "stat"],
            )

            skf = StratifiedKFold(
                n_splits=self.hparams["n_folds"],
                shuffle=True,
                random_state=self.hparams["kfold_seed"],
            )
            stratify = full_df["comp"] * 2 + full_df["stat"]
            train_idx, val_idx = list(skf.split(full_df, stratify))[
                self.hparams["fold_idx"]
            ]

            train_df = full_df.iloc[train_idx]
            val_df = full_df.iloc[val_idx]
            self.train_ds = TrainingDataset(train_df, train_transforms)
            self.val_ds = TrainingDataset(val_df, test_transforms)

    @override
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            persistent_workers=self.hparams["persistent_workers"],
            prefetch_factor=self.hparams["prefetch_factor"],
        )

    @override
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams["test_batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            persistent_workers=self.hparams["persistent_workers"],
            prefetch_factor=self.hparams["prefetch_factor"],
        )
