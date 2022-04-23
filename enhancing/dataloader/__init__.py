# ------------------------------------------------------------------------------------
# Modified from VQGAN (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import Optional
from omegaconf import OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..utils.general import initialize_from_config

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size: int, train: Optional[OmegaConf] = None,
                 validation: Optional[OmegaConf] = None,
                 test: Optional[OmegaConf] = None,
                 num_workers: Optional[int] = None):
        super().__init__()
        self.dataset_configs = dict()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            initialize_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, initialize_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)
