from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from . import cfg
from .dataset_a import DatasetA
from .dataset_b import DatasetB
from .dataset_c import DatasetC
from .dataset_d import DatasetD
from .base import collate_bb


@dataclass
class DataConfig:
    val_frac: float = 0.1
    batch_size: int = 4
    num_workers: int = 4

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataModule:
    def __init__(self, data_conf: DataConfig, num_channels=3, with_masks=True): 
        self.conf = data_conf
        self.with_masks = with_masks #load masks in ds that have no masks
        self.num_channels = num_channels
        self._init_datasets()

    def _init_datasets(self):
        ds_a_full = DatasetA(cfg.root_a_xml,cfg.root_a_img,cfg.label_map,cfg.max_size,self.num_channels, self.with_masks)
        ds_b_full = DatasetB(cfg.root_b,"train",cfg.label_map,cfg.max_size,num_channels=3) #b has train & test, so we split train into train and val, to use test set just for tests
        self.ds_c_test = DatasetC(cfg.root_c,"test",cfg.max_size, self.num_channels)
        n_val_a = max(1, int(len(ds_a_full) * self.conf.val_frac))
        n_val_b = max(1, int(len(ds_b_full) * self.conf.val_frac))
        g = torch.Generator().manual_seed(42)
        self.ds_a_train, self.ds_a_val = random_split(ds_a_full,[len(ds_a_full) - n_val_a, n_val_a],generator=g)
        self.ds_b_train, self.ds_b_val = random_split(ds_b_full,[len(ds_b_full) - n_val_b, n_val_b],generator=g)
        self.ds_b_test = DatasetB(cfg.root_b,"test",cfg.label_map,cfg.max_size,self.num_channels)
        self.ds_d_test =  DatasetD(cfg.root_d_xml,cfg.root_d_img,cfg.label_map,cfg.max_size,self.num_channels) #ds D is all test for now. 
        

    def make_loaders_b(self) -> Tuple[DataLoader, DataLoader]: #B train (with full masks)
        train_loader = DataLoader(self.ds_b_train,batch_size=self.conf.batch_size,shuffle=True,num_workers=self.conf.num_workers,collate_fn=collate_bb)
        val_loader = DataLoader(self.ds_b_val,batch_size=self.conf.batch_size,shuffle=False,num_workers=self.conf.num_workers,collate_fn=collate_bb,)
        return train_loader, val_loader

    def make_loaders_ab(self) -> Tuple[DataLoader, DataLoader]: #A train + B train / A val + Bv val
        train_ds = ConcatDataset([self.ds_a_train, self.ds_b_train])
        train_loader = DataLoader(train_ds,batch_size=self.conf.batch_size,shuffle=True,num_workers=self.conf.num_workers,collate_fn=collate_bb,)
        val_ds = ConcatDataset([self.ds_a_val, self.ds_b_val])
        val_loader = DataLoader(val_ds,batch_size=self.conf.batch_size,shuffle=False,num_workers=self.conf.num_workers,collate_fn=collate_bb)
        return train_loader, val_loader

    def make_loader_a_val(self) -> DataLoader:
        return DataLoader(self.ds_a_val,batch_size=self.conf.batch_size,shuffle=False,num_workers=self.conf.num_workers,collate_fn=collate_bb)

    def make_loader_b_test(self) -> DataLoader:
        return DataLoader(self.ds_b_test,batch_size=1,shuffle=False,num_workers=self.conf.num_workers,collate_fn=collate_bb,)

    def make_loader_c_test(self) -> DataLoader:
        return DataLoader(self.ds_c_test,batch_size=1,shuffle=False,num_workers=self.conf.num_workers,collate_fn=collate_bb,)

    def make_loader_d_test(self) -> DataLoader:
        return DataLoader(self.ds_d_test,batch_size=1,shuffle=False,num_workers=self.conf.num_workers,collate_fn=collate_bb,)
