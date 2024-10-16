import os
import sys

import motti

motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

import torch
import diffusers
from torchvision import transforms
import lightning as L
import os
from torch.utils.data import Dataset, DataLoader
import imageio.v2 as imageio
import numpy as np
from constant import (
    ED_MEAN,
    ED_STD
)


from args import opts
from model.DDPM import DiffusionModel

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint



class TrainingSet(Dataset):
    def __init__(self, data_dir: str, opts=None):
        self.data_dir = data_dir
        
        self.files = []
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
        
        for f in os.listdir(data_dir):
            fpath = os.path.join(data_dir, f)
            if os.path.isfile(fpath) and any(f.lower().endswith(ext) for ext in self.supported_formats):
                img = imageio.imread(fpath)
                self.files.append(img)
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=ED_MEAN, std=ED_STD),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f = self.files[idx]
        return self.transform(f)


class DiffusionData(L.LightningDataModule):
    def __init__(self, data_dir: str, opts=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = TrainingSet(data_dir)
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
        )


data = DiffusionData(opts.img_root, opts)
model = DiffusionModel(opts)

wandblogger = WandbLogger(
    name=f"{o_d}_{thisfile}_{opts.ps}", 
    save_dir=opts.log_dir,
    project=opts.project,
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1, save_last=True,
    dirpath=os.path.join(opts.ckpt_dir, o_d),
    monitor="val_loss", mode="min"
)


trainer = L.Trainer(
    max_epochs=opts.max_epochs,
    accelerator="gpu",
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    logger=wandblogger,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=opts.log_step,
    # callbacks=[checkpoint_callback, LogConfusionMatrix()],
    callbacks=[checkpoint_callback,],
)

trainer.fit(model, data)

