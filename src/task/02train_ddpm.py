import os
import motti
motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

from args import opts

import torch
import diffusers
import numpy as np
import pandas as pd
import lightning as L
from torchvision import transforms

from dataset.ACDC import QuadraACDCDataset
from model.DDPM import TrainableDDPM

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class DiffusionData(L.LightningDataModule):
    def __init__(self, data_dir: str, opts=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = QuadraACDCDataset(
            root_dir=data_dir,
            h5data="acdc_quadra.h5",
            metadata="quadra_per_slice_train.csv",
            transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((opts.img_size, opts.img_size), 
                    interpolation=transforms.InterpolationMode.NEAREST
                ),
            ])
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            collate_fn=QuadraACDCDataset.collate_fn,
        )

data = DiffusionData(opts.data_root, opts)
model = TrainableDDPM.from_config(opts.unet_config, opts.scheduler_config, opts)

wandblogger = WandbLogger(
    name=f"{o_d}_{thisfile}_{opts.ps}", 
    save_dir=opts.log_dir,
    project=opts.project,
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1, save_last=True,
    dirpath=os.path.join(opts.ckpt_dir, o_d),
    monitor="train_loss", mode="min"
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