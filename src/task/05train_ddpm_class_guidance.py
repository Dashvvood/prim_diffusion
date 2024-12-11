"""
python 05train_ddpm_class_guidance.py --unet_config \
../../config/ddpm_small/unet_class/ --scheduler_config ../../config/ddpm_sma \
ll/scheduler/ --batch_size 2 --warmup_epochs 10 --max_epochs 100 --num_worke \
rs 8 --device_num 1 --data_root ../../data/ACDC/quadra/ --ckpt_dir ../../ckp \
t/prim/ --log_dir ../../logs/ --lr 1e-4 --img_size 64 --project prim --log_step 1 --ps debug
"""

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
from model.DDPM import TrainableDDPMbyClass


from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class DiffusionData(L.LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        h5data="acdc_quadra.h5",
        train_metadata="quadra_per_slice_train.csv", 
        val_metadata="quadra_per_slice_val.csv", 
        test_metadata="quadra_per_slice_test.csv",
        opts=None
    ):
        super().__init__()
        self.data_dir = data_dir
        # self.dataset = QuadraACDCDataset(
        #     root_dir=data_dir,
        #     h5data="acdc_quadra.h5",
        #     metadata="quadra_per_slice_train.csv",
        #     transform= transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Resize((opts.img_size, opts.img_size), 
        #             interpolation=transforms.InterpolationMode.NEAREST
        #         ),
        #         transforms.Normalize((0.5,), (0.5,))
        #     ])
        # )
        
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((opts.img_size, opts.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.h5data = h5data
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        self.val_metadata = val_metadata
        
        
    def setup(self, stage=None):
        if stage == "fit":
            self.trainset = QuadraACDCDataset(
                root_dir=self.data_dir,
                h5data=self.h5data,
                metadata=self.train_metadata,
                transform=self.transform
            )
            self.valset = QuadraACDCDataset(
                root_dir=self.data_dir,
                h5data=self.h5data,
                metadata=self.val_metadata,
                transform=self.transform
            )

        elif stage == "test":
            self.testset = QuadraACDCDataset(
                root_dir=self.data_dir,
                h5data=self.h5data,
                metadata=self.test_metadata,
                transform=self.transform
            )
        
        elif stage == "predict":
            raise NotImplementedError("Predict not implemented")
        
        
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            collate_fn=QuadraACDCDataset.collate_fn,
        )
        
        
    def val_dataloader(self):
        return DataLoader(
            self.valset,
            shuffle=True,
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            collate_fn=QuadraACDCDataset.collate_fn,
        )


data = DiffusionData(
    data_dir=opts.data_root, 
    h5data="acdc_quadra.h5",
    train_metadata="quadra_per_slice_train_train.csv",
    val_metadata="quadra_per_slice_train_val.csv",
    test_metadata="quadra_per_slice_test.csv",
    opts=opts
)

model = TrainableDDPMbyClass.from_config(opts.unet_config, opts.scheduler_config, opts)

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
# TODO: val_loss

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