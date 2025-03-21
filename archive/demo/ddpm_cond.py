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

import numpy as np
import pandas as pd
from args import opts
from model.DDPM import DiffusionModel
from dataset import QuadraACDCDataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint



class DiffusionModel(L.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = diffusers.models.UNet2DModel(
            sample_size=opts.img_size,
            in_channels=opts.in_channels,
            out_channels=opts.out_channels,
            block_out_channels=[64, 128, 256, 512],
        )
        self.scheduler = diffusers.schedulers.DDPMScheduler()
        self.opts = opts
        
    def training_step(self, batch, batch_idx):
        images = batch[0]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.opts.lr,
            betas=[0.9, 0.999],
            weight_decay=1e-2,
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.opts.warmup_epochs,
            num_training_steps=self.opts.max_epochs,
        )
        return [optimizer], [scheduler]
    


class DiffusionData(L.LightningDataModule):
    def __init__(self, data_dir: str, opts=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = QuadraACDCDataset(
            data_dir,
            transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((opts.img_size, opts.img_size)),
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
model = DiffusionModel(opts)


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

