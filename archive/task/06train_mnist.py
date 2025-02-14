"""
python 06train_mnist.py \
--unet_config ../../config/ddpm_mnist/unet/ \
--scheduler_config ../../config/ddpm_mnist/scheduler/ \
--batch_size 5 --warmup_epochs 20 --max_epoch 200 \
--num_workers 8 --device_num 1 --data_root ../../data/ \
--ckpt_dir ../../ckpt/prim/ --log_dir ../../logs/ \
--lr 1e-5 --img_size 32 --project prim --log_step 5 --ps mnist
"""
import os
import motti
from pathlib import Path
motti.append_parent_dir(__file__)
thisfile = Path(__file__).stem
o_d = motti.o_d()

from args import opts
import lightning as L
from torchvision import transforms

from torch.utils.data import random_split, DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from torchvision.datasets import MNIST
import torch

from model.DDPM import TrainableDDPM

class DiffusionData(L.LightningDataModule):
    def __init__(self, data_dir, opts=None):
        super().__init__()
        self.data_dir = data_dir
        self.opts = opts
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((opts.img_size, opts.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform, download=False)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform, download=False)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform, download=False)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=opts.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=opts.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=opts.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=opts.batch_size)
        
        
data = DiffusionData(data_dir=opts.data_root, opts=opts)

model = TrainableDDPM.from_config(opts.unet_config, opts.scheduler_config, opts)

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
    callbacks=[checkpoint_callback,],
)

trainer.fit(model, data)