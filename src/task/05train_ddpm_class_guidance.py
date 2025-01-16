"""
CUDA_VISIBLE_DEVICES=2  python 05train_ddpm_class_guidance.py --unet_config ../../config/ddpm_medium/unet_class/ --scheduler_config ../../config/ddpm_medium/scheduler/ --batch_size 8 --warmup_epochs 20 --max_epochs 200 --num_workers 8 --device_num 1 --data_root ../../data/ACDC/quadra/ --ckpt_dir ../../ckpt/prim/ --log_dir ../../logs/ --lr 1e-4 --img_size 128 --project prim --log_step 5 --ps debug --p_uncond 0.5"""

import os
import motti
from pathlib import Path
motti.append_parent_dir(__file__)
thisfile = Path(__file__).stem
o_d = motti.o_d()

from args import opts

import lightning as L
from model.DDPM import TrainableDDPMbyClass
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from dataset.ACDC import DiffusionData

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