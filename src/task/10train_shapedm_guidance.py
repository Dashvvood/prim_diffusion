import os
import motti
from pathlib import Path
motti.append_parent_dir(__file__)
thisfile = Path(__file__).stem
o_d = motti.o_d()

from args import opts
import lightning as L
from model import TrainableShapeDM
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from dataset.ACDC import DiffusionData
from utils import load_config


data = DiffusionData(**opts.data, opts=opts)
model = TrainableShapeDM.from_config(config_dir=opts.model, opts=opts)

if opts.resume:
    model.load_state_from_ckpt(opts.ckpt_path)

wandblogger = WandbLogger(
    name=f"{o_d}_{thisfile}_{opts.ps}", 
    save_dir=opts.log_dir,
    project=opts.project,
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1, save_last=True,
    dirpath=os.path.join(opts.ckpt_dir, o_d),
    monitor="val_loss", mode="min",
    save_weights_only=True,
)

trainer = L.Trainer(
    max_epochs=opts.max_epochs,
    accelerator="gpu",
    strategy=opts.strategy,
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    logger=wandblogger,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=opts.log_step,
    callbacks=[checkpoint_callback,],
)

trainer.fit(model, data)
