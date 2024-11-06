import os
import sys

import motti

motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

import torch
from model.pipeline_ddpm import DDPMPipelineV2
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from args import opts


unet_config = UNet2DModel.load_config(opts.unet_config)
unet = UNet2DModel.from_config(unet_config)

scheduler_config = DDPMScheduler.load_config(opts.scheduler_config)
scheduler = DDIMScheduler.from_config(scheduler_config)