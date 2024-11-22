import os
import motti
motti.append_current_dir(os.path.abspath(''))

import numpy as np
import matplotlib.pyplot as plt
import torch
from model.pipeline_ddpm import DDPMPipelineV2
from model.pipeline_ddim import DDIMPipelineV2
from model.DDPM import DiffusionModel
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMPipeline, DDPMPipeline
from PIL import Image

import argparse

def local_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=".")
    parser.add_argument("--unet_config", type=str, default=".")
    parser.add_argument("--scheduler_config", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--output_type", type=str, default="pil")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--ddim", action='store_true', default=False)
    args, missing = parser.parse_known_args()
    print(f"Missing args: {missing}")
    return args

opts = local_args()

ckpt = torch.load(opts.ckpt_path, map_location="cpu")
d = {}
for k, v in ckpt["state_dict"].items():
    new_k = k.split('.', 1)[1]
    d[new_k] = v

unet_config = UNet2DModel.from_config(opts.unet_config)
unet = UNet2DModel.from_config(unet_config)
scheduler_config = DDPMScheduler.load_config(opts.scheduler_config)
ddpm_scheduler = DDPMScheduler.from_config(scheduler_config)


unet.load_state_dict(d)

if opts.ddim:
    ddim_scheduler = DDIMScheduler.from_config(scheduler_config)
    pipe = DDIMPipelineV2(unet, ddim_scheduler, opts)
else:
    pipe = DDPMPipelineV2(unet, ddpm_scheduler, opts)

