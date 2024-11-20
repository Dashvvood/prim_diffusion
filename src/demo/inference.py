import os
import sys

import motti

motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

import torch
from model.pipeline_ddpm import DDPMPipelineV2
from diffusers import (
    UNet2DModel, 
    DDPMScheduler, 
    DDIMScheduler,
    DDPMPipeline,
    DDPMPipeline
)
# from args import opts
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
    

unet_config = UNet2DModel.load_config(opts.unet_config)
unet = UNet2DModel.from_config(unet_config)

scheduler_config = DDPMScheduler.load_config(opts.scheduler_config)
scheduler = DDIMScheduler.from_config(scheduler_config)

