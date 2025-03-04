"""
Usage:
CUDA_VISIBLE_DEVICES=0 python 07inference_guide.py \
    --ckpt_path ../../ckpt/prim/20250116-141853/epoch=283-step=8236.ckpt \
    --unet_config ../../config/ddpm_medium/unet_class \
    --scheduler_config ../../config/ddpm_medium/scheduler \
    --batch_size 128 \
    --total_num 1024 \
    --num_inference_steps 1000 \
    --output_type numpy \
    --output_dir ../../output/ \
    --guidance_scale 1
"""
import os
import motti
motti.append_current_dir(os.path.abspath(''))
o_d = motti.o_d()

import numpy as np
import matplotlib.pyplot as plt
import torch

from model.ShapeDM import ShapeDM
from model.ShapeLDM import ShapeLDM
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMPipeline, DDPMPipeline
from PIL import Image
from tqdm import tqdm
import argparse
from utils.nms import NMS
from utils import binarize, closing
import json

def local_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=".")
    parser.add_argument("--config_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_num", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    parser.add_argument("--output_type", type=str, default="pil")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--ddim", action='store_true', default=False)
    parser.add_argument("--latent", action='store_true', default=False)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args, missing = parser.parse_known_args()
    print(f"Missing args: {missing}")
    return args

opts = local_args()
output_dir = os.path.join(opts.output_dir, o_d)
os.makedirs(output_dir, exist_ok=True)

raw_output_dir = os.path.join(output_dir, "raw")
binary_output_dir = os.path.join(output_dir, "binary")
os.makedirs(raw_output_dir, exist_ok=True)
os.makedirs(binary_output_dir, exist_ok=True)

os.makedirs(os.path.join(raw_output_dir, "123"), exist_ok=True)
os.makedirs(os.path.join(binary_output_dir, "123"), exist_ok=True)


for i in range(4):
    os.makedirs(os.path.join(raw_output_dir, str(i)), exist_ok=True)
    os.makedirs(os.path.join(binary_output_dir, str(i)), exist_ok=True)

with open(os.path.join(output_dir, "args.json"), "w") as f:
    json.dump(vars(opts), f, indent=4)

if opts.latent:
    pipe = ShapeLDM.from_config(config_dir=opts.config_dir)
else:
    pipe = ShapeDM.from_config(config_dir=opts.config_dir)
    
pipe.load_state_from_ckpt(opts.ckpt_path)
if opts.ddim:
    # raise NotImplementedError("DDIM not implemented")
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = ddim_scheduler
pipe = pipe.to("cuda")


num_class_embeds = pipe.unet.config["num_class_embeds"]
class_label = -1

for k in range(0, opts.total_num, opts.batch_size):
    batch_size = min(opts.total_num - k, opts.batch_size)
    
    class_labels = torch.IntTensor([(class_label + i + 1) % num_class_embeds for i in range(batch_size)]).to("cuda")
    with torch.no_grad():
        images = pipe(
            batch_size=batch_size,
            num_inference_steps=opts.num_inference_steps,
            output_type=opts.output_type,
            mean=opts.mean, # 0.5
            std=opts.std,  # 0.5
            class_labels=class_labels,
            guidance_scale=opts.guidance_scale,
        ).images

    if opts.latent:
        for i, raw in enumerate(images):
            class_label = class_labels[i].item()
            
            im123 = Image.fromarray((raw * 255).astype(np.uint8))
            im123.save(os.path.join(raw_output_dir, "123", f"C{class_label}_{k+i}.png"))
            
            for j in range(3):  # no background for raw output of latent model
                im = Image.fromarray((raw[..., j] * 255).astype(np.uint8))
                im.save(os.path.join(raw_output_dir, str(j+1), f"C{class_label}_{k+i}.png"))
            
            closed = closing(raw, kernel_size=5)
            binary = binarize(closed, threshold=0.5)
            
            im123 = Image.fromarray((binary * 255).astype(np.uint8))
            im123.save(os.path.join(binary_output_dir, "123", f"C{class_label}_{k+i}.png"))
            
            for j in range(3):
                im = Image.fromarray((binary[..., j] * 255).astype(np.uint8))
                im.save(os.path.join(binary_output_dir, str(j+1), f"C{class_label}_{k+i}.png"))
    else:
        for i, raw in enumerate(images):
            class_label = class_labels[i].item()
            
            im123 = Image.fromarray((raw[..., 1:] * 255).astype(np.uint8))
            im123.save(os.path.join(raw_output_dir, "123", f"C{class_label}_{k+i}.png"))
            
            for j in range(4):
                im = Image.fromarray((raw[..., j] * 255).astype(np.uint8))
                im.save(os.path.join(raw_output_dir, str(j), f"C{class_label}_{k+i}.png"))
            
            binary = binarize(raw)
            im123 = Image.fromarray((binary[..., 1:] * 255).astype(np.uint8))
            im123.save(os.path.join(binary_output_dir, "123", f"C{class_label}_{k+i}.png"))
            
            for j in range(4):
                im = Image.fromarray((binary[..., j] * 255).astype(np.uint8))
                im.save(os.path.join(binary_output_dir, str(j), f"C{class_label}_{k+i}.png"))