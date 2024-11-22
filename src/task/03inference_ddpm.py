import os
import motti
motti.append_current_dir(os.path.abspath(''))
o_d = motti.o_d()

import numpy as np
import matplotlib.pyplot as plt
import torch
from model.pipeline_ddpm import DDPMPipelineV2
from model.pipeline_ddim import DDIMPipelineV2
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMPipeline, DDPMPipeline
from PIL import Image
from tqdm import tqdm
import argparse
from utils.nms import NMS
import json

def local_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=".")
    parser.add_argument("--unet_config", type=str, default=".")
    parser.add_argument("--scheduler_config", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_num", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    parser.add_argument("--output_type", type=str, default="pil")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--ddim", action='store_true', default=False)
    args, missing = parser.parse_known_args()
    print(f"Missing args: {missing}")
    return args

opts = local_args()
output_dir = os.path.join(opts.output_dir, o_d)
os.makedirs(output_dir, exist_ok=True)

raw_output_dir = os.path.join(output_dir, "raw")
nms_output_dir = os.path.join(output_dir, "nms")
os.makedirs(raw_output_dir, exist_ok=True)
os.makedirs(nms_output_dir, exist_ok=True)

os.makedirs(os.path.join(raw_output_dir, "123"), exist_ok=True)
os.makedirs(os.path.join(nms_output_dir, "123"), exist_ok=True)

# # raw
# 0
# 1
# 2
# 3
# 123(rgb)
# # nms
# 0
# 1
# 2
# 3
# 123(rgb)

for i in range(4):
    os.makedirs(os.path.join(raw_output_dir, str(i)), exist_ok=True)
    os.makedirs(os.path.join(nms_output_dir, str(i)), exist_ok=True)

with open(os.path.join(output_dir, "args.json"), "w") as f:
    json.dump(vars(opts), f, indent=4)

ckpt = torch.load(opts.ckpt_path, map_location="cpu")
d = {}
for k, v in ckpt["state_dict"].items():
    new_k = k.split('.', 1)[1]
    d[new_k] = v

unet_config = UNet2DModel.load_config(opts.unet_config)
unet = UNet2DModel.from_config(unet_config)
unet.load_state_dict(d)

scheduler_config = DDPMScheduler.load_config(opts.scheduler_config)
ddpm_scheduler = DDPMScheduler.from_config(scheduler_config)


if opts.ddim:
    ddim_scheduler = DDIMScheduler.from_config(scheduler_config)
    pipe = DDIMPipelineV2(unet, ddim_scheduler).to("cuda")
else:
    pipe = DDPMPipelineV2(unet, ddpm_scheduler).to("cuda")


for k in range(0, opts.total_num, opts.batch_size):
    batch_size = min(opts.total_num - k, opts.batch_size)
    images = pipe(
        batch_size=batch_size,
        num_inference_steps=opts.num_inference_steps,
        output_type=opts.output_type,
        mean=opts.mean, # 0.5
        std=opts.std,  # 0.5
    ).images

    for i, raw in enumerate(images):
        breakpoint()
        im123 = Image.fromarray((raw[..., 1:] * 255).astype(np.uint8))
        im123.save(os.path.join(raw_output_dir, "123", f"{k+i}.png"))
        
        for j in range(4):
            im = Image.fromarray((raw[..., j] * 255).astype(np.uint8))
            im.save(os.path.join(raw_output_dir, str(j), f"{k+i}.png"))
        
        nms = NMS(raw)
        im123 = Image.fromarray((nms[..., 1:] * 255).astype(np.uint8))
        im123.save(os.path.join(nms_output_dir, "123", f"{k+i}.png"))
        
        for j in range(4):
            im = Image.fromarray((nms[..., j] * 255).astype(np.uint8))
            im.save(os.path.join(nms_output_dir, str(j), f"{k+i}.png"))
