from diffusers import (
    DDPMPipeline, 
    DDIMPipeline,
    UNet2DModel, 
    DDPMScheduler,
    DDIMScheduler
)
import torch
import argparse

# from args import opts

from model.pipeline_ddpm import DDPMPipelineV2

def load_ddpmpipeline(unet_config, unet_ckpt_path, scheduler_config, ):
    config = UNet2DModel.load_config(unet_config)
    unet = UNet2DModel.from_config(config)
    config = DDPMScheduler.load_config(scheduler_config)
    scheduler = DDPMScheduler.from_config(config)
    
    ckpt = torch.load(unet_ckpt_path, map_location="cpu")
    
    D = {}
    for k, v in ckpt['state_dict'].items():
        D[k.replace("model.", "")] = v
    unet.load_state_dict(D)
    
    pipeline = DDPMPipelineV2(unet, scheduler)
    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a DDPM pipeline with specified configurations.")
    parser.add_argument("--unet_config", type=str, required=True, help="Path to the UNet configuration file.")
    parser.add_argument("--unet_ckpt_path", type=str, required=True, help="Path to the UNet checkpoint file.")
    parser.add_argument("--scheduler_config", type=str, required=True, help="Path to the scheduler configuration file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--total_size", type=int, default=1, help="Total size of the dataset to generate.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM scheduler instead of DDPM scheduler.")
    args = parser.parse_args()
    
    pipeline = load_ddpmpipeline(args.unet_config, args.unet_ckpt_path, args.scheduler_config)
    
    if args.use_ddim:
        scheduler = DDIMScheduler() # eta = 1 means a DDPM type
        unet = pipeline.unet
        unet.sample_size = 64
        unet.config.sample_size = 64
        pipeline = DDIMPipeline(unet, scheduler)
    print("Pipeline loaded successfully.")
    for i in range(0, args.total_size, args.batch_size):
        if args.use_ddim:
            X = pipeline(batch_size=args.batch_size, eta=1.0, num_inference_steps=50).images
        else:
            X = pipeline(batch_size=args.batch_size).images
        for j in range(len(X)):
            X[j].save(f"{args.output_dir}/image_{i+j}.png")
            