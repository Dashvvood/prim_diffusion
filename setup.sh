#!/bin/bash

mkdir ckpt data doc logs misc

CUDA_VISIBLE_DEVICES=1 python 05train_ddpm_class_guidance.py --unet_config ../../config/ddpm_medium/unet_class/ --scheduler_config ../../config/ddpm_medium/scheduler/ --batch_size 16 --warmup_epochs 20 --max_epochs 200 --num_workers 8 --device_num 1  --lr 1e-5 
--img_size 64 --project prim  --log_step 5  --p_uncond 0.5 --ps test_small  