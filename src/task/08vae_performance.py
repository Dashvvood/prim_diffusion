import os
import motti
motti.append_current_dir(".")

import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from dataset import QuadraACDCDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import nms

Qdata = QuadraACDCDataset(
    root_dir="../../data/ACDC/quadra/",
    h5data="acdc_quadra.h5",
    metadata="quadra_per_slice_test.csv",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

dataloader = DataLoader(
    Qdata, batch_size=8, shuffle=False, num_workers=4,
    collate_fn=QuadraACDCDataset.collate_fn
)


device = "cuda:2"
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae", cache_dir="../model/vae/").to(device)
output_dir = "../../output/vae/stable-diffusion-3.5-large/"

# vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", cache_dir="../model/vae/").to(device)
# vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", cache_dir="../model/vae/").to(device)

g = torch.Generator().manual_seed(20250125)

output_dir = "../../output/vae/stable-diffusion-3.5-large/"

output_raw_dir = os.path.join(output_dir, "raw")
output_nms_dir = os.path.join(output_dir, "nms")

os.makedirs(output_raw_dir, exist_ok=True)
os.makedirs(output_nms_dir, exist_ok=True)

def psnr(target, prediction):
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        return 100  # No error
    max_pixel = 1.0  # assuming image is normalized between [0, 1]
    return 20 * np.log10(max_pixel / np.sqrt(mse))

raw_psnr_list = []
raw_ssim_list = []

nms_psrn_list = []
nms_ssim_list = []

with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader)):
        data, meta = batch
        X = data[:, 1:4, ...].to(device)
        posterior = vae.encode(X).latent_dist
        z = posterior.sample(generator=g)
        y = vae.decode(z).sample
        
        y01 = (y * 0.5 + 0.5).clamp(0, 1)
        imgs = y01.clone().cpu().permute(0, 2, 3, 1).numpy()
        
        images = motti.numpy_to_pil(imgs)
        for j, image in enumerate(images):
            name = meta.iloc[j].ID
            image.save(os.path.join(output_raw_dir, f"{name}.png"))
        
        for j, img in enumerate(imgs):
            name = meta.iloc[j].ID
            nms_img = nms(img)
            nms_image = motti.numpy_to_pil(nms_img)
            nms_image.save(os.path.join(output_nms_dir, f"{name}.png"))
        
        # metrics
        for Xi, yi in zip(X, y):
            Xi = (Xi * 0.5 + 0.5).clamp(0, 1)
            yi = (yi * 0.5 + 0.5).clamp(0, 1)
            
            Xi = Xi.permute(1, 2, 0).cpu().numpy()
            yi = yi.permute(1, 2, 0).cpu().numpy()

            psnr_Xi_yi = psnr(Xi, yi)
            ssim_Xi_yi = ssim(Xi, yi, data_range=1, channel_axis=-1)
            
            raw_psnr_list.append(psnr_Xi_yi)
            raw_ssim_list.append(ssim_Xi_yi)
            
            Xi_nms = Xi
            yi_nms = nms(yi)
            
            psnr_Xi_yi = psnr(Xi_nms, yi_nms)
            ssim_Xi_yi = ssim(Xi_nms, yi_nms, data_range=1, channel_axis=-1)
            
            nms_psrn_list.append(psnr_Xi_yi)
            nms_ssim_list.append(ssim_Xi_yi)

        
print(f"RAW PSNR: {np.mean(raw_psnr_list):.2f} ± {np.std(raw_psnr_list):.2f}")
print(f"RAW SSIM: {np.mean(raw_ssim_list):.2f} ± {np.std(raw_ssim_list):.2f}")

print(f"NMS PSNR: {np.mean(nms_psrn_list):.2f} ± {np.std(nms_psrn_list):.2f}")
print(f"NMS SSIM: {np.mean(nms_ssim_list):.2f} ± {np.std(nms_ssim_list):.2f}")

# rFID	PSNR	SSIM	PSIM
