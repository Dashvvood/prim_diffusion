import os
import motti
motti.append_parent_dir(__file__)

import torch
from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    DiffusionPipeline,
    AutoencoderKL,
    ImagePipelineOutput,
    get_cosine_schedule_with_warmup,
)
from diffusers.image_processor import VaeImageProcessor

from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union
import lightning as L

import wandb
from constant import CLASS2IDX, VAL_SEED
from torchvision.utils import make_grid


class ShapeLDM(DiffusionPipeline):
    def __init__(self, vae, unet, scheduler):
        super(ShapeLDM, self).__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.no_grad()
    def vae_encode(self, x: torch.Tensor):
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample() * self.vae.config.scaling_factor
        return z
    
    @torch.no_grad()
    def vae_decode(self, z: torch.Tensor):
        z = z / self.vae.config.scaling_factor
        x = self.vae.decode(z, return_dict=False)[0]
        return x
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        mean = 0.5,
        std = 0.5,
        callback=None,
        class_labels=None,
        guidance_scale = 7.5,
        guidance_rescale = 0.0,
    ):
        
        height = width = self.unet.config.sample_size
        num_channels_latents = self.unet.config.in_channels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height, width=width,
            device=self.device,
            generator=generator,
            dtype=torch.float32,
        )
        
        if class_labels is None:
            # 0 means random generation, w/o class conditioning
            class_condition = torch.zeros(batch_size, dtype=torch.long, device=self.device) 
        elif isinstance(class_labels, list):
            class_condition = torch.zeros(batch_size * 2, dtype=torch.long, device=self.device)
            class_condition[batch_size:] = torch.tensor(class_labels, dtype=torch.long, device=self.device)
        elif isinstance(class_labels, torch.Tensor):
            class_condition = torch.zeros(batch_size * 2, dtype=torch.long, device=self.device)
            class_condition[batch_size:] = class_labels.clone().detach().to(dtype=torch.long, device=self.device)
        else:
            raise ValueError("Invalid class_labels type")


        for t in self.progress_bar(self.scheduler.timesteps):
            
            # 0. double the image size
            latent_model_input = torch.cat([latents] * 2, dim=0) if class_labels is not None else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # 1. predict noise model_output
            noise_pred = self.unet(latent_model_input, t, class_labels=class_condition).sample
            
            if class_labels is not None:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
            # 2. compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
       
            # call back
            if callback is not None:
                callback(image, t)

        image = self.vae_decode(latents)
        
        # image = (image * 0.5 + 0.5).clamp(0, 1) # version originale 
        image = (image * std + mean).clamp(0, 1) # version changee
        
        # image = self.image_processor.postprocess(image, do_denormalize=do_denormalize)
        
        if output_type == "tensor":
            return (image.cpu(), )
        
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

class TrainableShapeLDM(L.LightningModule, ShapeLDM):
    def __init__(self, vae, unet, scheduler, opts=None):
        L.LightningModule.__init__(self)
        ShapeLDM.__init__(self, vae=vae, unet=unet, scheduler=scheduler)
        # self.save_hyperparameters(ignore=["ipython_dir"])
        self.save_hyperparameters(ignore=["ipython_dir"])
        self.opts = opts
    
    @classmethod
    def from_config(cls, config_dir, opts=None):
        unet_dir = os.path.join(config_dir, "unet")
        scheduler_dir = os.path.join(config_dir, "scheduler")
        
        unet_config = UNet2DModel.load_config(unet_dir)
        unet = UNet2DModel.from_config(unet_config)
        
        scheduler_config = DDPMScheduler.load_config(scheduler_dir)
        scheduler = DDPMScheduler.from_config(scheduler_config)
        
        vae = AutoencoderKL.from_pretrained(os.path.join(config_dir, "vae"))
        return cls(vae=vae, unet=unet, scheduler=scheduler, opts=opts)

    def _get_batch_class_idx_from_meta(self, meta):
        res = []
        for i, row in meta.iterrows():
            res.append(CLASS2IDX[(row.Group, row.Phase)])
        return torch.LongTensor(res)
    
    def training_step(self, batch, batch_idx):
        images = batch[0][:, 1:4, :, :]
        latents = self.vae_encode(images)
        class_labels = self._get_batch_class_idx_from_meta(batch[1]).to(self.device)
        
        random_uncond_mask = (torch.rand(size=(len(latents),))<=self.opts.p_uncond)
        class_labels[random_uncond_mask] = self.opts.p_uncond_label  # dropout class labels
        
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (latents.size(0),), device=self.device, )
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        residual = self.unet(sample=noisy_latents, timestep=steps, class_labels=class_labels).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True, batch_size=latents.size(0))
        return loss
    
    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.g = torch.Generator(self.device).manual_seed(VAL_SEED)
        
    def validation_step(self, batch, batch_idx):
        images = batch[0][:, 1:4, :, :]
        latents = self.vae_encode(images)
        class_labels = self._get_batch_class_idx_from_meta(batch[1]).to(self.device)
        
        random_uncond_mask = (torch.rand(size=(len(latents),))<=self.opts.p_uncond)
        class_labels[random_uncond_mask] = self.opts.p_uncond_label
        
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (latents.size(0),), device=self.device, generator=self.g)
        
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        residual = self.unet(sample=noisy_latents, timestep=steps, class_labels=class_labels).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("val_loss", loss, prog_bar=True, batch_size=latents.size(0))
        return loss
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.opts.inference_step != 0:
            return None
        else:
            self.pipe = self.pipe.to(self.device)  # make sure use gpu
            batch_size = min(8, self.opts.batch_size)
            latents = self.pipe(batch_size=batch_size, num_inference_steps=1000, generator=self.g, output_type="tensor")[0]
            images = self.vae_decode(latents)
            # images = images[:, 1:, :, :]
            grid = make_grid(images, nrow=batch_size)
            self.logger.experiment.log({
                "Sampling": wandb.Image(grid, caption=f"Epoch {self.current_epoch}"), 
            })
            return grid
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.opts.lr,
            betas=[0.9, 0.999],
            weight_decay=1e-2,
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.opts.warmup_epochs,
            num_training_steps=self.opts.max_epochs,
        )
        
        return [optimizer], [scheduler]
