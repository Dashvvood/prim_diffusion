import os
import motti
motti.append_parent_dir(__file__)

import torch
import diffusers
from diffusers import DDPMPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union

from torchvision.utils import make_grid 
import lightning as L

from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    get_cosine_schedule_with_warmup
)

from constant import (
    CLASS2IDX,
    VAL_SEED
)

import wandb

from .pipeline_ddpm import DDPMPipelineV2

from diffusers import DDPMPipeline

class TrainableDDPM(L.LightningModule):
    def __init__(self, unet, scheduler, opts=None):
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])
        
        self.unet = unet
        self.scheduler = scheduler
        self.opts = opts
        
        self.pipe = DDPMPipelineV2(unet, scheduler)
        
        #1 an mixed embedding = class embedding + Timestep embedding WITH unet2d
        #2 class embedding with ConditionalUNet2D
        # addaptive layer normalization in SELF-ATTENTION LAYER
        
        
    @classmethod
    def from_config(cls, unet_config, scheduler_config, opts):
        if isinstance(unet_config, str):
            config = UNet2DModel.load_config(unet_config)
            unet = UNet2DModel.from_config(config)
        elif isinstance(unet_config, dict):
            unet = UNet2DModel.from_config(unet_config)
        else:
            raise ValueError("unet_config must be a path or a dictionary")
        
        if isinstance(scheduler_config, str):
            config = DDPMScheduler.load_config(scheduler_config)
            scheduler = DDPMScheduler.from_config(config)
        elif isinstance(scheduler_config, dict):
            scheduler = DDPMScheduler(**scheduler_config)
        else:
            raise ValueError("scheduler_config must be a path or a dictionary")
        
        return cls(unet=unet, scheduler=scheduler, opts=opts)
    
    
    def training_step(self, batch, batch_idx):
        images = batch[0]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device, )
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.unet(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def on_validation_epoch_start(self):
        self.g = torch.Generator(self.device).manual_seed(VAL_SEED)
        
        
    def validation_step(self, batch, batch_idx):
        images = batch[0]
        noise = torch.randn_like(images)
        # TODO: fix timesteps for validation for each epoch: 
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device, generator=self.g) # TODO
        # steps = f(batch_idx)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.unet(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    
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
    
    def forward(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        mean = 0.5,
        std = 0.5,
        callback=None,
        class_labels: List[int]=None,
        guidance_scale = 7.5,
        guidance_rescale = 0.0,
    ) -> Union[ImagePipelineOutput, Tuple]:

        return self.pipe(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_dict=return_dict,
            mean=mean,
            std=std,
            callback=callback,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
        )

class TrainableDDPMbyClass(L.LightningModule):
    def __init__(self, unet, scheduler, opts=None):
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])
        
        self.unet = unet
        self.scheduler = scheduler
        self.opts = opts
        self.pipe = DDPMPipelineV2(unet, scheduler)
        self.unet.config["sample_size"] = opts.img_size
    
    @classmethod
    def from_config(cls, unet_config, scheduler_config, opts=None):
        if isinstance(unet_config, str):
            config = UNet2DModel.load_config(unet_config)
            unet = UNet2DModel.from_config(config)
        elif isinstance(unet_config, dict):
            unet = UNet2DModel.from_config(unet_config)
        else:
            raise ValueError("unet_config must be a path or a dictionary")
        
        if isinstance(scheduler_config, str):
            config = DDPMScheduler.load_config(scheduler_config)
            scheduler = DDPMScheduler.from_config(config)
        elif isinstance(scheduler_config, dict):
            scheduler = DDPMScheduler(**scheduler_config)
        else:
            raise ValueError("scheduler_config must be a path or a dictionary")
        
        return cls(unet=unet, scheduler=scheduler, opts=opts)
    
    def _get_batch_class_idx_from_meta(self, meta):
        res = []
        for i, row in meta.iterrows():
            res.append(CLASS2IDX[(row.Group, row.Phase)])
        return torch.LongTensor(res)
    
    def training_step(self, batch, batch_idx):
        images = batch[0]
        class_labels = self._get_batch_class_idx_from_meta(batch[1]).to(self.device)
        
        random_uncond_mask = (torch.rand(size=(len(images),))<=self.opts.p_uncond)
        class_labels[random_uncond_mask] = self.opts.p_uncond_label  # dropout class labels
        
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device, )
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.unet(sample=noisy_images, timestep=steps, class_labels=class_labels).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss
    
    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.g = torch.Generator(self.device).manual_seed(VAL_SEED)

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        class_labels = self._get_batch_class_idx_from_meta(batch[1]).to(self.device)
        
        random_uncond_mask = (torch.rand(size=(len(images),))<=self.opts.p_uncond)
        class_labels[random_uncond_mask] = self.opts.p_uncond_label
        
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device, generator=self.g)
        
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.unet(sample=noisy_images, timestep=steps, class_labels=class_labels).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("val_loss", loss, prog_bar=True, batch_size=images.size(0))
        return loss
    
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.opts.inference_step != 0:
            return None
        else:
            self.pipe = self.pipe.to(self.device)  # make sure use gpu
            batch_size = min(8, self.opts.batch_size)
            images = self.pipe(batch_size=batch_size, num_inference_steps=1000, generator=self.g, output_type="tensor")[0]
            images = images[:, 1:, :, :]
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
    
    def forward(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        mean = 0.5,
        std = 0.5,
        callback=None,
        class_labels: List[int]=None,
        guidance_scale = 7.5,
        guidance_rescale = 0.0,
    ) -> Union[ImagePipelineOutput, Tuple]:

        return self.pipe(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_dict=return_dict,
            mean=mean,
            std=std,
            callback=callback,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
        )
        
