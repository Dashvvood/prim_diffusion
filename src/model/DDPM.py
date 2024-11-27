import torch
import diffusers
import lightning as L

from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    get_cosine_schedule_with_warmup
)

class TrainableDDPM(L.LightningModule):
    def __init__(self, unet, scheduler, opts=None):
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])
        
        self.unet = unet
        self.scheduler = scheduler
        self.opts = opts
        
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
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.unet(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.unet(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def on_train_epoch_end(self):
        # TODO: sampling
        pass
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        pass
    
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
    
    
    