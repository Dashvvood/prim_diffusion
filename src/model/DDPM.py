import torch
import diffusers
import lightning as L

from utils.cos_warmup_scheduler import get_cosine_schedule_with_warmup

class DiffusionModel(L.LightningModule):
    def __init__(self, opts=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = diffusers.models.UNet2DModel(
            sample_size=opts.img_size,
            in_channels=opts.in_channels,
            out_channels=opts.out_channels,
            block_out_channels=[64, 128, 256, 512],
            center_input_sample=True, # 2 * x - 1.0 in forward fn
        )
        self.scheduler = diffusers.schedulers.DDPMScheduler()
        self.opts = opts
        
    def training_step(self, batch, batch_idx):
        images = batch[0]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
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
    