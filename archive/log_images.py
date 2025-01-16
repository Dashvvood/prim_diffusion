import wandb
import numpy as np
import torch
from torchvision.utils import make_grid 

# Initialize a W&B run
wandb.init(project="image_logging", name="numpy_images")

# Create dummy images
image_array = np.random.rand(64, 64, 3) * 255  # Random RGB image
image_array = image_array.astype(np.uint8)

# Log a single image
wandb.log({"example_image": wandb.Image(image_array, caption="Random Image")})

# Log a batch of images
image_batch = [np.random.rand(64, 64, 3) * 255 for _ in range(5)]
image_batch = [img.astype(np.uint8) for img in image_batch]

# Convert image_batch to a torch tensor with shape (batch_size, channels, width, height)
image_batch_tensor = torch.tensor(image_batch).permute(0, 3, 1, 2)

for i in range(5):
    
    grid = make_grid(image_batch_tensor, nrow=5)
    wandb.log({"example_images": wandb.Image(grid, caption=f"grid is {i}"), "sample_epoch": i},)

# Finish the run
wandb.finish()
