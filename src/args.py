import argparse
import motti

parser = argparse.ArgumentParser()

# Model and Training Parameters
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of epochs")
parser.add_argument("--from_epoch", type=int, default=0, help="Starting epoch")
parser.add_argument("--num_workers", type=int, default=16, help="Number of data loading workers")
parser.add_argument("--device_num", type=int, default=1, help="Device number to use (GPU/TPU)")
parser.add_argument("--config_dir", type=str, default="../config/", help="Random seed")
parser.add_argument("--unet_config", type=str, default=".", help="unet config json")
parser.add_argument("--scheduler_config", type=str, default=".", help="scheduler config json")

# Data and Input Parameters
parser.add_argument("--img_size", type=int, default=28, help="Size of input images")
parser.add_argument("--img_root", type=str, default=".", help="Root directory of images")
parser.add_argument("--data_root", type=str, default="../../data/ACDC/quadra/", help="Root directory of images")
# parser.add_argument("--in_channels", type=int, default=1, help="input channels of unet")
# parser.add_argument("--out_channels", type=int, default=1, help="output channels of unet")

# Checkpoint and Logging
parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
parser.add_argument("--ckpt_dir", type=str, default="../../ckpt/prim/", help="Directory to save checkpoints")
parser.add_argument("--log_dir", type=str, default="../../logs/", help="Directory to save logs")
parser.add_argument("--log_step", type=int, default=10, help="Logging step interval")
parser.add_argument("--inference_step", type=int, default=50, help="Logging inference interval")
# parser.add_argument("--config_dir", type=str, default="../../config/", help="Directory to save configs")

# Learning Rate and Optimizer
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients across batches")

# Miscellaneous
parser.add_argument("--project", type=str, default="prim", help="Project name")
parser.add_argument("--ps", type=str, default="postscript", help="Postscript")

# # Data Augmentation
# parser.add_argument("-p", "--proportion", type=float, default=1.0, help="Proportion of data used")
# parser.add_argument("--jitter_p", type=float, default=0.5, help="Probability of jitter in data augmentation")

# Boolean Flags
parser.add_argument("--fast", action="store_true", help="Use fast mode")
# parser.add_argument("--save_training_output", action="store_true", help="Save training output")
# parser.add_argument("--frozen", action="store_true", help="Freeze model layers")
parser.add_argument("--reuse", action="store_true", help="Reuse previous session")

parser.add_argument("--p_uncond", type=float, default=0.1, help="Probability of unconditional sampling")
parser.add_argument("--p_uncond_label", type=int, default=0, help="Probability of unconditional label sampling")

opts, missing = parser.parse_known_args()

print(f"{opts = }")
print(f"{missing = }")
