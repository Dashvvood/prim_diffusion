# some useful constant values
# ed mask mean and std
from diffusers import StableDiffusionPipeline
CLASS2IDX = {
    ('Random', 'Random'): 0,
    ('DCM', 'ed'): 1,
    ('DCM', 'es'): 2,
    ('HCM', 'ed'): 3,
    ('HCM', 'es'): 4,
    ('MINF', 'ed'): 5,
    ('MINF', 'es'): 6,
    ('NOR', 'ed'): 7,
    ('NOR', 'es'): 8,
    ('RV', 'ed'): 9,
    ('RV', 'es'): 10,
}

# TODO: guidance weight 
IDX2CLASS = {v: k for k, v in CLASS2IDX.items()}

# (256, 32, 32) -> (256, 1024) -> (1024, 256)
# (C, H, W) -> (C, H * W) -> (H * W, C), 
# W_q = (C, C)

# latent = (16, 16, 16) -> (16, 2, 2)

# 1. 5 + 1 classes
# 2. 2 + 1 phases
# 3. slice_position: float(0, 1) -> Linear Projection( embedding)
# sum all

# embed_ = 100

# [:50] = DCM ?
# [50:] = Phase?

VAL_SEED = 42

# VAE config on training set -- data/ACDC/quadra/quadra_per_slice_train_train.csv
vae_scaling_factor = 9.76