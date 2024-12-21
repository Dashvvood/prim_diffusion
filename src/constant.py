# some useful constant values
# ed mask mean and std
from diffusers import StableDiffusionPipeline
CLASS2IDX = {
    ('DCM', 'ed'): 0,
    ('DCM', 'es'): 1,
    ('HCM', 'ed'): 2,
    ('HCM', 'es'): 3,
    ('MINF', 'ed'): 4,
    ('MINF', 'es'): 5,
    ('NOR', 'ed'): 6,
    ('NOR', 'es'): 7,
    ('RV', 'ed'): 8,
    ('RV', 'es'): 9,
}

# TODO: guidance weight 

IDX2CLASS = {v: k for k, v in CLASS2IDX.items()}


# TODO

VAL_SEED = 42