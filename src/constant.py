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


IDX2CLASS = {v: k for k, v in CLASS2IDX.items()}

VAL_SEED = 42
