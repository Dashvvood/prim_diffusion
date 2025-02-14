from omegaconf import OmegaConf
from types import SimpleNamespace
import sys

def load_config(config_path):
    config = OmegaConf.load(config_path)
    
    overrides = OmegaConf.from_cli(sys.argv[1:])
    config = OmegaConf.merge(config, overrides)
    
    return SimpleNamespace(**config)
