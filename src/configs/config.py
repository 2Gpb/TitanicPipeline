from pathlib import Path
from omegaconf import OmegaConf

config_path = Path(__file__).parent / "config.yml"
config = OmegaConf.load(config_path)