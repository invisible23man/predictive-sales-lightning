from omegaconf import OmegaConf


def load_config(path="src/config/config.yaml"):
    return OmegaConf.load(path)
