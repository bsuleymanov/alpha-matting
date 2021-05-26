import hydra
from omegaconf import OmegaConf, DictConfig
import os, logging
from hydra import initialize, compose
from hydra.utils import instantiate


logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    logger.info(f"Working dir: {os.getcwd()}")
    print(OmegaConf.to_yaml(cfg))

net = instantiate(cfg.)

if __name__ == "__main__":
    train()