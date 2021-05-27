import hydra
from omegaconf import OmegaConf, DictConfig
import os, logging
from hydra import initialize, compose
from hydra.utils import instantiate
from dataclasses import dataclass
from typing import Any
import torch
from fvcore.nn import flop_count_table, flop_count_str
from fvcore.nn.flop_count import FlopCountAnalysis
from fvcore.nn.parameter_count import parameter_count

logger = logging.getLogger(__name__)

@dataclass
class Imageconfig:
    size: int
    channels: int

@dataclass
class DatasetConfig:
    image: Imageconfig

@dataclass
class TrainDataloaderConfig:
    _target_: Any
    image_path: str
    image_size: int
    batch_size: int
    mode: str

@dataclass
class ModelConfig:
    _target_: str

@dataclass
class MainConfig:
    dataset: DatasetConfig
    train_dataloader: TrainDataloaderConfig
    model: ModelConfig

cs = hydra.core.config_store.ConfigStore()
cs.store(name="config_schema", node=MainConfig)

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    logger.info(f"Working dir: {os.getcwd()}")
    print(OmegaConf.to_yaml(cfg))
    train_loader = instantiate(cfg.train_dataloader).loader()
    model = instantiate(cfg.model)
    dummy_input = torch.ones([2, 3, 256, 256])
    flops = FlopCountAnalysis(model, dummy_input)
    n_flops = flops.by_module()[''] * 1e-9
    n_params = dict(parameter_count(flops._model))[''] * 1e-6
    print(f"Flops: {n_flops:.3f}G, Parameters: {n_params: .3f}M")


if __name__ == "__main__":
    train()