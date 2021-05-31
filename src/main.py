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
import wandb
from pathlib import Path
from utils import mkdir_if_empty_or_not_exist
from hydra.utils import to_absolute_path



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

#cs = hydra.core.config_store.ConfigStore()
#cs.store(name="config_schema", node=MainConfig)

# @hydra.main(config_path="configs", config_name="config")
# def train(cfg: DictConfig):
#     logger.info(f"Working dir: {os.getcwd()}")
#     print(OmegaConf.to_yaml(cfg))
#     train_loader = instantiate(cfg.dataloader).loader()
#     model = instantiate(cfg.model)
#     dummy_input = torch.ones([2, 3, 256, 256])
#     flops = FlopCountAnalysis(model, dummy_input)
#     n_flops = flops.by_module()[''] * 1e-9
#     n_params = dict(parameter_count(flops._model))[''] * 1e-6
#     print(f"Flops: {n_flops:.3f}G, Parameters: {n_params: .3f}M")

@hydra.main(config_path="configs/maadaa/modnet", config_name="config")
def train(cfg: DictConfig):
    wandb.init(project=cfg.WANDB.PROJECT, entity=cfg.WANDB.USER)
    mode = "train" if cfg.IS_TRAIN else "test"
    model_save_path = Path(to_absolute_path(cfg.MODEL_SAVE_PATH))
    input_image_save_path = Path(to_absolute_path(cfg.INPUT_IMAGE_SAVE_PATH))
    mkdir_if_empty_or_not_exist(model_save_path)
    mkdir_if_empty_or_not_exist(input_image_save_path)

    #print(cfg.DATA.TRAIN.DATALOADER.image_path)
    dataloader = instantiate(cfg.DATA.TRAIN.DATALOADER).loader()
    print(cfg.DATA.TRAIN.DATALOADER)
    print(len(dataloader))
    # dataloader = MaadaaMattingLoaderV2(image_path, foreground_path,
    #                                    background_path, image_size,
    #                                    batch_size, mode).loader()



if __name__ == "__main__":
    train()