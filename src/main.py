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
from typing import Any, Optional

OmegaConf.register_new_resolver("if_null_default", lambda x, y: y if x is None else x)

logger = logging.getLogger(__name__)


@dataclass
class DataloaderConfig:
    _target_: Any
    image_path: str
    foreground_path: str
    background_path: str
    image_size: int
    batch_size: int
    drop_last: bool
    shuffle: bool
    mode: str
    num_workers: Optional = None
    shared_pre_transform: Optional = None
    composition_transform: Optional = None
    foreground_transform: Optional = None
    background_transform: Optional = None
    matte_transform: Optional = None
    shared_post_transform: Optional = None

@dataclass
class TransformsConfig:
    shared_pre: Any
    composition: Any
    shared_post: Any
    foreground: Optional = None
    background: Optional = None
    matte: Optional = None


@dataclass
class TrainDataConfig:
    NAME: str
    IMAGE_PATH: str
    FOREGROUND_PATH: str
    BACKGROUND_PATH: str
    DATALOADER: DataloaderConfig
    BATCH_SIZE: int
    SHUFFLE: bool
    IMAGE_SIZE: int
    DROP_LAST: bool
    NUM_WORKERS: int
    TRANSFORMS: TransformsConfig

@dataclass
class TestDataConfig:
    ...

@dataclass
class DataConfig:
    TRAIN: TrainDataConfig
    #TEST: TestDataConfig

@dataclass
class MainConfig:
    DATA: DataConfig
    MODE: str
    #train_dataloader: TrainDataloaderConfig
    #model: ModelConfig

cs = hydra.core.config_store.ConfigStore()
cs.store(name="config_schema", node=MainConfig)

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
    #wandb.init(project=cfg.WANDB.PROJECT, entity=cfg.WANDB.USER)
    #mode = "train" if cfg.IS_TRAIN else "test"
    #model_save_path = Path(to_absolute_path(cfg.MODEL_SAVE_PATH))
    #input_image_save_path = Path(to_absolute_path(cfg.INPUT_IMAGE_SAVE_PATH))
    #mkdir_if_empty_or_not_exist(model_save_path)
    #mkdir_if_empty_or_not_exist(input_image_save_path)

    #print(cfg.DATA.TRAIN.DATALOADER.image_path)
    train_dataloader = instantiate(cfg.DATA.TRAIN.DATALOADER).loader
    #val_dataloader = instantiate(cfg.DATA.TEST.DATALOADER).loader()

    print(len(train_dataloader))
    #print(len(val_dataloader))


if __name__ == "__main__":
    train()