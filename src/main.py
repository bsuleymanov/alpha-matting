import hydra
from omegaconf import OmegaConf, DictConfig, MISSING
import os, logging
from hydra import initialize, compose
from hydra.utils import instantiate
from dataclasses import dataclass, field
from typing import Any
import torch
from fvcore.nn import flop_count_table, flop_count_str
from fvcore.nn.flop_count import FlopCountAnalysis
from fvcore.nn.parameter_count import parameter_count
import wandb
from pathlib import Path
from utils import mkdir_if_empty_or_not_exist
from hydra.utils import to_absolute_path
from typing import Any, Optional, List

OmegaConf.register_new_resolver("if_null_default", lambda x, y: y if x is None else x)
#OmegaConf.register_new_resolver("kek", lambda x: x is None)

logger = logging.getLogger(__name__)


@dataclass
class DataloaderConfig:
    _target_: str = MISSING
    image_path: str = MISSING
    foreground_path: str = MISSING
    background_path: str = MISSING
    image_size: int = 256
    batch_size: int = 8
    drop_last: bool = False
    shuffle: bool = True
    mode: str = "train"
    num_workers: Optional[int] = 8
    shared_pre_transform: Optional = None
    composition_transform: Optional = None
    foreground_transform: Optional = None
    background_transform: Optional = None
    matte_transform: Optional = None
    shared_post_transform: Optional = None

@dataclass
class TransformsConfig:
    shared_pre: Any = None
    composition: Any = None
    shared_post: Any = None
    foreground: Optional = None
    background: Optional = None
    matte: Optional = None

@dataclass
class TrainDataConfig:
    #defaults: List[Any] = field(default_factory=lambda:[{}])
    num_workers: int = MISSING#: Optional[int]
    name: str = "Maadaa"
    image_path: str = MISSING
    foreground_path: str = MISSING
    background_path: str = MISSING
    dataloader: DataloaderConfig = MISSING
    batch_size: int = 8
    shuffle: bool = True
    image_size: int = 256
    drop_last: bool = False
    transforms: Optional[TransformsConfig] = None

# @dataclass
# class TestDataConfig:
#     num_workers: Optional[int]
#     name: str = "Maadaa"
#     image_path: str = MISSING
#     foreground_path: str = MISSING
#     background_path: str = MISSING
#     dataloader: DataloaderConfig = MISSING
#     batch_size: int = 8
#     shuffle: bool = True
#     image_size: int = 256
#     drop_last: bool = False
#     transforms: TransformsConfig = None


@dataclass
class DataConfig:
    #defaults: List[Any] = field(default_factory=lambda: [{"train": "maadaatrain"}])
    train: TrainDataConfig = TrainDataConfig
    #TEST: TestDataConfig

@dataclass
class MainConfig:
    defaults: List[Any] = field(default_factory=lambda: [{"data": "maadaadatamodule"}])
    data: DataConfig = MISSING
    mode: str = MISSING
    #train_dataloader: TrainDataloaderConfig
    #model: ModelConfig

cs = hydra.core.config_store.ConfigStore()
cs.store(name="config_schema", node=MainConfig)
#cs.store(group="data", name="maadaadatamodule", node=DataConfig)
#cs.store(group="data/train", name="maadaatrain", node=TrainDataConfig)

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

class Foo:
    def __init__(self, cfg):
        self.cfg = cfg.copy()

    def __call__(self):
        self.cfg.update({"num_workers": 111})
        return self.cfg

@hydra.main(config_path="configs/maadaa/modnet", config_name="config")
#@hydra.main(config_path=None, config_name="config")
def train(cfg: DictConfig):
    #wandb.init(project=cfg.WANDB.PROJECT, entity=cfg.WANDB.USER)
    #mode = "train" if cfg.IS_TRAIN else "test"
    #model_save_path = Path(to_absolute_path(cfg.MODEL_SAVE_PATH))
    #input_image_save_path = Path(to_absolute_path(cfg.INPUT_IMAGE_SAVE_PATH))
    #mkdir_if_empty_or_not_exist(model_save_path)
    #mkdir_if_empty_or_not_exist(input_image_save_path)

    #print(cfg.DATA.TRAIN.DATALOADER.image_path)
    # dataloader_config = cfg.data.train.dataloader.copy()
    # dataloader_config.update({"num_workers": 218})
    # train_dataloader = instantiate(dataloader_config).loader
    # print(train_dataloader.num_workers)
    # print(cfg.data.train.dataloader)
    dataloader_config = cfg.data.train.dataloader
    foo = Foo(dataloader_config)
    new_cfg = foo()
    print(dataloader_config)
    print(new_cfg)


    #print(train_dataloader)
    #print(train_dataloader)
    #val_dataloader = instantiate(cfg.DATA.TEST.DATALOADER).loader()
    #train_transforms = instantiate(cfg.data.train.transforms.shared_pre)

    #print(cfg.data.train.image_path
    #print(train_transforms)
    #print(OmegaConf.to_yaml(cfg))
    #print(cfg)
    #print(cfg.data.train.image_path)
    #data_config = dict(cfg.data)
    #print(type(data_config))
    #data_config["kakoy-to_kek"] = 10
    #print(data_config)
    #assert cfg.data.train.image_path
    #print(len(train_dataloader))
    #print(cfg.data.train.transforms)
    #print(OmegaConf.to_yaml(cfg))
    #print(len(val_dataloader))


if __name__ == "__main__":
    train()