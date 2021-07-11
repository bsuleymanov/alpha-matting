import hydra
from omegaconf import OmegaConf, DictConfig, MISSING
import omegaconf
import os, logging
from hydra import initialize, compose
from hydra.utils import instantiate

@hydra.main(config_path="configs/maadaa/modnet", config_name="full_experiment")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(type(cfg))
    print(cfg.network)
    cfg.update(network='kek')
    print(cfg.network)
    OmegaConf.update(cfg, "network.param", "keks", force_add=True)
    print(cfg.network)
    #cfg.training.is_ddp = True
    print(cfg.training)
    print(omegaconf.__version__)
    new_cfg = OmegaConf.create(dict(
        is_ddp='1',
        rank=2,
        world_size=3
    ))
    new_cfg1 = OmegaConf.create(dict(
        device="cuda"
    ))
    print(type(new_cfg))
    print(new_cfg)
    print(OmegaConf.merge(new_cfg1, new_cfg))

    #cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict_training = OmegaConf.to_object(cfg)["training"]
    cfg_dict_training.update(new_cfg)
    OmegaConf.update(cfg, "training", cfg_dict_training)

    print(cfg_dict_training)
    print(cfg.training)

if __name__ == "__main__":
    main()