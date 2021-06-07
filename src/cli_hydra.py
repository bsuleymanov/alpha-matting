from dataloader import MaadaaMattingLoader, MattingTestLoader, \
                       MattingLoaderDeprecated, MaadaaMattingLoaderV2
from utils import mkdir_if_empty_or_not_exist, generate_trimap_kornia
import time
import datetime
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from model import GaussianBlurLayer, MODNet
from utils import denorm, tensor_to_image
import wandb
import numpy as np
import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import sys
import cv2
from losses import ModNetLoss, modnet_loss
from functools import partial
import pandas as pd
import seaborn as sns
import plotly.express as px

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



class OptimizerWrapper:
    def __init__(self, cfg):
        self.cfg = cfg.copy()

    def __call__(self, params):
        self.cfg.update({"params": params})
        return instantiate(self.cfg)

class LRSchedulerWrapper:
    def __init__(self, cfg):
        self.cfg = cfg.copy()

    def __call__(self, optimizer=None):
        if self.optimizer is not None:
            self.cfg.update({"optimizer": optimizer})
        return instantiate(self.cfg)

def get_loss(loss_name):
    if loss_name == "modnet":
        return modnet_loss


@hydra.main(config_path="configs/maadaa/modnet", config_name="config")
# def train_from_folder(
#     data_dir="../data",
#     results_dir="../data/results",
#     models_dir="../",
#     image_size=256,
#     version="mobilenetv2",
#     total_step=150000,
#     batch_size=2,
#     val_batch_size = 3,
#     accumulation_steps=1,
#     n_workers=8,
#     learning_rate=0.01,
#     lr_decay=0.95,
#     beta1=0.5,
#     beta2=0.999,
#     test_size=2824,
#     model_name="model.pth",
#     pretrained_model=None,
#     is_train=True,
#     parallel=False,
#     use_tensorboard=False,
#     image_path="../data/dataset_split/train/",
#     foreground_path="../data/foregrounds_split/train/",
#     background_path="../data/backgrounds/",
#     val_image_path="../data/one_image_dataset_split/val/",
#     mask_path="../data/dataset/train/seg",
#     log_path="./logs",
#     model_save_path="./models",
#     sample_path="./sample_path",
#     input_image_save_path="./input_save",
#     test_image_path="../data/dataset/val/image",
#     test_mask_path="./test_results",
#     test_color_mask_path="./test_color_visualize",
#     log_step=10,
#     sample_step=100,
#     model_save_step=1.0,
#     device="cuda",
#     verbose=1,
#     dataset="matting",
#     semantic_scale=10.0,
#     detail_scale=10.0,
#     matte_scale=1.0,
#     visual_debug=False
# ):
def train_from_folder(cfg: DictConfig):
    #wandb.init(project="alpha-matting",  entity='bsuleymanov')#, settings=wandb.Settings(start_method="fork"))
    wandb.init(project=cfg.logging.project, entity=cfg.logging.entity)

    #config = wandb.config
    mode = cfg.training.mode

    #mode = "train" if is_train else "test"
    sample_path = Path(cfg.logging.sample_path)
    model_save_path = Path(cfg.logging.model_save_path)
    input_image_save_path = Path(cfg.logging.input_image_save_path)
    mkdir_if_empty_or_not_exist(sample_path)
    mkdir_if_empty_or_not_exist(model_save_path)
    mkdir_if_empty_or_not_exist(input_image_save_path)

    # dataloader = MaadaaMattingLoaderV2(image_path, foreground_path,
    #                                    background_path, image_size,
    #                                    batch_size, mode).loader()
    dataloader = instantiate(cfg.data.train.dataloader).loader
    #dataloader = MaadaaMattingLoader(image_path, image_size,
    #                                 batch_size, mode).loader()
    validation_dataloader = instantiate(cfg.data.test.dataloader).loader
    # validation_dataloader = MaadaaMattingLoader(val_image_path, image_size,
    #                                             val_batch_size, mode).loader()
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    total_epoch = cfg.training.total_step / cfg.training.step_per_epoch
    print(total_epoch)
    model_save_step = 500

    if cfg.pretrained_model:
        start = cfg.pretrained_model + 1
    else:
        start = 0

    blurer = GaussianBlurLayer(3, 3).to(cfg.training.device)
    #loss_fn = ModNetLoss(semantic_scale, detail_scale, matte_scale, blurer)
    loss_args = cfg.train.loss.copy()
    loss_fn = instantiate(cfg.train.loss)
    loss_args.update({"average": True})
    loss_list_fn = instantiate(loss_args)
    # loss_fn = ModNetLoss(semantic_scale=cfg.training.loss.semantic_scale,
    #                      detail_scale=cfg.training.loss.detail_scale,
    #                      matte_scale=cfg.training.loss.matte_scale, blurer=blurer, average=True)
    # loss_list_fn = ModNetLoss(semantic_scale=cfg.training.loss.semantic_scale,
    #                      detail_scale=cfg.training.loss.detail_scale,
    #                      matte_scale=cfg.training.loss.matte_scale, blurer=blurer, average=True)
    #network = MODNet().to(cfg.training.device)
    network = instantiate(cfg.model)
    network.freeze_backbone()
    if cfg.training.parallel:
        network = nn.DataParallel(network)
    #optimizer = torch.optim.Adam([
    #    {'params': network.backbone.parameters(), 'lr': 1e-4},
    #    {'params': iter([param for name, param in network.named_parameters()
    #                     if 'backbone' not in name]), "lr": 5e-4}
    #])
    #optimizer = torch.optim.AdamW(network.parameters(), lr=5e-4)
    #optimizer = torch.optim.Adam(network.parameters())
    #optimizer = instantiate(cfg.train.optimizer)
    optimizer = OptimizerWrapper(cfg.train.optimizer)([
        e for e in network.parameters() if e.requires_grad
    ])
    lr_scheduler = LRSchedulerWrapper(optimizer=optimizer)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * total_epoch), gamma=0.1)
    #optimizer = torch.optim.Adam([e for e in network.parameters()
    #							  if e.requires_grad])
    #optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * total_epoch), gamma=0.1)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    val_loss_arr = []
    start_time = time.time()
    for step in range(start, cfg.training.total_step):
        #print(f'step {step}')
        try:
            images, mattes_true, foregrounds, backgrounds = next(data_iter)
            #images, mattes_true = next(data_iter)
        except:
            data_iter = iter(dataloader)
            images, mattes_true, foregrounds, backgrounds = next(data_iter)
            #images, mattes_true = next(data_iter)

        images = images.to(cfg.training.device).float()
        mattes_true = mattes_true.to(cfg.training.device).float()
        trimaps_true = generate_trimap_kornia(mattes_true).float()

        if cfg.logging.visual_debug:
            save_image(make_grid(images), str(input_image_save_path / f"{step+1}_input.png"))

        network.train()
        network.freeze_bn()

        semantic_pred, detail_pred, matte_pred = network(images, "train")

        # optimization step
        loss = loss_fn(semantic_pred, detail_pred, matte_pred,
                       mattes_true, trimaps_true, images)

        #print(semantic_loss.item(), detail_loss.item(), matte_loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #lr_scheduler.step(loss.item())

        #if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        #    optimizer.step()
        #    optimizer.zero_grad()

        # logging
        if cfg.logging.verbose > 0:
            if (step + 1) % cfg.logging.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed [{elapsed}], step [{step + 1} / {cfg.training.total_step}], "
                      f"loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(),})
                           # "semantic loss": semantic_loss.item(),
                           # "detail loss": detail_loss.item(),
                           # "matte loss": matte_loss.item()})

        del semantic_pred, detail_pred, matte_pred
        torch.cuda.empty_cache()

        # new validation step

        if (step + 1) % cfg.logging.sample_step == 0:
            network.eval()
            val_loss_list = []
            val_loss = 0
            semantic_val_loss = 0
            detail_val_loss = 0
            matte_val_loss = 0
            val_dataset_size = len(validation_dataloader.dataset)
            images_to_save = []
            with torch.no_grad():
                for images, mattes_true, foregrounds, backgrounds in validation_dataloader:
                    images = images.to(cfg.testing.device).float()
                    mattes_true = mattes_true.to(cfg.testing.device).float()
                    trimaps_true = generate_trimap_kornia(mattes_true).float()

                    semantic_pred, detail_pred, matte_pred = network(images, "train")

                    current_batch_size = len(images)
                    #semantic_loss = semantic_loss * current_batch_size
                    #detail_loss = detail_loss * current_batch_size
                    #matte_loss = matte_loss * current_batch_size

                    loss = loss_fn(semantic_pred, detail_pred, matte_pred,
                                   mattes_true, trimaps_true, images) * current_batch_size
                    val_loss += loss.item()
                    val_loss_list.extend(loss_list_fn(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images).tolist())
                    # semantic_val_loss += semantic_loss.item()
                    # detail_val_loss += detail_loss.item()
                    # matte_val_loss += matte_loss.item()

                    #images_to_save = []
                    for k in range(len(matte_pred)):
                        images_to_save.append(wandb.Image(tensor_to_image(matte_pred[k]), caption="Label"))

                    del semantic_pred, detail_pred, matte_pred,
                    torch.cuda.empty_cache()
                #print(val_loss_list)
                val_loss_arr.extend([[x, y] for (x, y) in zip([step] * len(val_loss_list), val_loss_list)])
                df = pd.DataFrame(data=val_loss_arr, columns=['step', 'error'])
                print(df)
                fig = px.scatter(x=df.step.values, y=df.error.values)
                #print(np.array(val_loss_arr), np.array(val_loss_arr).shape)
                #table = wandb.Table(data=val_loss_arr, columns=['step', 'error'])
                #table = wandb.Table(dataframe=df)
                #wandb.log({"val loss": wandb.plot.scatter(table, "step", "error")})
                wandb.log({'val loss': fig})
                wandb.log({"examples": images_to_save})
                #wandb.log({"val loss": val_loss / val_dataset_size,})
                           # "val semantic loss": semantic_val_loss / val_dataset_size,
                           # "val detail loss": detail_val_loss / val_dataset_size,
                           # "val matte loss": matte_val_loss / val_dataset_size})

        # if (step + 1) % sample_step == 0:
        #     network.eval()
        #     with torch.no_grad():
        #         _, _, mattes_samples = network(images, "test")
        #     #print(mattes_samples.sum())
        #     #save_image(mattes_samples[0:1].data,
        #     #           str(sample_path / f"{step+1}_predict.png"))
        #     #save_image(mattes_true[0:1].data,
        #     #           str(sample_path / f"{step + 1}_true.png"))
        #     im_to_save = tensor_to_image(mattes_samples[0])
        #     print(np.array(im_to_save).shape)
        #     wandb.log({"examples": [wandb.Image(im_to_save, caption="Label")]})
        #     del mattes_samples
        #     torch.cuda.empty_cache()

        if (step + 1) % model_save_step == 0:
            torch.save(network.state_dict(),
                       str(model_save_path / f"{step + 1}_network.pth"))

        #if (step + 1) % step_per_epoch:
            #lr_scheduler.step()


def inference_from_folder(
    data_dir="../data",
    results_dir="../data/results",
    models_dir="../",
    image_size=512,
    version="mobilenetv2",
    total_step=1000000,
    batch_size=30,
    accumulation_steps=4,
    n_workers=8,
    learning_rate=0.0002,
    lr_decay=0.95,
    beta1=0.5,
    beta2=0.999,
    test_size=2824,
    model_name="model.pth",
    pretrained_model=None,
    is_train=False,
    parallel=False,
    use_tensorboard=False,
    image_path="../data/dataset/val/",
    mask_path="../data/dataset/train/seg",
    log_path="./logs",
    model_load_path="./models_adam_23042021",
    inference_path ="./result_submission",
    test_image_path="../data/dataset/val/image",
    test_mask_path="./test_results",
    test_color_mask_path="./test_color_visualize",
    log_step=10,
    sample_step=100,
    model_save_step=1.0,
    device="cuda",
    verbose=1,
    dataset="matting",
    semantic_scale=10.0,
    detail_scale=10.0,
    matte_scale=1.0
):
    inference_path = Path(inference_path)
    model_load_path = Path(model_load_path)
    mkdir_if_empty_or_not_exist(inference_path)
    #mkdir_if_empty_or_not_exist(model_load_path)

    dataloader = MattingTestLoader(image_path, image_size,
                                   batch_size, is_train).loader()
    data_iter = iter(dataloader)

    network = MODNet(backbone_pretrained=False).to(device)
    network.load_state_dict(torch.load(f"{model_load_path}/2393_network.pth"))
    network.eval()
    #network.freeze_bn()

    if verbose > 0:
        print(network)

    start_time = time.time()
    for step in tqdm.tqdm(range(len(dataloader))):
        try:
            images, paths, orig_sizes = next(data_iter)
        except:
            data_iter = iter(dataloader)
            images, paths, orig_sizes = next(data_iter)
        images = images.to(device)
        # semantic loss
        with torch.no_grad():
            _, _, mattes_pred = network(images, "test")
            for k, matte_pred in enumerate(mattes_pred):
                #print('/'.join([str(inference_path)]+paths[k].split('/')[4:-1]))
                Path('/'.join([str(inference_path)]+paths[k].split('/')[4:-1])).mkdir(parents=True, exist_ok=True)
                width, height = orig_sizes[0][k], orig_sizes[1][k]
                save_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((height, width)),
                    transforms.ToTensor()
                ])
                matte_pred = save_transform(matte_pred.data)
                save_image(matte_pred,
                           f"{inference_path}/{'/'.join(paths[k].split('/')[4:])[:-4]}.png")


def main():
    train_from_folder()
    #inference_from_folder()


if __name__ == "__main__":
    main()






















