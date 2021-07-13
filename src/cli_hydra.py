#from dataloader import MaadaaMattingLoader, MattingTestLoader, \
#                       MattingLoaderDeprecated, MaadaaMattingLoaderV2
from utils import mkdir_if_empty_or_not_exist, generate_trimap_kornia, \
    generate_trimap
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
import gc

import hydra
from omegaconf import OmegaConf, DictConfig, MISSING
import os, logging
from hydra import initialize, compose
from hydra.utils import instantiate
from dataclasses import dataclass, field
from typing import Any
import torch
#from fvcore.nn import flop_count_table, flop_count_str
#from fvcore.nn.flop_count import FlopCountAnalysis
#from fvcore.nn.parameter_count import parameter_count
import wandb
from pathlib import Path
from utils import mkdir_if_empty_or_not_exist
from hydra.utils import to_absolute_path
from typing import Any, Optional, List

from hydra.utils import to_absolute_path
from kornia.enhance.normalize import denormalize, normalize

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import random
from copy import deepcopy
#from addict import Adict


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def full_step():
    ...

def train_step():
    ...

def validation_step():
    ...

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#@hydra.main(config_path="configs/maadaa/modnet", config_name="full_experiment")
def run_training(rank, world_size, seed, cfg):
    set_seed(seed)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    print(f"{rank + 1}/{world_size} process initialized.")

    if rank == 0:
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity)

    if rank == 0:
        sample_path = Path(to_absolute_path(cfg.logging.sample_path))
        model_save_path = Path(to_absolute_path(cfg.logging.model_save_path))
        input_image_save_path = Path(to_absolute_path(cfg.logging.input_image_save_path))
        mkdir_if_empty_or_not_exist(sample_path)
        mkdir_if_empty_or_not_exist(model_save_path)
        mkdir_if_empty_or_not_exist(input_image_save_path)

    # distributed dataloader
    dataloader_class = instantiate(cfg.data.train.dataloader, world_size=world_size, rank=rank)
    dataloader = dataloader_class.loader
    validation_dataloader = instantiate(cfg.data.validation.dataloader, world_size=world_size, rank=rank).loader
    data_iter = iter(dataloader)
    #data_iter = dataloader
    step_per_epoch = len(dataloader)
    total_epoch = cfg.training.total_step / step_per_epoch
    print(total_epoch)
    model_save_step = 500

    # if cfg.pretrained_model:
    #     start = cfg.pretrained_model + 1
    # else:
    #     start = 0
    loss_fn = instantiate(cfg.training.loss, detailed=True, device=rank)
    print(loss_fn)
    loss_list_fn = instantiate(
        cfg.training.loss, average=False, device=rank)
    loss_fn_valid = instantiate(
        cfg.training.loss, detailed=True, device=rank)

    network = instantiate(cfg.network, rank=rank).to(rank)#.to(cfg.training.device)
    network.freeze_backbone()
    if cfg.training.parallel:
        network = DDP(network, device_ids=[rank], output_device=rank,
                      find_unused_parameters=False)
    network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
    if rank == 0:
        wandb.watch(network)
    #print(dir(network))

    # if cfg.training.parallel:
    #     network = nn.DataParallel(network)

    optimizer = instantiate(
        cfg.training.optimizer,
        params=[e for e in network.parameters()
                if e.requires_grad])
    print(optimizer)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * total_epoch), gamma=0.1)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    val_loss_arr = []
    start_time = time.time()
    start = 0
    for step in range(start, cfg.training.total_step):
        #print(dataloader.sampler)
        #print(dataloader_class.data_sampler)
        try:
            (images, mattes_true, foregrounds, backgrounds,
             foregrounds_names, background_names, matte_names) = next(data_iter)
        except:
            # if rank == 0:
            #     print("------")
            dataloader.sampler.set_epoch(step // step_per_epoch)
            data_iter = iter(dataloader)
            (images, mattes_true, foregrounds, backgrounds,
             foregrounds_names, background_names, matte_names) = next(data_iter)
            #print(f"rank {rank}", foregrounds_names)

        images = images.to(rank).float() #.to(cfg.training.device).float()
        #print(f"Image size: {images.size()}")
        images = normalize(images, torch.tensor([0.5, 0.5, 0.5], device=rank),
                                   torch.tensor([0.5, 0.5, 0.5], device=rank))
        mattes_true = mattes_true.to(rank).float()#.to(cfg.training.device).float()
        trimaps_true, eroded, dilated = generate_trimap_kornia(mattes_true, rank)#.float()
        #print("Eroded min-max:", eroded.min(), eroded.max())
        #print("Dilated min-max:", dilated.min(), dilated.max())
        # trimaps_true = mattes_true.clone().cpu().numpy()
        # eroded = mattes_true.clone().cpu().numpy()
        # dilated = mattes_true.clone().cpu().numpy()
        # for j, matte_true in enumerate(mattes_true):
        #     trimaps_true[j], eroded[j], dilated[j] = generate_trimap(matte_true.cpu().numpy())
        # trimaps_true = torch.cuda.FloatTensor(trimaps_true)
        # eroded = torch.cuda.FloatTensor(eroded)
        # dilated = torch.cuda.FloatTensor(dilated)

        #if cfg.logging.visual_debug:
        #    save_image(make_grid(images), str(input_image_save_path / f"{step+1}_input.png"))

        network.train()
        #network.freeze_bn()
        #print(f"IMAGES", images.size())
        semantic_pred, detail_pred, matte_pred = network(images, "train")
        (semantic_loss, detail_loss,
         matte_loss, loss) = loss_fn(semantic_pred, detail_pred, matte_pred,
                                     mattes_true, trimaps_true, images)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        #    optimizer.step()
        #    optimizer.zero_grad()

        # logging
        if cfg.logging.verbose > 0 and rank == 0:
            if (step + 1) % cfg.logging.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed [{elapsed}], step [{step + 1} / {cfg.training.total_step}], "
                      f"loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(),
                           "semantic loss": semantic_loss.item(),
                           "detail loss": detail_loss.item(),
                           "matte loss": matte_loss.item()})
        ### del semantic_pred, detail_pred
        ### torch.cuda.empty_cache()
        # new validation step
        if (step + 1) % cfg.logging.sample_step == 0:
            del semantic_pred, detail_pred
            torch.cuda.empty_cache()
            if rank == 0:
                train_images_to_save = []
                for k in range(len(images)):
                    train_images_to_save.append(wandb.Image(tensor_to_image(images[k]), caption=matte_names[k]))
                wandb.log({"train pic examples": train_images_to_save})
                del images, train_images_to_save
                train_images_to_save = []
                for k in range(len(matte_pred)):
                    train_images_to_save.append(wandb.Image(tensor_to_image(matte_pred[k]), caption=matte_names[k]))
                wandb.log({"train examples": train_images_to_save})
                del matte_pred, train_images_to_save
                trimaps_to_save = []
                for k in range(len(trimaps_true)):
                    trimaps_to_save.append(wandb.Image(tensor_to_image(trimaps_true[k]), caption=f"Trimap {matte_names[k]}"))
                wandb.log({"trimaps": trimaps_to_save})
                del trimaps_true, trimaps_to_save
                eroded_to_save = []
                for k in range(len(eroded)):
                    eroded_to_save.append(
                        wandb.Image(tensor_to_image(eroded[k]), caption=f"Eroded {matte_names[k]}"))
                wandb.log({"eroded": eroded_to_save})
                del eroded, eroded_to_save
                dilated_to_save = []
                for k in range(len(dilated)):
                    dilated_to_save.append(
                        wandb.Image(tensor_to_image(dilated[k]), caption=f"Dilated {matte_names[k]}"))
                wandb.log({"dilated": dilated_to_save})
                del dilated, dilated_to_save
                train_true_images_to_save = []
                for k in range(len(mattes_true)):
                    train_true_images_to_save.append(wandb.Image(tensor_to_image(mattes_true[k]), caption=matte_names[k]))
                wandb.log({"train true examples": train_true_images_to_save})
                del mattes_true, train_true_images_to_save
                gc.collect()
                torch.cuda.empty_cache()
            network.eval()
            val_loss_list = []
            val_loss = 0
            semantic_val_loss = 0
            detail_val_loss = 0
            matte_val_loss = 0
            val_dataset_size = len(validation_dataloader.dataset)
            images_to_save = []
            for images, mattes_true, foregrounds, backgrounds, \
                foregrounds_names, background_names, matte_names in validation_dataloader:
                with torch.no_grad():

                    images = images.to(rank)#.to(cfg.testing.device).float()
                    images = normalize(images, torch.tensor([0.5, 0.5, 0.5], device=rank),
                                       torch.tensor([0.5, 0.5, 0.5], device=rank)).float()
                    mattes_true = mattes_true.to(rank).float()#.to(cfg.testing.device).float()
                    trimaps_true, eroded, dilated = generate_trimap_kornia(mattes_true, rank)#.float()
                    semantic_pred, detail_pred, matte_pred = network(images)
                    current_batch_size = len(images)
                    #semantic_loss = semantic_loss * current_batch_size
                    #detail_loss = detail_loss * current_batch_size
                    #matte_loss = matte_loss * current_batch_size

                    (semantic_loss, detail_loss,
                     matte_loss, loss) = loss_fn_valid(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images)
                    val_loss += loss.item() * current_batch_size
                    val_loss_list.extend(loss_list_fn(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images).tolist())
                    semantic_val_loss += semantic_loss.item() * current_batch_size
                    detail_val_loss += detail_loss.item() * current_batch_size
                    matte_val_loss += matte_loss.item() * current_batch_size
                    #images_to_save = []
                    if rank == 0 and len(images_to_save) < (108 - len(matte_pred)):
                        for k in range(len(matte_pred)):
                            images_to_save.append(wandb.Image(tensor_to_image(matte_pred[k]), caption=matte_names[k]))
                    del semantic_pred, detail_pred, matte_pred,
                    torch.cuda.empty_cache()
                #print(val_loss_list)
            if rank == 0:
                val_loss_arr.extend([[x, y] for (x, y) in zip([step] * len(val_loss_list), val_loss_list)])
                df = pd.DataFrame(data=val_loss_arr, columns=['step', 'error'])
                #print(df)

                fig = px.scatter(x=df.step.values, y=df.error.values)
                #print(np.array(val_loss_arr), np.array(val_loss_arr).shape)
                #table = wandb.Table(data=val_loss_arr, columns=['step', 'error'])
                #table = wandb.Table(dataframe=df)
                #wandb.log({"val loss": wandb.plot.scatter(table, "step", "error")})
                wandb.log({'val loss': fig})
                #wandb.log({'val loss scalar': val_loss / val_dataset_size})
                wandb.log({"examples": images_to_save})
                #print(len(images_to_save))
                wandb.log({"val loss total": val_loss / val_dataset_size,
                           "val semantic loss": semantic_val_loss / val_dataset_size,
                           "val detail loss": detail_val_loss / val_dataset_size,
                           "val matte loss": matte_val_loss / val_dataset_size})

        if (step + 1) % model_save_step == 0 and rank == 0:
            torch.save(network.state_dict(),
                       str(model_save_path / f"{step + 1}_network.pth"))
        #if (step + 1) % step_per_epoch:
        #    lr_scheduler.step()
    cleanup()
    wandb.finish()
    #return

#@hydra.main(config_path="configs/maadaa/modnet", config_name="full_experiment")
def train_from_folder_distributed():
    #cfg = train_from_folder_distributed_subfunc()
    #print(cfg)
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    global_seed = 228
    initialize(config_path="configs/maadaa/modnet", job_name="test_app")
    cfg = compose(config_name="full_experiment")
    mp.spawn(run_training,
             args=(world_size, global_seed, cfg),
             nprocs=torch.cuda.device_count(),
             join=True)


#@hydra.main(config_path="configs/maadaa/modnet", config_name="full_experiment")
def train_from_folder_distributed_subfunc():
    #print(OmegaConf.to_yaml(cfg))
    #cfg_copy = deepcopy(cfg)
    initialize(config_path="configs/maadaa/modnet", job_name="test_app")
    cfg = compose(config_name="full_experiment")
    print(OmegaConf.to_yaml(cfg))
    return cfg
    #train_from_folder_distributed_subfunc(cfg)
    # world_size = torch.cuda.device_count()
    # global_seed = 228
    # mp.spawn(run_training,
    #          args=(world_size, global_seed, cfg),
    #          nprocs=torch.cuda.device_count(),
    #          join=True)
    # print("kek spawn 353")

@hydra.main(config_path="configs/maadaa/modnet", config_name="full_experiment")
def train_from_folder(cfg: DictConfig):
    wandb.init(project=cfg.logging.project, entity=cfg.logging.entity)
    sample_path = Path(to_absolute_path(cfg.logging.sample_path))
    model_save_path = Path(to_absolute_path(cfg.logging.model_save_path))
    input_image_save_path = Path(to_absolute_path(cfg.logging.input_image_save_path))
    mkdir_if_empty_or_not_exist(sample_path)
    mkdir_if_empty_or_not_exist(model_save_path)
    mkdir_if_empty_or_not_exist(input_image_save_path)

    dataloader = instantiate(cfg.data.train.dataloader).loader
    validation_dataloader = instantiate(cfg.data.validation.dataloader).loader
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    total_epoch = cfg.training.total_step / step_per_epoch
    print(total_epoch)
    model_save_step = 500

    # if cfg.pretrained_model:
    #     start = cfg.pretrained_model + 1
    # else:
    #     start = 0
    loss_fn = instantiate(cfg.training.loss, detailed=True)
    print(loss_fn)
    loss_list_fn = instantiate(
        cfg.training.loss, average=False,)
    loss_fn_valid = instantiate(
        cfg.training.loss, detailed=True,)

    network = instantiate(cfg.network).to(cfg.training.device)
    # if cfg.training.multi_gpu:
    #     network = DDP(network)
    network.freeze_backbone()
    if cfg.training.parallel:
        network = nn.DataParallel(network)

    optimizer = instantiate(
        cfg.training.optimizer,
        params=[e for e in network.parameters()
                if e.requires_grad])
    print(optimizer)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * total_epoch), gamma=0.1)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    val_loss_arr = []
    start_time = time.time()
    start = 0
    for step in range(start, cfg.training.total_step):
        try:
            (images, mattes_true, foregrounds, backgrounds,
             foregrounds_names, background_names, matte_names) = next(data_iter)
        except:
            data_iter = iter(dataloader)
            (images, mattes_true, foregrounds, backgrounds,
             foregrounds_names, background_names, matte_names) = next(data_iter)

        images = images.to(cfg.training.device).float()
        images = normalize(images, torch.cuda.FloatTensor([0.5, 0.5, 0.5]), torch.cuda.FloatTensor([0.5, 0.5, 0.5]))
        mattes_true = mattes_true.to(cfg.training.device).float()
        trimaps_true, eroded, dilated = generate_trimap_kornia(mattes_true)#.float()
        #print("Eroded min-max:", eroded.min(), eroded.max())
        #print("Dilated min-max:", dilated.min(), dilated.max())
        # trimaps_true = mattes_true.clone().cpu().numpy()
        # eroded = mattes_true.clone().cpu().numpy()
        # dilated = mattes_true.clone().cpu().numpy()
        # for j, matte_true in enumerate(mattes_true):
        #     trimaps_true[j], eroded[j], dilated[j] = generate_trimap(matte_true.cpu().numpy())
        # trimaps_true = torch.cuda.FloatTensor(trimaps_true)
        # eroded = torch.cuda.FloatTensor(eroded)
        # dilated = torch.cuda.FloatTensor(dilated)

        if cfg.logging.visual_debug:
            save_image(make_grid(images), str(input_image_save_path / f"{step+1}_input.png"))

        network.train()
        network.freeze_bn()
        #print(f"IMAGES", images.size())
        semantic_pred, detail_pred, matte_pred = network(images, "train")
        (semantic_loss, detail_loss,
         matte_loss, loss) = loss_fn(semantic_pred, detail_pred, matte_pred,
                                     mattes_true, trimaps_true, images)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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
                wandb.log({"loss": loss.item(),
                           "semantic loss": semantic_loss.item(),
                           "detail loss": detail_loss.item(),
                           "matte loss": matte_loss.item()})

        del semantic_pred, detail_pred

        torch.cuda.empty_cache()

        # new validation step

        if (step + 1) % cfg.logging.sample_step == 0:

            train_images_to_save = []
            for k in range(len(images)):
                train_images_to_save.append(wandb.Image(tensor_to_image(images[k]), caption=matte_names[k]))
            wandb.log({"train pic examples": train_images_to_save})
            del images, train_images_to_save

            train_images_to_save = []
            for k in range(len(matte_pred)):
                train_images_to_save.append(wandb.Image(tensor_to_image(matte_pred[k]), caption=matte_names[k]))
            wandb.log({"train examples": train_images_to_save})
            del matte_pred, train_images_to_save

            trimaps_to_save = []
            for k in range(len(trimaps_true)):
                trimaps_to_save.append(wandb.Image(tensor_to_image(trimaps_true[k]), caption=f"Trimap {matte_names[k]}"))
            wandb.log({"trimaps": trimaps_to_save})
            del trimaps_true, trimaps_to_save

            eroded_to_save = []
            for k in range(len(eroded)):
                eroded_to_save.append(
                    wandb.Image(tensor_to_image(eroded[k]), caption=f"Eroded {matte_names[k]}"))
            wandb.log({"eroded": eroded_to_save})
            del eroded, eroded_to_save

            dilated_to_save = []
            for k in range(len(dilated)):
                dilated_to_save.append(
                    wandb.Image(tensor_to_image(dilated[k]), caption=f"Dilated {matte_names[k]}"))
            wandb.log({"dilated": dilated_to_save})
            del dilated, dilated_to_save

            train_true_images_to_save = []
            for k in range(len(mattes_true)):
                train_true_images_to_save.append(wandb.Image(tensor_to_image(mattes_true[k]), caption=matte_names[k]))
            wandb.log({"train true examples": train_true_images_to_save})
            del mattes_true, train_true_images_to_save

            gc.collect()
            torch.cuda.empty_cache()

            network.eval()
            val_loss_list = []
            val_loss = 0
            semantic_val_loss = 0
            detail_val_loss = 0
            matte_val_loss = 0
            val_dataset_size = len(validation_dataloader.dataset)
            images_to_save = []
            with torch.no_grad():
                for images, mattes_true, foregrounds, backgrounds, \
                    foregrounds_names, background_names, matte_names in validation_dataloader:

                    images = images.to(cfg.testing.device).float()
                    images = normalize(images, torch.cuda.FloatTensor([0.5, 0.5, 0.5]),
                                       torch.cuda.FloatTensor([0.5, 0.5, 0.5]))
                    mattes_true = mattes_true.to(cfg.testing.device).float()
                    trimaps_true, eroded, dilated = generate_trimap_kornia(mattes_true)#.float()

                    semantic_pred, detail_pred, matte_pred = network(images, "train")

                    current_batch_size = len(images)
                    #semantic_loss = semantic_loss * current_batch_size
                    #detail_loss = detail_loss * current_batch_size
                    #matte_loss = matte_loss * current_batch_size

                    (semantic_loss, detail_loss,
                     matte_loss, loss) = loss_fn_valid(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images)
                    val_loss += loss.item() * current_batch_size
                    val_loss_list.extend(loss_list_fn(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images).tolist())
                    semantic_val_loss += semantic_loss.item() * current_batch_size
                    detail_val_loss += detail_loss.item() * current_batch_size
                    matte_val_loss += matte_loss.item() * current_batch_size

                    #images_to_save = []
                    for k in range(len(matte_pred)):
                        images_to_save.append(wandb.Image(tensor_to_image(matte_pred[k]), caption=matte_names[k]))

                    del semantic_pred, detail_pred, matte_pred,
                    torch.cuda.empty_cache()
                #print(val_loss_list)
                val_loss_arr.extend([[x, y] for (x, y) in zip([step] * len(val_loss_list), val_loss_list)])
                df = pd.DataFrame(data=val_loss_arr, columns=['step', 'error'])
                #print(df)

                fig = px.scatter(x=df.step.values, y=df.error.values)
                #print(np.array(val_loss_arr), np.array(val_loss_arr).shape)
                #table = wandb.Table(data=val_loss_arr, columns=['step', 'error'])
                #table = wandb.Table(dataframe=df)
                #wandb.log({"val loss": wandb.plot.scatter(table, "step", "error")})
                wandb.log({'val loss': fig})
                #wandb.log({'val loss scalar': val_loss / val_dataset_size})
                wandb.log({"examples": images_to_save})
                #print(len(images_to_save))
                wandb.log({"val loss total": val_loss / val_dataset_size,
                           "val semantic loss": semantic_val_loss / val_dataset_size,
                           "val detail loss": detail_val_loss / val_dataset_size,
                           "val matte loss": matte_val_loss / val_dataset_size})

        # if (step + 1) % cfg.logging.sample_step == 0:
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

        if (step + 1) % step_per_epoch:
            lr_scheduler.step()


@hydra.main(config_path="configs/maadaa/modnet", config_name="full_experiment")
def inference_from_folder(cfg: DictConfig):
    inference_path = Path(cfg.inference.sample_path)
    model_load_path = Path(cfg.inference.model_load_path)
    mkdir_if_empty_or_not_exist(inference_path)

    dataloader = instantiate(cfg.data.inference.dataloader).loader
    data_iter = iter(dataloader)

    network = instantiate(cfg.network).to(cfg.inference.device)
    network.load_state_dict(torch.load(model_load_path / cfg.inference.saved_model_name))
    network.eval()

    if cfg.inference.verbose > 0:
        print(network)

    start_time = time.time()
    for step in tqdm.tqdm(range(len(dataloader))):
        try:
            (images, image_names) = next(data_iter)
        except:
            data_iter = iter(dataloader)
            (images, image_names) = next(data_iter)
        images = images.to(cfg.inference.device).float()
        # semantic loss
        with torch.no_grad():
            _, _, mattes_pred = network(images, "test")
            for k, matte_pred in enumerate(mattes_pred):
                save_image(tensor_to_image(matte_pred),
                           f"{cfg.inference.sample_path}/{image_names[k]}")


def main():
    train_from_folder_distributed()
    #train_from_folder()
    #inference_from_folder()


if __name__ == "__main__":
    main()






















