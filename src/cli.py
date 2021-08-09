import time
import datetime
import tqdm
import pandas as pd
import plotly.express as px
import gc
import hydra
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.utils import instantiate, to_absolute_path
from pathlib import Path
import wandb

import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from kornia.enhance.normalize import normalize
from kornia.geometry.transform import resize

from image_utils import (generate_trimap_kornia,
                         tensor_to_image,
                         mkdir_if_empty_or_not_exist,
                         denorm)

from distributed_utils import (setup, cleanup,
                               is_main_process, set_seed)


def run_training(rank, world_size, seed, cfg):
    set_seed(seed)
    setup(rank=rank, world_size=world_size)

    print(f"{rank + 1}/{world_size} process initialized.")

    if is_main_process(rank):
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity)

    if is_main_process(rank):
        sample_path = Path(to_absolute_path(cfg.logging.sample_path))
        model_save_path = Path(to_absolute_path(cfg.logging.model_save_path))
        input_image_save_path = Path(to_absolute_path(cfg.logging.input_image_save_path))
        mkdir_if_empty_or_not_exist(sample_path)
        mkdir_if_empty_or_not_exist(model_save_path)
        mkdir_if_empty_or_not_exist(input_image_save_path)

    dataloader = instantiate(cfg.data.train.dataloader,
                             world_size=world_size, rank=rank).loader
    validation_dataloader = instantiate(cfg.data.validation.dataloader,
                                        world_size=world_size, rank=rank).loader
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    total_epoch = cfg.training.total_step / step_per_epoch
    model_save_step = 500

    loss_fn = instantiate(cfg.training.loss, detailed=True, device=rank)
    loss_list_fn = instantiate(
        cfg.training.loss, average=False, device=rank)
    loss_fn_valid = instantiate(
        cfg.training.loss, detailed=True, device=rank)
    print("kek")
    network = instantiate(cfg.network, rank=rank).to(rank)
    print("kek")
    network.freeze_backbone()
    if cfg.training.parallel:
        network = DDP(network, device_ids=[rank], output_device=rank,
                      find_unused_parameters=False)
    network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
    if is_main_process(rank):
        wandb.watch(network)

    optimizer = instantiate(
        cfg.training.optimizer,
        params=[e for e in network.parameters()
                if e.requires_grad])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.25 * total_epoch), gamma=0.1)

    val_loss_arr = []
    start_time = time.time()
    start = 0

    for step in range(start, cfg.training.total_step):
        try:
            (images, mattes_true, foregrounds, backgrounds,
             foregrounds_names, background_names, matte_names) = next(data_iter)
        except:
            dataloader.sampler.set_epoch(step // step_per_epoch)
            data_iter = iter(dataloader)
            (images, mattes_true, foregrounds, backgrounds,
             foregrounds_names, background_names, matte_names) = next(data_iter)

        images = images.to(rank).float()
        images = normalize(images, torch.tensor([0.5, 0.5, 0.5], device=rank),
                                   torch.tensor([0.5, 0.5, 0.5], device=rank))
        mattes_true = mattes_true.to(rank).float()
        trimaps_true, eroded, dilated = generate_trimap_kornia(mattes_true, rank)#.float()

        network.train()
        semantic_pred, detail_pred, matte_pred = network(images, "train")
        (semantic_loss, detail_loss,
         matte_loss, loss) = loss_fn(semantic_pred, detail_pred, matte_pred,
                                     mattes_true, trimaps_true, images)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if cfg.logging.verbose > 0 and is_main_process(rank):
            if (step + 1) % cfg.logging.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed [{elapsed}], step [{step + 1} / {cfg.training.total_step}], "
                      f"loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(),
                           "semantic loss": semantic_loss.item(),
                           "detail loss": detail_loss.item(),
                           "matte loss": matte_loss.item()})

        if (step + 1) % cfg.logging.sample_step == 0:
            del semantic_pred, detail_pred
            torch.cuda.empty_cache()
            if is_main_process(rank):
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

                    images = images.to(rank)
                    images = normalize(images, torch.tensor([0.5, 0.5, 0.5], device=rank),
                                       torch.tensor([0.5, 0.5, 0.5], device=rank)).float()
                    mattes_true = mattes_true.to(rank).float()
                    trimaps_true, eroded, dilated = generate_trimap_kornia(mattes_true, rank)
                    semantic_pred, detail_pred, matte_pred = network(images)
                    current_batch_size = len(images)

                    (semantic_loss, detail_loss,
                     matte_loss, loss) = loss_fn_valid(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images)
                    val_loss += loss.item() * current_batch_size
                    val_loss_list.extend(loss_list_fn(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images).tolist())
                    semantic_val_loss += semantic_loss.item() * current_batch_size
                    detail_val_loss += detail_loss.item() * current_batch_size
                    matte_val_loss += matte_loss.item() * current_batch_size

                    if is_main_process(rank) and len(images_to_save) < (108 - len(matte_pred)):
                        for k in range(len(matte_pred)):
                            images_to_save.append(wandb.Image(tensor_to_image(matte_pred[k]), caption=matte_names[k]))
                    del semantic_pred, detail_pred, matte_pred,
                    torch.cuda.empty_cache()

            if is_main_process(rank):
                val_loss_arr.extend([[x, y] for (x, y) in zip([step] * len(val_loss_list), val_loss_list)])
                df = pd.DataFrame(data=val_loss_arr, columns=['step', 'error'])
                fig = px.scatter(x=df.step.values, y=df.error.values)
                wandb.log({'val loss': fig})
                wandb.log({"examples": images_to_save})
                wandb.log({"val loss total": val_loss / val_dataset_size,
                           "val semantic loss": semantic_val_loss / val_dataset_size,
                           "val detail loss": detail_val_loss / val_dataset_size,
                           "val matte loss": matte_val_loss / val_dataset_size})

        if (step + 1) % model_save_step == 0 and is_main_process(rank):
            torch.save(network.state_dict(),
                       str(model_save_path / f"{step + 1}_network.pth"))

    cleanup()
    wandb.finish()


def run_training_single_device(cfg, rank):

    if is_main_process(rank):
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity)

    if is_main_process(rank):
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
    model_save_step = 500

    print(cfg.training.loss)

    loss_fn = instantiate(cfg.training.loss, detailed=True, device=rank)
    loss_list_fn = instantiate(
        cfg.training.loss, average=False, device=rank)
    loss_fn_valid = instantiate(
        cfg.training.loss, detailed=True, device=rank)
    print("kek")
    network = instantiate(cfg.network, rank=rank).to(rank)
    print("kek")
    network.freeze_backbone()
    if is_main_process(rank):
        wandb.watch(network)

    optimizer = instantiate(
        cfg.training.optimizer,
        params=[e for e in network.parameters()
                if e.requires_grad])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.25 * total_epoch), gamma=0.1)

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

        images = images.to(rank).float()
        images = normalize(images, torch.tensor([0.5, 0.5, 0.5], device=rank),
                                   torch.tensor([0.5, 0.5, 0.5], device=rank))
        mattes_true = mattes_true.to(rank).float()
        trimaps_true, eroded, dilated = generate_trimap_kornia(mattes_true, rank)#.float()

        network.train()
        semantic_pred, detail_pred, matte_pred = network(images, "train")
        (semantic_loss, detail_loss,
         matte_loss, loss) = loss_fn(semantic_pred, detail_pred, matte_pred,
                                     mattes_true, trimaps_true, images)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if cfg.logging.verbose > 0 and is_main_process(rank):
            if (step + 1) % cfg.logging.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed [{elapsed}], step [{step + 1} / {cfg.training.total_step}], "
                      f"loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(),
                           "semantic loss": semantic_loss.item(),
                           "detail loss": detail_loss.item(),
                           "matte loss": matte_loss.item()})

        if (step + 1) % cfg.logging.sample_step == 0:
            del semantic_pred, detail_pred
            torch.cuda.empty_cache()
            if is_main_process(rank):
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

                    images = images.to(rank)
                    images = normalize(images, torch.tensor([0.5, 0.5, 0.5], device=rank),
                                       torch.tensor([0.5, 0.5, 0.5], device=rank)).float()
                    mattes_true = mattes_true.to(rank).float()
                    trimaps_true, eroded, dilated = generate_trimap_kornia(mattes_true, rank)
                    semantic_pred, detail_pred, matte_pred = network(images)
                    current_batch_size = len(images)

                    (semantic_loss, detail_loss,
                     matte_loss, loss) = loss_fn_valid(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images)
                    val_loss += loss.item() * current_batch_size
                    val_loss_list.extend(loss_list_fn(semantic_pred, detail_pred, matte_pred,
                                         mattes_true, trimaps_true, images).tolist())
                    semantic_val_loss += semantic_loss.item() * current_batch_size
                    detail_val_loss += detail_loss.item() * current_batch_size
                    matte_val_loss += matte_loss.item() * current_batch_size

                    if is_main_process(rank) and len(images_to_save) < (108 - len(matte_pred)):
                        for k in range(len(matte_pred)):
                            images_to_save.append(wandb.Image(tensor_to_image(matte_pred[k]), caption=matte_names[k]))
                    del semantic_pred, detail_pred, matte_pred,
                    torch.cuda.empty_cache()
                break

            if is_main_process(rank):
                val_loss_arr.extend([[x, y] for (x, y) in zip([step] * len(val_loss_list), val_loss_list)])
                df = pd.DataFrame(data=val_loss_arr, columns=['step', 'error'])
                fig = px.scatter(x=df.step.values, y=df.error.values)
                wandb.log({'val loss': fig})
                wandb.log({"examples": images_to_save})
                wandb.log({"val loss total": val_loss / val_dataset_size,
                           "val semantic loss": semantic_val_loss / val_dataset_size,
                           "val detail loss": detail_val_loss / val_dataset_size,
                           "val matte loss": matte_val_loss / val_dataset_size})

        if (step + 1) % model_save_step == 0 and is_main_process(rank):
            torch.save(network.state_dict(),
                       str(model_save_path / f"{step + 1}_network.pth"))

    wandb.finish()


def train_from_folder_distributed():
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    global_seed = 228
    initialize(config_path="configs/modnet", job_name="test_app")
    cfg = compose(config_name="config")
    mp.spawn(run_training,
             args=(world_size, global_seed, cfg),
             nprocs=torch.cuda.device_count(),
             join=True)

def train_from_folder():
    global_seed = 228
    initialize(config_path="configs/modnet", job_name="test_app")
    cfg = compose(config_name="config")
    run_training_single_device(cfg, rank=0)

@hydra.main(config_path="configs/modnet", config_name="config")
def inference_from_folder(cfg: DictConfig):
    inference_path = Path(cfg.inference.sample_path)
    model_load_path = Path(cfg.inference.model_load_path)
    mkdir_if_empty_or_not_exist(inference_path)

    dataloader = instantiate(cfg.data.inference.dataloader).loader
    data_iter = iter(dataloader)

    network = instantiate(cfg.network).to(cfg.inference.device)
    #network.load_state_dict(torch.load(model_load_path / cfg.inference.saved_model_name))

    state_dict = torch.load(model_load_path / cfg.inference.saved_model_name)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    network.load_state_dict(new_state_dict)
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
        images = normalize(images, torch.tensor([0.5, 0.5, 0.5], device=cfg.inference.device),
                           torch.tensor([0.5, 0.5, 0.5], device=cfg.inference.device))
        with torch.no_grad():
            _, _, mattes_pred = network(images, "test")
            for k, matte_pred in enumerate(mattes_pred):
                matte_pred = denorm(images[k]) * matte_pred + (1 - mattes_pred)
                if image_names[k] == "me.jpg":
                    matte_pred = resize(matte_pred, (1280, 958))

                save_image(matte_pred,
                           f"{cfg.inference.sample_path}/{image_names[k]}")


def main():
    #train_from_folder()
    #train_from_folder_distributed()
    inference_from_folder()


if __name__ == "__main__":
    main()
