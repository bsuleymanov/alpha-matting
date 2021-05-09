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



def train_from_folder(
    data_dir="../data",
    results_dir="../data/results",
    models_dir="../",
    image_size=256,
    version="mobilenetv2",
    total_step=150000,
    batch_size=3,
    accumulation_steps=1,
    n_workers=8,
    learning_rate=0.01,
    lr_decay=0.95,
    beta1=0.5,
    beta2=0.999,
    test_size=2824,
    model_name="model.pth",
    pretrained_model=None,
    is_train=True,
    parallel=False,
    use_tensorboard=False,
    image_path="../data/dataset/train/",
    foreground_path="../data/foregrounds/train/",
    background_path="../data/backgrounds/",
    mask_path="../data/dataset/train/seg",
    log_path="./logs",
    model_save_path="./models",
    sample_path="./sample_path",
    input_image_save_path="./input_save",
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
    matte_scale=1.0,
    visual_debug=False
):
    wandb.init(project="alpha-matting",  entity='bsuleymanov')#, settings=wandb.Settings(start_method="fork"))

    config = wandb.config

    mode = "train" if is_train else "test"
    sample_path = Path(sample_path)
    model_save_path = Path(model_save_path)
    input_image_save_path = Path(input_image_save_path)
    mkdir_if_empty_or_not_exist(sample_path)
    mkdir_if_empty_or_not_exist(model_save_path)
    mkdir_if_empty_or_not_exist(input_image_save_path)

    dataloader = MaadaaMattingLoaderV2(image_path, foreground_path,
                                       background_path, image_size,
                                       batch_size, mode).loader()
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    total_epoch = total_step / step_per_epoch
    print(total_epoch)
    model_save_step = 500

    if pretrained_model:
        start = pretrained_model + 1
    else:
        start = 0

    blurer = GaussianBlurLayer(3, 3).to(device)
    network = MODNet().to(device)
    network.freeze_backbone()
    if parallel:
        network = nn.DataParallel(network)
    #optimizer = torch.optim.Adam([
    #    {'params': network.backbone.parameters(), 'lr': 1e-4},
    #    {'params': iter([param for name, param in network.named_parameters()
    #                     if 'backbone' not in name]), "lr": 5e-4}
    #])
    #optimizer = torch.optim.AdamW(network.parameters(), lr=5e-4)
    #optimizer = torch.optim.Adam(network.parameters())
    optimizer = torch.optim.Adam([e for e in network.parameters()
    							  if e.requires_grad])
    #optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * total_epoch), gamma=0.1)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    start_time = time.time()
    for step in range(start, total_step):
        #print(f'step {step}')
        try:
            images, mattes_true, foregrounds, backgrounds = next(data_iter)
        except:
            data_iter = iter(dataloader)
            images, mattes_true, foregrounds, backgrounds = next(data_iter)

        images = images.to(device).float()
        mattes_true = mattes_true.to(device).float()
        trimaps_true = generate_trimap_kornia(mattes_true).float()

        if visual_debug:
            save_image(make_grid(images), str(input_image_save_path / f"{step+1}_input.png"))

        network.train()
        network.freeze_bn()
        # semantic loss
        semantics_pred, details_pred, mattes_pred = network(images, "train")
        boundaries = (trimaps_true == 0) + (trimaps_true == 1)
        semantics_true = F.interpolate(mattes_true, scale_factor=1/16, mode="bilinear")
        semantics_true = blurer(semantics_true)
        semantic_loss = torch.mean(F.mse_loss(semantics_pred, semantics_true))
        semantic_loss = semantic_scale * semantic_loss

        # detail loss
        boundary_detail_pred = torch.where(boundaries, trimaps_true, details_pred)
        details_true = torch.where(boundaries, trimaps_true, mattes_true)
        detail_loss = torch.mean(F.l1_loss(boundary_detail_pred, details_true))
        detail_loss = detail_scale * detail_loss

        # matte loss
        boundary_mattes_pred = torch.where(boundaries, trimaps_true, mattes_pred)
        matte_l1_loss = (F.l1_loss(mattes_pred, mattes_true) +
                         4.0 * F.l1_loss(boundary_mattes_pred, mattes_true))
        matte_compositional_loss = (F.l1_loss(images * mattes_pred, images * mattes_true) +
                                    4.0 * F.l1_loss(images * boundary_mattes_pred, images * mattes_true))
        matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
        matte_loss = matte_scale * matte_loss

        # optimization step
        loss = semantic_loss + detail_loss + matte_loss

        #print(semantic_loss.item(), detail_loss.item(), matte_loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #lr_scheduler.step(loss.item())

        #if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        #    optimizer.step()
        #    optimizer.zero_grad()

        # logging
        if verbose > 0:
            if (step + 1) % log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed [{elapsed}], step [{step + 1} / {total_step}], "
                      f"loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(),
                           "semantic loss": semantic_loss.item(),
                           "detail loss": detail_loss.item(),
                           "matte loss": matte_loss.item()})

        del semantics_pred, details_pred, mattes_pred, \
            semantics_true, boundary_detail_pred, details_true, \
            boundary_mattes_pred
        torch.cuda.empty_cache()

        if (step + 1) % sample_step == 0:
            network.eval()
            with torch.no_grad():
                _, _, mattes_samples = network(images, "test")
            #print(mattes_samples.sum())
            #save_image(mattes_samples[0:1].data,
            #           str(sample_path / f"{step+1}_predict.png"))
            #save_image(mattes_true[0:1].data,
            #           str(sample_path / f"{step + 1}_true.png"))
            im_to_save = tensor_to_image(mattes_samples[0])
            print(np.array(im_to_save).shape)
            wandb.log({"examples": [wandb.Image(im_to_save, caption="Label")]})
            del mattes_samples
            torch.cuda.empty_cache()

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






















