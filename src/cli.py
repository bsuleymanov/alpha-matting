from dataloader import MattingLoader
from utils import mkdir_if_empty_or_not_exist
import time
import datetime
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from model import GaussianBlurLayer, MODNet



def train_from_folder(
    data_dir="../data",
    results_dir="../data/results",
    models_dir="../",
    image_size=512,
    version="mobilenetv2",
    total_step=1000000,
    batch_size=5,
    accumulation_steps=4,
    n_workers=8,
    learning_rate=0.0002,
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
    mask_path="../data/dataset/train/seg",
    log_path="./logs",
    model_save_path="./models",
    sample_path="./samples",
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
    sample_path = Path(sample_path)
    model_save_path = Path(model_save_path)
    mkdir_if_empty_or_not_exist(sample_path)
    mkdir_if_empty_or_not_exist(model_save_path)

    dataloader = MattingLoader(image_path, image_size,
                               batch_size, is_train).loader()
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    model_save_step = int(model_save_step * step_per_epoch)

    if pretrained_model:
        start = pretrained_model + 1
    else:
        start = 0

    blurer = GaussianBlurLayer(1, 3).to(device)
    network = MODNet().to(device)
    if parallel:
        network = nn.DataParallel(network)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, network.parameters()),
        learning_rate, [beta1, beta2]
    )
    if verbose > 0:
        print(network)

    start_time = time.time()
    for step in range(start, total_step):
        try:
            images, trimaps_true, mattes_true = next(data_iter)
        except:
            data_iter = iter(dataloader)
            images, trimaps_true, mattes_true = next(data_iter)
        images = images.to(device)
        trimaps_true = trimaps_true.to(device)
        mattes_true = mattes_true.to(device)

        network.train()
        # semantic loss
        semantics_pred, details_pred, mattes_pred = network(images, "train")
        boundaries = (trimaps_true < 0.5) + (trimaps_true > 0.5)
        semantics_true = F.interpolate(mattes_true, scale_factor=1/16, mode="bilinear")
        semantics_true = blurer(semantics_true)
        #print(semantics_true.size(), semantics_pred.size())
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
        loss.backward()

        if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()
            optimizer.zero_grad()

        # logging
        if verbose > 0:
            if (step + 1) % log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed [{elapsed}], step [{step + 1} / {total_step}], "
                      f"loss: {loss.item():.4f}")


        if (step + 1) % sample_step == 0:
            with torch.no_grad():
                network.eval()
                save_image(mattes_pred[0:1].data,
                           str(sample_path / f"{step+1}_predict.png"))
                save_image(mattes_true[0:1].data,
                           str(sample_path / f"{step + 1}_true.png"))

        if (step + 1) % model_save_step == 0:
            torch.save(network.state_dict(),
                       str(model_save_path / f"{step + 1}_network.pth"))


def main():
    train_from_folder()


if __name__ == "__main__":
    main()






















