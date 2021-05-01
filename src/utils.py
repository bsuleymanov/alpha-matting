import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from pathlib import Path
import torch
import torchvision
from PIL import Image
import random
import cv2


def generate_trimap_from_alpha(masks):
    batch_size, height, width = masks.shape
    boundaries = np.zeros_like(masks)
    for i in range(batch_size):
        side = int((height + width) / 2 * 0.05)
        dilated_mask = grey_dilation(masks[i, ...], size=(side, side))
        eroded_mask = grey_erosion(masks[i, ...], size=(side, side))
        boundaries[i, np.where(dilated_mask - eroded_mask != 0)] = 1
    return boundaries

def generate_trimap(mattes):
    batch_size, height, width = mattes.shape
    trimaps = np.zeros_like(mattes)
    for i, matte in enumerate(range(batch_size)):
        kernel_size = random.choice(range(1, 5))
        iterations = random.randint(1, 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        dilated = cv2.dilate(matte, kernel, iterations)
        eroded = cv2.erode(matte, kernel, iterations)
        trimap = np.zeros_like(matte)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0
        trimaps[i] = trimap
    return trimaps


def mkdir_if_empty_or_not_exist(dir_name):
    if (not dir_name.exists() or
        next(dir_name.iterdir(), None) is None):
        Path.mkdir(dir_name, exist_ok=True)
    else:
        raise Exception


def denorm(tensor):
    out = (tensor + 1) / 2
    return out.clamp_(0, 1)


def tensor_to_image(tensor):
    grid = torchvision.utils.make_grid(tensor)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im