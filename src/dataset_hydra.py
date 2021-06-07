import torch
from torchvision import transforms
from PIL import Image
import jpeg4py as jpeg
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import generate_trimap, set_transform
import cv2
import random
from hydra.utils import to_absolute_path



class MaadaaMattingDatasetWOTrimapV2:
    def __init__(self, image_dir, foreground_dir, background_dir,
                 shared_pre_transform, composition_transform=None,
                 foreground_transform=None, background_transform=None,
                 matte_transform=None, shared_post_transform=None,
                 bg_per_fg=10, mode="train", verbose=0):
        self.bg_per_fg = bg_per_fg
        self.foreground_list = list(map(str, Path(foreground_dir).rglob("*_foreground.jpg")))
        self.background_list = list(map(str, Path(background_dir).rglob("*.jpg")))
        self.matte_list = list(map(str, Path(image_dir).rglob("*.png")))
        # hack
        #self.foreground_list = [x[:-4]+"_foreground.jpg" for x in self.matte_list]
        print(len(self.foreground_list), len(self.background_list), len(self.matte_list))

        self.shared_pre_transform = set_transform(shared_pre_transform)
        self.composition_transform = set_transform(composition_transform)
        self.foreground_transform = set_transform(foreground_transform)
        self.background_transform = set_transform(background_transform)
        self.matte_transform = set_transform(matte_transform)
        self.shared_post_transform = set_transform(shared_post_transform)

        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.verbose = verbose

        self.preprocess()

    def preprocess(self, bg_per_fg=10):
        for i in range(len(self.foreground_list)):
            foreground_path = self.foreground_list[i]
            matte_path = self.matte_list[i]
            current_background_paths = random.sample(self.background_list, bg_per_fg)
            for background_path in current_background_paths:
                if self.verbose > 0:
                    print(foreground_path, background_path, matte_path)
                if self.mode == "train":
                    self.train_dataset.append([foreground_path, background_path, matte_path])
                else:
                    self.test_dataset.append([foreground_path, background_path, matte_path])
        random.shuffle(self.train_dataset)
        random.shuffle(self.test_dataset)
        if self.verbose > 0:
            print("Finished preprocessing the MaadaaMatting dataset...")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        foreground_path, background_path, matte_path = dataset[index]
        foreground = jpeg.JPEG(foreground_path).decode()
        background = jpeg.JPEG(background_path).decode()

        matte = Image.open(matte_path)
        if matte.mode == "L":
            matte = np.array(matte)
            matte = np.stack([matte, matte, matte], axis=2)
        elif matte.mode == "RGBA":
            matte = np.array(matte)
            matte = matte[:, :, :3]
        else:
            matte = np.array(matte)

        matte = matte / 255.
        foreground = foreground / 255.
        background = background / 255.

        foreground = self.foreground_transform(image=foreground)#["image"]
        background = self.background_transform(image=background)#["image"]

        height, width = foreground.shape[:2]
        background = cv2.resize(background, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        fg_channels = foreground.shape[2]
        bg_channels = background.shape[2]
        composition = np.zeros((height, width, fg_channels + bg_channels))
        composition[:, :, :fg_channels] = foreground
        composition[:, :, fg_channels:] = background
        #composition = matte * foreground + (1 - matte) * background
        #del foreground, background

        composition, matte = self.shared_pre_transform(image=composition, mask=matte).values()
        #composition = self.composition_transform(image=composition)['image']
        matte = self.matte_transform(image=matte)#['image']
        composition, matte = self.shared_post_transform(image=composition, mask=matte).values()

        foreground = composition[:fg_channels, :, :]
        background = composition[fg_channels:, :, :]
        matte = matte.permute(2, 0, 1)
        #print(matte.shape, foreground.shape, background.shape)
        composition = matte * foreground + (1 - matte) * background

        return (composition, matte, foreground, background)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)