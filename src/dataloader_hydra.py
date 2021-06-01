import torch
from torchvision import transforms
from PIL import Image
import jpeg4py as jpeg
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor
from utils import generate_trimap
import cv2
import random
from hydra.utils import to_absolute_path



class MattingDatasetDeprecated:
    def __init__(self, image_dir, image_transform, trimap_transform,
                 matte_transform, mode, verbose=0):
        self.image_list = list(map(str, Path(image_dir).rglob("*.jpg")))
        self.trimap_list = list(map(str, Path(image_dir).rglob("*trimap_true.png")))
        self.matte_list = list(map(str, Path(image_dir).rglob("*.png")))
        self.matte_list = [x for x in self.matte_list if "trimap" not in x]
        # hack
        self.image_list = [x[:-4]+".jpg" for x in self.matte_list]
        self.trimap_list = [x[:-4]+"_trimap_true.png" for x in self.matte_list]
        print(self.trimap_list[0])
        print(len(self.image_list), len(self.trimap_list), len(self.matte_list))
        self.image_transform = image_transform
        self.trimap_transform = trimap_transform
        self.matte_transform = matte_transform
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.verbose = verbose

        self.preprocess()

        if mode == "train":
            self.n_images = len(self.train_dataset)
        else:
            self.n_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len(self.image_list)):
            image_path = self.image_list[i]
            trimap_path = self.trimap_list[i]
            matte_path = self.matte_list[i]
            if self.verbose > 0:
                print(image_path, trimap_path, matte_path)
            if self.mode == "train":
                self.train_dataset.append([image_path, trimap_path, matte_path])
            else:
                self.test_dataset.append([image_path, trimap_path, matte_path])

        if self.verbose > 0:
            print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        image_path, trimap_path, matte_path = dataset[index]
        #image = jpeg.JPEG(image_path).decode()
        image = Image.open(image_path)
        trimap = Image.open(trimap_path)
        #trimap = np.array(trimap)
        #if len(trimap) == 3:
        #    print(trimap.shape)
        #    trimap = trimap[:, :, 0]
        #trimap = Image.fromarray(trimap)

        matte = Image.open(matte_path)
        if matte.mode == "L":
            matte = np.array(matte)
            #print(matte.shape)
            #print(matte.shape)
            matte = np.stack([matte, matte, matte], axis=2)
            matte = Image.fromarray(matte, mode="RGB")
        elif matte.mode == "RGBA":
            matte = np.array(matte)
            #print(matte.shape)
            #print(matte.shape)
            matte = matte[:, :, :3]
            matte = Image.fromarray(matte, mode="RGB")

        return (self.image_transform(image), self.trimap_transform(trimap),
                self.matte_transform(matte))

    def __len__(self):
        return self.n_images


class MattingTestDataset:
    def __init__(self, image_dir, image_transform, verbose=0):
        self.image_list = list(map(str, Path(image_dir).rglob("*.jpg")))
        print(len(self.image_list))
        self.image_transform = image_transform
        self.test_dataset = []
        self.verbose = verbose

        self.preprocess()

        self.n_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len(self.image_list)):
            image_path = self.image_list[i]
            if self.verbose > 0:
                print(image_path)
            self.test_dataset.append(image_path)

        if self.verbose > 0:
            print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        dataset = self.test_dataset
        image_path = dataset[index]
        #image = jpeg.JPEG(image_path).decode()
        image = Image.open(image_path)
        size = image.size

        return self.image_transform(image), image_path, size

    def __len__(self):
        return self.n_images


class MattingLoaderDeprecated:
    def __init__(self, image_path, image_size, batch_size, mode):
        self.image_dir = Path(image_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode

    def transform(self):
        image_transform = transforms.Compose([
            #transforms.Resize((1024, self.image_size)),
            transforms.Resize((self.image_size, self.image_size)),
            #transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trimap_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        matte_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        return image_transform, trimap_transform, matte_transform

    def loader(self):
        image_transform, trimap_transform, matte_transform = self.transform()
        dataset = MattingDatasetDeprecated(self.image_dir, image_transform, trimap_transform,
                                 matte_transform, self.mode)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=True,
            num_workers=4, drop_last=False)
        return data_loader


class MattingTestLoader:
    def __init__(self, image_path, image_size, batch_size, mode):
        self.image_dir = Path(image_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode

    def transform(self):
        image_transform = transforms.Compose([
            transforms.Resize((1024, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # trimap_transform = transforms.Compose([
        #     transforms.Resize((self.image_size, self.image_size)),
        #     transforms.ToTensor(),
        # ])
        # matte_transform = transforms.Compose([
        #     transforms.Resize((self.image_size, self.image_size)),
        #     transforms.ToTensor(),
        # ])

        return image_transform#, trimap_transform, matte_transform

    def loader(self):
        image_transform = self.transform()
        dataset = MattingTestDataset(self.image_dir, image_transform, self.mode)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=False,
            num_workers=8, drop_last=False)
        return data_loader



class MaadaaMattingDataset:
    def __init__(self, image_dir, shared_transform,  image_transform,
                 matte_transform=None, trimap_transform=None, mode="train",
                 verbose=0):
        self.image_list = list(map(str, Path(image_dir).rglob("*.jpg")))
        self.matte_list = list(map(str, Path(image_dir).rglob("*.png")))
        # hack
        self.image_list = [x[:-4]+".jpg" for x in self.matte_list]
        print(len(self.image_list), len(self.matte_list))
        self.image_transform = image_transform
        self.shared_transform = shared_transform
        self.matte_transform = matte_transform
        self.trimap_transform = trimap_transform
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.verbose = verbose

        self.preprocess()

        if mode == "train":
            self.n_images = len(self.train_dataset)
        else:
            self.n_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len(self.image_list)):
            image_path = self.image_list[i]
            matte_path = self.matte_list[i]
            if self.verbose > 0:
                print(image_path, matte_path)
            if self.mode == "train":
                self.train_dataset.append([image_path, matte_path])
            else:
                self.test_dataset.append([image_path, matte_path])

        if self.verbose > 0:
            print("Finished preprocessing the MaadaaMatting dataset...")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        image_path, matte_path = dataset[index]
        image = jpeg.JPEG(image_path).decode()
        #image = np.array(Image.open(image_path))

        matte = Image.open(matte_path)
        if matte.mode == "L":
            matte = np.array(matte)
            matte = np.stack([matte, matte, matte], axis=2)
            #matte = Image.fromarray(matte, mode="RGB")
        elif matte.mode == "RGBA":
            matte = np.array(matte)
            matte = matte[:, :, :3]
            #matte = Image.fromarray(matte, mode="RGB")
        else:
            matte = np.array(matte)


        #image = image / 255.
        matte = matte / 255.

        image, matte = self.shared_transform(image=image, mask=matte).values()
        trimap = generate_trimap(matte)
        trimap = trimap / 255.

        image = self.image_transform(image=image)['image']
        matte = self.matte_transform(image=matte)['image']
        trimap = self.trimap_transform(image=trimap)['image']

        return (image, trimap, matte)

    def __len__(self):
        return self.n_images


class MaadaaMattingDatasetWOTrimap:
    def __init__(self, image_dir, shared_transform,  image_transform,
                 matte_transform=None, trimap_transform=None, mode="train",
                 verbose=0):
        self.image_list = list(map(str, Path(image_dir).rglob("*.jpg")))
        self.matte_list = list(map(str, Path(image_dir).rglob("*.png")))
        # hack
        self.image_list = [x[:-4]+".jpg" for x in self.matte_list]
        print(len(self.image_list), len(self.matte_list))
        self.image_transform = image_transform
        self.shared_transform = shared_transform
        self.matte_transform = matte_transform
        self.trimap_transform = trimap_transform
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.verbose = verbose

        self.preprocess()

        if mode == "train":
            self.n_images = len(self.train_dataset)
        else:
            self.n_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len(self.image_list)):
            image_path = self.image_list[i]
            matte_path = self.matte_list[i]
            if self.verbose > 0:
                print(image_path, matte_path)
            if self.mode == "train":
                self.train_dataset.append([image_path, matte_path])
            else:
                self.test_dataset.append([image_path, matte_path])

        if self.verbose > 0:
            print("Finished preprocessing the MaadaaMatting dataset...")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        image_path, matte_path = dataset[index]
        image = jpeg.JPEG(image_path).decode()
        #image = np.array(Image.open(image_path))

        matte = Image.open(matte_path)
        if matte.mode == "L":
            matte = np.array(matte)
            matte = np.stack([matte, matte, matte], axis=2)
            #matte = Image.fromarray(matte, mode="RGB")
        elif matte.mode == "RGBA":
            matte = np.array(matte)
            matte = matte[:, :, :3]
            #matte = Image.fromarray(matte, mode="RGB")
        else:
            matte = np.array(matte)


        #image = image / 255.
        matte = matte / 255.

        image, matte = self.shared_transform(image=image, mask=matte).values()

        image = self.image_transform(image=image)['image']
        matte = self.matte_transform(image=matte)['image']

        return (image, matte)

    def __len__(self):
        return self.n_images


class MaadaaMattingLoader:
    def __init__(self, image_path, image_size, batch_size, mode):
        self.image_dir = Path(image_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode

    def transform(self):
        shared_transform = A.Compose([
            A.SmallestMaxSize(self.image_size),
            #A.Resize(self.image_size, self.image_size)
            A.RandomCrop(self.image_size, self.image_size)
        ])
        image_transform = A.Compose([
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
            # ToTensor(normalize={
            #     "mean": (0.5, 0.5, 0.5),
            #     "std": (0.5, 0.5, 0.5)
            # })
        ])
        trimap_transform = A.Compose([
            A.Normalize((0., 0., 0.), (1., 1., 1.)),
            ToTensorV2(),
        ])
        matte_transform = A.Compose([
            #A.Normalize((0., 0., 0.), (1., 1., 1.)),
            ToTensorV2(),
        ])

        return shared_transform, image_transform, trimap_transform, matte_transform

    def loader(self):
        shared_transform, image_transform, trimap_transform, matte_transform = self.transform()
        dataset = MaadaaMattingDatasetWOTrimap(self.image_dir, shared_transform, image_transform,
                                               matte_transform, trimap_transform, self.mode)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=True,
            num_workers=32, drop_last=False)
        return data_loader


class MaadaaMattingDatasetWOTrimap:
    def __init__(self, image_dir, shared_transform,  image_transform,
                 matte_transform=None, trimap_transform=None, mode="train",
                 verbose=0):
        #self.image_list = list(map(str, Path(image_dir).rglob("*.jpg")))
        self.matte_list = list(map(str, Path(image_dir).rglob("*.png")))
        # hack
        self.image_list = [x[:-4]+".jpg" for x in self.matte_list]

        print(len(self.image_list), len(self.matte_list))
        self.image_transform = image_transform
        self.shared_transform = shared_transform
        self.matte_transform = matte_transform
        self.trimap_transform = trimap_transform
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.verbose = verbose

        self.preprocess()

        if mode == "train":
            self.n_images = len(self.train_dataset)
        else:
            self.n_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len(self.image_list)):
            image_path = self.image_list[i]
            matte_path = self.matte_list[i]
            if self.verbose > 0:
                print(image_path, matte_path)
            if self.mode == "train":
                self.train_dataset.append([image_path, matte_path])
            else:
                self.test_dataset.append([image_path, matte_path])

        if self.verbose > 0:
            print("Finished preprocessing the MaadaaMatting dataset...")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        image_path, matte_path = dataset[index]
        image = jpeg.JPEG(image_path).decode()
        #image = np.array(Image.open(image_path))

        matte = Image.open(matte_path)
        if matte.mode == "L":
            matte = np.array(matte)
            matte = np.stack([matte, matte, matte], axis=2)
            #matte = Image.fromarray(matte, mode="RGB")
        elif matte.mode == "RGBA":
            matte = np.array(matte)
            matte = matte[:, :, :3]
            #matte = Image.fromarray(matte, mode="RGB")
        else:
            matte = np.array(matte)


        #image = image / 255.
        matte = matte / 255.

        image, matte = self.shared_transform(image=image, mask=matte).values()

        image = self.image_transform(image=image)['image']
        matte = self.matte_transform(image=matte)['image']

        return (image, matte)

    def __len__(self):
        return self.n_images


class MaadaaMattingLoader:
    def __init__(self, image_path, image_size, batch_size, mode):
        self.image_dir = Path(image_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode

    def transform(self):
        shared_transform = A.Compose([
            A.SmallestMaxSize(self.image_size),
            #A.Resize(self.image_size, self.image_size)
            A.RandomCrop(self.image_size, self.image_size)
        ])
        image_transform = A.Compose([
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
            # ToTensor(normalize={
            #     "mean": (0.5, 0.5, 0.5),
            #     "std": (0.5, 0.5, 0.5)
            # })
        ])
        trimap_transform = A.Compose([
            A.Normalize((0., 0., 0.), (1., 1., 1.)),
            ToTensorV2(),
        ])
        matte_transform = A.Compose([
            #A.Normalize((0., 0., 0.), (1., 1., 1.)),
            ToTensorV2(),
        ])

        return shared_transform, image_transform, trimap_transform, matte_transform

    def loader(self):
        shared_transform, image_transform, trimap_transform, matte_transform = self.transform()
        dataset = MaadaaMattingDatasetWOTrimap(self.image_dir, shared_transform, image_transform,
                                               matte_transform, trimap_transform, self.mode)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=True,
            num_workers=8, drop_last=True)
        return data_loader


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

        self.shared_pre_transform = shared_pre_transform
        self.composition_transform = composition_transform
        self.foreground_transform = foreground_transform
        self.background_transform = background_transform
        self.matte_transform = matte_transform
        self.shared_post_transform = shared_post_transform

        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.verbose = verbose

        self.preprocess()

        if mode == "train":
            self.n_images = len(self.train_dataset)
        else:
            self.n_images = len(self.test_dataset)

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
        return self.n_images


class MaadaaMattingLoaderV2:
    def __init__(self, image_path, foreground_path, background_path,
                 image_size, batch_size, mode):
        self.image_dir = Path(to_absolute_path(image_path))
        self.foreground_dir = Path(to_absolute_path(foreground_path))
        self.background_dir = Path(to_absolute_path(background_path))
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode

    def transform(self):
        shared_pre_transform = A.Compose([
            A.SmallestMaxSize(self.image_size),
            A.RandomCrop(self.image_size, self.image_size)
        ])
        composition_transform = A.Compose([
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        #foreground_transform = A.Compose([
        #    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #])
        #background_transform = A.Compose([
        #    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #])
        foreground_transform = lambda image: image
        background_transform = lambda image: image
        matte_transform = lambda image: image#A.Compose([])
        shared_post_transform = A.Compose([
            ToTensorV2(),
        ])

        return (shared_pre_transform, composition_transform, foreground_transform,
                background_transform, matte_transform, shared_post_transform)

    def loader(self):
        (shared_pre_transform, composition_transform,
         foreground_transform, background_transform,
         matte_transform, shared_post_transform) = self.transform()
        dataset = MaadaaMattingDatasetWOTrimapV2(
            self.image_dir, self.foreground_dir, self.background_dir,
            shared_pre_transform, composition_transform,
            foreground_transform, background_transform,
            matte_transform, shared_post_transform, self.mode)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=True,
            num_workers=8, drop_last=False)
        return data_loader




