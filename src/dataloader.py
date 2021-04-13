import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path


class MattingDataset:
    def __init__(self, image_dir, image_transform, trimap_transform,
                 matte_transform, mode, verbose=0):
        self.image_list = list(map(str, Path(image_dir).rglob("*.jpg")))
        self.trimap_list = list(map(str, Path(image_dir).rglob("*trimap_true.png")))
        self.matte_list = list(map(str, Path(image_dir).rglob("*.png")))
        self.matte_list = [x for x in self.matte_list if "trimap" not in x]
        self.image_transform = image_transform
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
        image = Image.open(image_path)
        trimap = Image.open(trimap_path)
        matte = Image.open(matte_path)

        return (self.image_transform(image), self.trimap_transform(trimap),
                self.matte_transform(matte))

    def __len__(self):
        return self.n_images


class MattingLoader:
    def __init__(self, image_path, image_size, batch_size, mode):
        self.image_dir = Path(image_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode

    def transform(self):
        image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        trimap_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        matte_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        return image_transform, trimap_transform, matte_transform

    def loader(self):
        image_transform, trimap_transform, matte_transform = self.transform()
        dataset = MattingDataset(self.image_dir, image_transform, trimap_transform,
                                 matte_transform, self.mode)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=(self.mode == "train"),
            num_workers=2, drop_last=False)
        return data_loader




















