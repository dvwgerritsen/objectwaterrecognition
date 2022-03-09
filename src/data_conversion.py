import os
from pathlib import Path

import PIL
import torch
import pandas as pd
from matplotlib import image as mpimg
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

path = Path("../data/annotations.csv")
imagesDir = "../data/images/"

df = pd.read_csv(path)


def showItem(idx):
    """Show image with landmarks"""
    img_dir = imagesDir + df.iloc[idx, 0]
    landmarks = df.iloc[idx, 1:5]
    class_names = df.iloc[idx, 5]
    landmarks = np.asarray(landmarks)
    image = mpimg.imread(img_dir)
    plt.imshow(image)
    plt.scatter(landmarks[0], landmarks[1], s=15, marker='.', c='r')
    plt.scatter(landmarks[2], landmarks[1], s=15, marker='.', c='r')
    plt.scatter(landmarks[0], landmarks[3], s=15, marker='.', c='r')
    plt.scatter(landmarks[2], landmarks[3], s=15, marker='.', c='r')
    plt.pause(0.001)
    plt.show()


# uiteindelijk naar dataset object omzetten, met tensor voor images

class AquaTrashDataset(Dataset):

    def __init__(self, df, imagesDir, transform=None):
        self.df = df
        self.imagesDir = imagesDir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = df.iloc[idx, 0]
        img = PIL.Image.open(imagesDir+img_name)
        landmarks = df.iloc[idx, 1:5]
        class_name = df.iloc[idx, 5]
        landmarks = np.asarray(landmarks)
        img_array = np.array(img)
        img_array = img_array.transpose((2, 0, 1))
        sample = {'image': img_array, 'landmarks': landmarks, 'class': class_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


# class ToTensor(object):
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'landmarks': torch.from_numpy(landmarks)}


aquaTrash = AquaTrashDataset(df, imagesDir)
#showItem(0)
#dataloader = DataLoader(aquaTrash, batch_size=4, shuffle=True, num_workers=4)

print(aquaTrash.__getitem__(468))
print(aquaTrash.__len__())