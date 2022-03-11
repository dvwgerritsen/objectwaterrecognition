import os
from pathlib import Path

import PIL
import torch
import pandas as pd
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
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

# Train/test split om accuracy van localisatie in de toekomst te traceren
train, test = train_test_split(df, test_size=0.1, random_state=0)


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


# test/train split toepassen
aquaTrash = AquaTrashDataset(df, imagesDir)

# data per item
print(aquaTrash.__getitem__(468))
# lengte dataset
print(aquaTrash.__len__())

#Laadt de data in
#dataloader = DataLoader(aquaTrash, batch_size=4, shuffle=True, num_workers=4)

#Geeft foto met border weer
#showItem(0)