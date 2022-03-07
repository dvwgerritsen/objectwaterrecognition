import os
from pathlib import Path

import torch
import pandas as pd
from matplotlib import image as mpimg
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

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
    """Face Landmarks dataset."""

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
        landmarks = df.iloc[idx, 1:5]
        class_names = df.iloc[idx, 5]
        landmarks = np.asarray(landmarks)
        sample = {'image': img_name, 'landmarks': landmarks, 'class': class_names}

        if self.transform:
            sample = self.transform(sample)

        return sample


aquaTrash = AquaTrashDataset(df, imagesDir)
showItem(0)