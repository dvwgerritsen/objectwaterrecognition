import glob
import os
from pathlib import Path

import cv2 as cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

path = Path("../data/annotations.csv")

df = pd.read_csv(path)


def rescaleImages(size):
    for index, row in tqdm(df.iterrows()):
        image_name = str(row['image_name'])
        # defining the image path
        image_path = '../data/images/' + image_name
        # reading the image
        img = cv2.imread(image_path)
        if img.shape[0] != img.shape[1]:
            color = [255, 255, 255]
            delta = abs(img.shape[0] - img.shape[1])
            if img.shape[0] > img.shape[1]:
                width = size * img.shape[1] / img.shape[0]
                height = size
                image_bordered = cv2.copyMakeBorder(img, 0, 0, 0, int(delta), cv2.BORDER_CONSTANT, value=color)
            else:
                width = size
                height = size * img.shape[0] / img.shape[1]
                image_bordered = cv2.copyMakeBorder(img, 0, int(delta), 0, 0, cv2.BORDER_CONSTANT, value=color)
            image_resized = resize(image_bordered, (size, size), anti_aliasing=True)

        else:
            height = size
            width = size
            image_resized = resize(img, (size, size), anti_aliasing=True)

        df.at[index, 'x_min'] = int((width / img.shape[1]) * row['x_min'])
        df.at[index, 'x_max'] = int((width / img.shape[1]) * row['x_max'])
        df.at[index, 'y_min'] = int((height / img.shape[0]) * row['y_min'])
        df.at[index, 'y_max'] = int((height / img.shape[0]) * row['y_max'])
        image_converted = np.float32(image_resized)
        image_converted = cv2.cvtColor(image_converted, cv2.COLOR_RGB2BGR)
        io.imsave('../data/resized_Images/' + image_name, image_converted)
    df.to_csv("../data/output.csv", index=False)


def processImages():
    rescaledDf = pd.read_csv("../data/output.csv")
    train, test = train_test_split(rescaledDf, test_size=0.1, random_state=0)
    for index, row in tqdm(train.iterrows()):
        image_name = str(row['image_name'])
        image_imp_path = '../data/resized_Images/' + image_name
        image_dest_path = '../data/processed_Images/' + image_name
        image = cv2.imread(image_imp_path, 0)
        image_medianBlurred = cv2.medianBlur(image, 7)
        img_converted = cv2.adaptiveThreshold(image_medianBlurred, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        img_colored = cv2.cvtColor(img_converted, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(image_dest_path, img_colored)
    for index, row in tqdm(test.iterrows()):
        image_name = str(row['image_name'])
        image_imp_path = '../data/resized_Images/' + image_name
        image_dest_path = '../data/processed_Images/' + image_name
        image = cv2.imread(image_imp_path)
        cv2.imwrite(image_dest_path, image)


#rescaleImages(256)
processImages()
