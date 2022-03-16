import glob
import os
from pathlib import Path

import cv2 as cv2
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
        img = imread(image_path)
        aspect_ratio = img.shape[0] / img.shape[1]
        height = size * aspect_ratio
        image_resized = resize(img, (height, size), anti_aliasing=True)
        df.at[index, 'x_min'] = int((size / img.shape[1]) * row['x_min'])
        df.at[index, 'x_max'] = int((size / img.shape[1]) * row['x_max'])
        df.at[index, 'y_min'] = int((height / img.shape[0]) * row['y_min'])
        df.at[index, 'y_max'] = int((height / img.shape[0]) * row['y_max'])

        #img = image_resized.astype('float32')

        io.imsave('../data/resized_Images/' + image_name, image_resized)
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


#rescaleImages(256)
processImages()