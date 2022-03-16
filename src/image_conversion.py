from pathlib import Path

import pandas as pd
from skimage import io
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

path = Path("../data/annotations.csv")

df = pd.read_csv(path)

# loading training images
train_img = []
for index, row in tqdm(df.iterrows()):
    image_name = str(row['image_name'])

    # defining the image path
    image_path = '../data/images/' + image_name
    # reading the image
    img = imread(image_path)
    aspect_ratio = img.shape[0] / img.shape[1]
    height = 400 * aspect_ratio
    image_resized = resize(img, (height, 400), anti_aliasing=True)
    df.at[index, 'x_min'] = int((400 / img.shape[1]) * row['x_min'])
    df.at[index, 'x_max'] = int((400 / img.shape[1]) * row['x_max'])
    df.at[index, 'y_min'] = int((height / img.shape[0]) * row['y_min'])
    df.at[index, 'y_max'] = int((height / img.shape[0]) * row['y_max'])

    # , as_gray=True)
    # normalizing the pixel values
    # img /= 255.0
    # converting the type of pixel to float 32
    img = image_resized.astype('float32')
    # appending the image into the list
    io.imsave('../data/processed_Images/' + image_name, img)

df.to_csv("../data/output.csv", index=False)
