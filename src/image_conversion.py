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
for img_name in tqdm(df['image_name']):
    # defining the image path
    image_path = '../data/images/' + str(img_name)
    # reading the image
    img = imread(image_path)
    #image_resized = resize(img, (400, 400),anti_aliasing=True)
    #, as_gray=True)
    # normalizing the pixel values
    #img /= 255.0
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    io.imsave('../data/processed_Images/' + str(img_name), img)