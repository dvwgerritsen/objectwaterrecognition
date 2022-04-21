# import the necessary packages
import torch
import os

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "data"

# path to AquaTrash images (https://github.com/Harsh9524/AquaTrash)
AQUATRASH_IMAGES_PATH = os.path.sep.join([BASE_PATH, "Images"])
# Path to taco images. The annotations.json and batch folders should be there.
# (https://github.com/pedropro/TACO or direct download https://doi.org/10.5281/zenodo.3587843)
TACO_IMAGES_PATH = os.path.sep.join([BASE_PATH, "taco-data"])

ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations.csv"])
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label encoder, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# determine the current device and based on that set the pin memory
# flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 4e-4
NUM_EPOCHS = 100
BATCH_SIZE = 32
# specify the loss weights
LABELS = 1.0
BBOX = 2.0
