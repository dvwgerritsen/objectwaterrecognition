import pickle
from pathlib import Path

import cv2
import torch
from torch.nn import Identity, Sequential, Linear, ReLU, Sigmoid, Module

from src import config

if __name__ == '__main__':
    # Ignore warnings
    from torchvision import transforms
    import time
    import warnings
    from pathlib import Path

    import PIL
    # from imutils import paths
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from matplotlib import image as mpimg
    from sklearn.model_selection import train_test_split
    from torch.nn import MSELoss
    from torch.nn import Module, Sequential, ReLU, Linear, Sigmoid, Identity
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    from torchvision.models import resnet50
    from tqdm import tqdm

    from src import config

    warnings.filterwarnings("ignore")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False

    path = Path("../data/output.csv")
    imagesDir = "../data/resized_Images/"

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
            self.transform = transforms.Compose([transforms.ToTensor()])

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = df.iloc[idx, 0]
            img = PIL.Image.open(imagesDir + img_name)
            landmarks = df.iloc[idx, 1:5]
            class_name = df.iloc[idx, 5]
            box = np.asarray(landmarks)
            boxCords = np.vstack(box).astype(np.float)
            boxTensor = torch.from_numpy(boxCords)
            img_array = np.array(img)
            img_array = img_array.transpose((2, 0, 1))
            return self.transform(img_array), boxTensor


    class ObjectDetector(Module):
        def __init__(self, baseModel):  # , numClasses):
            super(ObjectDetector, self).__init__()
            # initialize the base model and the number of classes
            self.baseModel = baseModel
            # self.numClasses = numClasses

            # build the regressor head for outputting the bounding box
            # coordinates
            self.regressor = Sequential(
                Linear(baseModel.fc.in_features, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 32),
                ReLU(),
                Linear(32, 4),
                Sigmoid()
            )
            # # build the classifier head to predict the class labels
            # self.classifier = Sequential(
            #     Linear(baseModel.fc.in_features, 512),
            #     ReLU(),
            #     Dropout(),
            #     Linear(512, 512),
            #     ReLU(),
            #     Dropout(),
            #     Linear(512, self.numClasses)
            # )
            # set the classifier of our base model to produce outputs
            # from the last convolution block
            self.baseModel.fc = Identity()

        def forward(self, x):
            # pass the inputs through the base model and then obtain
            # predictions from two different branches of the network
            features = self.baseModel(x)
            bboxes = self.regressor(features)
            # classLogits = self.classifier(features)
            # return the outputs as a tuple
            return (bboxes)  # ,classLogits)


    # data per item
    # print(aquaTrash.__getitem__(468))
    # lengte dataset
    # print(aquaTrash.__len__())

    # Laadt de data in
    # dataloader = DataLoader(aquaTrash, batch_size=4, shuffle=True, num_workers=4)

    # Geeft foto met border weer
    # showItem(4)

    trainDS = AquaTrashDataset(train, imagesDir)

    testDS = AquaTrashDataset(test, imagesDir)

    # print(trainDS.__getitem__(0))

    print("[INFO] total training samples: {}...".format(len(trainDS)))
    print("[INFO] total test samples: {}...".format(len(testDS)))

    trainSteps = len(trainDS) // config.BATCH_SIZE
    valSteps = len(testDS) // config.BATCH_SIZE

    trainLoader = DataLoader(trainDS, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=0, pin_memory=config.PIN_MEMORY)
    testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE,
                            num_workers=0, pin_memory=config.PIN_MEMORY)

    print("[INFO] loading object detector...")
    model = torch.load('detector.pt').to(config.DEVICE)
    model.eval()
    # le = pickle.loads(open(config.LE_PATH, "rb").read())
    index = 0
    for img, box in testLoader:
        if index == 0:
            image = mpimg.imread(r"C:\Users\31611\Desktop\Github\water-object-recognition\data\processed_Images\000000_jpg.rf.beffaf3b548106ccf1da5dc629bc9504.jpg")
            img = torch.permute(img, (0, 2, 1, 3))
            boxPred = model(img)
            print(boxPred[0])
            size = 256

            # for i in range(len(boxPred[0])):
            #     print(size * boxPred[0][i])
            plt.scatter(int(boxPred[0][0] * size), int(boxPred[0][1] * size), s=15, marker='.', c='r')
            plt.scatter(int(boxPred[0][2] * size), int(boxPred[0][1] * size), s=15, marker='.', c='r')
            plt.scatter(int(boxPred[0][0] * size), int(boxPred[0][3] * size), s=15, marker='.', c='r')
            plt.scatter(int(boxPred[0][2] * size), int(boxPred[0][3] * size), s=15, marker='.', c='r')
            plt.imshow(image)
            plt.show()
        index += 1
