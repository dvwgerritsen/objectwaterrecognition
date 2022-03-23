from torchvision import transforms

if __name__ == '__main__':
    # Ignore warnings
    import os
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
    from torch.nn import CrossEntropyLoss
    from torch.nn import MSELoss
    from torch.nn import Module, Sequential, ReLU, Linear, Sigmoid, Dropout, Identity
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
    #showItem(4)

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


    # load the ResNet50 network
    resnet = resnet50(pretrained=True)
    # freeze all ResNet50 layers so they will *not* be updated during the
    # training process
    for param in resnet.parameters():
        param.requires_grad = False

    # create our custom object detector model and flash it to the current
    # device
    objectDetector = ObjectDetector(resnet)
    objectDetector = objectDetector.to(config.DEVICE)
    # define our loss functions
    #classLossFunc = CrossEntropyLoss()
    bboxLossFunc = MSELoss()
    # initialize the optimizer, compile the model, and show the model
    # summary
    opt = Adam(objectDetector.parameters(), lr=config.INIT_LR)
    print(objectDetector)
    # initialize a dictionary to store training history
    H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
         "val_class_acc": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        objectDetector.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        #trainCorrect = 0
        valCorrect = 0

        # train_features, train_labels = next(iter(trainLoader))
        # print(f"Feature batch shape: {train_features.size()}")
        # print(f"Labels batch shape: {train_labels.size()}")

        # loop over the training set
        for img, box in trainLoader:
            #send the input to the device
            (img, box) = (img.to(config.DEVICE), box.to(config.DEVICE))
            img = torch.permute(img, (0, 2, 1, 3))
            # perform a forward pass and calculate the training loss
            predictions = objectDetector(img)
            bboxLoss = bboxLossFunc(predictions[0].float(), box.float())
            totalLoss = (config.BBOX * bboxLoss)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            totalLoss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += totalLoss
            # trainCorrect += (predictions[1].argmax(1) == labels).type(
            #     torch.float).sum().item()
            # switch off autograd
            with torch.no_grad():
                # set the model in evaluation mode
                objectDetector.eval()
                # loop over the validation set
                for (img, box) in testLoader:
                    # send the input to the device
                    (img,box) = (img.to(config.DEVICE), box.to(config.DEVICE))
                    img = torch.permute(img, (0, 2, 1, 3))
                # make the predictions and calculate the validation loss
                    predictions = objectDetector(img)
                    bboxLoss = bboxLossFunc(predictions[0], box)
                    totalLoss = (config.BBOX * bboxLoss)
                    totalValLoss += totalLoss

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        #trainCorrect = trainCorrect / len(trainDS)
        #valCorrect = valCorrect / len(testDS)
        # update our training history
        H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        #H["train_class_acc"].append(trainCorrect)
        H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_class_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        #print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            #avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
            avgValLoss, valCorrect))
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    torch.save(objectDetector, 'tada')
