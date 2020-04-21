#####################################################################################################
##   Note: this code is based on the following tutorial:                                          ##
##    - Author: adventuresinML                                                                     ##
##    - URL: adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch     ##
##    - Title: Convolutional Neural Networks Tutorial in PyTorch                                   ##
#####################################################################################################

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np

# Set hyperparameters
numEpochs = 5       # Number of times complete dataset is passed forward and backward through NN
numClasses = 10     # Number of classes being classified at NN's output
batchSize = 100     # Number of samples from dataset to use
learningRate = 0.001    # Rate at which weights are updated (adjusts step size toward local minimum each iteration)

# Folders for PyTorch to save MNIST dataset and model's trained parameter
MNIST_PATH = os.getcwd() + "\mnist"
MODEL_PATH = os.getcwd() + "\model"

# Use Compose() to apply the following transformations on dataset:
#   - Convert dataset to a Tensor (multi-dimensional matrix)
#   - Normalise data within a range (usually -1 to 1 or 0 to 1) using: Mean = 0.1307, Std deviation = 0.3081
#       - Note: a mean and std deviation needs to be set for each input channel (MNIST dataset only has one channel)
appliedTransfroms = transforms.Compose([ transforms.ToTensor(), transforms.Normalize( (0.1307,), (0.3081,) ) ])

# Create datasets from MNIST data for training and testing CNN
trainingDataset = torchvision.datasets.MNIST(root = MNIST_PATH, train = True, transform = appliedTransfroms, download = True)
testingDataset = torchvision.datasets.MNIST(root = MNIST_PATH, train = False, transform = appliedTransfroms)

# Load the datasets created above into data loader to create DataLoader objects
trainingLoader = DataLoader(dataset = trainingDataset, batch_size = batchSize, shuffle = True)
testingLoader = DataLoader(dataset = testingDataset, batch_size = batchSize, shuffle = False)

# Class for NN model that inheirits from nn.Module
class DigitClassifierNN(nn.Module):
    def __init__(self):
        super(DigitClassifierNN, self).__init__()

        # Create layer objects
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2), # Convolutional filter with 1 input channel, 32 output channels and a filter size of 5 x 5  
                                    nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size = 2, stride = 2))  # Size of window is 2 x 2, default padding = 0
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2),
                                    nn.ReLU,
                                    nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.dropOut = nn.Dropout()     # Create drop-out layer to avoid over-fitting
        self.fullyConnected1 = nn.Linear(7 * 7 * 64, 1000)      # First argument = no. of nodes in the layer, second argument = no. of nodes in the next layer
        self.fullyConnected2 = nn.Linear(1000, 10)

# Define how data flows through NN when performing a forward pass
# Overrides default forward function in nn.Module 
def forward(self, data):       # data = a batch of data
    dataOut = self.layer1(data)
    dataOut = self.layer2(dataOut)
    dataOut = dataOut.reshape(dataOut.size(0), -1)      # Flatten data dimensions from 7 x 7 x 64 to 3164 x 7
    dataOut = self.dropOut(dataOut)
    dataOut = self.fullyConnected1(dataOut)
    dataOut = self.fullyConnected2(dataOut)
    return dataOut

# Create instance of CNN
model = DigitClassifierNN()

criterion = nn.CrossEntropyLoss()       # Loss operation
                                        # CrossEntropyLoss() combines SoftMax activation and a cross entropy loss function 
                                        # (i.e. generates probability for each class and then measures NN's performance)

# Define Adam optimiser to adjust NN's trainable parameters during training
optimiser = torch.optim.Adam(model.parameters(), lr = learningRate)     # nn.Module class provides parameters() to automatically track all trainable paramters in model