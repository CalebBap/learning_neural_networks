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

# Create a class for NN model that inheirits from nn.Module
class DigitClassifierNN(nn.Module):
    def __init__(self):
        super(DigitClassifierNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2), 
                                    nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2),
                                    nn.ReLU,
                                    nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.dropOut = nn.Dropout()
        self.fullyConnected1 = nn.Linear(7 * 7 * 64, 1000)
        self.fullyConnected2 = nn.Linear(1000, 10)