import os
#import torch

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