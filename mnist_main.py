import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import _pickle as pickle
import torch.cuda
from bnn import *



# Network parameters
NET_DIM_H1  = 150
NET_DIM_H2  = 150
CATEGORIES  = 10
BATCH_SIZE  = 32
LEARN_RATE  = 0.3
EPOCHS      = 1
DEVICE      = 'cuda'
CHAIN_SMAPLES = 100

# Data parameters
# These will put our datasets in the same range and help with training on MNIST
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081
MNIST_TRANS = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])

if __name__ == "__main__":
    print("Initializing datasets...")
    train_set = datasets.MNIST('C:/data', 
                        train=True, 
                        download=True, 
                        transform=MNIST_TRANS)

    test_set = datasets.MNIST('C:/data', 
                        train=False, 
                        transform=MNIST_TRANS)

    fashion_set = datasets.FashionMNIST('C:/data',
                        train=False,
                        download=True,
                        transform=MNIST_TRANS)

    print("Initializing Network...")
    net = MCMCBayesNet(NET_DIM_H1, NET_DIM_H2, CATEGORIES, train_set, DEVICE)

    print("Network initialized, maximizing posterior on MNIST...")
    net.maximize_posterior(EPOCHS, BATCH_SIZE, LEARN_RATE)

    print("Testing MAP estimate...")
    net.test_network(test_set, BATCH_SIZE)

    print("Initializing Markov Chain and Monte Carlo samples...")
    net.init_predictive(CHAIN_SMAPLES)

    print("Testing the estimated posterior predictive distribution on MNIST...")
    net.test_predictive(test_set)

    print("Now testing on Fashion MNIST, coefficient of deviation should be higher...")
    net.test_predictive(fashion_set)