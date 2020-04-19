import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from bnn import *
import _pickle as pickle
import matplotlib.pyplot as plt

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

print("Loading data")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', 
                    train=True, 
                    download=True, 
                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])), 
    shuffle=False)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', 
                    train=False, 
                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])), 
    shuffle=False)

print("Initializing Network...")
bnn = BayesNet(train_loader, 150, 150, 10)
print("Network initialized, Beginning training")

# Train, or recover previous training.
recover = True
if recover:
    with open('samples.txt', 'rb') as file:
        bnn.samples = pickle.load(file)
else:
    bnn.train(500, 280)
    with open('samples.txt', 'wb') as file:
        pickle.dump(bnn.samples, file)

results = [0]*10

print("Testing predictive distribution...")
for x, y in test_loader:
    print("Actual:")
    print(y)
    print("Output")
    out = bnn.predictive(x)
    results[out.item()] += 1
    print()

ax = fig.add_axes([0,0,1,1])
labels = [str(x) for x in range(10)]
ax.bar(labels, results)
plt.show()