# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# PyTorch provides 2 data primitives
#   torch.utils.data.Dataset -> stores the samples and their corresponding labels
#   torch.utils.data.DataLoader -> wraps iterable around Dataset

# loading a dataset
training_data = datasets.FashionMNIST(
    root="data", # path where the train/test data is stroed
    train=True, # specifies training or test dataset
    download=True, # download dataset if it's not available at root
    transform=ToTensor() # transformations
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# iterating and visualizing
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols = 3
rows = 3

for i in range(1, cols * rows + 1):
    sample_idx= torch.randint(len(training_data), size=(1,)).item() # get random index
    img, label = training_data[sample_idx] # get image and lable
    figure.add_subplot(rows, cols, i)    
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Dataset retrieves our dataset's features and labels one sample at a time,
# however, we typically want to pass samples in minibatches,
# reshuffle data at every epoch to reduce overfitting

# DataLoader is an iterable that abstracts this complexity with an easy api

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)