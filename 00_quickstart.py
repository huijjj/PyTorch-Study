# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# download or load datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# craeting dataloaders (wrapping datasets so we can use it)
BATCH_SIZE = 64 # single batch consists of 64 elements(images)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

# checking data shape
for X, y in test_dataloader:
    print(f"shape of X [N, C, H, W]: {X.shape}")
    print(f"shape of y: {y.shape}")
    break

# setting up device, use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# defining neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # first Fully Connected layer, 28*28(input H * W) -> 512
            nn.ReLU(), # ReLU activation layer
            nn.Linear(512, 512), # second Fully Connected layer, 512 -> 512
            nn.ReLU(), # ReLU activation layer
            nn.Linear(512, 10) # last Fully Connected layer, 512 -> 10
        )

    def forward(self, x): # define the forward pass of the network
        x = self.flatten(x) # flatten the input into vector (1-d tensor)
        logits = self.linear_relu_stack(x) # pass the input to neural network and get the output
        return logits # return the output of the network

# model instantiation on available device
model = NeuralNetwork().to(device) 
print(model) # our model is ready to go, but weights are not trained yet


# we need a loss function and an optimizer to train a model
loss_fn = nn.CrossEntropyLoss() # use cross entropy loss for a loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # use stochastic gradient descent, with learning rate(lr) 1e-3

# define training code
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # set model on train mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # move tensors to device

        # compute the prediction error
        pred = model(X) # prediction (model output)
        loss = loss_fn(pred, y) # loss

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:5d}]")

# define test code
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # set model on evaluation mode
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test: \n Accuracy: {(100*correct):0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5 # run 5 epochs(iterate full dataset for 5 times)
for t in range(epochs):
    print(f"Epoch {t+1}\n")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)