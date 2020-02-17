import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
# import torchvision
import torchvision.transforms as transforms
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import torchnet

import time

DATASET = "fashion_mnist"
TRAINSIZE = 60000
SEED = None
DROPOUT = 0.25
BATCH_SIZE = 500
DEPTH = 5
WIDTH = 100
OUTPUT_COUNT = 10
LR = 0.001
L1REG = 1e-5
MEMORY_SHARE = 0.05
ITERS = 30
EVALUATION_CHECKPOINT = 1
AUGMENTATION = False
SESSION_NAME = "sinusoidal_5_100_KP_{}_{}".format(DROPOUT, time.strftime('%Y%m%d-%H%M%S'))
BN_WEIGHT = 0
COV_WEIGHT = 0
CLASSIFIER_TYPE = "dense"  # "conv" / "dense"
LOG_DIR = "logs/%s" % SESSION_NAME
EVALUATE_USEFULNESS = True
USEFULNESS_EVAL_SET_SIZE = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

starttime = time.time()

train_dataset = FashionMNIST('.', train=True, download=True,
                             transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = FashionMNIST('.', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))
eval_dataset = torch.utils.data.Subset(test_dataset, range(USEFULNESS_EVAL_SET_SIZE))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

eval_loader = DataLoader(dataset=eval_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)


def test_epoch(net):
    net = net.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        correct += torch.sum(torch.argmax(net(images), dim=1) == labels).numpy()
        total += labels.size(0)                  # Increment the total count
    return(correct/total)


def calculate_l1loss(net):
    # l1loss = torch.autograd.Variable(torch.tensor(0, dtype=torch.float, requires_grad=True))
    l1loss = 0.
    for param in net.parameters():
        if param.requires_grad:
            l1loss += param.abs().sum()

    return(l1loss * L1REG)


net = torchnet.FFNet(WIDTH, DEPTH, DROPOUT, OUTPUT_COUNT)
net.to(device)

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

minibatches = len(train_dataset) // BATCH_SIZE

for epoch in range(ITERS):  # loop over the dataset multiple times
    running_predictions = 0
    running_loss = 0.0
    running_l1loss = 0.0
    hidden_activations = []
    epochtime = time.time()
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(images)
        hidden_activations += [net.hidden_activations]
        # l1loss = calculate_l1loss(net)
        l1loss = calculate_l1loss(net)
        loss = criterion(outputs, labels) + l1loss

        running_l1loss += l1loss
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss
        running_predictions += torch.sum(torch.argmax(net(images), dim=1) == labels)
    print(f'{epoch + 1:03d}/{ITERS:03d} Train loss: {running_loss / minibatches:.3f}\t\
    Train accuracy: {running_predictions.numpy()/len(train_dataset):.3f}\
    Test accuracy: {test_epoch(net):.3f}\tEpoch time: {time.time()-epochtime:.2f}\
    L1Loss: {running_l1loss / minibatches:.3f}\tWeight sum: {sum([p.abs().sum() for p in net.parameters() if p.requires_grad]):.3f}')
    net = net.train()
    running_loss = 0.0
    running_predictions = 0
    running_l1loss = 0.0


hidden_activations_new = np.concatenate(hidden_activations, axis=1)
endtime = time.time()
print(f'Training took {endtime-starttime:.2f} seconds')
