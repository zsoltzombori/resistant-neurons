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
BN_DO = 'DO'  # "BN" (batchnorm), "DO" (dropout), None
DROPOUT = 0.25  # KEEP PROBABILITY, means 1 is no dropout
BATCH_SIZE = 500
DEPTH = 5
WIDTH = 100
OUTPUT_COUNT = 10
LR = 0.001
MEMORY_SHARE = 0.05
ITERS = 2
EVALUATION_CHECKPOINT = 1
AUGMENTATION = False
SESSION_NAME = "sinusoidal_5_100_KP_{}_{}".format(DROPOUT, time.strftime('%Y%m%d-%H%M%S'))
BN_WEIGHT = 0
COV_WEIGHT = 0
CLASSIFIER_TYPE = "dense"  # "conv" / "dense"
LOG_DIR = "logs/%s" % SESSION_NAME
EVALUATE_USEFULNESS = True
USEFULNESS_EVAL_SET_SIZE = 1000


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
    correct = 0
    total = 0
    for images, labels in test_loader:

        correct += torch.sum(torch.argmax(net(images), dim=1) == labels).numpy()
        total += labels.size(0)                  # Increment the total count
    return(correct/total)


net = torchnet.FFNet(WIDTH, DEPTH, DROPOUT, OUTPUT_COUNT)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

minibatches = len(train_dataset) // BATCH_SIZE

for epoch in range(ITERS):  # loop over the dataset multiple times
    running_predictions = 0
    running_loss = 0.0
    hidden_activations = []
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(images)
        hidden_activations += [net.hidden_activations]
        # print(labels.size())
        # print(outputs.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        running_predictions += torch.sum(torch.argmax(net(images), dim=1) == labels)
        if i % minibatches == minibatches-1:
            print(f'''{epoch + 1}/{ITERS}\tTrain loss: {running_loss / minibatches:.3f}\t
            Train accuracy: {running_predictions.numpy()/len(train_dataset):.3f}
            Test accuracy: {test_epoch(net):.3f}''')
            running_loss = 0.0
            running_predictions = 0

hidden_activations_new = np.concatenate(hidden_activations, axis=1)
