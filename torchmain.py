import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
# import torchvision
import torchvision.transforms as transforms
import torch.optim
from torch.utils.data import DataLoader

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
ITERS = 100
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


class FFNet(nn.Module):

    def __init__(self):
        super(FFNet, self).__init__()
        self.fcin = nn.Linear(28 * 28, WIDTH)
        self.do1 = nn.Dropout(p=DROPOUT)
        self.relu1 = nn.ReLU()

        self.fc_layers = []
        for d in range(DEPTH-1):
            self.fc_layers += [nn.Linear(WIDTH, WIDTH), nn.Dropout(p=DROPOUT), nn.ReLU()]
        self.fc_layers = nn.ModuleList(self.fc_layers)
        # print(self.fc_layers)
        self.fclass = nn.Linear(WIDTH, OUTPUT_COUNT)

    def forward(self, x):
        hidden_activations = []
        x = torch.flatten(x, 1)
        x = self.fcin(x)
        hidden_activations += [x.detach().numpy()]
        x = self.do1(x)
        x = self.relu1(x)
        for l in self.fc_layers:
            print(l)
            print(isinstance(l, nn.Linear))
            # I really-really don't like this
            x = l(x)
            print(x.size())
        out = self.fclass(x)
        out = F.softmax(out, dim=1)
        return(out)


net = FFNet()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

minibatches = len(train_dataset) // BATCH_SIZE
for epoch in range(ITERS):  # loop over the dataset multiple times
    running_predictions = 0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(images)
        # print(labels.size())
        # print(outputs.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break
        # print statistics
        running_loss += loss.item()
        running_predictions += torch.sum(torch.argmax(net(images), dim=1) == labels)
        if i % minibatches == minibatches-1:
            print(f'''{epoch + 1}/{ITERS}\tTrain loss: {running_loss / minibatches:.3f}\t
            Train accuracy: {running_predictions.numpy()/len(train_dataset):.3f}
            Test accuracy: {test_epoch(net):.3f}''')
            running_loss = 0.0
            running_predictions = 0
print('Finished Training')


params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

ex = torch.randn(5, 6)
ex.squeeze()

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
