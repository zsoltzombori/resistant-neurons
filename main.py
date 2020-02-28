import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
# import torchvision
import torchvision.transforms as transforms
import torch.optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchnet
import matplotlib.pyplot as plt

import time

DATASET = "fashion_mnist"
TRAINSIZE = 60000
SEED = None
DROPOUT = 0.25
BATCH_SIZE = 500
DEPTH = 5
WIDTH = 100
OUTPUT_COUNT = 10
LR = 0.002
L1REG = 0.01
L2REG = 0.01
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


def test_epoch(net, device):
    net.eval()
    correct = 0.
    total = len(test_dataset)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # torch.argmax(net(images), dim=1) == labels
        correct += torch.sum(torch.argmax(net(images), dim=1) == labels).cpu().numpy()

    return(correct/total)


def calculate_l1loss(net):
    # l1loss = torch.autograd.Variable(torch.tensor(0, dtype=torch.float, requires_grad=True))
    l1loss = 0.
    for name, param in net.named_parameters():
        if param.requires_grad and 'weight' in name:
            l1loss += torch.mean(torch.abs(param))

    return(l1loss * L1REG)

def calculate_l2loss(net):
    l2loss = 0.
    for name, param in net.named_parameters():
        if param.requires_grad and 'weight' in name:
            l2loss += torch.mean(param ** 2)

    return(l2loss * L2REG)


def get_weights_for_position(pos, net, direction='input'):

    d, p = pos
    weight_layers = []
    for name, weights in net.named_parameters():
        if 'weight' in name:
            weight_layers += [weights.cpu().detach().numpy()]

    return(weight_layers[d][p, :])


def get_grad_for_position(pos, net, direction='input'):

    d, p = pos
    grads = []
    for name, weights in net.named_parameters():
        if 'weight' in name:
            grads += [weights.cpu().detach().numpy()]

    return(grads[d][p, :])


def zero_grad_for_neuron(pos, net, direction='input'):
    d, p = pos
    i = 0
    for name, weights in net.named_parameters():
        if 'weight' in name:
            if i == d:
                # print(weights.grad[p, :].size())
                weights.grad[p, :] = 0
                return()
            i += 1


class VanishingDataset(Dataset):
    def __init__(self, list_of_image_target_tuples):
        self.data = torch.stack([x[0] for x in list_of_image_target_tuples])
        self.targets = torch.stack([x[1] for x in list_of_image_target_tuples])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return(self.data[idx], self.targets[idx])


def get_and_add_topn_activations(net, layer, frozen_neurons, topn, hidden_activations):

    hidden_activations_for_layer = np.concatenate([x[layer] for x in hidden_activations], axis=0)
    sum_activations_per_neuron = np.sum(np.abs(hidden_activations_for_layer), axis=0)
    sorted_by_activation = np.argsort(sum_activations_per_neuron)[::-1]
    old_frozen_neurons = [x[1] for x in frozen_neurons if x[0] == layer]
    new_frozen_neurons = []
    for n_i in sorted_by_activation:
        if len(new_frozen_neurons) < topn:
            # print(n_i)
            if n_i in old_frozen_neurons:
                pass
            else:
                new_frozen_neurons += [n_i]
        else:
            break

    frozen_neurons += [(layer, n_i) for n_i in new_frozen_neurons]
    return(frozen_neurons)


net = torchnet.FFNet(WIDTH, DEPTH, DROPOUT, OUTPUT_COUNT)
net.to(device)

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0)

minibatches = len(train_dataset) // BATCH_SIZE

neurons_to_freeze = []
vanish_dataloader = train_loader

for epoch in range(ITERS):  # loop over the dataset multiple times
    running_predictions = 0.
    running_loss = 0.0
    running_l1loss = 0.0
    hidden_activations_for_epoch = []
    epochtime = time.time()
    net = net.train()
    samples_seen = 0

    list_of_data = []

    for i, data in enumerate(vanish_dataloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        images, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(images)

        predictions = torch.argmax(outputs, dim=1)
        for j in range(data[0].shape[0]):
            im, tag, pred = data[0][j].squeeze(), data[1][j].squeeze(), predictions[j]
            if tag != pred:
                list_of_data += [tuple([im, tag])]

        hidden_activations_for_epoch += [net.hidden_activations]
        # l1loss = calculate_l1loss(net)
        l1loss = calculate_l1loss(net)
        l2loss = calculate_l2loss(net)
        loss = criterion(outputs, labels) + l1loss + l2loss

        running_l1loss += l1loss
        loss.backward()
        # we have the gradients at this point, and they are encoded in param.grad where param is net.parameters()
        if epoch >= 0:
            for pos in neurons_to_freeze:
                zero_grad_for_neuron(pos, net)

        optimizer.step()
        # print statistics
        running_loss += loss
        running_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).cpu().numpy()
        samples_seen += images.shape[0]

    #  = (0, 5)
    # print(np.sum(np.abs(get_weights_for_position(pos, net))))
    print(f'number of bad guesses: {len(list_of_data)}')

    print(f'{epoch + 1:03d}/{ITERS:03d} Train loss: {running_loss.cpu() / minibatches:.3f}\
    Train accuracy: {running_predictions/samples_seen:.3f}\
    Test accuracy: {test_epoch(net, device):.3f}\tEpoch time: {time.time()-epochtime:.2f}\
    L1Loss: {running_l1loss.cpu() / minibatches:.3f}\tWeight sum: {sum([p.abs().sum() for p in net.parameters()]):.3f}')

    layer = 1

    ratio_to_freeze = 0.949

    if epoch == 15:
        vanishing_dataset = VanishingDataset(list_of_data)
        vanish_dataloader = DataLoader(dataset=vanishing_dataset, batch_size=BATCH_SIZE, shuffle=True)

        neurons_to_freeze = get_and_add_topn_activations(
            net, 0, neurons_to_freeze, int(28*28*ratio_to_freeze), hidden_activations_for_epoch)
        neurons_to_freeze = get_and_add_topn_activations(
            net, DEPTH-1, neurons_to_freeze, int(OUTPUT_COUNT), hidden_activations_for_epoch)
        for l in range(1, DEPTH-1):
            topn = int(WIDTH * ratio_to_freeze)
            neurons_to_freeze = get_and_add_topn_activations(
                net, l, neurons_to_freeze, topn, hidden_activations_for_epoch)

        neurons_to_freeze = sorted(neurons_to_freeze, key=lambda x: x[0])
        print(f'length of the remaining images: {len(vanishing_dataset)}')
        print(neurons_to_freeze)

    if False:
        hidden_activations_for_layer = np.concatenate([x[layer] for x in hidden_activations_for_epoch], axis=0)
        plt.plot(figsize=(12, 6), facecolor='w')
        plt.hist(np.sum(np.abs(hidden_activations_for_layer), axis=0))
        plt.title(f'sum of activations in epoch {epoch}')
        plt.grid()
        plt.show()


endtime = time.time()
print(f'Training took {endtime-starttime:.2f} seconds')
