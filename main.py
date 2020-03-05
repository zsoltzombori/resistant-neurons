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
import matplotlib.pyplot as plt
import helper_functions as H

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

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
eval_loader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=False)


# net = torchnet.FFNet(WIDTH, DEPTH, DROPOUT, OUTPUT_COUNT)
net = torchnet.LeNet()
net.to(device)

print(net)

print(f'layer dimensions: '
      f'{[(list(layer.size())) for layer in net.parameters() if layer.requires_grad]}')

print(f'number of trainable parameters: '
      f'{sum([np.prod(list(layer.size())) for layer in net.parameters() if layer.requires_grad])}')

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
        l1loss = H.calculate_l1loss(net, L1REG)
        l2loss = H.calculate_l2loss(net, L2REG)
        loss = criterion(outputs, labels) + l1loss + l2loss

        running_l1loss += l1loss
        loss.backward()
        # we have the gradients at this point, and they are encoded in param.grad where param is net.parameters()
        if epoch >= 0:
            for pos in neurons_to_freeze:
                H.zero_grad_for_neuron(pos, net)

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
    Test accuracy: {H.test_epoch(net, device, test_loader):.3f}\tEpoch time: {time.time()-epochtime:.2f}\
    L1Loss: {running_l1loss.cpu() / minibatches:.3f}\tWeight sum: {sum([p.abs().sum() for p in net.parameters()]):.3f}')

    ratio_to_freeze = 1

    if epoch == 35:
        # vanishing_dataset = H.VanishingDataset(list_of_data)
        # vanish_dataloader = DataLoader(dataset=vanishing_dataset, batch_size=BATCH_SIZE, shuffle=True)

        neurons_to_freeze = H.get_and_add_topn_activations(
            net, 0, neurons_to_freeze, int(28*28*ratio_to_freeze), hidden_activations_for_epoch)
        neurons_to_freeze = H.get_and_add_topn_activations(
            net, DEPTH-1, neurons_to_freeze, int(OUTPUT_COUNT), hidden_activations_for_epoch)
        for l in range(1, DEPTH-1):
            topn = int(WIDTH * ratio_to_freeze)
            neurons_to_freeze = H.get_and_add_topn_activations(
                net, l, neurons_to_freeze, topn, hidden_activations_for_epoch)

        neurons_to_freeze = sorted(neurons_to_freeze, key=lambda x: x[0])
        # print(f'length of the remaining images: {len(vanishing_dataset)}')
        print(neurons_to_freeze)

    if False:
        layer = 1
        hidden_activations_for_layer = np.concatenate([x[layer] for x in hidden_activations_for_epoch], axis=0)
        plt.plot(figsize=(12, 6), facecolor='w')
        plt.hist(np.sum(np.abs(hidden_activations_for_layer), axis=0))
        plt.title(f'sum of activations in epoch {epoch}')
        plt.grid()
        plt.show()


endtime = time.time()
print(f'Training took {endtime-starttime:.2f} seconds')
