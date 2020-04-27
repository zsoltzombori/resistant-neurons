import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision.datasets import CIFAR10
# import torchvision
import torchvision.transforms as transforms
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import torchnet
import matplotlib.pyplot as plt
import helper_functions as H
import resnet

from collections import defaultdict
import os

import time

DATASET = "fashion_mnist"
TRAINSIZE = 50000
SEED = None
DROPOUT = 0.25
BATCH_SIZE = 500
DEPTH = 5
WIDTH = 100
OUTPUT_COUNT = 10
LR = 0.01
L1REG = 0.01
L2REG = 0.01
MEMORY_SHARE = 0.05
ITERS = 300
EVALUATION_CHECKPOINT = 1
AUGMENTATION = False
SESSION_NAME = f'{time.strftime("%Y%m%d-%H%M%S")}'
BN_WEIGHT = 0
COV_WEIGHT = 0
CLASSIFIER_TYPE = "dense"  # "conv" / "dense"
LOG_DIR = "logs/%s" % SESSION_NAME
EVALUATE_USEFULNESS = True
USEFULNESS_EVAL_SET_SIZE = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

starttime = time.time()

# TODO put this to some other file
# train_dataset = FashionMNIST('.', train=True, download=True,
#                              transform=transforms.Compose([transforms.ToTensor()]))
# test_dataset = FashionMNIST('.', train=False, download=True,
#                             transform=transforms.Compose([transforms.ToTensor()]))
# eval_dataset = torch.utils.data.Subset(test_dataset, range(USEFULNESS_EVAL_SET_SIZE))

# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# eval_loader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_dataset = CIFAR10(root='./.datasets', train=True, download=True, transform=transform_train)
test_dataset = CIFAR10(root='./.datasets', train=False, download=True, transform=transform_test)
eval_dataset = torch.utils.data.Subset(test_dataset, range(USEFULNESS_EVAL_SET_SIZE))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

eval_loader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# net = torchnet.FFNet(WIDTH, DEPTH, DROPOUT, OUTPUT_COUNT)
net = resnet.resnet18(pretrained=False, progress=True)
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
weights = []

# here comes the hooks

os.mkdir(f'neuron_logs/{SESSION_NAME}')

hidden_activations = defaultdict(list)


def get_activation(name):
    def hook(model, input, output):
        hidden_activations[name] += [output.cpu().detach().numpy()]
    return hook


net.conv1.register_forward_hook(get_activation('conv1'))
net.maxpool.register_forward_hook(get_activation('maxpool1'))

net.layer1[0].conv1.register_forward_hook(get_activation('layer1_block1_conv1'))
net.layer1[0].conv2.register_forward_hook(get_activation('layer1_block1_conv2'))
net.layer1[1].conv1.register_forward_hook(get_activation('layer1_block2_conv1'))
net.layer1[1].conv2.register_forward_hook(get_activation('layer1_block2_conv2'))

net.layer2[0].conv1.register_forward_hook(get_activation('layer2_block1_conv1'))
net.layer2[0].conv2.register_forward_hook(get_activation('layer2_block1_conv2'))
net.layer2[1].conv1.register_forward_hook(get_activation('layer2_block2_conv1'))
net.layer2[1].conv2.register_forward_hook(get_activation('layer2_block2_conv2'))

net.layer3[0].conv1.register_forward_hook(get_activation('layer3_block1_conv1'))
net.layer3[0].conv2.register_forward_hook(get_activation('layer3_block1_conv2'))
net.layer3[1].conv1.register_forward_hook(get_activation('layer3_block2_conv1'))
net.layer3[1].conv2.register_forward_hook(get_activation('layer3_block2_conv2'))

net.layer4[0].conv1.register_forward_hook(get_activation('layer4_block1_conv1'))
net.layer4[0].conv2.register_forward_hook(get_activation('layer4_block1_conv2'))
net.layer4[1].conv1.register_forward_hook(get_activation('layer4_block2_conv1'))
net.layer4[1].conv2.register_forward_hook(get_activation('layer4_block2_conv2'))

net.fc.register_forward_hook(get_activation('fc'))

for epoch in range(ITERS):  # loop over the dataset multiple times
    running_predictions = 0.
    running_loss = 0.0
    running_l1loss = 0.0
    hidden_activations_for_epoch = []
    epochtime = time.time()
    net = net.train()
    samples_seen = 0

    list_of_data = []
    for param_group in optimizer.param_groups:
        if epoch < 100:
            param_group['lr'] = LR
        elif epoch >= 100 and epoch <= 200:
            param_group['lr'] = LR/10
        elif epoch >= 200:
            param_group['lr'] = LR/100

    current_weights = [layer.cpu().detach().numpy() for layer in net.parameters() if layer.requires_grad]
    weights += [current_weights]
    np.save(f'neuron_logs/{SESSION_NAME}_epoch_{epoch:03}.npy', weights)

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

        # hidden_activations_for_epoch += [net.hidden_activations]
        # l1loss = calculate_l1loss(net)
        l1loss = H.calculate_l1loss(net, L1REG)
        l2loss = H.calculate_l2loss(net, L2REG)
        loss = criterion(outputs, labels) + l1loss + l2loss

        running_l1loss += l1loss
        loss.backward()
        # we have the gradients at this point, and they are encoded in param.grad where param is net.parameters()
        # if epoch >= 0:
        #     for pos in neurons_to_freeze:
        #         H.zero_grad_for_neuron(pos, net)

        optimizer.step()
        # print statistics
        running_loss += loss
        running_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).cpu().numpy()
        samples_seen += images.shape[0]
        
    np.save(f'neuron_logs/{SESSION_NAME}/{SESSION_NAME}_activations_epoch_{epoch:03}.npy', hidden_activations)
    #  = (0, 5)
    # print(np.sum(np.abs(get_weights_for_position(pos, net))))
    print(f'number of bad guesses: {len(list_of_data)}')

    print(f'{epoch + 1:03d}/{ITERS:03d} Train loss: {running_loss.cpu() / minibatches:.3f}\
    Train accuracy: {running_predictions/samples_seen:.3f}\
    Test accuracy: {H.test_epoch(net, device, test_loader):.3f}\tEpoch time: {time.time()-epochtime:.2f}\
    L1Loss: {running_l1loss.cpu() / minibatches:.3f}\tWeight sum: {sum([p.abs().sum() for p in net.parameters()]):.3f}')


endtime = time.time()
print(f'Training took {endtime-starttime:.2f} seconds')
