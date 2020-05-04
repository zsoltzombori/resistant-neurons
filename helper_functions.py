import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn


def calculate_l1loss(net, L1REG) -> float:
    l1loss = 0.
    for name, param in net.named_parameters():
        if param.requires_grad and 'weight' in name:
            l1loss += torch.mean(torch.abs(param))

    return(l1loss * L1REG)


def calculate_l2loss(net, L2REG) -> float:
    l2loss = 0.
    for name, param in net.named_parameters():
        if param.requires_grad and 'weight' in name:
            l2loss += torch.mean(param ** 2)

    return(l2loss * L2REG)


def get_weights_for_position(pos, net, direction='input') -> np.ndarray:

    d, p = pos
    weight_layers = []
    for name, weights in net.named_parameters():
        if 'weight' in name:
            weight_layers += [weights.cpu().detach().numpy()]

    return(weight_layers[d][p, :])


def get_grad_for_position(pos, net, direction='input') -> np.ndarray:

    d, p = pos
    grads = []
    for name, weights in net.named_parameters():
        if 'weight' in name:
            grads += [weights.cpu().detach().numpy()]

    return(grads[d][p, :])


class VanishingDataset(Dataset):
    def __init__(self, list_of_image_target_tuples):
        self.data = torch.stack([x[0] for x in list_of_image_target_tuples])
        self.targets = torch.stack([x[1] for x in list_of_image_target_tuples])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return(self.data[idx], self.targets[idx])


def get_and_add_topn_activations(net, layer, frozen_neurons, topn, hidden_activations) -> list:

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


def test_epoch(net, device, test_loader) -> float:
    net.eval()
    correct = 0.
    total = 0

    for images, labels in test_loader:
        total += images.shape[0]
        images = images.to(device)
        labels = labels.to(device)
        # torch.argmax(net(images), dim=1) == labels
        correct += torch.sum(torch.argmax(net(images), dim=1) == labels).cpu().numpy()

    return(correct/total)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))


class CustomResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, width_per_group=64):
        super(CustomResNet, self).__init__()
        self.base_width = width_per_group
        self.in_planes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),)

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, groups=1,
                            base_width=self.base_width, dilation=self.dilation))
        # the magic 1 refers to number of groups, and we don't have those at resnet18
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=1,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
