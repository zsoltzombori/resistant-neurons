import torch
from torch.utils.data import Dataset
import numpy as np


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
