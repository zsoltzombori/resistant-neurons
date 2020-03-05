from torch import nn
import torch
import torch.nn.functional as F


class FFNet(nn.Module):

    def __init__(self, width, depth, dropout, output_count):
        super(FFNet, self).__init__()
        self.hidden_activations = []
        self.fcin = nn.Linear(28 * 28, width)
        self.do1 = nn.Dropout(p=dropout)
        self.relu1 = nn.ReLU()
        self.fc_layers = []
        for d in range(depth-2):
            self.fc_layers += [nn.Linear(width, width), nn.Dropout(p=dropout), nn.ReLU()]
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fclass = nn.Linear(width, output_count)

    def forward(self, x):
        self.hidden_activations = []
        x = torch.flatten(x, 1)
        x = self.fcin(x)
        self.hidden_activations += [x.cpu().detach().numpy()]
        x = self.do1(x)
        x = self.relu1(x)
        for l in self.fc_layers:
            if isinstance(l, nn.Linear):
                self.hidden_activations += [x.cpu().detach().numpy()]
            x = l(x)

        x = self.fclass(x)
        self.hidden_activations += [x.cpu().detach().numpy()]
        out = F.softmax(x, dim=1)
        return(out)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        self.hidden_activations = []
        x = self.conv1(x)
        self.hidden_activations += [x.cpu().detach().numpy()]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        self.hidden_activations += [x.cpu().detach().numpy()]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        self.hidden_activations += [x.cpu().detach().numpy()]
        x = F.relu(x)
        x = self.fc2(x)
        self.hidden_activations += [x.cpu().detach().numpy()]
        x = F.relu(x)
        x = self.fc3(x)
        self.hidden_activations += [x.cpu().detach().numpy()]
        # I'd have a softmax, please and thank you
        out = F.softmax(x, dim=1)
        return out
