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
        for d in range(depth-1):
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
        out = F.softmax(x, dim=1)
        return(out)
