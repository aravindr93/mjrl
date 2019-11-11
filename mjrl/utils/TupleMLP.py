
import torch
import torch.nn as nn

class TupleMLP(nn.Module):

    def __init__(self, d_in, d_out, hidden_sizes, non_linearity=nn.ReLU):
        super(TupleMLP, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_sizes = hidden_sizes

        modules = [nn.Linear(self.d_in, hidden_sizes[0]), non_linearity()]

        for i in range(len(hidden_sizes) - 1):
            modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            modules.append(non_linearity())

        modules.append(nn.Linear(hidden_sizes[-1], d_out))

        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.model(x)


