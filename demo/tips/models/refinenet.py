"""TIPS: Text-Induced Pose Synthesis

Stage-2 network: RefineNet regressor
Created on Thu Nov 18 10:00:00 2021
Author: Prasun Roy | https://prasunroy.github.io
GitHub: https://github.com/prasunroy/tips

"""


import torch
import torch.nn as nn


class RefineNet(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(RefineNet, self).__init__()
        self.linear1 = nn.Linear(in_features, 128, bias=bias)
        self.linear2 = nn.Linear(128, 128, bias=bias)
        self.linear3 = nn.Linear(128, 128, bias=bias)
        self.linear4 = nn.Linear(128, out_features, bias=bias)
    
    def forward(self, x):
        y = torch.relu(self.linear1(x))
        y = torch.relu(self.linear2(y))
        y = torch.relu(self.linear3(y))
        y = torch.tanh(self.linear4(y))
        return y
