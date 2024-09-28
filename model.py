import torch
import torch.nn as nn
import numpy as np

rows = 2
columns = 7

hidden_layers = [25, 25, 25]
output_dimension = columns

class Network(nn.Module):
    def __init__(self, weights1=None, weights2=None, mult1=1, mult2=1, input_dim=(rows, columns), hidden_layers=hidden_layers, output_dim=output_dimension, activation_fn=nn.ReLU):
        super.__init__()
        
        kernel_size = 3
        layers = [
            nn.Conv2d(1, hidden_layers[0], kernel_size),
            activation_fn(),
            nn.Flatten(),
            nn.Linear(hidden_layers[0] * (input_dim[0] - kernel_size + 1) * (input_dim[1] - kernel_size + 1), hidden_layers[0]),
            activation_fn()
        ]

        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation_fn())

        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        self.sequential = nn.Sequential(*layers)


