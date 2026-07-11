import torch
import torch.nn as nn

# Input
torch.manual_seed(42)
inputs = torch.tensor([1.0, 2.0, 3.0])

neuron = nn.Linear(in_features = 3, out_features = 1)
output = neuron(inputs)

print(output)

# Verify .... Formula
neuron.weight @ inputs + neuron.bias
neuron.bias