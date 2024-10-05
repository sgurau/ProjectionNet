# projection_net.py

import torch
import torch.nn as nn
import torch.utils.data as data

class ProjectionNet(nn.Module):
    def __init__(self, input_size, projection_size, num_layers, activation_fn):
        super(ProjectionNet, self).__init__()
        self.layers = nn.Sequential()
        
        for i in range(num_layers):
            
            # Linear layer
            layer = nn.Linear(input_size if i == 0 else projection_size, projection_size)
            self.layers.add_module(f"Linear_{i}", layer)
            
            # Batch normalization
            self.layers.add_module(f"BatchNorm_{i}", nn.BatchNorm1d(projection_size))
             
            # Add the activation function directly without calling it
            self.layers.add_module(f"Activation_{i}", activation_fn) 
            
            # Dropout
            self.layers.add_module(f"Dropout_{i}", nn.Dropout(p=0.5))
            
        # Final Projection Layer    
        self.final = nn.Linear(projection_size, projection_size)

    def forward(self, x):
        x = self.layers(x)
        x = self.final(x)
        return x

class PairwiseDataset(data.Dataset):
    def __init__(self, distances_matrix):
        self.distances_matrix = distances_matrix

    def __getitem__(self, index):
        n = self.distances_matrix.size(0)
        i, j = index // n, index % n
        return self.distances_matrix[i], self.distances_matrix[j], self.distances_matrix[i, j]

    def __len__(self):
        return self.distances_matrix.size(0) ** 2
