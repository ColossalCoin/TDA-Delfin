import torch.nn as nn
import torch

class LeNet5Top(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Capa de entrada
        self.conv1 = nn.Conv2d(1 + 756, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        
        # Capa 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        
        # Capa 3
        self.conv3 = nn.Conv2d(16, 120, 5)
        
        # Capa densa 1
        self.dense1 = nn.Linear(120 * 4 * 4 + 756, 84)
        
        # Capa densa 2
        self.dense2 = nn.Linear(84, 2)
        
    def forward(self, x, v):
        
        # Pasa la imagen por las capas convolucionales
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        
        # Aplana la salida
        x = x.view(-1, 120 * 4 * 4)
        x = torch.cat(x, v)
        
        # Pasa la entrada por las capas densas
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x
        