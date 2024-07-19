import torch.nn as nn

class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1 (224, 224, 1) -> (220, 220, 6)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5)
        # P1: (220, 220, 6) -> (110, 110, 6)
        self.pool1 = nn.MaxPool2d(2)
        
        # C2: (110, 110, 6) -> (106, 106, 16)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # P2: (106, 106, 16) -> (53, 53, 16)
        self.pool2 = nn.MaxPool2d(2)
        
        # C3: (53, 53, 16) -> (49, 49, 120)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        
        # FC1: (228 120) -> (120)
        self.dense1 = nn.Linear(228120, 84)
        # FC2: (120) -> (84)
        self.dense2 = nn.Linear(84, 2)
        
    def forward(self, x):
        # Se aplican las convoluciones
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Se apalana la salida
        x = x.view(-1, 228120)
        
        # Se aplican las capas densas
        x = self.dense1
        x= self.dense2
        
        return x