import torch
from torch import nn


# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 200)  # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        # Ingresso: Batch x 3 canali x 224 x 224 pixel
        x = self.conv1(x)   # -> 64 x 224 x 224
        x = torch.relu(x)
        x = self.maxpool(x) # -> 64 x 112 x 112 (dimezzato)

        # Phase 2
        x = self.conv2(x)   # -> 128 x 112 x 112
        x = torch.relu(x)
        x = self.maxpool(x) # -> 128 x 56 x 56 (dimezzato)

        # Phase 3
        x = self.conv3(x)   # -> 256 x 56 x 56
        x = torch.relu(x)

        # Final Phase
        x = self.global_pool(x) # -> 256 x 1 x 1
        x = self.flatten(x)     # -> 256
        x = self.fc1(x)         # -> 200

        return x

