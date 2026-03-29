import torch.nn as nn

class CycleDiscriminator(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1,   4, padding=1)
        )

    def forward(self, x):
        return self.model(x)