import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3136,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        z = self.fc2(x)
        return z

if __name__ == "__main__":
    import torch
    net = MnistNet()
    out = net(torch.randn(1, 1, 28, 28))
    print(out.shape)
    print(sum(p.numel() for p in net.parameters()))
