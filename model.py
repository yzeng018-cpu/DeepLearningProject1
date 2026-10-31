"""Modified ResNet for CIFAR-10 (<= 5M trainable parameters).

Based on the ResNet-18 layout from He et al., "Deep Residual Learning for
Image Recognition" (CVPR 2016), adapted to 32x32 inputs (3x3 stem, no max-pool)
as in https://github.com/kuangliu/pytorch-cifar, with reduced channel widths.
"""
import torch.nn as nn
import torch.nn.functional as F

PARAM_LIMIT = 5_000_000


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, blocks=(2, 2, 2, 2), widths=(64, 128, 256, 240), num_classes=10, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        layers, in_ch = [], widths[0]
        for i, (n, w) in enumerate(zip(blocks, widths)):
            stride = 1 if i == 0 else 2
            for j in range(n):
                layers.append(BasicBlock(in_ch, w, stride if j == 0 else 1))
                in_ch = w
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ResNet()
    n = count_params(m)
    print(f"Trainable parameters: {n:,} (limit {PARAM_LIMIT:,}) -> {'OK' if n <= PARAM_LIMIT else 'EXCEEDS LIMIT'}")
