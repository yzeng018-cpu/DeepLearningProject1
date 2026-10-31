"""Modified ResNet for CIFAR-10 (<= 5M trainable parameters).

Based on the ResNet-18 layout from He et al., "Deep Residual Learning for
Image Recognition" (CVPR 2016), adapted to 32x32 inputs (3x3 stem, no max-pool)
as in https://github.com/kuangliu/pytorch-cifar, with reduced channel widths.

Print the parameter count of any architecture, e.g.:
    python model.py
    python model.py --widths 80,160,320 --blocks 2,2,2
"""
import argparse

import torch.nn as nn
import torch.nn.functional as F

PARAM_LIMIT = 5_000_000


def projection(in_ch, out_ch, stride, kind):
    """'conv': strided 1x1 conv. 'pool': avg-pool then 1x1 conv (ResNet-D style)."""
    if kind == "pool" and stride != 1:
        return [nn.AvgPool2d(stride), nn.Conv2d(in_ch, out_ch, 1, bias=False)]
    return [nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)]


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel=3, skip=True, shortcut="conv"):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = skip
        self.shortcut = nn.Sequential()
        if skip and (stride != 1 or in_ch != out_ch):
            self.shortcut = nn.Sequential(*projection(in_ch, out_ch, stride, shortcut), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip:
            out = out + self.shortcut(x)
        return F.relu(out)


class PreActBlock(nn.Module):
    """Pre-activation block (He et al., ECCV 2016): bn-relu-conv-bn-relu-conv + shortcut."""

    def __init__(self, in_ch, out_ch, stride=1, kernel=3, skip=True, shortcut="conv"):
        super().__init__()
        pad = kernel // 2
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel, 1, pad, bias=False)
        self.skip = skip
        self.shortcut = None
        if skip and (stride != 1 or in_ch != out_ch):
            self.shortcut = nn.Sequential(*projection(in_ch, out_ch, stride, shortcut))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + sc if self.skip else out


class ResNet(nn.Module):
    def __init__(self, blocks=(2, 2, 2, 2), widths=(64, 128, 256, 240), num_classes=10, dropout=0.0,
                 kernel=3, stem_kernel=3, block="basic", skip=True, shortcut="conv", pool="avg"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], stem_kernel, 1, stem_kernel // 2, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        Block = PreActBlock if block == "preact" else BasicBlock
        layers, in_ch = [], widths[0]
        for i, (n, w) in enumerate(zip(blocks, widths)):
            stride = 1 if i == 0 else 2
            for j in range(n):
                layers.append(Block(in_ch, w, stride if j == 0 else 1, kernel, skip, shortcut))
                in_ch = w
        self.layers = nn.Sequential(*layers)
        # pre-activation nets need a final BN-ReLU before pooling
        self.final = nn.Sequential(nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True)) if block == "preact" else nn.Identity()
        self.pool = nn.AdaptiveMaxPool2d(1) if pool == "max" else nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.final(self.layers(self.stem(x)))
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def int_list(s):
    return tuple(int(v) for v in str(s).strip("()[] ").split(","))


def add_model_args(p):
    g = p.add_argument_group("architecture")
    g.add_argument("--widths", type=int_list, default=(64, 128, 256, 240), help="channels per stage, e.g. 64,128,256,240")
    g.add_argument("--blocks", type=int_list, default=(2, 2, 2, 2), help="residual blocks per stage, e.g. 2,2,2,2")
    g.add_argument("--width-mult", type=float, default=1.0, help="scale all widths")
    g.add_argument("--kernel", type=int, default=3, help="conv kernel size inside residual blocks")
    g.add_argument("--stem-kernel", type=int, default=3)
    g.add_argument("--block", choices=["basic", "preact"], default="basic")
    g.add_argument("--no-skip", action="store_true", help="remove skip connections (plain CNN)")
    g.add_argument("--shortcut", choices=["conv", "pool"], default="conv", help="projection shortcut type")
    g.add_argument("--pool", choices=["avg", "max"], default="avg", help="global pooling")
    g.add_argument("--dropout", type=float, default=0.0, help="dropout before the classifier")


def build_model(cfg):
    """Build a ResNet from a run config dict (vars(args) or a checkpoint's 'args').
    Missing keys fall back to the defaults, so checkpoints from older runs still load."""
    widths = int_list(cfg.get("widths", (64, 128, 256, 240)))
    widths = tuple(max(8, round(w * cfg.get("width_mult", 1.0))) for w in widths)
    return ResNet(blocks=int_list(cfg.get("blocks", (2, 2, 2, 2))), widths=widths,
                  dropout=cfg.get("dropout", 0.0), kernel=cfg.get("kernel", 3),
                  stem_kernel=cfg.get("stem_kernel", 3), block=cfg.get("block", "basic"),
                  skip=not cfg.get("no_skip", False), shortcut=cfg.get("shortcut", "conv"),
                  pool=cfg.get("pool", "avg"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_model_args(p)
    m = build_model(vars(p.parse_args()))
    n = count_params(m)
    print(f"Trainable parameters: {n:,} (limit {PARAM_LIMIT:,}) -> {'OK' if n <= PARAM_LIMIT else 'EXCEEDS LIMIT'}")
