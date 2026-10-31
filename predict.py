"""Generate submission.csv from a trained checkpoint (test images used for inference only)."""
import argparse
import csv
import os

import torch

from data import ROOT, load_test, to_tensor
from model import ResNet, count_params
from train import get_device


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "best.pt"))
    p.add_argument("--out", default=os.path.join(ROOT, "submission.csv"))
    p.add_argument("--no-tta", action="store_true", help="disable horizontal-flip test-time augmentation")
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    model = ResNet(dropout=ckpt["args"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt} (epoch {ckpt['epoch']}, {count_params(model):,} params)")

    mean = torch.tensor(ckpt["mean"], device=device)
    std = torch.tensor(ckpt["std"], device=device)
    x, ids = load_test()
    xt = to_tensor(x, mean, std, device)

    preds = []
    for i in range(0, len(xt), 500):
        xb = xt[i:i + 500]
        probs = model(xb).softmax(1)
        if not args.no_tta:
            probs = probs + model(xb.flip(3)).softmax(1)
        preds.append(probs.argmax(1).cpu())
    preds = torch.cat(preds).tolist()

    assert len(preds) == len(ids)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, lab in zip(ids, preds):
            w.writerow([i, int(lab)])
    print(f"Wrote {len(preds)} predictions to {args.out}")
    print("Predicted class counts:", torch.bincount(torch.tensor(preds), minlength=10).tolist())


if __name__ == "__main__":
    main()
