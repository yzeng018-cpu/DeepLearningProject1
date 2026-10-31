"""Generate runs/<run-name>/submission.csv from that run's checkpoint.

Test images are used for inference only.

Example:
    python predict.py --run-name run01_baseline
    python predict.py                 # most recently modified run
"""
import argparse
import csv
import json
import os

import torch

from data import ROOT, load_test, to_tensor
from model import ResNet, count_params
from train import get_device


def latest_run(runs_dir):
    runs = [d for d in os.listdir(runs_dir) if os.path.exists(os.path.join(runs_dir, d, "best.pt"))]
    if not runs:
        raise SystemExit(f"no trained run found in {runs_dir}")
    return max(runs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d, "best.pt")))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None, help="run folder under --runs-dir (default: latest)")
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    p.add_argument("--ckpt", default="best.pt", help="checkpoint file inside the run folder")
    p.add_argument("--no-tta", action="store_true", help="disable horizontal-flip test-time augmentation")
    args = p.parse_args()

    run_name = args.run_name or latest_run(args.runs_dir)
    run_dir = os.path.join(args.runs_dir, run_name)
    ckpt_path = os.path.join(run_dir, args.ckpt)
    out_path = os.path.join(run_dir, "submission.csv")

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet(dropout=ckpt["args"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, {count_params(model):,} params)")

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
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, lab in zip(ids, preds):
            w.writerow([i, int(lab)])
    with open(os.path.join(run_dir, "predict_config.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "epoch": ckpt["epoch"], "tta": not args.no_tta}, f, indent=1)
    print(f"Wrote {len(preds)} predictions to {out_path}")
    print("Predicted class counts:", torch.bincount(torch.tensor(preds), minlength=10).tolist())


if __name__ == "__main__":
    main()
