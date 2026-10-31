"""Train the modified ResNet on CIFAR-10 training images only.

Each run writes everything to runs/<run-name>/:
    config.json   hyper-parameters of this run
    train_log.txt console output
    history.json  per-epoch metrics
    summary.json  best val acc / epoch, parameter count
    best.pt, last.pt

Example:
    python train.py --epochs 100 --run-name run01_baseline              # 90/10 train/val split
    python train.py --epochs 100 --val-frac 0 --run-name run05_full50k  # train on all 50k images
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from data import ROOT, augment, load_train, split_train_val, to_tensor
from model import PARAM_LIMIT, ResNet, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate (OneCycle)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cutout", type=int, default=8, help="cutout square size, 0 disables")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of train data held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None, help="output folder name (default: run_<timestamp>)")
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"), help="parent folder of all runs")
    p.add_argument("--overwrite", action="store_true", help="allow reusing an existing run folder")
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Logger:
    """Print to the console and append to <run_dir>/train_log.txt."""

    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(x), batch_size):
        out = model(x[i:i + batch_size])
        loss_sum += nn.functional.cross_entropy(out, y[i:i + batch_size], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / len(x), loss_sum / len(x)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    args.run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, args.run_name)
    if os.path.exists(os.path.join(run_dir, "config.json")) and not args.overwrite:
        sys.exit(f"{run_dir} already exists; choose another --run-name or pass --overwrite")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    log = Logger(os.path.join(run_dir, "train_log.txt"))
    log(f"run: {args.run_name}  ({run_dir})")
    log("config: " + json.dumps(vars(args)))

    x, y = load_train()
    (xtr, ytr), val = split_train_val(x, y, args.val_frac, args.seed)
    # normalization statistics from the training split only
    mean = torch.tensor(xtr.reshape(-1, 3).mean(0) / 255, dtype=torch.float32, device=device)
    std = torch.tensor(xtr.reshape(-1, 3).std(0) / 255, dtype=torch.float32, device=device)
    xtr_t, ytr_t = to_tensor(xtr, mean, std, device), torch.from_numpy(ytr).to(device)
    if val is not None:
        xva_t, yva_t = to_tensor(val[0], mean, std, device), torch.from_numpy(val[1]).to(device)
    log(f"device={device}  train={len(ytr)}  val={0 if val is None else len(val[1])}")

    model = ResNet(dropout=args.dropout).to(device)
    n_params = count_params(model)
    log(f"Trainable parameters: {n_params:,}")
    assert n_params <= PARAM_LIMIT, "model exceeds the 5M parameter limit"

    # no weight decay on BN / bias parameters
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0}],
        lr=args.lr, momentum=args.momentum, nesterov=True,
    )
    steps_per_epoch = len(ytr) // args.batch_size
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.15)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_meta = {"mean": mean.tolist(), "std": std.tolist(), "args": vars(args), "params": n_params}
    history, best_acc, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(ytr_t), device=device)
        tot_loss, tot_correct, seen = 0.0, 0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            xb, yb = augment(xtr_t[idx], cutout=args.cutout), ytr_t[idx]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(xb)
                loss = criterion(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += (out.argmax(1) == yb).sum().item()
            seen += len(yb)

        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "train_loss": tot_loss / seen,
               "train_acc": tot_correct / seen, "time": time.time() - t0}
        if val is not None:
            rec["val_acc"], rec["val_loss"] = evaluate(model, xva_t, yva_t)
        history.append(rec)
        log("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        state = {"model": model.state_dict(), "epoch": epoch, **ckpt_meta}
        torch.save(state, os.path.join(run_dir, "last.pt"))
        score = rec.get("val_acc", epoch)  # without validation, the latest epoch is kept
        if score > best_acc:
            best_acc, best_epoch = score, epoch
            torch.save(state, os.path.join(run_dir, "best.pt"))

    summary = {"run_name": args.run_name, "params": n_params, "epochs": args.epochs, "best_epoch": best_epoch,
               "best_val_acc": best_acc if val is not None else None,
               "final_train_acc": history[-1]["train_acc"], "total_time_min": sum(r["time"] for r in history) / 60}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if val is not None:
        log(f"Best val acc: {best_acc:.4f} (epoch {best_epoch})")
    log(f"Saved to {run_dir}. Next: python predict.py --run-name {args.run_name}")


if __name__ == "__main__":
    main()
