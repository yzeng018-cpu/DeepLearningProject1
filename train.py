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
    python train.py --epochs 100 --run-name run01_baseline --resume     # continue after a disconnect
    python train.py --help                                              # all experiment options
"""
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from data import ROOT, apply_policy, augment, load_train, make_policy, mix_batch, split_train_val, to_tensor
from model import PARAM_LIMIT, add_model_args, build_model, count_params


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD / RMSprop momentum")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--cutout", type=int, default=8, help="cutout square size, 0 disables")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of train data held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None, help="output folder name (default: run_<timestamp>)")
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"), help="parent folder of all runs")
    p.add_argument("--overwrite", action="store_true", help="allow reusing an existing run folder")
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    add_model_args(p)

    g = p.add_argument_group("optimization")
    g.add_argument("--optimizer", choices=["sgd", "adam", "adamw", "rmsprop"], default="sgd")
    g.add_argument("--sched", choices=["onecycle", "cosine", "step", "constant"], default="onecycle")
    g.add_argument("--warmup-epochs", type=int, default=2, help="linear warm-up for cosine / step / constant")
    g.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights, e.g. 0.999 (0 disables)")

    g = p.add_argument_group("augmentation")
    g.add_argument("--no-crop", action="store_true", help="disable random crop")
    g.add_argument("--no-flip", action="store_true", help="disable horizontal flip")
    g.add_argument("--autoaug", choices=["none", "autoaugment", "randaugment", "trivialaugment"], default="none")
    g.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0 disables)")
    g.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha (0 disables)")

    g = p.add_argument_group("knowledge distillation")
    g.add_argument("--teacher", default="", help="teacher run name (uses its best.pt) or a checkpoint path")
    g.add_argument("--kd-alpha", type=float, default=0.5, help="weight of the distillation loss")
    g.add_argument("--kd-T", type=float, default=4.0, help="distillation temperature")

    g = p.add_argument_group("run control")
    g.add_argument("--resume", action="store_true", help="continue from runs/<run-name>/last.pt")
    g.add_argument("--ckpt-every", type=int, default=1, help="save last.pt every N epochs")
    g.add_argument("--allow-over-budget", action="store_true", help="allow > 5M params (teacher models only)")
    return p


def get_args():
    return build_parser().parse_args()


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


def make_optimizer(model, args):
    # no weight decay on BN / bias parameters
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0}]
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, nesterov=args.momentum > 0)
    if args.optimizer == "adam":
        return torch.optim.Adam(groups, lr=args.lr)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr)
    return torch.optim.RMSprop(groups, lr=args.lr, momentum=args.momentum)


def make_scheduler(opt, args, steps_per_epoch):
    total = args.epochs * steps_per_epoch
    if args.sched == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                   steps_per_epoch=steps_per_epoch, pct_start=0.15)
    warm = args.warmup_epochs * steps_per_epoch

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        if args.sched == "cosine":
            return 0.5 * (1 + np.cos(np.pi * t))
        if args.sched == "step":
            return 0.1 ** ((t >= 0.5) + (t >= 0.75))  # x0.1 at 50% and 75% of training
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def device_rng_state(device):
    if device.type == "cuda":
        return torch.cuda.get_rng_state()
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def set_device_rng_state(device, state):
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)


def load_teacher(args, device):
    path = args.teacher if os.path.isfile(args.teacher) else os.path.join(args.runs_dir, args.teacher, "best.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)  # our own checkpoint
    teacher = build_model(ckpt["args"]).to(device)
    teacher.load_state_dict(ckpt["model"])
    return teacher.eval(), ckpt, path


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    args.run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, args.run_name)
    if os.path.exists(os.path.join(run_dir, "config.json")) and not (args.overwrite or args.resume):
        sys.exit(f"{run_dir} already exists; choose another --run-name, pass --resume to continue it, or --overwrite")
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pt")
    resume = None
    if args.resume and os.path.exists(last_path):
        resume = torch.load(last_path, map_location="cpu", weights_only=False)  # our own checkpoint
        if "opt" not in resume:
            sys.exit(f"{last_path} was saved by an older train.py without optimizer state; it cannot be resumed")
    log = Logger(os.path.join(run_dir, "train_log.txt"))
    if resume is None:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=1)
        log(f"run: {args.run_name}  ({run_dir})")
        log("config: " + json.dumps(vars(args)))
    else:
        changed = {k: (v, vars(args).get(k)) for k, v in resume["args"].items()
                   if k not in ("resume", "overwrite") and vars(args).get(k) != v}
        log(f"resuming {args.run_name} from epoch {resume['epoch']}"
            + (f"  WARNING, args differ from the original run (old, new): {changed}" if changed else ""))

    x, y = load_train()
    (xtr, ytr), val = split_train_val(x, y, args.val_frac, args.seed)
    # normalization statistics from the training split only
    mean = torch.tensor(xtr.reshape(-1, 3).mean(0) / 255, dtype=torch.float32, device=device)
    std = torch.tensor(xtr.reshape(-1, 3).std(0) / 255, dtype=torch.float32, device=device)
    xtr_t, ytr_t = to_tensor(xtr, mean, std, device), torch.from_numpy(ytr).to(device)
    if val is not None:
        xva_t, yva_t = to_tensor(val[0], mean, std, device), torch.from_numpy(val[1]).to(device)
    log(f"device={device}  train={len(ytr)}  val={0 if val is None else len(val[1])}")
    policy = make_policy(args.autoaug)

    model = build_model(vars(args)).to(device)
    n_params = count_params(model)
    log(f"Trainable parameters: {n_params:,}")
    assert n_params <= PARAM_LIMIT or args.allow_over_budget, "model exceeds the 5M parameter limit"

    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema > 0 else None
    eval_model = ema.module if ema is not None else model  # what gets evaluated and saved

    teacher = None
    if args.teacher:
        teacher, t_ckpt, t_path = load_teacher(args, device)
        # re-normalize student inputs with the teacher's statistics
        t_mean = torch.tensor(t_ckpt["mean"], device=device).view(1, 3, 1, 1)
        t_std = torch.tensor(t_ckpt["std"], device=device).view(1, 3, 1, 1)
        kd_scale, kd_shift = std.view(1, 3, 1, 1) / t_std, (mean.view(1, 3, 1, 1) - t_mean) / t_std
        log(f"teacher: {t_path} ({count_params(teacher):,} params)")
        t_args = t_ckpt["args"]
        if val is not None and (t_args.get("seed") != args.seed or t_args.get("val_frac") != args.val_frac):
            log("WARNING: teacher used a different train/val split, so it has seen part of this validation set")

    opt = make_optimizer(model, args)
    steps_per_epoch = len(ytr) // args.batch_size
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    sched = make_scheduler(opt, args, steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    ckpt_meta = {"mean": mean.tolist(), "std": std.tolist(), "args": vars(args), "params": n_params}
    history, best_acc, best_epoch, start_epoch = [], -1.0, 0, 1
    if resume is not None:
        model.load_state_dict(resume["train_model"])
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        scaler.load_state_dict(resume["scaler"])
        if ema is not None:
            ema.load_state_dict(resume["ema"])
        history, best_acc, best_epoch = resume["history"], resume["best_acc"], resume["best_epoch"]
        start_epoch = resume["epoch"] + 1
        torch.set_rng_state(resume["torch_rng"])
        np.random.set_state(resume["np_rng"])
        set_device_rng_state(device, resume.get("device_rng"))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(ytr_t), device=device)
        tot_loss, tot_correct, seen = 0.0, 0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            if policy is not None:
                xb = to_tensor(apply_policy(policy, xtr[idx.cpu().numpy()]), mean, std, device)
            else:
                xb = xtr_t[idx]
            xb = augment(xb, cutout=args.cutout, crop=not args.no_crop, flip=not args.no_flip)
            yb = ytr_t[idx]
            ya, yb2, lam = yb, yb, 1.0
            if args.mixup > 0 or args.cutmix > 0:
                xb, ya, yb2, lam = mix_batch(xb, yb, args.mixup, args.cutmix)
            with autocast:
                out = model(xb)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb2)
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(xb * kd_scale + kd_shift)
                    kd = F.kl_div(F.log_softmax(out.float() / args.kd_T, 1), F.softmax(t_out.float() / args.kd_T, 1),
                                  reduction="batchmean") * args.kd_T ** 2  # Hinton et al., 2015
                    loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot_loss += loss.item() * len(yb)
            tot_correct += (out.argmax(1) == yb).sum().item()
            seen += len(yb)

        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "train_loss": tot_loss / seen,
               "train_acc": tot_correct / seen, "time": time.time() - t0}
        if val is not None:
            rec["val_acc"], rec["val_loss"] = evaluate(eval_model, xva_t, yva_t)
        history.append(rec)
        log("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        state = {"model": eval_model.state_dict(), "epoch": epoch, **ckpt_meta}
        score = rec.get("val_acc", epoch)  # without validation, the latest epoch is kept
        if score > best_acc:
            best_acc, best_epoch = score, epoch
            torch.save(state, os.path.join(run_dir, "best.pt"))
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            state.update({"train_model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                          "scaler": scaler.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                          "history": history, "best_acc": best_acc, "best_epoch": best_epoch,
                          "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                          "device_rng": device_rng_state(device)})
            torch.save(state, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)  # atomic, so a disconnect mid-save can't corrupt it

    summary = {"run_name": args.run_name, "params": n_params, "epochs": args.epochs, "best_epoch": best_epoch,
               "best_val_acc": best_acc if val is not None else None,
               "final_val_acc": history[-1].get("val_acc") if history else None,
               "final_train_acc": history[-1]["train_acc"], "total_time_min": sum(r["time"] for r in history) / 60}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if val is not None:
        log(f"Best val acc: {best_acc:.4f} (epoch {best_epoch})")
    log(f"Saved to {run_dir}. Next: python predict.py --run-name {args.run_name}")


if __name__ == "__main__":
    main()
