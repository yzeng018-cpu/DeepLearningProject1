"""Print a Markdown table comparing all runs in runs/ (for the report).

Only the settings that differ between runs are shown (use --all-columns to show every one).

Example:
    python compare_runs.py
"""
import argparse
import json
import os

from data import ROOT
from train import build_parser

COLUMNS = ["epochs", "batch_size", "optimizer", "lr", "sched", "weight_decay", "label_smoothing", "ema",
           "widths", "blocks", "width_mult", "kernel", "block", "no_skip", "shortcut", "pool", "dropout",
           "no_crop", "no_flip", "cutout", "autoaug", "mixup", "cutmix", "teacher", "val_frac", "seed"]


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(v):
    if isinstance(v, (list, tuple)):
        return ",".join(map(str, v))
    return "" if v is None else str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    p.add_argument("--all-columns", action="store_true", help="show every setting, not only those that differ")
    args = p.parse_args()

    runs = []
    for run in sorted(os.listdir(args.runs_dir)):
        run_dir = os.path.join(args.runs_dir, run)
        cfg = load_json(os.path.join(run_dir, "config.json"))
        summ = load_json(os.path.join(run_dir, "summary.json"))
        if cfg or summ:
            runs.append((run, cfg, summ))

    defaults = vars(build_parser().parse_args([]))  # runs made before an option existed used its default
    values = {c: [fmt(cfg.get(c, defaults.get(c))) for _, cfg, _ in runs] for c in COLUMNS}
    columns = [c for c in COLUMNS if args.all_columns or len(runs) < 2 or len(set(values[c])) > 1]

    header = ["run"] + columns + ["params", "best_val_acc", "final_val_acc", "best_epoch", "time_min"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for i, (run, cfg, summ) in enumerate(runs):
        acc, final = summ.get("best_val_acc"), summ.get("final_val_acc")
        time_min = summ.get("total_time_min")
        row = [run] + [values[c][i] for c in columns] + [
            f"{summ['params']:,}" if "params" in summ else "",
            f"{acc:.4f}" if acc is not None else "",
            f"{final:.4f}" if final is not None else "",
            str(summ.get("best_epoch", "")),
            f"{time_min:.1f}" if time_min is not None else "",
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
