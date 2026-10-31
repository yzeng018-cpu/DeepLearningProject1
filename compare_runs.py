"""Print a Markdown table comparing all runs in runs/ (for the report).

Example:
    python compare_runs.py
"""
import argparse
import json
import os

from data import ROOT

COLUMNS = ["epochs", "batch_size", "lr", "weight_decay", "label_smoothing", "cutout", "dropout", "val_frac"]


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    args = p.parse_args()

    header = ["run"] + COLUMNS + ["params", "best_val_acc", "best_epoch", "time_min"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for run in sorted(os.listdir(args.runs_dir)):
        run_dir = os.path.join(args.runs_dir, run)
        cfg = load_json(os.path.join(run_dir, "config.json"))
        summ = load_json(os.path.join(run_dir, "summary.json"))
        if not cfg and not summ:
            continue
        acc = summ.get("best_val_acc")
        time_min = summ.get("total_time_min")
        row = [run] + [str(cfg.get(c, "")) for c in COLUMNS] + [
            f"{summ['params']:,}" if "params" in summ else "",
            f"{acc:.4f}" if acc is not None else "",
            str(summ.get("best_epoch", "")),
            f"{time_min:.1f}" if time_min is not None else "",
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
