#!/bin/bash
# Member 1 optional experiments (30 epochs each). Run inside the Singularity container
# from the project folder. Every run uses --resume, so re-running this script after a
# preemption / timeout skips finished runs and continues the unfinished one.
# Locally (e.g. macOS, where only python3 exists):  PYTHON=python3 bash hpc/run_optional_A.sh
set -e

EXPS=(
  "A1_shallow4|--widths 64,128,256,512 --blocks 1,1,1,1"
  "A1_last1|--widths 64,128,256,320 --blocks 2,2,2,1"
  "A4_k5|--kernel 5 --widths 38,76,152,148"
  "A5_preact|--block preact"
  "A5_poolshortcut|--shortcut pool"
  "A5_maxpool|--pool max"
  "C1_step|--sched step"
  "C2_adam|--optimizer adam --lr 1e-3"
  "C2_rmsprop|--optimizer rmsprop --lr 1e-3"
)

for e in "${EXPS[@]}"; do
  name="${e%%|*}"
  extra="${e#*|}"
  echo "===== $name: $extra ====="
  # $extra is intentionally unquoted so it splits into separate flags
  ${PYTHON:-python} train.py --epochs 30 $extra --resume --run-name "$name"
done
