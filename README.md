# Efficient ResNet on CIFAR-10 (≤ 5M parameters)

ECE-GY 7123 Deep Learning mini-project: a modified ResNet trained from scratch on the CIFAR-10 training images, under a 5M trainable-parameter budget.

**Trainable parameters: 4,949,514** (limit 5,000,000) — verify with `python model.py`.

## Repository layout
| File | Purpose |
|---|---|
| `model.py` | Modified ResNet-18: 3×3 stem (no max-pool), BasicBlocks `[2,2,2,2]`, widths `[64,128,256,240]`, global avg-pool + linear head. All of these are configurable (see *Experiment options*) |
| `data.py` | Loads training images/labels and test images; batched GPU augmentation, AutoAugment-style policies, mixup / CutMix |
| `train.py` | Training loop with stratified train/val split; writes all outputs of a run to `runs/<run-name>/` |
| `predict.py` | Inference with flip TTA; writes `runs/<run-name>/submission.csv` in `sample_submission.csv` order |
| `compare_runs.py` | Prints a Markdown table comparing the configs and val accuracy of all runs |
| `colab_run.ipynb` | One-click runner for Google Colab |

## Data
The competition data is not included in this repo. Place it in the repository root:
```
train_images/        # 50,000 PNGs (32x32 RGB)
test_images/         # 2,000 PNGs
train_labels.csv     # id,filename,label
test.csv             # id,filename
sample_submission.csv
```
- Training and validation use **only** `train_images/` + `train_labels.csv`. Validation is a stratified 10% hold-out of the training set; normalization statistics come from the training split.
- `test_images/` is read only by `predict.py` to produce predictions — never for training or model selection.
- No external data and no pretrained weights.

## Training configuration (defaults)
| Setting | Value |
|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov |
| LR schedule | OneCycle, peak LR 0.1, 15% warm-up |
| Weight decay | 5e-4 (not applied to BN / bias) |
| Batch size | 128 |
| Epochs | 100 |
| Loss | Cross-entropy, label smoothing 0.1 |
| Augmentation | Random crop (reflect pad 4), horizontal flip, Cutout 8×8 |
| Mixed precision | fp16 autocast on CUDA |
| Test-time | Horizontal-flip TTA |

All settings are command-line flags; see `python train.py --help`.

## Experiment options
Every option defaults to the baseline above, so a run only lists what it changes, e.g.
`python train.py --epochs 30 --no-skip --run-name A3_no_skip`.
Check the parameter count of an architecture without training: `python model.py --widths 80,160,320 --blocks 2,2,2`.

| Group | Flag | Default | Choices / examples |
|---|---|---|---|
| Architecture | `--widths` | `64,128,256,240` | channels per stage; the number of stages = list length |
| | `--blocks` | `2,2,2,2` | residual blocks per stage |
| | `--width-mult` | `1.0` | scale all widths, e.g. `0.25`, `0.5`, `0.75` |
| | `--kernel` | `3` | conv kernel inside blocks, e.g. `5` |
| | `--stem-kernel` | `3` | first conv kernel |
| | `--block` | `basic` | `preact` (pre-activation block) |
| | `--no-skip` | off | remove skip connections (plain CNN) |
| | `--shortcut` | `conv` | `pool` (avg-pool + 1×1 conv projection) |
| | `--pool` | `avg` | `max` global pooling |
| | `--dropout` | `0` | dropout before the classifier |
| Optimization | `--optimizer` | `sgd` | `adam`, `adamw`, `rmsprop` |
| | `--lr` | `0.1` | peak LR; use ~`1e-3`–`3e-3` for Adam-type optimizers |
| | `--sched` | `onecycle` | `cosine`, `step` (×0.1 at 50% and 75%), `constant` |
| | `--warmup-epochs` | `2` | linear warm-up for `cosine` / `step` / `constant` |
| | `--ema` | `0` | EMA decay of the weights, e.g. `0.999`; the EMA weights are evaluated and saved |
| | `--batch-size`, `--momentum`, `--weight-decay`, `--label-smoothing` | 128, 0.9, 5e-4, 0.1 | |
| Augmentation | `--no-crop`, `--no-flip` | off | disable random crop / horizontal flip |
| | `--cutout` | `8` | `0` disables |
| | `--autoaug` | `none` | `autoaugment` (CIFAR-10 policy), `randaugment`, `trivialaugment` |
| | `--mixup`, `--cutmix` | `0` | alpha, e.g. `0.2` / `1.0`; with both on, one is picked per batch |
| Distillation | `--teacher` | – | run name of a teacher (uses its `best.pt`) or a checkpoint path |
| | `--kd-alpha`, `--kd-T` | `0.5`, `4` | weight and temperature of the distillation loss |
| Run control | `--resume` | off | continue `runs/<run-name>/` from `last.pt` after a disconnect |
| | `--ckpt-every` | `1` | save `last.pt` every N epochs |
| | `--allow-over-budget` | off | allow > 5M params (teacher models only; the submitted model must be ≤ 5M) |

A teacher should be trained with the same `--seed` and `--val-frac` as the student, otherwise it has seen the student's validation images (`train.py` warns about this).

## Reproduce
```bash
pip install -r requirements.txt
python model.py                                          # parameter count
python train.py --epochs 100 --run-name run01_baseline   # 45k train / 5k val, keeps best val checkpoint
python predict.py --run-name run01_baseline              # -> runs/run01_baseline/submission.csv
python compare_runs.py                                   # table of all runs
```
To train the final model on all 50k training images, add `--val-frac 0` (then `best.pt` is the last epoch).

### Experiment outputs
Every run gets its own folder, so runs never overwrite each other:
```
runs/<run-name>/
  config.json          # all hyper-parameters of the run
  train_log.txt        # console output
  history.json         # per-epoch loss / accuracy / lr
  summary.json         # best val acc, best epoch, params, training time
  predict_config.json  # checkpoint + TTA used for the submission
  best.pt, last.pt     # checkpoints (not committed); last.pt also holds optimizer state for --resume
  submission.csv       # Kaggle submission (not committed)
```
`train.py` refuses to reuse an existing run name unless `--resume` (continue it) or `--overwrite` is given.
`compare_runs.py` only shows the settings that differ between runs (`--all-columns` shows all).

### Google Colab
1. Zip the data files listed above into `kaggle_data.zip` and upload it to `MyDrive/`.
2. Open `colab_run.ipynb` in Colab, select a GPU runtime, set `RUN_NAME` and `ARGS`, and run all cells. After a disconnect, set `RESUME = True` and run the training cell again. The notebook clones this repo, unzips the data to local disk, trains, and writes the run folder to `MyDrive/kaggle_DL_output/runs/<RUN_NAME>/`.

## Results
| Run | Config | Val acc | Kaggle public |
|---|---|---|---|
| run01_baseline | Default, 100 epochs | TBD | TBD |

## References
- K. He, X. Zhang, S. Ren, J. Sun. "Deep Residual Learning for Image Recognition." CVPR 2016.
- CIFAR ResNet layout adapted from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (MIT License).
- T. DeVries, G. Taylor. "Improved Regularization of Convolutional Neural Networks with Cutout." arXiv:1708.04552, 2017.
- L. Smith, N. Topin. "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates." arXiv:1708.07120, 2017.
- C. Szegedy et al. "Rethinking the Inception Architecture for Computer Vision." CVPR 2016 (label smoothing).
- K. He, X. Zhang, S. Ren, J. Sun. "Identity Mappings in Deep Residual Networks." ECCV 2016 (pre-activation blocks).
- E. D. Cubuk et al. "AutoAugment: Learning Augmentation Strategies from Data." CVPR 2019; "RandAugment." CVPR Workshops 2020.
- S. G. Müller, F. Hutter. "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation." ICCV 2021.
- H. Zhang et al. "mixup: Beyond Empirical Risk Minimization." ICLR 2018.
- S. Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." ICCV 2019.
- G. Hinton, O. Vinyals, J. Dean. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531, 2015.
