# Efficient ResNet on CIFAR-10 (≤ 5M parameters)

ECE-GY 7123 Deep Learning mini-project: a modified ResNet trained from scratch on the CIFAR-10 training images, under a 5M trainable-parameter budget.

**Trainable parameters: 4,949,514** (limit 5,000,000) — verify with `python model.py`.

## Repository layout
| File | Purpose |
|---|---|
| `model.py` | Modified ResNet-18: 3×3 stem (no max-pool), BasicBlocks `[2,2,2,2]`, widths `[64,128,256,240]`, global avg-pool + linear head |
| `data.py` | Loads training images/labels and test images; batched GPU augmentation |
| `train.py` | Training loop with stratified train/val split; saves `best.pt`, `last.pt`, `history.json` |
| `predict.py` | Inference with flip TTA; writes `submission.csv` in `sample_submission.csv` order |
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

## Reproduce
```bash
pip install -r requirements.txt
python model.py                    # parameter count
python train.py --epochs 100       # 45k train / 5k val, keeps best val checkpoint
python predict.py                  # -> submission.csv
```
To train the final model on all 50k training images: `python train.py --epochs 100 --val-frac 0` (then `best.pt` is the last epoch).

### Google Colab
1. Zip the data files listed above into `kaggle_data.zip` and upload it to `MyDrive/`.
2. Open `colab_run.ipynb` in Colab, select a GPU runtime, and run all cells. The notebook clones this repo, unzips the data to local disk, trains, and writes checkpoints / `submission.csv` to `MyDrive/kaggle_DL_output/`.

## Results
| Config | Val acc | Kaggle public |
|---|---|---|
| Default, 100 epochs | TBD | TBD |

## References
- K. He, X. Zhang, S. Ren, J. Sun. "Deep Residual Learning for Image Recognition." CVPR 2016.
- CIFAR ResNet layout adapted from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (MIT License).
- T. DeVries, G. Taylor. "Improved Regularization of Convolutional Neural Networks with Cutout." arXiv:1708.04552, 2017.
- L. Smith, N. Topin. "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates." arXiv:1708.07120, 2017.
- C. Szegedy et al. "Rethinking the Inception Architecture for Computer Vision." CVPR 2016 (label smoothing).
