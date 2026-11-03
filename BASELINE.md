# CIFAKE baseline training and data exploration

This guide covers the trained plain CNN and its notebook workflow for Team Kandy. See [README.md](README.md) for the team project overview.

## Working notebook

Open **[baseline_training.ipynb](baseline_training.ipynb)** for the team's baseline workflow. It includes editable training settings, the split audit, data exploration, model architecture, training and resume cells, automatic checkpoints, learning curves, validation examples, final test metrics, and export for sharing. Saved outputs show the completed baseline. The notebook calls the shared Python implementation, so it uses the same split and checkpoint logic as the scripts.

Select the project's `.venv/bin/python` kernel in VS Code, or start JupyterLab from the repository directory:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name cifake --display-name "Python (CIFAKE)"
.venv/bin/jupyter lab baseline_training.ipynb
```

The notebook defaults to `MODE = "review"`, which reads saved results without retraining or repeating test evaluation. To train a separate model, set `MODE = "train"` and choose a new `RUN_NAME`. To recover an interrupted run, use `MODE = "resume"` with that run's name. Data preparation, final test evaluation, and export each have an explicit flag in the configuration cell. Teammates can review the exported results and load the compact model without downloading the full dataset; validation-image previews need the local prepared data.

Section 4 explores the training data with split-count charts, exact-duplicate examples, balanced random galleries, brightness/contrast/saturation/sharpness distributions, sharpness extremes, a scatterplot linked to image thumbnails, and normalized Fourier spectra. Property distributions and spectra use all 88,702 cleaned training images; galleries and the 1,600-point scatterplot use fixed samples. This section was added after the baseline run and does not change training or test results. Test pixels are not used for exploration; the split audit uses metadata from all splits.

Run the notebook with a live kernel for the Plotly thumbnail explorer: choose its axes, then click a point to inspect the corresponding training image. The interactive view requires the pinned `plotly`, `anywidget`, and `ipywidgets` packages. Static plots are also embedded for GitHub previews and saved with a numerical summary in [results/eda_baseline_seed1/](results/eda_baseline_seed1/). Without local training data, the notebook displays these saved plots. Class differences are descriptive associations, not proof that the CNN uses those properties.

## Download the dataset

With Python 3.12 installed, run from the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_dataset.py
```

The script downloads CIFAKE to `data/cifake/`, containing `train/REAL`, `train/FAKE`, `test/REAL`, and `test/FAKE`. The dataset and local Python environment are excluded from Git. If the environment is already set up, activate it and run the final command.

## Plain CNN baseline

The first experiment trains from scratch **without any data augmentation**. Images stay at 32×32 RGB; preprocessing only scales pixels to `[0, 1]`. This establishes the detector's behavior before an augmentation comparison. Failure under a new transformation measures sensitivity; by itself, it does not prove shortcut learning.

The CNN is inspired by [Bird and Lotfi's CIFAKE paper, Section 4.2 and Figure 5](https://arxiv.org/html/2303.14126v1): two convolutional layers with 32 filters and a 64-unit dense layer. Our implementation uses 3×3 valid convolutions, ReLU and 2×2 max pooling after each convolution, followed by flattening, the dense layer, and one output logit: **84,001 parameters**. `FAKE=0`, `REAL=1`, so sigmoid(output) is P(REAL). There is no dropout, batch normalization, weight decay, or pretraining. This is a documented paper-inspired baseline, not an exact reproduction of the authors' training procedure.

The data audit hashes **decoded RGB pixels**, rather than filenames or JPEG bytes. It found 378 training images identical to official test images and another 920 repeated training copies. All 1,298 excluded copies are FAKE; source files are preserved. A seeded holdout of 5,000 images per label is then drawn from the unique remaining training images:

| Split | FAKE | REAL | Total |
| --- | ---: | ---: | ---: |
| Training | 43,702 | 45,000 | 88,702 |
| Validation | 5,000 | 5,000 | 10,000 |
| Official test | 10,000 | 10,000 | 20,000 |

There is **zero exact pixel overlap** across these splits. The official test set contains 38 repeated copies; final evaluation reports both the full official set and its 19,962 unique images. Near-duplicates have not been screened. Stratification uses REAL/FAKE labels; semantic class names are not inferred from filenames.

Training uses seed 1, Adam at learning rate 0.001, batch size 128, and binary cross-entropy with logits. It runs for up to 30 epochs, halves the learning rate after a validation-loss plateau, and stops after five epochs without a validation-loss improvement of at least 0.0001. The checkpoint with the lowest validation loss is selected. The decision threshold is fixed at 0.5. **The training program never loads test images; test evaluation is a separate final step.**

```bash
# For the existing uv-managed environment (which may not have pip):
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m unittest -v test_baseline test_resume
.venv/bin/python prepare_baseline_data.py
.venv/bin/python train_baseline.py
.venv/bin/python evaluate_baseline.py
.venv/bin/python plot_baseline.py
.venv/bin/python export_baseline.py
```

The device is selected automatically: CUDA, Apple MPS, then CPU. Override with `--device cpu`, `--device mps`, or `--device cuda`. On macOS, `caffeinate -i .venv/bin/python train_baseline.py` can keep the machine awake during training. Seeds and environment versions are recorded, but bitwise reproducibility across hardware/software is not guaranteed.

Prepared data is stored in `data/baseline_seed1/`, including `audit.json`, a manifest of every original image and exclusion, and pixel caches with integrity hashes. **Reuse this directory** for future comparisons; preparation refuses to overwrite it. Runs similarly refuse to overwrite an existing output directory; use `--output runs/<new-name>` for a separate training run. Test evaluation refuses to overwrite existing results.

The default run directory is `runs/baseline_plain_seed1/`. It contains `best.pt`, `last.pt`, every epoch checkpoint, source snapshots, configuration and package versions, training history/logs, validation/test metrics, and per-image probabilities for later visualization. `status.json` reports progress. Local data, caches, and runs are excluded from Git. Use validation images for developing robustness probes or choosing later training settings; retain the test set for fixed evaluations.

## Checkpoints and team sharing

After each complete epoch, training saves `checkpoints/epoch_NNN.pt`, updates `last.pt`, and updates `best.pt` if validation loss improves. Each contains weights, optimizer state, learning-rate scheduler state, random-generator states, validation metrics, and configuration. New saves are written to a temporary file and atomically renamed; future saves also flush the checkpoint to disk before replacement. An interruption can lose the unfinished epoch, but completed epochs remain available.

To continue an interrupted, **not yet test-evaluated** run:

```bash
.venv/bin/python train_baseline.py --resume --output runs/<run-name>
# On a teammate's computer, if the prepared data lives elsewhere:
.venv/bin/python train_baseline.py --resume --output runs/<run-name> --data-dir data/baseline_seed1
```

Resume restores the last complete epoch and checks data identity by hashes, not absolute paths. It retains the original hyperparameters and stopping rule. A completed, test-evaluated baseline is kept fixed; later training comparisons should use a new run directory. A synthetic-data test verifies that uninterrupted and resumed CPU training produce identical weights and metrics. Cross-device bitwise equivalence is not promised.

For the team's [GitHub repository](https://github.com/adamp-22/kandy):

- **Keep in Git:** `baseline_training.ipynb`, the shared Python scripts, pinned requirements, this guide (`BASELINE.md`), `ai-log.jsonl`, `models/baseline_plain_seed1.pt` (compact CPU inference weights), `results/baseline_plain_seed1/` (metrics, audit, history, and learning curves), and `results/eda_baseline_seed1/` (exploration figures and summary).
- **Keep out of normal Git history:** downloaded images, prepared pixel caches, the virtual environment, and full training runs. All are covered by `.gitignore`.
- **Full checkpoint history:** `export_baseline.py` produces `artifacts/baseline_plain_seed1.tar.gz` and a SHA-256 checksum. This can be shared as a GitHub Release asset or through team storage. Extract it under `runs/` to recover the run. Exporting creates local files; it does not publish a release or push to GitHub.

Cloning or pulling this repository downloads `models/baseline_plain_seed1.pt` along with the code. This compact checkpoint contains the trained weights selected at epoch 19 and is about 333 KiB; it can run on a CPU. No training dataset or full checkpoint archive is needed to load the model or predict on supplied images.

After installing `requirements.txt`, open `baseline_training.ipynb`, keep `MODE = "review"`, and run the cells. Section 9 loads the shared model automatically when local full-run checkpoints are absent. Saved plots and metrics can also be viewed directly on GitHub. The interactive training-image explorer and validation examples need the separately downloaded and prepared dataset.

To load the same model directly in Python, run this from the repository directory:

```python
import torch
from baseline import BaselineCNN

checkpoint = torch.load("models/baseline_plain_seed1.pt", map_location="cpu", weights_only=True)
model = BaselineCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
# Input: [batch, 3, 32, 32] float32 RGB pixels divided by 255.
# p_real = torch.sigmoid(model(images)); p_fake = 1 - p_real.
```

The first plain baseline completed 24 epochs and selected **epoch 19 by validation loss**: validation accuracy **94.54%**, official test accuracy **94.29%**, and test ROC-AUC **0.9871**. It correctly detected 94.18% of fake images and flagged 5.60% of real images as fake. Unique-image test accuracy was 94.294%. See [saved results](results/baseline_plain_seed1/summary.json) and [learning curves](results/baseline_plain_seed1/learning_curves.png). These are single-seed results on clean CIFAKE; robustness has not yet been measured.
