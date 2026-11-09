# Saved data used by the notebooks

These files let a reader run the figures without downloading CIFAKE or training the models again. The notebooks calculate and draw their plots from these arrays; the results are from the completed experiments.

| File | Contents and source |
| --- | --- |
| `training_data_extra_bundle.npz` | Properties, object labels, mean images and other saved measurements for all 88,702 cleaned training images. The data notebook shows and checks the four property calculations it uses. |
| `training_preview.npz` | Twelve original 32×32 RGB training images, their labels and their row indices in the audited training cache. Six images per label were sampled with NumPy seed 2026, FAKE first and REAL second. No validation or test pixels are included. |
| `cnn_training_report.json` | Actual CNN epoch history, selected epoch and clean evaluation, copied from the embedded original training records. |
| `resnet_training_report.json` | Actual ResNet18 epoch history, selected epoch and clean evaluation, copied from the completed run's report. |
| `phase1_behaviour_bundle.npz` | Both selected models' REAL logits for the same 10,000 validation images under 29 conditions, edited-image properties and 600 preview images. The previews contain 30 images per label and object class. |

In all arrays, FAKE is label 0 and REAL is label 1. A negative REAL logit predicts FAKE. Training properties use native 32×32 pixels scaled to 0–1. The data notebook recomputes its four properties on the twelve previews and checks them against the saved full-training measurements.

The models used the same audited cache, `data/baseline_seed1/`, prepared by `baseline_training.ipynb` at the repository root. That original notebook retains the data preparation and duplicate audit. Bundle-building code is preserved in Git history at commit `d0be3c0`, in the exploration and shared behaviour notebooks. The previous versions are also archived locally under `variants/previous-phase1/`.

The reports identify the original model and split. Review reads these saved results. A new training run writes into its own `runs/` folder and does not replace the reports or behaviour predictions used for the submission figures.
