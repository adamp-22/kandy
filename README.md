# Kandy

Team Kandy's project for the AI and Visualization Lab at JKU (Visual Data Science Lab), winter semester 2026.

## Topic: What fools an AI-image detector?

Robustness and shortcut learning in a classifier that separates real photos from AI-generated images.

The CIFAKE authors' own Grad-CAM analysis suggested that their detector ignores the object and focuses on small imperfections in the background. If that is true, simple edits such as JPEG compression, resizing or blurring could erase that trace and let fakes pass. We use visualization to find out what fools the detector and what it actually looks at.

### Data and models

- **Dataset:** [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), 120,000 images at 32×32: 60k real photos from CIFAR-10 and 60k generated with Stable Diffusion, in the same 10 classes. We add our own test variants (JPEG at several quality levels, resizing, blur, noise, screenshots).
- **Models:** two real vs. AI-generated classifiers, to compare pretrained and from scratch:
  - [ResNet-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), pretrained on ImageNet and fine-tuned on CIFAKE
  - our own small CNN, trained from scratch on CIFAKE at 32×32
- **Optional:** [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) for a comparison on larger images.

### Plan

**Phase I: how does the detector behave?** (black box)

- How fast detection collapses under JPEG, blur, resizing and noise (robustness curves per class)
- Which classes are hardest, and which real photos get flagged as fake
- How real and fake training images differ in simple statistics (sharpness, colour, noise)

**Phase II: what did it learn?** (inside the model)

- Grad-CAM: share of saliency on the object vs. the background, for correct vs. fooled images
- Frequency analysis: Fourier spectra of real vs. fake images, low-pass and high-pass filter tests
- Per-layer representation maps and nearest neighbours in the training set
- Checkpoints: when during training the shortcut appears

**AI4VIS:** one pair works with generative AI and the other without, and the pairs swap after Phase I. The AI-assisted pair keeps a log of prompts and of which suggestions were accepted, modified or rejected.

## Team

| Name | GitHub |
| --- | --- |
| Kavindu Malshan Pathirage Don | [kavimalshan](https://github.com/kavimalshan) |
| G.L. | [LordNassel](https://github.com/LordNassel) |
| Adam Parkanyi | [adamp-22](https://github.com/adamp-22) |

## Deadlines

| Date | Deliverable |
| --- | --- |
| 11.11.2026 | Midpoint (end of Phase I): code from both pairs, AI interaction log, progress video (max. 5 min) |
| 09.12.2026 | Final (end of Phase II): all code, AI logs from both phases, 10-minute recorded presentation and slides |

## Contents

| Path | Phase | What it contains |
| --- | --- | --- |
| `phase1/01_data_exploration.ipynb` | I | Load and visualize the training data: examples, counts, colour, sharpness, frequencies and pixel projections |
| `phase1/02_cnn_training_and_visualization.ipynb` | I | CNN architecture and training; saved learning curves and evaluation; Part A standard figures; Part B AI-assisted behaviour analysis |
| `phase1/03_resnet18_finetuning_and_visualization.ipynb` | I | ResNet18 classifier warmup and full fine-tuning; saved learning curves and evaluation; Part A standard figures; Part B AI-assisted analysis and CNN comparison |
| `phase1/data/` | I | Saved training reports, measurements, predictions and preview images used by the three notebooks |
| `phase2/01_cnn_internals.ipynb` | II | CNN Grad-CAM, neuron contributions, occlusion, filters, channel removal |
| `phase2/02_resnet18_internals.ipynb` | II | ResNet18 Grad-CAM, output contributions, occlusion, stem filters, channel interventions, explorer |
| `baseline_training.ipynb`, `resnet18_finetuning.ipynb` | Original pipelines | Original data preparation, training/resume and model-export tools; retained for reproducibility and Phase II |
| `docs/visualization_overview.md` | I + II | Where to find the training steps and figures, and how the two visualization sections are labelled |
| `team-info.json`, `workload.json` | — | Team registration |

**Phase I now has exactly three notebooks.** Open `phase1/` in VS Code and read them in order.
Training is included directly in `02` and `03`; each model notebook then has its own two figure sections.
Code is divided into small cells with `#` comments and explanations between steps.

Part A is labelled **Standard visualizations (feasible without AI)**. This describes the techniques, not verified human-only authorship. The existing notes do not establish which figures were made without AI, and this revision was edited with AI assistance. Part B contains the explicitly AI-assisted follow-up analyses. This grouping does not replace the team's AI interaction log.

All three notebooks include executed outputs. Their default **Run All** uses the bundled results and does not train, download weights or need the CIFAKE dataset. Use the project's Python environment and keep the notebooks inside the repository: they find its root through `team-info.json`.

To train again, prepare the audited `data/baseline_seed1/` cache using the original root data pipeline, then set `RUN_TRAINING = True` in `02` or `03`. The simplified loops contain the model, optimizer, validation, early stopping and checkpoint selection. They save into a new run folder; ResNet18 downloads ImageNet weights when training starts. The submission figures continue to show the recorded experiment and are labelled accordingly.
