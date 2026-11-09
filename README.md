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

## Shorter Phase I variant

This branch, **`phase1-simple-variant`**, is a shorter walkthrough for review. The fuller version remains on [`phase1-new-variant`](https://github.com/adamp-22/kandy/tree/phase1-new-variant).

Open the three notebooks inside `phase1/` in this order:

| File | What the reader sees |
| --- | --- |
| `01_data_exploration.ipynb` | Load data; image examples; property calculations and distributions; differences by object class; average images. |
| `02_cnn_training_and_visualization.ipynb` | CNN definition and training; learning curves; edits and population behaviour; AI-assisted sharpness and colour follow-up. |
| `03_resnet18_finetuning_and_visualization.ipynb` | ResNet18 preprocessing, classifier warmup and fine-tuning; the same investigation; a comparison with CNN scores. |
| `helpers.py` | Shared loading, batching, output-file handling and the pixel edits used in the galleries. Models, training loops and plotting stay in the notebooks. |
| `data/` | Recorded measurements, training reports, predictions and previews. `data/README.md` explains each file. |

Each question leads into a short calculation, a figure and a brief interpretation. The model notebooks each have five main figures. Supporting score-distribution checks sit at the bottom; the ResNet notebook also compares both models. The extra maps, clustering, interactive tools and repeated summaries are preserved in the fuller branch rather than repeated here.

Part A groups the earlier standard visualizations. Part B contains the explicitly AI-assisted follow-up investigations. This revision's code and prose were edited with AI assistance; the grouping does not establish human-only authorship for Part A. Keep the team's interaction log and actual contribution record with the submission.

### Running the notebooks

The saved outputs are already visible. **Run All** also works without the full dataset, trained weights or downloads. Select the project's Python environment with NumPy, Matplotlib, Pillow, scikit-learn, PyTorch and torchvision. Open Jupyter from the repository root or `phase1/` so the small helper module can be imported.

Training code is included in both model notebooks. To train again, prepare `data/baseline_seed1/` using the original root data pipeline, then set `RUN_TRAINING = True`. A new run uses a fresh folder and saves the selected weights plus its history. ResNet18 downloads ImageNet weights only when training starts. The submission figures explicitly use the recorded experiment; they do not silently switch to a new run.

The original root training notebooks and the two `phase2/` notebooks remain available. Their detailed development tools and model-internal analyses are outside this shorter Phase I reading path. See `docs/visualization_overview.md` for the figure guide.
