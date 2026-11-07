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
| `baseline_training.ipynb` | Training | Small CNN: data split and audit, training, epoch report, learning curves, official test evaluation (weights bundled) |
| `resnet18_finetuning.ipynb` | Training | ResNet18 fine-tuning pipeline, epoch report, confusion matrix, clean test (weights from the team release) |
| `phase1/01_training_data_exploration.ipynb` | I | How REAL and FAKE training images differ: properties, spectra, average images, colour channels, cues per object class, pixel-space projection; hypotheses for the behaviour tests |
| `phase1/data/training_data_extra_bundle.npz` | I | Saved statistics used by `01` Section 5 (runs without the dataset) |
| `phase1/02_cnn_behaviour.ipynb` | I | CNN under JPEG, blur, resize and noise on 10,000 validation images |
| `phase1/03_resnet18_behaviour.ipynb` | I | ResNet18 under the same edits, next to the CNN |
| `phase1/04_what_fools_the_detectors.ipynb` | I | Both models, 29 conditions: the sharpness dial, confidence shift, ICE curves, classes and colour, a map of the fakes, model agreement |
| `phase1/data/phase1_behaviour_bundle.npz` | I | Saved logits and image descriptors used by `04` (runs without the dataset) |
| `phase1/05_human_vs_ai_figures.ipynb` | I | All Phase I figures in one place, split into feasible without AI (Part A) and AI-assisted (Part B) |
| `phase2/01_cnn_internals.ipynb` | II | CNN Grad-CAM, neuron contributions, occlusion, filters, channel removal |
| `phase2/02_resnet18_internals.ipynb` | II | ResNet18 Grad-CAM, output contributions, occlusion, stem filters, channel interventions, explorer |
| `docs/visualization_overview.md` | I + II | Every visualization with its phase, lecture technique and Phase I verdict, plus the Phase II plan |
| `team-info.json`, `workload.json` | — | Team registration |

The analysis uses only the CIFAKE dataset with our audited split and our two models (baseline CNN, fine-tuned ResNet18). All notebooks open in review mode: they load saved weights and results and do not train. Run them from their own folder; the phase notebooks find the repository root through `team-info.json`.
