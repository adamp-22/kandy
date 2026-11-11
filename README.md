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
| Gabor Levai | [LordNassel](https://github.com/LordNassel) |
| Adam Parkanyi | [adamp-22](https://github.com/adamp-22) |

## Deadlines

| Date | Deliverable |
| --- | --- |
| 11.11.2026 | Midpoint (end of Phase I): code from both pairs, AI interaction log, progress video (max. 5 min) |
| 09.12.2026 | Final (end of Phase II): all code, AI logs from both phases, 10-minute recorded presentation and slides |

## Contents

- `team-info.json`, `workload.json`: team registration
