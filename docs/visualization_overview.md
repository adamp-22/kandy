# Visualization overview: Phase I and Phase II

This document lists every visualization we currently have, says which phase it belongs to and which lecture technique it uses, and checks whether it meets the Phase I brief. It also describes the new Phase I notebook and a lecture-based plan for Phase II.

**Phase I brief (summary).** Treat the model as a black box (inputs and outputs only), use enough data to show systematic behaviour, and explain *patterns* in depth: what distinguishes a group of images, and which sub-groups exist. "Not a model performance report": a table of benchmark scores is not enough. Training data may be used as a source of hypotheses.

## 1. Repository layout after the split

| Notebook | Phase | Content | Runs without the dataset? |
| --- | --- | --- | --- |
| `baseline_training.ipynb` | Training | CNN definition, data split and audit, training, epoch report, learning curves, official test evaluation | Yes (weights bundled) |
| `resnet18_finetuning.ipynb` | Training | ResNet18 fine-tuning pipeline, epoch report, learning curves, confusion matrix, clean test | Yes (downloads the release package) |
| `phase1/01_training_data_exploration.ipynb` | I | Training-data EDA (formerly baseline Section 4) and the hypotheses it gives | Yes (figures bundled) |
| `phase1/02_cnn_behaviour.ipynb` | I | CNN under JPEG, blur, resize and noise (formerly baseline Sections 10–11) | Yes |
| `phase1/03_resnet18_behaviour.ipynb` | I | ResNet18 under the same edits, next to the CNN (formerly ResNet Section 6) | Yes (downloads the release package) |
| `phase1/04_what_fools_the_detectors.ipynb` | I | **New.** Both models, 29 conditions, patterns behind the errors | Yes (`phase1/data/phase1_behaviour_bundle.npz`) |
| `phase2/01_cnn_internals.ipynb` | II | CNN Grad-CAM, contributions, occlusion, filters, channel removal (formerly baseline Sections 12–13) | Yes |
| `phase2/02_resnet18_internals.ipynb` | II | ResNet18 Grad-CAM, contributions, occlusion, filters, channel interventions, explorer (formerly ResNet Sections 7–9) | Yes (downloads the release package) |

The moved cells are unchanged apart from section numbers and cross-references. Each phase notebook finds the repository root by walking up to `team-info.json`, reads its own embedded data, and was executed in a clean folder without `data/` or `runs/` to check that it runs on a fresh clone. The training notebooks keep everything their pipelines need. The ResNet runtime cells still contain all analysis functions, because the training pipeline uses them to build the saved analysis package.

## 2. Are the existing visualizations fine for Phase I?

Verdict key: **✔ Phase I**: black-box behaviour, fits the brief. **◐ Supporting**: useful context, but not a Phase I finding on its own. **✘ Performance**: a performance report, which the brief excludes, so it stays in the training notebooks. **II**: needs model internals, so it belongs to Phase II.

### Baseline CNN (formerly `baseline_training.ipynb`)

| Former section | Visualization | Lecture technique | Verdict | Now in |
| --- | --- | --- | --- | --- |
| 3 | Split table, exclusions, duplicate overlap | Data audit | ◐ Documents the data | training notebook, `phase1/01` |
| 4.1 | Split balance bars, duplicate image pairs | Bar chart, examples (L02) | ◐ Data quality | `phase1/01` |
| 4.2 | Balanced random gallery | Look at the data (L02) | ✔ Hypothesis source | `phase1/01` |
| 4.3 | Brightness, contrast, saturation, sharpness distributions (REAL vs FAKE) | Overlaid histograms (L02) | ✔ Hypothesis source | `phase1/01` |
| 4.4 | Lowest and highest sharpness images per class | Extremes gallery | ✔ Connects the statistic to images | `phase1/01` |
| 4.5 | Sharpness × contrast scatter, linked thumbnail explorer | Scatter, brushing and linking (L02), explorable (L05) | ✔ Hypothesis source (explorer needs local data) | `phase1/01` |
| 4.6 | Mean Fourier spectra and radial profile | Frequency view | ✔ Motivates the blur and resize tests | `phase1/01` |
| 5, 7, 8 | Architecture, epoch report, learning curves | — | ✘ Training and performance | training notebook |
| 10 | Eight validation thumbnails with P(FAKE) | Instance examples | ◐ Anecdotal (8 images) | `phase1/02` |
| 11.1–11.2 | Same fake / real image at four strengths of each edit | Instance-level, ICE-like (L04) | ✔ But chosen for sensitivity, so needs the population views | `phase1/02` |
| 11.3 | Fake detection and real false alarms per edit strength | Line charts by class | ✔ Separates the two error directions; still close to a metric | `phase1/02` |
| 11.4 | Original vs modified P(FAKE) per image | Scatter | ✔ Shows the size and direction of each shift | `phase1/02` |
| 11.5 | Prediction-change map (images × settings) | Heatmap | ✔ Shows whether the same images fail across edits | `phase1/02` |
| 11.6 | Modification explorer | Explorable explanation (L05) | ✔ Needs a live kernel | `phase1/02` |
| 11.7 | Summary table of rates | Table | ◐ A score table; fine as a summary, not as the result | `phase1/02` |
| 12.1 | Grad-CAM on original / edited pairs | Grad-CAM (L04) | II | `phase2/01` |
| 12.2 | 64 neuron contributions to the logit | Exact linear decomposition | II | `phase2/01` |
| 12.3 | Occlusion maps | Perturbation attribution (L04/L05) | II | `phase2/01` |
| 12.4 | Explanation explorer | Explorable (L05) | II | `phase2/01` |
| 13.1–13.2 | First-layer filters, one convolution step | Weight visualization | II | `phase2/01` |
| 13.3–13.4 | Feature maps after resize, top-activating patches, synthetic checks | Activation maps, dataset examples (L05) | II | `phase2/01` |
| 13.5 | Single-channel removal under clean, resize, JPEG | Ablation | II | `phase2/01` |
| 13.6 | Filter explorer | Explorable (L05) | II | `phase2/01` |
| 14 | Test metrics table, confusion matrix | — | ✘ Performance | training notebook |

### ResNet18 (formerly `resnet18_finetuning.ipynb`)

| Former section | Visualization | Lecture technique | Verdict | Now in |
| --- | --- | --- | --- | --- |
| 5 | Epoch report, learning curves, confusion matrix | — | ✘ Training and performance | training notebook |
| 6 | Both models: accuracy, fakes detected, reals flagged for 4 edits × 4 strengths | Small multiples (L02) | ✔ Comparative and split by direction | `phase1/03` |
| 7 | Grad-CAM (7×7), 512 output contributions, occlusion | Grad-CAM (L04), decomposition, perturbation | II | `phase2/02` |
| 8 | Stem filters, channel responses, top examples, channel interventions | Weight and activation views, ablation | II | `phase2/02` |
| 9 | Interactive comparison | Explorable (L05) | II | `phase2/02` |

### Teammates' notebooks

These are listed for completeness and were **not changed** in this PR.

| Notebook | Author | Content | Phase | Notes |
| --- | --- | --- | --- | --- |
| `CIFAKE_plotting.ipynb` | kavimalshan | Training-data EDA: class counts, sample grid, brightness/contrast, RGB and saturation, extremes, mean images and difference map, Fourier spectra, edge strength, optional PCA / t-SNE on pixels | I (hypothesis source) | The committed version has no saved outputs; run and commit it so the figures can be read on GitHub. Overlaps with `phase1/01`; the PCA / t-SNE part is the projection technique from L03. |
| `Plotting_01.ipynb` | LordNassel | Class counts, brightness/contrast, extremes, mean images, scatter, RGB channels, Fourier spectra, edge density, t-SNE on pixels, Grad-CAM | I (EDA), II (Grad-CAM) | The Grad-CAM cell uses an ImageNet model without our trained CIFAKE weights (its own comment says so), so it does not explain our detectors. `phase2/` has Grad-CAM for both trained models. Images are sampled in `os.listdir` order, which can differ between machines; sort the list and use a seed. |
| `Plotting_02.ipynb` | LordNassel | Class counts and proportions, sample grids, mean images and difference maps | I (EDA) | Plot titles refer to *faces*; check which dataset it reads before using it in the CIFAKE story. |
| `ResNet18.ipynb` | LordNassel | ResNet-18 training, confusion matrix, confidence distribution, Grad-CAM | Training / II | Uses a different dataset (about 2,000 images), so it cannot be compared with the CIFAKE models. |

### Verdict

The existing behaviour notebooks are a solid start for Phase I. They treat the models as black boxes, use all 10,000 validation images, separate missed fakes from false alarms, follow single images and include an explorer. The EDA gives clear hypotheses. Measured against the brief, five things were missing:

1. **Patterns behind the errors.** The figures show *that* accuracy drops, not *which* images fail or what distinguishes them from the images that survive.
2. **Sub-groups.** There was no per-class view (planned in the README) and no grouping of images by their properties.
3. **A test of the main hypothesis in the opposite direction.** If the models read fine detail, *adding* detail should make real photos look fake. No edit in the original sweep adds detail.
4. **Image-level comparison of the two models.** The models were only compared through aggregate rates.
5. **Too much performance reporting.** Learning curves, test tables and confusion matrices are performance reports, so they now stay in the training notebooks.

## 3. New Phase I visualizations (`phase1/04_what_fools_the_detectors.ipynb`)

All figures use only images and output logits, on the 10,000 validation images, for both models and 29 input conditions (the original 17 plus sharpening, blur-then-sharpen, saturation and blur-then-desaturate). The data are stored in a 7 MB bundle, so the notebook runs in about 30 seconds without the dataset, the weights or a GPU. `RECOMPUTE = True` rebuilds the bundle from local data.

| § | Figure | Lecture technique | Question | Main finding |
| --- | --- | --- | --- | --- |
| 1 | Missed fakes and flagged reals per **sharpness decile**, with 95 % Wilson intervals and counts | Conditioning on a hypothesis variable (L02) | Do clean errors follow the training-data sharpness gap? | CNN misses 14.6 % of the smoothest fakes vs 2.4 % of the sharpest, and flags 22.7 % of the sharpest reals vs 2.3 % of the smoothest. Within-decile ROC-AUC stays 0.98–0.99 (CNN) and ≥ 0.99 (ResNet18), against 0.73 for sharpness alone. |
| 2 | **Sharpness dial**: median sharpness vs share predicted FAKE, all edit families as connected lines; counterfactual sharpening and blur → sharpen arrows | Connected scatter, counterfactual (L02/L04) | Is there one mechanism behind JPEG, blur, resize and noise? | Almost every edit moves the output along one fine-detail axis. Sharpening real photos by 200 % raises false alarms from 5.5 % to 83.7 % (CNN) and 2.3 % to 66.9 % (ResNet18). Sharpening blurred fakes restores detection (33.8 % → 88.6 %, 27.4 % → 90.6 %). Noise on fakes is the exception. |
| 3 | REAL-logit distributions (box and whiskers) per edit and strength, with ROC-AUC | Distribution comparison | Lost information or a confidence shift? | Mostly a shift: ResNet18 after resizing to 28 px keeps ROC-AUC 0.92 at 55.8 % accuracy. |
| 4 | **Centered ICE curves** with PDP, coloured by original sharpness | c-ICE, PDP (L04) | Does every image react the same way? | Sharper fakes lose more under blur (ρ = +0.65 / +0.70). They start more confidently FAKE but end closer to REAL. Saturation does not change the size of the shift but gives the CNN a head start. |
| 5 | **Class slope chart**: detection after blur → blur + desaturate, per object class; contrast-pair gallery | Bump / slope chart (L02), counterfactual, examples | Which classes survive blur, and is colour the reason? | Deer 65 % → 28 % and frog 51 % → 19 % (CNN) once colour is removed. The drop correlates with class saturation (ρ = +0.75 CNN, +0.56 ResNet18). On clean images saturation alone barely matters. |
| 6 | **Map of the 5,000 fakes** from 14 model-independent descriptors (t-SNE at two perplexities and PCA), coloured by outcome after blur; k-means group profiles; group examples | t-SNE / PCA with their caveats (L03), k-means (L02), profiles | What distinguishes survivors from fooled fakes? Which sub-groups exist? | Survival ranges from 9.5 % (G6) to 29.2 % (G1) of fakes detected by both models. There are two kinds of survivors: spectra rich in fine detail relative to low contrast, and a saturated frog/deer group. Groups cut across object classes. |
| 7 | CNN vs ResNet18 logit hexbin after resize; error overlap (Jaccard) vs independence baseline | Agreement plot | Do both detectors fail on the same images? | Yes: blur 0.6 has Jaccard 0.68 (2,819 fakes fool both), resize has r = 0.61. Strong noise is the exception (72.6 % vs 18.3 % of reals flagged). |
| 8 | Findings, limits and Phase II questions | Storytelling (L02) | — | — |

The galleries are chosen by stated, seeded rules, and the t-SNE caveats from Lecture 3 are applied: two perplexities, a PCA check, and groups taken from k-means on the descriptors rather than from the picture.

## 4. Phase II plan

The existing Phase II notebooks already cover Grad-CAM, exact output contributions, occlusion, filter and activation views, channel removal and explorers for both models. Phase I raised concrete questions that these tools can answer, and the lectures suggest several additions.

| Planned addition | Lecture | Phase I question it answers |
| --- | --- | --- |
| **Sanity checks for saliency** (model and label randomization) before trusting Grad-CAM or occlusion | L05 (Adebayo et al.) | Are the existing heatmaps model-dependent at all? |
| **Vanilla gradients / SmoothGrad / Integrated Gradients** on blur and sharpen pairs | L05 | Where does fine detail become FAKE evidence? |
| **Frequency-band sensitivity**: low-pass and high-pass input filters, gradient energy per band | README plan, L05 | Is the sharpness dial a specific frequency band? Why does noise behave differently from sharpening? |
| **Projections of penultimate features** (PCA / t-SNE / UMAP) for clean and edited images, coloured by label, edit and Phase I group; trajectories as blur increases | L03 (Rauber 2016, Hinterreiter 2020) | Do edited fakes move into the REAL region, or to a new region? This is the internal version of the confidence shift. |
| **Training-time trajectories** of the same projection across checkpoints | L03, README plan | When during training does the sharpness shortcut appear? |
| **Nearest neighbours and prototypes / criticisms** in the training set for survivors and fooled fakes | L05, README plan | What do the training images closest to a fooled fake look like? |
| **Network-Dissection-style unit alignment**: correlate channel activations with the Phase I descriptors (sharpness, saturation, spectral bands) | L05 | Which channels carry the sharpness dial, and which (if any) carry colour in the CNN? |
| **LIME (superpixels) / SHAP** on enlarged images, with the 32×32 limits stated | L04 | Is evidence local (object or background) or global texture? |
| **Object vs border share of saliency**, using the centre/border proxy from Phase I | README plan | Does the detector look at the object or the background, as the CIFAKE paper suggests? |

## 5. Not included in this PR

The midpoint submission also needs the AI interaction record, the GenAI reflection report and the video (at most 5 minutes). Those are not part of this PR.
