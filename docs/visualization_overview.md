# Where to find the Phase I work

Phase I contains three notebooks. They use the same audited CIFAKE split: 88,702 training images, 10,000 validation images and 20,000 official test images. FAKE is label 0; REAL is label 1.

| Notebook | What to read |
| --- | --- |
| `phase1/01_data_exploration.ipynb` | Load the training measurements and image examples, then explore the data. No model training. |
| `phase1/02_cnn_training_and_visualization.ipynb` | CNN training in Sections 1–4, recorded results in Section 5, then Part A and Part B visualizations. |
| `phase1/03_resnet18_finetuning_and_visualization.ipynb` | ResNet18 fine-tuning in Sections 1–4, recorded results in Section 5, then Part A and Part B visualizations. The final comparisons bring both models together. |

The old separate behaviour, shared-analysis and figure-collection notebooks have been consolidated into these files. Training code is included in the model notebooks, so it can be read alongside the results. The original root notebooks remain available for data preparation, interrupted-run recovery and export; they are not additional notebooks inside the Phase I folder.

## Data exploration

Notebook 01 loads saved training measurements and gallery images. If the audited local cache exists, it uses those pixels for the galleries. It contains:

- Split counts, removed duplicate counts and example duplicate pairs.
- REAL/FAKE image galleries and sharpness extremes.
- Brightness, contrast, saturation and sharpness distributions and scatterplots.
- Fourier spectra and radial frequency profiles.
- Average images and colour channels, overall and by object class.
- Per-class image cues, edge density, file size and pixel-space PCA/t-SNE.

These plots describe the inputs and suggest hypotheses. They do not use model activations or establish which cue a detector learned. The old interactive thumbnail explorer and bundle-rebuilding cells have been removed from this reading version.

## Training inside each model notebook

Both model notebooks contain the data loader, architecture, optimizer, training epoch, validation, early stopping and checkpoint selection in short commented cells. The CNN starts from random weights. ResNet18 starts from ImageNet weights, trains its new classifier for two epochs, then fine-tunes all layers with a smaller learning rate for the backbone.

The completed CNN selected epoch 19 from 24 epochs; ResNet18 selected epoch 5 from 10. Selection uses the lowest validation BCE, with a fixed prediction threshold of 0.5. The two small JSON reports in `phase1/data/` preserve the recorded histories and clean evaluation. Learning curves and test scores document training; the behaviour figures below supply the Phase I investigation.

`RUN_TRAINING = False` is the default. Review works without the dataset, trained weights or model downloads. A new run needs the audited cache and uses a fresh folder. The figures explicitly show the recorded experiment rather than silently using a newly trained model. The shorter loops keep the recorded architecture and hyperparameters; they do not include the original pipelines' background execution or interrupted-run recovery.

## The two visualization sections

**Part A — Standard visualizations (feasible without AI)** includes learning curves, clean evaluation, example predictions, sensitive-image edit galleries, population rate curves and original-versus-edited score scatterplots. This is a grouping by technique, not evidence of human-only authorship. The existing project notes do not identify which original figures were made without AI. The code and prose in this revision were edited with AI assistance.

**Part B — AI-assisted visualizations** contains the follow-up questions:

| Section | Question | Visual approach |
| --- | --- | --- |
| B1, both models | Do clean errors follow image sharpness? | Sharpness deciles, error rates with Wilson intervals, group counts and within-decile AUC. |
| B2, both models | Do adding and removing detail move predictions in opposite directions? | Connected sharpness-versus-prediction plots, sharpening and blur-then-sharpen controls. |
| B3, both models | Is an edit changing confidence or destroying label separation? | Logit distributions and AUC at each edit strength. |
| B4, both models | Does every image react equally? | Centred individual response curves, coloured by original sharpness, with the population mean. |
| B5, both models | Which object classes survive blur, and does colour help? | Per-class trajectories through blur and reduced saturation, with a colour-only control. |
| B6, both models | Do the same images fail under different edits? | A fixed-sample prediction-change heatmap. |
| B7, ResNet notebook | What kinds of fakes survive both detectors? | Image-descriptor t-SNE at two perplexities, PCA, k-means group profiles and class summaries. |
| B8, ResNet notebook | Do CNN and ResNet18 make errors on the same images? | Paired logit hexbin and Jaccard error overlap against an independence baseline. |

Both notebooks use the unchanged behaviour bundle: the same 10,000 validation images, 29 input conditions, labels and saved logits. The previews are a fixed sample of 600 images; sensitive examples are chosen by a stated score-change rule. Population plots use all relevant validation images. The map uses pixel measurements, not hidden network features.

Blur and resizing sharply reduce fake detection for both models. Sharpening often makes real photos look fake. Colour helps some fakes remain detectable after blur, particularly for the CNN. However, noise does not behave like sharpening, and several edits affect multiple image properties. These observations suggest mechanisms to investigate; they do not prove one internal explanation.

## Phase II

`phase2/01_cnn_internals.ipynb` and `phase2/02_resnet18_internals.ipynb` contain Grad-CAM, output contributions, occlusion, filters, activations and channel interventions. Those methods inspect the network and belong to Phase II. Follow-up questions include which channels carry the fine-detail response, whether colour is a separate cue, and how edited inputs move through the learned representation.

The three Phase I notebooks do not replace the team's required AI interaction log, reflection or submission video.
