# Shorter Phase I reading guide

Read the three notebooks in `phase1/`. The data notebook has four figures. Each model notebook has five main figures, with short supporting checks at the bottom. Code cells contain at most 27 lines; training, important measurements and plotting remain visible.

| Notebook/section | Question | Evidence |
| --- | --- | --- |
| Data exploration | What differs between REAL and FAKE inputs? | Seeded training examples, four property distributions, class medians and average images. |
| Model training | How was the detector trained and selected? | Architecture, preprocessing, optimizer, training and validation functions, validation-loss selection and early stopping. |
| Part A1 | What happened during training? | Complete epoch tables, recorded learning curves, a clearly marked checkpoint, selection reasoning and clean accuracy. These document the experiment rather than constitute the behaviour investigation. |
| Part A2 | Can a small edit change one prediction? | A fake sensitive to blur and a real photo sensitive to sharpening, selected by a stated rule from the 600 previews. |
| Part A3 | How common is the failure? | Fake detection and real-photo false alarms under JPEG, blur, resizing and noise, using all 10,000 validation images. |
| Part B1 | What happens when detail is added instead of removed? | Sharpness versus predictions, sharpening and blur-then-sharpen controls, with the noise exception visible. |
| Part B2 | Does colour help some fakes survive blur? | Object-class trajectories through blur and reduced saturation, plus a colour-only control. |
| Supporting checks | Is it a score shift, and do both models respond similarly? | Resize logit distributions with AUC; paired CNN/ResNet scores in notebook 03. |

Both models use 88,702 training images and 10,000 validation images, with the official 20,000-image test set kept for evaluation. CNN selected epoch 19 out of 24; ResNet18 selected epoch 5 out of 10. FAKE is label 0 and REAL is label 1. The model output is a REAL logit, so negative values predict FAKE.

`helpers.py` contains common loading, batching, file-saving and gallery pixel edits. `data/README.md` identifies the saved files and their sources. Review does not need the full dataset or model weights. New training runs use the audited local cache and new output folders. The figures continue to show the recorded experiment.

Part A groups earlier standard plots and Part B groups AI-assisted follow-up. This shorter revision was edited with AI assistance. Human-only authorship of the earlier figures is not established by the existing notes; the team's interaction log should document the actual contributions.

The full previous version is preserved on [phase1-new-variant](https://github.com/adamp-22/kandy/tree/phase1-new-variant), including the additional data views, sharpness deciles, individual response curves, prediction-change maps, clustering and error-overlap analysis. Phase II remains in `phase2/`, where Grad-CAM, contributions, occlusion, filters and channel interventions inspect the models internally.

These notebooks cover the code and figures. The submission also needs the team's interaction log, reflection and video.
