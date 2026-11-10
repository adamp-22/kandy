# Phase I: how we used AI

**Project:** Kandy, real versus AI-generated image detection  
**Tools:** Codex for code revisions, visualization support and writing  
**Report prepared:** 10 November 2026

This report was drafted with AI assistance from the available project conversation and the three Phase I notebooks. The prompt excerpts below are paraphrases with spelling cleaned up, not a full verbatim chat export. We have kept reconstructed prompt examples in a separate section. We do not have a complete record of every earlier interaction or each team member's contribution.

## What we wanted help with

Our main question was what makes an image detector change its prediction. Both models already performed well on unchanged CIFAKE images. We wanted figures that showed where that performance broke down and helped us decide what to investigate next.

We also wanted the notebooks to be readable. Some earlier versions had long code cells, a lot of setup and too many optional parts. Our requests to AI were often quite practical: keep the important calculation visible, split it into smaller steps, and explain the less obvious lines. We wanted to understand the code well enough to present it ourselves.

The current submission has one data-exploration notebook, one CNN notebook, and one ResNet18 notebook. The model notebooks include training and checkpoint selection before the behaviour figures. The figures use saved predictions from the recorded experiments. Editing the plotting code did not mean training new models.

## Requests that changed the work

These requests are present in the available conversation. The wording below is shortened and paraphrased.

| Prompt or request | What we accepted or changed | Why it mattered |
| --- | --- | --- |
| “Make the code simpler, split the big chunks, and add comments in the middle too.” | We refined the notebook structure and comments. Shared loading and file handling moved into a small helper, while model definitions, training and plotting stayed visible. | A short notebook is not useful if the main work disappears into functions we cannot follow. |
| “Phase I should have three notebooks: data exploration, CNN training and visualization, and ResNet fine-tuning and visualization.” | We replaced the earlier reading path with those three notebooks. | The previous behaviour notebook referred to training elsewhere, which made the submission confusing to read on its own. |
| “Include the training epochs for both models so the teacher can see the accuracy and how we picked one model.” | We added complete epoch tables, loss and accuracy curves, and the reason for selecting a checkpoint. | The final epoch and the highest-accuracy epoch were not necessarily the selected ones. |
| “Based on the slides, add one or two simple visualizations that reveal something a normal table would not show. Tell me what you would add first.” | The resulting A4 and A5 sections follow individual blur responses and pair blur and resize scores for the same images. We asked for both views in both model notebooks. | Averages hide variation, and two separate detection rates do not tell us whether the same images fail. |
| “Add one more visualization to the AI-assisted part based on the slides.” | We added the B3 PCA view using four measured image properties. The positions stay fixed while the prediction colours change. | This broadens the analysis from overall rates to groups of images with different responses. |

The record supports these requests and the resulting notebook changes. It does not establish who originally wrote every older figure. In particular, **Part A means standard visualizations, not verified human-only authorship**. AI also helped revise that part. Part B identifies the explicitly AI-assisted follow-up investigations.

## How the figures affected our reasoning

**Individual blur responses, A4.** We kept the same seeded sample of 20 fake images for both models and used all 5,000 fakes for the average. This makes the comparison easier to follow. The lines show that images do not all respond in the same way. Some scores drop early, and some rise again at stronger blur. That stopped us from describing the result as a simple, steady decrease. The idea follows the individual-response plots in Lecture 04, slides 20–23, although blur changes many pixels at once.

**Paired blur and resize scores, A5.** We kept one dot per fake image and marked the two decision thresholds. For the CNN, 3,241 of 5,000 fakes are missed under both blur 0.6 px and resizing to 28 px and back. For ResNet18, the number is 3,609. This shows a large shared group of failures that two separate summary rates would hide. The visual inspection idea connects to Lecture 02, slide 7. We compare these particular settings; the edits are not matched for equal visual change.

**Sharpness and colour, B1–B2.** These sections already existed in the broader analysis. The available conversation does not contain their full original prompt history, so we cannot reconstruct that history as fact. Their results guide the later discussion: sharpening blurred fakes restores much of their detection, while sharpening real photos causes false alarms. Colour changes also affect some object classes more than others. However, noise does not follow a simple sharpness story. We therefore describe fine detail and colour as possible cues, rather than claiming that one measurement explains the network.

**PCA, B3.** We standardized brightness, contrast, saturation and log sharpness, then used the original fake images to calculate one shared layout. Predictions only determine the point colours. The two axes retain about 74.3% of the standardized variation. We kept that number and the limitation in the explanation: nearby points only look similar in this compressed description. This is a map of input properties, not the model's learned representation. The lecture connection is Lecture 03, slides 6–7, 14 and 17.

Our extra AI-assisted work therefore **broadened the analysis**. We chose static comparisons that run from the saved data, rather than claiming that the final submission contains a new interactive dashboard.

## What did not work well

The clearest problem was that AI assistance could make the project harder to read. The conversation includes repeated requests for shorter code cells, simpler structure and more useful comments. More code and more plots were not automatically an improvement. We kept the fuller versions locally and used a clearer three-notebook submission.

The optional loading and bundle code also needed explanation. It helps reproduce figures without downloading the full dataset, but it can distract from the actual experiment. We moved routine handling into the helper and explained the saved files separately.

Version names caused another practical problem. Several notebooks had similar names in different working folders, and screenshots showed uncertainty about which version was open. We eventually made one canonical `phase1` branch and a clearly named VS Code workspace.

There were also limits to the analysis that the writing had to respect. A mean probability is different from a detection rate. A gallery chosen for a large score change does not represent a typical image. PCA leaves out variation. Input edits show sensitivity, but they do not prove which internal feature caused a decision. These are checks we retained in the explanations, not claims that AI made every one of these mistakes during the session.

## What we would do again

- Ask for the question and proposed figure before asking for code. That makes it easier to reject extra complexity early.
- Keep the same images, axes and thresholds when comparing models. Otherwise, an apparent difference might come from the comparison itself.
- Calculate numbers from the saved arrays and check the accompanying explanation against them. A plausible sentence is not evidence.
- Keep controls and exceptions visible. Colour-only edits and the noise results help limit the sharpness claim.
- Request comments that explain a choice or intermediate step, rather than comments that simply repeat the Python line.
- Record prompts and decisions as the work happens. A later summary is useful, but it cannot replace missing interaction history or identify undocumented contributions.

AI helped extend the figures and revise the code quickly. We did not measure time saved or run a controlled comparison against a pair working without AI, so we do not claim a numerical productivity benefit. The useful outcome was a broader set of comparisons that we could still explain.

## Reconstructed prompt examples

The following are **newly written examples that match the finished figures**. They are not historical messages and should not be presented as evidence that these exact prompts were used.

> For both models, plot P(FAKE) for the same 20 randomly selected fake images across the tested blur strengths. Add the mean over all 5,000 fakes and the 0.5 decision threshold. Keep the code simple and explain why the mean can hide different responses.

> Pair each fake image's score after blur 0.6 with its score after resizing to 28 pixels and back. Show the four threshold outcomes and count the images in each group. Explain what this adds beyond comparing two detection rates.

> Make one PCA layout from standardized brightness, contrast, saturation and log sharpness of the original fake images. Reuse the positions for both models and for before and after blur. Colour by P(FAKE), report the retained variation, and explain what the projection cannot tell us.

## Files behind this report

- [Data exploration](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/01_data_exploration.ipynb)
- [CNN training and visualization](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/02_cnn_training_and_visualization.ipynb)
- [ResNet18 fine-tuning and visualization](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/03_resnet18_finetuning_and_visualization.ipynb)
- [Saved data and experiment sources](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/data/README.md)

The notebook version used for this report is commit `7d32a79` on the `phase1` branch. This report and the accompanying presentation also used AI assistance. The team should attach its original interaction record where available and confirm individual contributions before submission.
