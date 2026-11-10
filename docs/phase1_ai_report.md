# Phase I: AI-assisted visualization

Our aim was to understand how the CNN and ResNet18 react to image changes. Overall accuracy gave us a starting point, but it did not show whether individual images failed in the same way or which image properties were associated with those failures.

The AI-assisted analysis includes individual response curves, paired prediction plots, sharpening and colour controls, and a PCA view. These comparisons broaden the investigation beyond performance summaries.

## Prompt and decision examples

The following prompts and iterations are **reconstructed scenarios that match the implemented visualizations**. They illustrate possible analytical decisions, not a verified history of past AI interactions.

### 1. Looking beyond the average response

> “The average fake score drops after blur. How could we see whether individual images follow that same pattern?”

**Initial AI proposal:** Draw a response curve for every fake image.

**Refine:** Show the same random sample of 20 images for both models, together with the mean over all 5,000 fakes. Keep the decision threshold visible.

**Reject:** Drawing all 5,000 lines in one plot. The overlap would make individual responses difficult to follow.

**Why this helps:** Some images lose their fake score quickly, while others stay high or rise again at stronger blur. The individual curves reveal differences hidden by the mean. This is the ICE-inspired view in A4.

### 2. Checking whether two edits fool the same images

> “Blur and resizing both reduce detection. Can we compare their effects on each image, instead of comparing two overall percentages?”

**Initial AI proposal:** Put the two detection rates in a bar chart.

**Reject:** Using that chart as the main answer. It would still hide which images fail under both edits.

**Accept:** Pair the two P(FAKE) scores in a scatter plot. Add the 0.5 thresholds and count the four outcomes.

**Why this helps:** At blur 0.6 px and resize to 28 px and back, both edits fool the CNN on 3,241 of 5,000 fakes and ResNet18 on 3,609. The paired plot in A5 shows this overlap and the images that react differently.

### 3. Testing the sharpness hypothesis in both directions

> “Fakes tend to have more fine detail. What comparison would test whether the detectors respond to that detail?”

**Initial AI proposal:** Compare sharpness with fake scores.

**Refine:** Add interventions. Blur the images, sharpen the originals, and sharpen images that have already been blurred. Include real photos as well as fakes.

**Accept:** Keep the noise results in the comparison, even when they do not fit the main pattern.

**Reject:** Concluding that sharpness alone explains the detector.

**Why this helps:** Sharpening blurred fakes restores much of their detection, while sharpening real photos creates false alarms. Noise behaves differently. B1 therefore supports sensitivity to fine detail without reducing the whole model to one measurement.

### 4. Separating colour effects from blur effects

> “Some object classes remain easier to detect after blur. Could colour help explain that difference?”

**Initial AI proposal:** Reduce saturation after blur and compare the object classes.

**Accept:** Follow the same images through each edit and show a separate line for each object class.

**Refine:** Include saturation changes without blur as a control. Otherwise, we cannot compare colour alone with colour after detail has been removed.

**Why this helps:** The combination affects some classes more strongly, especially for the CNN. B2 suggests a secondary colour cue, while the control helps limit the interpretation.

### 5. Comparing groups with PCA

> “Can we use brightness, contrast, saturation and sharpness to see which kinds of fakes keep high scores after blur?”

**Initial AI proposal:** Create separate projections for the original and edited images.

**Reject:** Separate layouts for this comparison. A point could move because the projection changed, making the prediction change harder to interpret.

**Refine:** Standardize the four properties and fit one PCA to the original fake images. Reuse the positions for both models and conditions. Only the prediction colours change.

**Accept:** Show how much variation the projection retains and keep predictions out of the PCA inputs.

**Why this helps:** B3 connects broad groups of input properties with model responses. The two axes retain about 74.3% of the standardized variation. The map remains a partial view of the images, rather than an explanation of the network's internal features.

## Failure patterns and limits

The recorded results show that strong clean performance can hide substantial sensitivity to edits. Blur 0.6 px reduces fake detection from 94.6% to 33.8% for the CNN and from 98.1% to 27.4% for ResNet18. Individual responses are not always monotonic, and the paired plots reveal many shared failures.

The follow-up analysis also exposes limits to simple explanations. Noise does not consistently follow the sharpness pattern. High and low prediction scores overlap in the PCA view. These observations keep the conclusions narrower than “the model only uses sharpness” or “PCA separates the failure cases.”

For AI-assisted analysis, the main lesson is that a plausible visualization idea still needs checks. An aggregate chart can miss the pairing we care about, separate projections can make comparisons ambiguous, and an appealing explanation can ignore exceptions. The reconstructed examples above illustrate those risks without claiming they all occurred as documented AI errors.

## Practices worth keeping

- Start with the analytical question and ask what the proposed figure would reveal beyond a table.
- Compare the same images with fixed models and thresholds.
- Include controls that could weaken the preferred explanation.
- Check suggested interpretations against the plotted values and exceptions.
- State what a projection leaves out and distinguish input properties from learned features.

## Analysis sources

- [CNN notebook: A4–A5 and B1–B3](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/02_cnn_training_and_visualization.ipynb)
- [ResNet18 notebook: A4–A5 and B1–B3](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/03_resnet18_finetuning_and_visualization.ipynb)
