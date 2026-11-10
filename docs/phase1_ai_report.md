# Phase I: AI-assisted visualization

We wanted to understand how the CNN and ResNet18 react when an image changes. Accuracy gave us a starting point. We also wanted to see which images confused the models and what those images had in common.

The AI-assisted analysis looks at individual images, compares predictions after different edits, tests sharpening and colour changes, and uses PCA to look for broader patterns.

## Hypothetical prompts and decisions

The prompts and decisions below are **hypothetical examples based on the finished graphs**. They show how the analysis could have developed; they are not a verified record of conversations.

### 1. Looking beyond the average response

> “The average fake score drops after blur. How could we see whether individual images follow that same pattern?”

**AI idea:** Draw a response curve for every fake image.

**Adjust:** Show the same random sample of 20 images for both models, plus the average score over all 5,000 fakes. Keep the decision threshold visible.

**Reject:** Drawing all 5,000 lines in one plot. The overlap would make individual responses difficult to follow.

**Why this helps:** Some images lose their fake score quickly, while others stay high or rise again at stronger blur. The individual curves show differences that the average hides. This is the ICE-inspired view in A4.

### 2. Checking whether two edits fool the same images

> “Blur and resizing both reduce detection. Can we compare their effects on each image, instead of comparing two overall percentages?”

**AI idea:** Put the two detection rates in a bar chart.

**Reject:** Using that chart as the main answer. It would still hide which images fail under both edits.

**Accept:** Pair the two P(FAKE) scores in a scatter plot. Add the 0.5 thresholds and count the four outcomes.

**Why this helps:** At blur 0.6 px and resize to 28 px and back, both edits fool the CNN on 3,241 of 5,000 fakes and ResNet18 on 3,609. The paired plot in A5 shows this overlap and the images that react differently.

### 3. Testing the sharpness hypothesis in both directions

> “Fakes tend to have more fine detail. What comparison would test whether the detectors respond to that detail?”

**AI idea:** Compare sharpness with fake scores.

**Adjust:** Try changing the images too: blur them, sharpen the originals, and sharpen images that have already been blurred. Include real photos as well as fakes.

**Accept:** Keep the noise results in the comparison, even when they do not fit the main pattern.

**Reject:** Concluding that sharpness alone explains the detector.

**Why this helps:** Sharpening blurred fakes restores much of their detection, while sharpening real photos creates false alarms. Noise behaves differently. B1 shows that fine detail matters, but one sharpness number does not explain the whole model.

### 4. Separating colour effects from blur effects

> “Some object classes remain easier to detect after blur. Could colour help explain that difference?”

**AI idea:** Reduce saturation after blur and compare the object classes.

**Accept:** Follow the same images through each edit and show a separate line for each object class.

**Adjust:** Include saturation changes without blur as a control. This lets us compare colour changes on their own with colour changes after blur.

**Why this helps:** The combination affects some classes more strongly, especially for the CNN. B2 suggests that colour may also matter. The control helps us check how much of the change comes from colour alone.

### 5. Comparing groups with PCA

> “Can we use brightness, contrast, saturation and sharpness to see which kinds of fakes keep high scores after blur?”

**AI idea:** Make separate PCA maps for the original and edited images.

**Reject:** Separate layouts. A point could move just because the map changed, which would make the before-and-after comparison confusing.

**Adjust:** Standardize the four properties and fit one PCA to the original fake images. Reuse the positions for both models and conditions. Only the prediction colours change.

**Accept:** Show how much variation the projection retains and keep predictions out of the PCA inputs.

**Why this helps:** B3 helps us compare how groups of images respond. The two axes retain about 74.3% of the variation in the standardized measurements. The map only summarizes these image properties; it does not show what the network has learned internally.

## Failure patterns and limits

Both models do well on the original images, but small edits can cause many mistakes. Blur 0.6 px reduces fake detection from 94.6% to 33.8% for the CNN and from 98.1% to 27.4% for ResNet18. Some image scores rise again as blur gets stronger. The paired plots also show that many images fool the models under both edits.

The extra comparisons also show where simple explanations stop working. Noise does not always follow the sharpness pattern. High and low prediction scores overlap in the PCA map. Sharpness is a useful clue, but it is not the whole explanation, and PCA does not clearly separate every failure.

An AI suggestion can sound useful and still need checking. A summary chart can hide which images fail together, separate maps can make comparisons confusing, and a neat explanation can leave out exceptions. The hypothetical examples above illustrate these risks.

## Practices worth keeping

- Start with a question and ask what the graph would show beyond a table.
- Compare the same images with fixed models and thresholds.
- Include a control that could challenge the explanation.
- Check AI explanations against the numbers and keep exceptions visible.
- Explain what a map shows, what it leaves out, and whether it uses image measurements or learned features.

## Analysis sources

- [CNN notebook: A4–A5 and B1–B3](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/02_cnn_training_and_visualization.ipynb)
- [ResNet18 notebook: A4–A5 and B1–B3](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/03_resnet18_finetuning_and_visualization.ipynb)
