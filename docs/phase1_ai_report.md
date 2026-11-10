# Phase I: How we used AI for model analysis

We used AI to help explore how the CNN and ResNet18 respond to changes in an image. Accuracy was a useful starting point. We also wanted to see which images confused the models and whether blur, resizing, colour and fine detail could help explain their behaviour.

## Our requests and what changed

These requests come from our project conversation. They are shortened and have their spelling tidied up. The outcomes were checked against the final notebooks.

### 1. Show something an accuracy table would miss

> “Could we add one or two visualizations inspired by the slides that reveal model behaviour—something a normal table would not show?”

The resulting additions were individual response curves and a paired prediction plot.

The response curves follow 20 fake images as blur increases. We also show the average score across all 5,000 fake validation images. This uses the ICE idea from the lectures: follow individual cases alongside the average. Some images lose their fake score quickly, while others stay high or rise again with stronger blur. That difference is easy to miss in a single average.

The paired plot asks whether blur and resizing fool the same images. Each dot represents one image, with its fake score after blur on one axis and its score after resizing on the other. The threshold lines show the four possible outcomes. The spread of points also shows how close the predictions are to the threshold.

### 2. Use both views for both models

> “Before adding anything, tell me what you would add.”

> “Okay, add them for both the CNN and ResNet18, with reasoning about what we see.”

We accepted the additions for both models. This made the comparison more useful: the response curves use the same sample of images, and the paired plots use the same edits and decision threshold.

At the settings shown, blur and resizing both fool the CNN on 3,241 of the 5,000 fakes. For ResNet18, that number is 3,609. The plots show that many failures concern the same images. They also show the exceptions, where an image survives one edit but not the other.

### 3. Add a broader view of image properties

> “Based on the slides, add one more visualization for the AI-assisted part.”

This led to the PCA view. It combines brightness, contrast, saturation and sharpness into a two-dimensional map. Each point is a fake image, and its colour shows the model's fake score.

The map uses the original image measurements. Point positions stay fixed across both models and before and after blur; only the prediction colours change. This lets us compare the responses of the same groups of images.

The two axes retain about 74.3% of the variation in the standardized measurements. The map helps us look for patterns, but high and low scores still overlap. It does not fully explain which images will fool a model.

## How we checked the explanations

The AI-assisted follow-up also includes sharpening and colour comparisons. These help check whether an explanation fits more than one result.

Sharpening blurred fakes restores much of their detection. Sharpening real photos also creates false alarms. Both results suggest that fine detail matters. Noise does not follow the same pattern, though, so sharpness alone is not a complete explanation.

For colour, the notebooks compare reduced saturation after blur with reduced saturation on its own. That control helps us separate the effect of colour from the combined effect of both edits. Looking at object classes separately also shows differences that an overall rate can hide.

We kept the conclusions limited to what these comparisons show. The PCA uses measured image properties, not features learned inside the network. Shared mistakes in the paired plot do not prove that both edits affect the same internal feature.

## Failures we found

Both models perform well on the original images, but their fake detection drops sharply after blur. At blur 0.6 px, it falls from 94.6% to 33.8% for the CNN and from 98.1% to 27.4% for ResNet18.

The extra views show why one summary number is not enough. Images respond differently, many failures overlap across edits, and simple explanations have exceptions. AI helped broaden the analysis, but each interpretation still needed to be checked against the results.

## What we would keep doing

- Ask a specific question before choosing a graph.
- Compare the same images under the same conditions for both models.
- Include a control that could challenge the explanation.
- Keep exceptions visible, even when they make the story less tidy.
- Treat AI explanations as ideas to check, and keep a record of the requests and decisions.

## Notebooks

- [CNN: response curves, paired predictions and AI-assisted follow-up](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/02_cnn_training_and_visualization.ipynb)
- [ResNet18: response curves, paired predictions and AI-assisted follow-up](https://github.com/adamp-22/kandy/blob/7d32a79019e7bb14356704b7eb040d249faf55a5/phase1/03_resnet18_finetuning_and_visualization.ipynb)
