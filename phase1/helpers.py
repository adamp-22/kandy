"""Small shared utilities. Models, training loops and plots stay in the notebooks."""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Resolve paths from this file so they do not depend on where Jupyter was opened.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "phase1/data"
REAL_COLOR, FAKE_COLOR = "#0072B2", "#D55E00"


def load_saved(filename):
    """Read the named arrays from a saved experiment."""
    with np.load(DATA / filename, allow_pickle=False) as saved:
        # Read the arrays while the file is open; later cells can then use them directly.
        return {name: saved[name] for name in saved.files}


def load_report(model):
    # The report holds the completed run's history and evaluation results.
    return json.loads((DATA / f"{model}_training_report.json").read_text())


def load_split(split):
    """Load local training pixels in the same order as the audited split."""
    data = ROOT / "data/baseline_seed1"
    manifest = data / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError("Run python phase1/prepare_data.py from the repository root first.")
    # Both models need the same audited split that was used in the recorded experiment.
    expected = load_report("cnn")["manifest_sha256"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError("The local split differs from the recorded experiment.")
    # Keep only the manifest rows for the requested split.
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    # Memory mapping lets training read batches without loading every pixel at once.
    images = np.load(data / f"{split}_images.npy", mmap_mode="r")
    labels = np.load(data / f"{split}_labels.npy")
    assert images.shape == (len(labels), 3, 32, 32)
    # Check that each label still lines up with the correct image row.
    assert np.array_equal(labels, [int(row["label"]) for row in rows])
    return images, labels


def batches(images, labels, order, batch_size, device):
    """Convert one batch of 0–255 pixels to 0–1 floats."""
    import torch

    for start in range(0, len(order), batch_size):
        # Take the next group from the shuffled order.
        indices = order[start:start + batch_size]
        # Scale the pixels to 0–1 only when this batch is needed.
        x = torch.tensor(np.asarray(images[indices]), dtype=torch.float32, device=device) / 255
        y = torch.tensor(labels[indices], dtype=torch.float32, device=device)
        yield x, y


def new_run(name, settings):
    """Set the seed and make a fresh output folder, without overwriting an old run."""
    import torch

    folder = ROOT / "runs" / name
    # Fail if this name already exists, so an earlier run is not overwritten.
    folder.mkdir(parents=True, exist_ok=False)
    # Python, NumPy and PyTorch each have their own random generator.
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    return folder


def save_best(model, folder, epoch, settings):
    """Write the selected weights atomically; this version does not implement resume."""
    import torch

    # Store which epoch these weights came from alongside the settings.
    state = {"epoch": epoch, "model_state_dict": model.state_dict(), "settings": settings}
    temporary = folder / "best.tmp"
    torch.save(state, temporary)
    # Replace the selected checkpoint only after the new file has been fully written.
    temporary.replace(folder / "best.pt")


def condition_index(conditions, family, value):
    # Find the matching edit without assuming which row it occupies.
    return next(i for i, item in enumerate(conditions)
                if item["family"] == family and item["value"] == value)


def family_indices(conditions, family):
    # Begin each edit curve with the unchanged-image reference.
    return [0] + [i for i, item in enumerate(conditions) if item["family"] == family]


def edit_image(image, condition):
    """Apply the blur, resize or sharpen edit shown in the notebook galleries."""
    family, value = condition["family"], condition["value"]
    if family == "Original":
        return image.copy()
    # Pillow expects the colour channel last, unlike the stored model input.
    source = Image.fromarray(image.transpose(1, 2, 0))
    if family == "Blur":
        changed = source.filter(ImageFilter.GaussianBlur(radius=value))
    elif family == "Resize":
        # Shrink the image first, then bring it back to the model's 32×32 input size.
        changed = source.resize((value, value), Image.Resampling.BILINEAR)
        changed = changed.resize((32, 32), Image.Resampling.BILINEAR)
    elif family == "Sharpen":
        changed = source.filter(ImageFilter.UnsharpMask(radius=1, percent=value, threshold=0))
    else:
        raise ValueError(f"This gallery does not use {family}.")
    # Return to the colour-first layout used by the saved previews.
    return np.asarray(changed, dtype=np.uint8).transpose(2, 0, 1).copy()


def show_plot():
    import matplotlib.pyplot as plt

    plt.show()
    # Start the next plot with a clean figure instead of leaving old ones open.
    plt.close("all")
