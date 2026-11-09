"""Small shared utilities. Models, training loops and plots stay in the notebooks."""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "phase1/data"
REAL_COLOR, FAKE_COLOR = "#0072B2", "#D55E00"


def load_saved(filename):
    """Read the named arrays from a saved experiment."""
    with np.load(DATA / filename, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def load_report(model):
    return json.loads((DATA / f"{model}_training_report.json").read_text())


def load_split(split):
    """Load local training pixels in the same order as the audited split."""
    data = ROOT / "data/baseline_seed1"
    manifest = data / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError("Prepare data/baseline_seed1 with the root baseline_training notebook first.")
    expected = load_report("cnn")["manifest_sha256"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError("The local split differs from the recorded experiment.")
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    images = np.load(data / f"{split}_images.npy", mmap_mode="r")
    labels = np.load(data / f"{split}_labels.npy")
    assert images.shape == (len(labels), 3, 32, 32)
    assert np.array_equal(labels, [int(row["label"]) for row in rows])
    return images, labels


def batches(images, labels, order, batch_size, device):
    """Convert one batch of 0–255 pixels to 0–1 floats."""
    import torch

    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        x = torch.tensor(np.asarray(images[indices]), dtype=torch.float32, device=device) / 255
        y = torch.tensor(labels[indices], dtype=torch.float32, device=device)
        yield x, y


def new_run(name, settings):
    """Set the seed and make a fresh output folder, without overwriting an old run."""
    import torch

    folder = ROOT / "runs" / name
    folder.mkdir(parents=True, exist_ok=False)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    return folder


def save_best(model, folder, epoch, settings):
    """Write the selected weights atomically; this version does not implement resume."""
    import torch

    state = {"epoch": epoch, "model_state_dict": model.state_dict(), "settings": settings}
    temporary = folder / "best.tmp"
    torch.save(state, temporary)
    temporary.replace(folder / "best.pt")


def condition_index(conditions, family, value):
    return next(i for i, item in enumerate(conditions)
                if item["family"] == family and item["value"] == value)


def family_indices(conditions, family):
    return [0] + [i for i, item in enumerate(conditions) if item["family"] == family]


def edit_image(image, condition):
    """Apply the blur, resize or sharpen edit shown in the notebook galleries."""
    family, value = condition["family"], condition["value"]
    if family == "Original":
        return image.copy()
    source = Image.fromarray(image.transpose(1, 2, 0))
    if family == "Blur":
        changed = source.filter(ImageFilter.GaussianBlur(radius=value))
    elif family == "Resize":
        changed = source.resize((value, value), Image.Resampling.BILINEAR)
        changed = changed.resize((32, 32), Image.Resampling.BILINEAR)
    elif family == "Sharpen":
        changed = source.filter(ImageFilter.UnsharpMask(radius=1, percent=value, threshold=0))
    else:
        raise ValueError(f"This gallery does not use {family}.")
    return np.asarray(changed, dtype=np.uint8).transpose(2, 0, 1).copy()


def show_plot():
    import matplotlib.pyplot as plt

    plt.show()
    plt.close("all")
