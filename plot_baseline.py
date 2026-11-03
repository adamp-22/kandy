"""Plot saved training history without rerunning training or test evaluation."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/baseline_plain_seed1"))
    args = parser.parse_args()
    with (args.run_dir / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    summary = json.loads((args.run_dir / "summary.json").read_text())
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    fig.suptitle("CIFAKE · Plain CNN baseline · No augmentation", fontsize=15, fontweight="bold")
    for ax, metric, title, scale in zip(axes, ["loss", "accuracy"], ["Binary cross-entropy", "Accuracy (%)"], [1, 100]):
        ax.plot(epochs, [float(r[f"train_{metric}"])*scale for r in history], color="#3465a4", label="Training", linewidth=2)
        ax.plot(epochs, [float(r[f"val_{metric}"])*scale for r in history], color="#d96b27", label="Validation", linewidth=2)
        ax.axvline(summary["best_epoch"], color="#777777", linestyle="--", linewidth=1, label=f"Selected epoch {summary['best_epoch']}")
        ax.set(xlabel="Epoch", ylabel=title, title=title, xlim=(1, max(epochs) + 0.2))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(args.run_dir / "learning_curves.png", dpi=180)
    fig.savefig(args.run_dir / "learning_curves.svg")
    plt.close(fig)
    print(args.run_dir / "learning_curves.png")


if __name__ == "__main__":
    main()
