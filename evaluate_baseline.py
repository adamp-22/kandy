"""One final clean-test evaluation of the checkpoint selected by validation loss."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline import (BaselineCNN, choose_device, classification_metrics, file_sha256,
                      load_split, predict, save_predictions, verify_run_data, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/baseline_plain_seed1"))
    parser.add_argument("--data-dir", type=Path, help="Override the original data path on another machine")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args(argv)
    summary = json.loads((args.run_dir / "summary.json").read_text())
    if summary["status"] != "completed":
        raise SystemExit("Training must finish before test evaluation")
    if (args.run_dir / "test_metrics.json").exists():
        raise SystemExit("Test already evaluated; read test_metrics.json instead of using it to tune the model.")
    config = json.loads((args.run_dir / "config.json").read_text())
    data_dir = args.data_dir or Path(config["data_dir"])
    if args.data_dir is None and not data_dir.exists():
        data_dir = Path("data/baseline_seed1")
    verify_run_data(data_dir, args.run_dir, config)
    device = choose_device(args.device)
    torch.set_num_threads(4)
    model = BaselineCNN().to(device)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=True)
    if checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError("Checkpoint does not match selected epoch")
    model.load_state_dict(checkpoint["model_state_dict"])
    images, labels, rows = load_split(data_dir, "test")
    logits = predict(model, images, labels, config["batch_size"], device)
    metrics = classification_metrics(labels, logits)
    # Preserve the official benchmark, and also report each unique test image once.
    seen, unique_indices = set(), []
    for idx, row in enumerate(rows):
        if row["pixel_sha256"] not in seen:
            unique_indices.append(idx)
            seen.add(row["pixel_sha256"])
    indices = np.array(unique_indices)
    result = {"selected_epoch": checkpoint["epoch"], "checkpoint_sha256": file_sha256(args.run_dir / "best.pt"),
              "official_test": metrics, "unique_test": classification_metrics(labels[indices], logits[indices]),
              "note": "Fixed threshold 0.5. No test-driven tuning. Clean CIFAKE performance only, not robustness or unseen-generator performance."}
    save_predictions(args.run_dir / "test_predictions.csv", rows, logits)
    write_json(args.run_dir / "test_metrics.json", result)
    summary["test_evaluated"] = True
    summary["test_metrics"] = "test_metrics.json"
    write_json(args.run_dir / "summary.json", summary)
    write_json(args.run_dir / "status.json", summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
