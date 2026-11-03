"""Export compact CPU weights/results for Git and a full local checkpoint archive."""

import argparse
import json
from pathlib import Path
import shutil
import tarfile

import torch

from baseline import BaselineCNN, file_sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/baseline_plain_seed1"))
    args = parser.parse_args(argv)
    name = args.run_dir.name
    model_path, result_dir = Path("models") / f"{name}.pt", Path("results") / name
    archive = Path("artifacts") / f"{name}.tar.gz"
    for target in (model_path, result_dir, archive):
        if target.exists():
            raise SystemExit(f"Refusing to overwrite {target}")
    config = json.loads((args.run_dir / "config.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    test = json.loads((args.run_dir / "test_metrics.json").read_text())
    if summary["status"] != "completed" or file_sha256(args.run_dir / "best.pt") != test["checkpoint_sha256"]:
        raise ValueError("Run must be complete, and weights must match the evaluated checkpoint")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    model = BaselineCNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model_path.parent.mkdir(exist_ok=True)
    result_dir.mkdir(parents=True)
    archive.parent.mkdir(exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "architecture": config["architecture"],
               "class_to_index": config["class_to_index"], "epoch": checkpoint["epoch"],
               "preprocessing": config["preprocessing"], "augmentation": None,
               "manifest_sha256": config["manifest_sha256"],
               "note": "Inference weights; full optimizer and training state are in the separate run archive."}
    torch.save(payload, model_path)
    # Verify the exported weights rather than evaluating the test set again.
    restored = torch.load(model_path, map_location="cpu", weights_only=True)
    for key, value in checkpoint["model_state_dict"].items():
        if not torch.equal(value, restored["model_state_dict"][key]):
            raise AssertionError(f"Export changed weights: {key}")
    for filename in ("history.csv", "validation_metrics.json", "test_metrics.json", "learning_curves.png"):
        shutil.copy2(args.run_dir / filename, result_dir / filename)
    audit = json.loads((args.run_dir / "data_audit.json").read_text())
    audit.pop("source_root", None)
    write_json(result_dir / "data_audit.json", audit)
    report = {"run": name, "seed": config["seed"], "model_path": model_path.as_posix(),
              "model_sha256": file_sha256(model_path), "architecture": config["architecture"],
              "parameters": config["parameters"], "augmentation": None,
              "split_counts": audit["split_counts"], "selected_epoch": summary["best_epoch"],
              "epochs_run": summary["epochs_run"], "selection": config["checkpoint_selection"],
              "stop_reason": summary.get("stop_reason", "early_stopping"),
              "validation": summary["validation"], "official_test": test["official_test"],
              "unique_test": test["unique_test"], "manifest_sha256": config["manifest_sha256"],
              "training_source_sha256": config["source_sha256"],
              "training_parameters": {k: config[k] for k in ("batch_size", "learning_rate", "epochs", "patience", "min_delta")},
              "environment": config["environment"], "limitations": test["note"]}
    write_json(result_dir / "summary.json", report)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(args.run_dir, arcname=name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{file_sha256(archive)}  {archive.name}\n")
    print(f"Git-ready weights: {model_path} ({model_path.stat().st_size / 1024:.0f} KiB)")
    print(f"Git-ready results: {result_dir}")
    print(f"Full run archive: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
