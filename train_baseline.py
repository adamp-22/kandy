"""Train the plain CNN using training/validation data only; never load the test set."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

from baseline import (BaselineCNN, LABELS, batches, choose_device, classification_metrics,
                      file_sha256, load_split, predict, save_predictions, verify_run_data, write_json)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_resume(args):
    if (args.output / "test_metrics.json").exists():
        raise SystemExit("This baseline has already been test-evaluated. Preserve it; use a separate run for new experiments.")
    config = json.loads((args.output / "config.json").read_text())
    status_path = args.output / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "running":
            try:
                os.kill(status["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise SystemExit(f"Training process {status['pid']} is still active")
    checkpoint = torch.load(args.output / "last.pt", map_location="cpu", weights_only=True)
    for key in ("seed", "batch_size", "learning_rate", "patience", "min_delta"):
        requested = getattr(args, key)
        if requested is not None and requested != config[key]:
            raise SystemExit(f"Cannot change --{key.replace('_', '-')} when resuming an existing experiment")
        setattr(args, key, config[key])
    args.epochs = args.epochs if args.epochs is not None else config["epochs"]
    if args.epochs < checkpoint["epoch"]:
        raise SystemExit("Requested epoch limit is earlier than the saved checkpoint")
    if args.data_dir is None:
        original = Path(config["data_dir"])
        args.data_dir = original if original.exists() else Path("data/baseline_seed1")
    verify_run_data(args.data_dir, args.output, config)
    with (args.output / "history.csv").open(newline="") as handle:
        history = [{k: int(v) if k == "epoch" else float(v) for k, v in row.items()}
                   for row in csv.DictReader(handle) if int(row["epoch"]) <= checkpoint["epoch"]]
    if [r["epoch"] for r in history] != list(range(1, checkpoint["epoch"] + 1)):
        raise ValueError("History is incomplete for the saved checkpoint")
    # The epoch file is written before last.pt, so this also repairs a best.pt
    # update interrupted between those atomic saves.
    best_file = args.output / "checkpoints" / f"epoch_{checkpoint['best_epoch']:03d}.pt"
    best = torch.load(best_file, map_location="cpu", weights_only=True)
    save_checkpoint(args.output / "best.pt", best)
    return checkpoint, config, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/baseline_plain_seed1"))
    parser.add_argument("--resume", action="store_true", help="Restore last.pt, optimizer, scheduler, and RNG states")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-delta", type=float)
    args = parser.parse_args(argv)
    resume, original_config, history = None, None, []
    if args.resume:
        resume, original_config, history = read_resume(args)
    else:
        if args.output.exists():
            raise SystemExit(f"Refusing to overwrite {args.output}; use --resume or choose a new --output.")
        defaults = {"data_dir": Path("data/baseline_seed1"), "seed": 1, "epochs": 30,
                    "batch_size": 128, "learning_rate": 0.001, "patience": 5, "min_delta": 0.0001}
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.learning_rate <= 0 or args.min_delta < 0:
        raise SystemExit("Invalid training parameters")
    device = choose_device(args.device)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)

    train_images, train_labels, _ = load_split(args.data_dir, "train")
    val_images, val_labels, val_rows = load_split(args.data_dir, "val")
    model = BaselineCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "checkpoints").mkdir(exist_ok=args.resume)
    source_dir = args.output / (f"source_resume_{time.time_ns()}" if args.resume else "source")
    source_dir.mkdir()
    for name in ("baseline.py", "prepare_baseline_data.py", "train_baseline.py", "evaluate_baseline.py", "requirements.txt"):
        shutil.copy2(Path(__file__).parent / name, source_dir / name)
    log = logging.getLogger("cifake.training")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in list(log.handlers):
        handler.close()
        log.removeHandler(handler)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(args.output / "training.log")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data_dir": str(args.data_dir.resolve()), "device_used": str(device),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "Conv(3,32,3)-ReLU-Pool(2)-Conv(32,32,3)-ReLU-Pool(2)-Flatten-Linear(1152,64)-ReLU-Linear(64,1)",
        "parameters": sum(p.numel() for p in model.parameters()),
        "class_to_index": LABELS, "preprocessing": "native RGB 32x32; float32 / 255", "augmentation": None,
        "optimizer": "Adam, weight_decay=0", "loss": "BCEWithLogitsLoss; logits represent REAL",
        "scheduler": "ReduceLROnPlateau(validation_loss, factor=0.5, patience=2, min_lr=1e-5)",
        "checkpoint_selection": "Lowest validation BCE; fixed 0.5 probability threshold",
        "test_usage": "Not loaded by this program; evaluate only after model selection is complete",
        "paper": "https://arxiv.org/html/2303.14126v1",
        "reproduction_note": "Paper-inspired architecture, not an exact reproduction. Split, optimizer, initialization, kernel/pooling implementation, scheduler, and early stopping are explicitly specified for this experiment.",
        "determinism": "Seeded Python, NumPy and Torch; deterministic algorithms requested with warn_only. Bitwise identity across hardware/software is not guaranteed.",
        "manifest_sha256": file_sha256(args.data_dir / "manifest.csv"),
        "audit_sha256": file_sha256(args.data_dir / "audit.json"),
        "source_sha256": {p.name: file_sha256(p) for p in source_dir.iterdir()},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}},
    }
    if resume is not None:
        invocation = {"resumed_utc": config["started_utc"], "from_epoch": resume["epoch"],
                      "device": str(device), "source_directory": source_dir.name,
                      "source_sha256": config["source_sha256"], "environment": config["environment"]}
        config = {**original_config, "data_dir": str(args.data_dir.resolve()), "epochs": args.epochs,
                  "resumes": original_config.get("resumes", []) + [invocation]}
    write_json(args.output / "config.json", config)
    if not args.resume:
        shutil.copy2(args.data_dir / "audit.json", args.output / "data_audit.json")
    write_json(args.output / "status.json", {"status": "running", "pid": os.getpid(), "epoch": resume["epoch"] if resume else 0})
    log.info("Device=%s; parameters=%s; train=%s; validation=%s; no augmentation", device, config["parameters"], len(train_labels), len(val_labels))
    best_loss, early_stop_best, best_epoch, stale_epochs = float("inf"), float("inf"), 0, 0
    start_epoch = 1
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        rng.bit_generator.state = resume["numpy_generator_state"]
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if "python_rng_state" in resume:
            random.setstate(resume["python_rng_state"])
        if "numpy_legacy_rng_state" in resume:
            legacy = resume["numpy_legacy_rng_state"]
            np.random.set_state((legacy[0], np.array(legacy[1], dtype=np.uint32), *legacy[2:]))
        previous_device = original_config.get("resumes", [{}])[-1].get("device", original_config["device_used"])
        if resume.get("accelerator_rng_state") is not None and previous_device == str(device):
            if device.type == "mps":
                torch.mps.set_rng_state(resume["accelerator_rng_state"].cpu())
            elif device.type == "cuda":
                torch.cuda.set_rng_state(resume["accelerator_rng_state"].cpu())
        best_loss, best_epoch, stale_epochs = resume["best_val_loss"], resume["best_epoch"], resume["stale_epochs"]
        for row in history:
            if row["val_loss"] < early_stop_best - args.min_delta:
                early_stop_best = row["val_loss"]
        start_epoch = resume["epoch"] + 1
        log.info("Restored checkpoint at epoch %s, including optimizer, scheduler, and RNG state", resume["epoch"])
        if stale_epochs >= args.patience:
            log.info("Early stopping already reached; restoring final reports without additional training")
            start_epoch = args.epochs + 1
    started = time.perf_counter()
    prior_epoch_seconds = sum(row["seconds"] for row in history)
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_roc_auc_fake", "learning_rate", "seconds"]
    try:
        with (args.output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
            handle.flush()
            for epoch in range(start_epoch, args.epochs + 1):
                epoch_started = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"]
                model.train()
                total_loss = torch.zeros((), device=device)
                total_correct = torch.zeros((), device=device)
                for x, y in batches(train_images, train_labels, args.batch_size, device, rng.permutation(len(train_labels))):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("Non-finite training loss")
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach() * len(y)
                    total_correct += ((logits.detach() >= 0) == y.bool()).sum()
                val_logits = predict(model, val_images, val_labels, args.batch_size, device)
                metrics = classification_metrics(val_labels, val_logits)
                scheduler.step(metrics["loss"])
                row = {"epoch": epoch, "train_loss": total_loss.item() / len(train_labels),
                       "train_accuracy": total_correct.item() / len(train_labels), "val_loss": metrics["loss"],
                       "val_accuracy": metrics["accuracy"], "val_roc_auc_fake": metrics["roc_auc_fake"],
                       "learning_rate": lr, "seconds": time.perf_counter() - epoch_started}
                writer.writerow(row)
                handle.flush()
                history.append(row)
                improved = metrics["loss"] < best_loss
                if improved:
                    best_loss, best_epoch = metrics["loss"], epoch
                if metrics["loss"] < early_stop_best - args.min_delta:
                    early_stop_best, stale_epochs = metrics["loss"], 0
                else:
                    stale_epochs += 1
                checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                              "validation_metrics": metrics, "best_epoch": best_epoch, "best_val_loss": best_loss,
                              "stale_epochs": stale_epochs, "numpy_generator_state": rng.bit_generator.state,
                              "early_stop_best": early_stop_best, "python_rng_state": random.getstate(),
                              "numpy_legacy_rng_state": [np.random.get_state()[0], np.random.get_state()[1].tolist(),
                                                         *np.random.get_state()[2:]],
                              "torch_rng_state": torch.get_rng_state(),
                              "accelerator_rng_state": torch.mps.get_rng_state() if device.type == "mps" else
                                  torch.cuda.get_rng_state() if device.type == "cuda" else None,
                              "config": config}
                save_checkpoint(args.output / "checkpoints" / f"epoch_{epoch:03d}.pt", checkpoint)
                save_checkpoint(args.output / "last.pt", checkpoint)
                if improved:
                    save_checkpoint(args.output / "best.pt", checkpoint)
                write_json(args.output / "status.json", {"status": "running", "pid": os.getpid(), "epoch": epoch,
                           "best_epoch": best_epoch, "best_val_loss": best_loss, "last_epoch": row})
                log.info("Epoch %02d/%02d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%% | lr %.5f | %.1fs%s",
                         epoch, args.epochs, row["train_loss"], 100*row["train_accuracy"], row["val_loss"],
                         100*row["val_accuracy"], lr, row["seconds"], " | best" if improved else "")
                if stale_epochs >= args.patience:
                    log.info("Early stopping after %s epochs without sufficient validation-loss improvement", args.patience)
                    break
        best = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        logits = predict(model, val_images, val_labels, args.batch_size, device)
        validation = classification_metrics(val_labels, logits)
        save_predictions(args.output / "validation_predictions.csv", val_rows, logits)
        write_json(args.output / "validation_metrics.json", validation)
        summary = {"status": "completed", "best_epoch": best_epoch, "epochs_run": len(history),
                   "elapsed_seconds": prior_epoch_seconds + time.perf_counter() - started, "validation": validation,
                   "stop_reason": "early_stopping" if stale_epochs >= args.patience else "max_epochs",
                   "test_evaluated": False, "best_checkpoint": "best.pt"}
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "status.json", summary)
        log.info("Complete. Selected epoch %s by validation loss. Test set has not been evaluated.", best_epoch)
    except BaseException as exc:
        write_json(args.output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   "error": str(exc), "completed_epochs": len(history)})
        raise


if __name__ == "__main__":
    main()
