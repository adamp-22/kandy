"""Check that resuming reproduces uninterrupted CPU training on synthetic data."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from baseline import file_sha256, write_json


class ResumeCheck(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        project = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="cifake-resume-") as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            rng, rows = np.random.default_rng(17), []
            for split, n in (("train", 16), ("val", 8)):
                images = rng.integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8)
                labels = np.arange(n, dtype=np.int64) % 2
                np.save(data / f"{split}_images.npy", images)
                np.save(data / f"{split}_labels.npy", labels)
                rows.extend({"path": f"{split}/{i}.jpg", "split": split, "label": int(labels[i]),
                             "class_name": "REAL" if labels[i] else "FAKE", "pixel_sha256": f"{split}-{i}"}
                            for i in range(n))
            with (data / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(data / "audit.json", {"file_sha256": {p.name: file_sha256(p) for p in data.iterdir()}})
            continuous, resumed = root / "continuous", root / "resumed"

            def run(*args):
                result = subprocess.run([sys.executable, "train_baseline.py", *map(str, args)],
                                        cwd=project, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run("--data-dir", data, "--output", continuous, "--device", "cpu", "--batch-size", 4, "--epochs", 2)
            run("--data-dir", data, "--output", resumed, "--device", "cpu", "--batch-size", 4, "--epochs", 1)
            # A teammate can put the same prepared data at a different path.
            relocated = root / "relocated"
            data.rename(relocated)
            audit = json.loads((relocated / "audit.json").read_text())
            audit["source_root"] = "a-different-machine"
            write_json(relocated / "audit.json", audit)
            run("--resume", "--data-dir", relocated, "--output", resumed, "--device", "cpu", "--epochs", 2)
            full = torch.load(continuous / "last.pt", map_location="cpu", weights_only=True)
            restarted = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(full["epoch"], restarted["epoch"])
            self.assertEqual(full["validation_metrics"], restarted["validation_metrics"])
            self.assertEqual(full["scheduler_state_dict"], restarted["scheduler_state_dict"])
            for key, value in full["model_state_dict"].items():
                self.assertTrue(torch.equal(value, restarted["model_state_dict"][key]), key)
            # A tested baseline must be kept fixed.
            write_json(resumed / "test_metrics.json", {"test": "already evaluated"})
            refused = subprocess.run([sys.executable, "train_baseline.py", "--resume", "--output", str(resumed)],
                                     cwd=project, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("already been test-evaluated", refused.stderr)


if __name__ == "__main__":
    unittest.main()
