"""Small correctness checks for leakage handling, labels, and gradient flow."""

import copy
import unittest

import numpy as np
import torch

from baseline import BaselineCNN, classification_metrics
from prepare_baseline_data import assign_splits


class BaselineChecks(unittest.TestCase):
    def test_split_removes_duplicates_and_preserves_test(self):
        records = []
        for label in (0, 1):
            for i in range(5):
                records.append({"label": label, "source_split": "train", "pixel_sha256": f"{label}-{i}"})
        records += [{"label": 0, "source_split": "train", "pixel_sha256": "0-1"},
                    {"label": 0, "source_split": "test", "pixel_sha256": "0-0"},
                    {"label": 0, "source_split": "test", "pixel_sha256": "0-0"},
                    {"label": 1, "source_split": "test", "pixel_sha256": "test-real"}]
        first = assign_splits(copy.deepcopy(records), validation_per_class=1, seed=1)
        self.assertEqual(first, assign_splits(copy.deepcopy(records), validation_per_class=1, seed=1))
        self.assertEqual(sum(r["split"] == "test" for r in first), 3)
        self.assertEqual(sum(r["split"] == "val" for r in first), 2)
        self.assertEqual(sum(r["split"] == "excluded" for r in first), 2)
        sets = [{r["pixel_sha256"] for r in first if r["split"] == s} for s in ("train", "val", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_conflicting_identical_labels_fail(self):
        with self.assertRaises(ValueError):
            assign_splits([{"label": 0, "source_split": "train", "pixel_sha256": "same"},
                           {"label": 1, "source_split": "test", "pixel_sha256": "same"}], 1)

    def test_metrics_class_direction_and_extreme_logits(self):
        metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([-1000., 2., -2., 1000.]))
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["fake_detection_rate"], 0.5)
        self.assertEqual(metrics["real_flagged_fake_rate"], 0.5)
        self.assertTrue(np.isfinite(metrics["loss"]))
        perfect = classification_metrics(np.array([0, 1]), np.array([-4., 4.]))
        self.assertEqual(perfect["roc_auc_fake"], 1)

    def test_model_backpropagation(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        model = BaselineCNN()
        x, y = torch.rand(4, 3, 32, 32), torch.tensor([0., 1., 0., 1.])
        before = model.features[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4,))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, y).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.features[0].weight))


if __name__ == "__main__":
    unittest.main()
