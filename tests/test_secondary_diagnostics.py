"""Hand-calculated checks of secondary estimands and preservation safeguards."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location("secondary_diagnostics", CODE / "secondary_diagnostics.py")
secondary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(secondary)


class SecondaryEstimandTests(unittest.TestCase):
    def setUp(self):
        # Host a has two errors1/3, host b one error−10.
        self.frame = pd.DataFrame({"host_group": ["a", "a", "b"],
                                   "observed_price": [10., 20., 50.],
                                   "prediction": [11., 23., 40.]})

    def test_listing_and_host_estimands_against_hand_calculation(self):
        listing = secondary.point_metrics(self.frame, "listing")
        host = secondary.point_metrics(self.frame, "equal_host")
        self.assertAlmostEqual(listing["MAE"], 14 / 3)
        self.assertAlmostEqual(listing["signed_bias"], -2)
        self.assertAlmostEqual(listing["RMSE"], np.sqrt(110 / 3))
        self.assertAlmostEqual(host["MAE"], 6)  # ((1+3)/2+10)/2
        self.assertAlmostEqual(host["signed_bias"], -4)
        self.assertAlmostEqual(host["RMSE"], np.sqrt(52.5))
        self.assertAlmostEqual(host["mean_observed_price"], 32.5)
        self.assertAlmostEqual(host["R2"], 1 - 105 / 637.5)
        self.assertEqual(host["n_listings"], 3)
        self.assertEqual(host["n_hosts"], 2)

    def test_equal_host_weight_invariance_when_replicating_one_hosts_rows(self):
        duplicated_host = pd.concat([self.frame, self.frame.iloc[:2]], ignore_index=True)
        before = secondary.point_metrics(self.frame, "equal_host")
        after = secondary.point_metrics(duplicated_host, "equal_host")
        for metric in secondary.POINT_METRICS:
            self.assertAlmostEqual(before[metric], after[metric])
        self.assertNotAlmostEqual(secondary.point_metrics(self.frame, "listing")["MAE"],
                                  secondary.point_metrics(duplicated_host, "listing")["MAE"])

    def test_single_listing_hosts_equal_listing_weights_and_constant_target_r2_undefined(self):
        frame = self.frame.assign(host_group=["a", "b", "c"])
        self.assertEqual(secondary.point_metrics(frame, "listing"), secondary.point_metrics(frame, "equal_host"))
        self.assertTrue(np.isnan(secondary.point_metrics(frame.assign(observed_price=10), "equal_host")["R2"]))

    def test_intervals_inclusive_coverage_and_host_weighted_width(self):
        frame = self.frame.assign(lower=[10., 19., 40.], upper=[11., 20., 45.],
                                  width=[1., 1., 5.], covered=[True, True, False])
        validated = secondary.validate_intervals(frame)
        listing = secondary.interval_metrics(validated, "listing")
        host = secondary.interval_metrics(validated, "equal_host")
        self.assertAlmostEqual(listing["coverage"], 2 / 3)
        self.assertAlmostEqual(host["coverage"], .5)
        self.assertAlmostEqual(listing["mean_width"], 7 / 3)
        self.assertAlmostEqual(host["mean_width"], 3)
        self.assertAlmostEqual(host["interval_point_MAE"], 6)
        self.assertEqual(host["median_width"], 1)  # Prespecified inverse-CDF median.
        with self.assertRaisesRegex(ValueError, "coverage"):
            secondary.validate_intervals(frame.assign(covered=False))
        with self.assertRaisesRegex(ValueError, "widths"):
            secondary.validate_intervals(frame.assign(width=99))
        with self.assertRaisesRegex(ValueError, "reversed"):
            secondary.validate_intervals(frame.assign(lower=99))

    def test_neighbourhood_threshold_keeps_small_groups_and_exact_boundary(self):
        frame = pd.concat([self.frame.iloc[[0]]] * 59, ignore_index=True)
        frame["neighbourhood"] = ["Alphabetical small"] * 29 + ["Zebra boundary"] * 30
        frame["design"], frame["repeat"] = "random", 1
        result = secondary.aggregate(frame, "neighbourhood")
        self.assertEqual(len(result), 4)  # Both weights, both groups; none removed.
        self.assertFalse(result.loc[result.neighbourhood.eq("Alphabetical small"), "interpretable_n_ge_30"].any())
        self.assertTrue(result.loc[result.neighbourhood.eq("Zebra boundary"), "interpretable_n_ge_30"].all())

    def test_weighted_median_exact_half_cdf_uses_lower_value(self):
        # One host with19listings followed by a single-listing host: exactly
        # half the host mass is reached at value19, even with recurring1/19.
        values = np.arange(1, 21, dtype=float)
        counts = np.array([19] * 19 + [1])
        self.assertEqual(secondary.weighted_host_median(values, counts), 19)
        # Four hosts; counts1,1,1,3. First two singleton observations reach
        # exactly half and must not advance to the next unequal-width row.
        self.assertEqual(secondary.weighted_host_median(np.array([10., 20., 30., 40., 50., 60.]),
                                                       np.array([1, 1, 1, 3, 3, 3])), 20)

    def test_repetition_summary_is_descriptive_and_rejects_duplicate_or_missing_repeats(self):
        frame = self.frame.assign(design="random", repeat=1)
        first = secondary.aggregate(frame, "overall")
        second = secondary.aggregate(frame.assign(repeat=2, prediction=[10., 20., 50.]), "overall")
        result = secondary.descriptive_summary(pd.concat([first, second]), "overall")
        listing = result.loc[result.weighting.eq("listing")].iloc[0]
        self.assertAlmostEqual(listing.MAE_mean, 7 / 3)
        self.assertAlmostEqual(listing.MAE_min, 0)
        self.assertAlmostEqual(listing.MAE_max, 14 / 3)
        self.assertEqual(listing.n_listings, 3)
        self.assertEqual(listing.n_repeats, 2)
        with self.assertRaisesRegex(ValueError, "exactly"):
            secondary.descriptive_summary(pd.concat([first, first]), "overall")
        with self.assertRaisesRegex(ValueError, "exactly"):
            secondary.descriptive_summary(first, "overall")


class SecondaryInputIntegrityTests(unittest.TestCase):
    def test_primary_hash_guard_detects_changes_and_rejects_escape(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            protected = root / "protected.csv"
            protected.write_text("price\n100\n")
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({"files": {"protected.csv": secondary.sha256(protected)}}))
            self.assertEqual(secondary.verify_primary(root, baseline)["files_verified"], 1)
            protected.write_text("price\n101\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                secondary.verify_primary(root, baseline)
            baseline.write_text(json.dumps({"files": {"../escape.csv": "0" * 64}}))
            with self.assertRaisesRegex(ValueError, "outside"):
                secondary.verify_primary(root, baseline)

    def test_oof_complete_coverage_and_target_guards(self):
        data = pd.DataFrame({"row_id": ["a", "b", "c"], "price": [10., 20., 30.]})
        frames = []
        splits = {}
        for design in secondary.DESIGNS:
            for repeat in secondary.REPEATS:
                frames.append(pd.DataFrame({"row_id": data.row_id, "observed_price": data.price,
                                            "prediction": data.price + 1, "design": design,
                                            "repeat": repeat, "fold": [1, 2, 3]}))
                for fold, rid in enumerate(data.row_id, 1):
                    splits[(design, repeat, fold)] = {rid}
        valid = pd.concat(frames, ignore_index=True)
        secondary.validate_prediction_slices(valid, data, splits)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            secondary.validate_prediction_slices(pd.concat([valid, valid.iloc[[0]]]), data, splits)
        with self.assertRaisesRegex(ValueError, "targets"):
            secondary.validate_prediction_slices(valid.assign(observed_price=999), data, splits)
        broken = valid.copy()
        broken.loc[0, "fold"] = 2
        with self.assertRaisesRegex(ValueError, "fold"):
            secondary.validate_prediction_slices(broken, data, splits)


if __name__ == "__main__":
    unittest.main()
