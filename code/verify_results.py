#!/usr/bin/env python3
"""Independent integrity checks of saved predictions and split membership."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


def metrics(frame):
    y = frame.observed_price.to_numpy(float)
    p = frame.prediction.to_numpy(float)
    residual = y - p
    return {"MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "R2": float(1 - np.sum(residual ** 2) / np.sum((y - np.mean(y)) ** 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = pd.read_csv(args.data).set_index("row_id", drop=False)
    config = json.loads((args.results / "frozen_config.json").read_text())
    fingerprint_config = dict(config)
    fingerprint_config.pop("fingerprint")
    assert hashlib.sha256(json.dumps(fingerprint_config, sort_keys=True).encode()).hexdigest() == config["fingerprint"]
    assert json.loads((args.results / "versions.json").read_text()) == config["versions"]
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == config["hashes"]["input_csv"]
    for name, filename in [("reanalysis_code", "reanalyse.py"),
                           ("preparation_code", "prepare_data.py"),
                           ("feature_code", "features.py"), ("protocol", "PROTOCOL.md")]:
        path = Path(__file__).parent / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config["hashes"][name]
    summary = json.loads((args.results / "run_summary.json").read_text())
    complete = sorted((args.results / "folds").glob("*/complete.json"))
    assert len(complete) == 18 == summary["completed_outer_folds"]
    assert summary["regressor_fits"] == config["estimated_regressor_fits"] == 2916
    core = config["core_families"]
    split_count = 0
    actual_cv_fits = 0
    fold_metric_checks = 0
    calibration_checks = 0
    outer_membership = {}
    for flag in complete:
        folder = flag.parent
        info = json.loads(flag.read_text())
        assert info["fingerprint"] == config["fingerprint"]
        manifest = json.loads((folder / "splits.json").read_text())
        outer = manifest["outer"][0]
        train, test = set(outer["train_row_id"]), set(outer["test_row_id"])
        assert not train & test
        assert train | test == set(data.index)
        design = info["design"]
        labels = (design, info["repeat"], info["fold"])
        outer_membership[labels] = test
        group = {"host": "host_group", "neighbourhood": "neighbourhood"}.get(design)
        for part in manifest["inner"]:
            a, b = set(part["train_row_id"]), set(part["test_row_id"])
            assert not a & b and a | b == train
            if group:
                assert not set(data.loc[list(a), group]) & set(data.loc[list(b), group])
            split_count += 1
        if group:
            assert not set(data.loc[list(train), group]) & set(data.loc[list(test), group])
        winner = min(core, key=lambda k: (info["inner_MAE_by_family"][k], core.index(k)))
        assert winner == info["selected_family_inner"]
        # Read the actual CV output rather than trusting its copied winner summary.
        for cv_path in folder.rglob("cv_results.csv"):
            cv = pd.read_csv(cv_path)
            selected = json.loads((cv_path.parent / "selection.json").read_text())
            inner_columns = [c for c in cv if re.fullmatch(r"split\d+_test_score", c)]
            assert len(inner_columns) == 3
            assert np.isfinite(cv[inner_columns].to_numpy()).all()
            assert np.allclose(cv[inner_columns].mean(axis=1), cv.mean_test_score)
            assert np.isclose(selected["inner_MAE"], -cv.mean_test_score.max())
            best = cv.loc[cv.mean_test_score.idxmax()]
            assert selected["best_params"] == json.loads(best.params_json)
            count = len(cv) * len(inner_columns) + 1
            assert count == selected["regressor_fits"]
            actual_cv_fits += count
        for family in core:
            selected = json.loads((folder / "full" / family / "selection.json").read_text())
            assert np.isclose(selected["inner_MAE"], info["inner_MAE_by_family"][family])
        fold_predictions = pd.read_csv(folder / "predictions.csv")
        fold_metrics = pd.read_csv(folder / "metrics.csv")
        chosen_predictions = fold_predictions[fold_predictions.method == winner + "_full"].set_index("row_id")
        selected_predictions = fold_predictions[fold_predictions.method == "selected_full"].set_index("row_id")
        assert np.allclose(chosen_predictions.loc[list(test), "prediction"], selected_predictions.loc[list(test), "prediction"])
        assert set(selected_predictions.family) == {winner}
        for method, frame in fold_predictions.groupby("method"):
            assert not frame.row_id.duplicated().any() and set(frame.row_id) == test
            assert set(zip(frame.design, frame["repeat"], frame.fold)) == {labels}
            assert np.allclose(frame.observed_price, data.loc[frame.row_id, "price"])
            row = fold_metrics[fold_metrics.method == method]
            assert len(row) == 1
            for name, value in metrics(frame).items():
                assert np.isclose(value, row.iloc[0][name], rtol=1e-10, atol=1e-10)
            fold_metric_checks += 1
        cal_splits = json.loads((folder / "interval/splits.json").read_text())
        fitting, calibration = set(cal_splits["fitting_row_id"]), set(cal_splits["calibration_row_id"])
        assert not fitting & calibration and fitting | calibration == train
        assert not (fitting | calibration) & test
        if group:
            assert not set(data.loc[list(fitting), group]) & set(data.loc[list(calibration), group])
        for part in cal_splits["inner"]:
            a, b = set(part["train_row_id"]), set(part["test_row_id"])
            assert not a & b and a | b == fitting
            if group:
                assert not set(data.loc[list(a), group]) & set(data.loc[list(b), group])
        cal = pd.read_csv(folder / "interval/calibration_residuals.csv")
        cal_info = json.loads((folder / "interval/summary.json").read_text())
        assert not cal.row_id.duplicated().any() and set(cal.row_id) == calibration
        assert np.allclose(cal.observed_price, data.loc[cal.row_id, "price"])
        assert np.allclose(cal.absolute_residual, np.abs(cal.observed_price - cal.prediction))
        expected_floor = .25 * data.loc[list(fitting), "price"].median()
        assert np.isclose(expected_floor, cal_info["scale_floor_proper_fit_only"])
        assert np.allclose(cal.proper_fit_price_scale, np.maximum(cal.prediction, expected_floor))
        assert np.allclose(cal.relative_absolute_residual, cal.absolute_residual / cal.proper_fit_price_scale)
        expected_tertiles = data.loc[list(fitting), "price"].quantile([1/3, 2/3]).to_numpy()
        assert np.allclose(expected_tertiles, cal_info["price_tertile_thresholds_proper_fit_only"])
        assert np.isclose(data.loc[list(fitting), "number_of_reviews"].median(), cal_info["review_threshold_fit_only"])
        calibration_scores = {family: json.loads((folder / "interval" / family / "selection.json").read_text())["inner_MAE"] for family in core}
        calibration_winner = min(core, key=lambda k: (calibration_scores[k], core.index(k)))
        assert calibration_winner == cal_info["selected_family"]
        rank = math.ceil((len(cal) + 1) * .8)
        for col, key in [("absolute_residual", "absolute_residual_quantile"),
                         ("relative_absolute_residual", "relative_residual_quantile")]:
            quantile = np.sort(cal[col].to_numpy())[rank - 1]
            assert np.isclose(quantile, cal_info[key])
        interval_predictions = pd.read_csv(folder / "interval_predictions.csv")
        for method, frame in interval_predictions.groupby("interval_method"):
            assert set(frame.row_id) == test and not frame.row_id.duplicated().any()
            assert set(zip(frame.design, frame["repeat"], frame.fold)) == {labels}
            assert set(frame.family) == {calibration_winner}
            if method == "absolute_residual":
                half_width = np.repeat(cal_info["absolute_residual_quantile"], len(frame))
            else:
                assert method == "relative_residual"
                half_width = cal_info["relative_residual_quantile"] * np.maximum(frame.prediction, expected_floor)
            assert np.allclose(frame.lower, np.maximum(frame.prediction - half_width, 0.))
            assert np.allclose(frame.upper, np.maximum(frame.prediction + half_width, 0.))
            assert np.isclose(frame.covered.mean(), cal_info["bands"][method]["coverage"])
            assert np.isclose(frame.width.mean(), cal_info["bands"][method]["mean_width"])
            for name, value in metrics(frame).items():
                assert np.isclose(value, cal_info["point_metrics"][name])
            predicted_band = np.where(frame.prediction <= expected_tertiles[0], "lower", np.where(frame.prediction <= expected_tertiles[1], "middle", "higher"))
            observed_band = np.where(frame.observed_price <= expected_tertiles[0], "lower", np.where(frame.observed_price <= expected_tertiles[1], "middle", "higher"))
            assert np.array_equal(frame.predicted_price_band, predicted_band)
            assert np.array_equal(frame.observed_price_band_diagnostic, observed_band)
            calibration_checks += 1
    # The remaining eighteen fits are the training-median baselines.
    assert actual_cv_fits + len(complete) == 2916
    predictions = pd.read_csv(args.results / "oof_predictions.csv")
    assert not predictions.duplicated(["design", "repeat", "method", "row_id"]).any()
    assert np.isfinite(predictions[["observed_price", "prediction"]].to_numpy()).all()
    assert np.allclose(predictions.observed_price.to_numpy(), data.loc[predictions.row_id, "price"].to_numpy())
    assert len(predictions) == len(data) * 2 * 3 * 15
    for labels, frame in predictions.groupby(["design", "repeat", "fold"]):
        for _, method_frame in frame.groupby("method"):
            assert set(method_frame.row_id) == outer_membership[labels]
    pooled = pd.read_csv(args.results / "pooled_metrics_by_repeat.csv")
    checks = 0
    for key, frame in predictions.groupby(["design", "repeat", "method"]):
        assert len(frame) == len(data) and set(frame.row_id) == set(data.index)
        row = pooled[(pooled.design == key[0]) & (pooled["repeat"] == key[1]) & (pooled.method == key[2])]
        assert len(row) == 1
        for name, value in metrics(frame).items():
            assert np.isclose(value, row.iloc[0][name], rtol=1e-10, atol=1e-10)
        checks += 1
    intervals = pd.read_csv(args.results / "interval_oof_predictions.csv")
    assert not intervals.duplicated(["design", "repeat", "interval_method", "row_id"]).any()
    assert len(intervals) == len(data) * 2 * 3 * 2
    assert np.isfinite(intervals[["observed_price", "prediction", "lower", "upper", "width"]].to_numpy()).all()
    assert np.allclose(intervals.observed_price, data.loc[intervals.row_id, "price"])
    assert (intervals.lower >= 0).all() and (intervals.upper >= intervals.lower).all()
    covered = ((intervals.observed_price >= intervals.lower) & (intervals.observed_price <= intervals.upper))
    assert np.array_equal(covered.to_numpy(), intervals.covered.to_numpy())
    assert np.allclose(intervals.width, intervals.upper - intervals.lower)
    for _, frame in intervals.groupby(["design", "repeat", "interval_method"]):
        assert len(frame) == len(data) and set(frame.row_id) == set(data.index)
    for labels, frame in intervals.groupby(["design", "repeat", "fold"]):
        for _, method_frame in frame.groupby("interval_method"):
            assert set(method_frame.row_id) == outer_membership[labels]
    interval_summary = pd.read_csv(args.results / "interval_summary.csv")
    interval_summary_checks = 0
    for _, row in interval_summary.iterrows():
        frame = intervals[(intervals.design == row.design) & (intervals["repeat"] == row["repeat"]) & (intervals.interval_method == row.interval_method)]
        if row.grouping != "overall":
            field = {"occupancy": "occupancy_type", "review_band": "review_band", "predicted_price_band": "predicted_price_band",
                     "observed_price_band_diagnostic": "observed_price_band_diagnostic"}[row.grouping]
            frame = frame[frame[field] == row[field]]
        assert len(frame) == row.n
        assert np.isclose(frame.covered.mean(), row.coverage)
        assert np.isclose(frame.width.mean(), row.mean_width)
        for name, value in metrics(frame).items():
            assert np.isclose(value, row[name], rtol=1e-10, atol=1e-10)
        interval_summary_checks += 1
    report = {"passed": True, "complete_outer_folds": len(complete),
              "primary_inner_split_checks": split_count,
              "fold_metric_rows_verified": fold_metric_checks,
              "cv_refits_counted_from_saved_candidate_tables": actual_cv_fits,
              "calibration_band_checks": calibration_checks,
              "interval_summary_rows_verified": interval_summary_checks,
              "regressor_fits": summary["regressor_fits"],
              "prediction_rows": len(predictions), "pooled_metric_rows_verified": checks,
              "interval_rows": len(intervals),
              "negative_point_predictions_by_method": predictions.assign(negative=predictions.prediction < 0).groupby("method").negative.sum().astype(int).to_dict(),
              "prepared_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "scope": "Integrity of saved computations and split isolation; not independent external validation or confirmation of upstream measurement."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
