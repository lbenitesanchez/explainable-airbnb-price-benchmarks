#!/usr/bin/env python3
"""Nested, shared-fold evaluation; see PROTOCOL.md before interpreting results.

Default invocation only records the frozen plan. Add --execute after review.
Original Dropbox files are read only. Predictions are for advertised prices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from features import ListingText, SafeTextSVD

SEEDS = (20261018, 20261019)
CORE = ("Ridge", "ElasticNet", "GradientBoosting", "RandomForest")
FULL = "SLRAT"
BLOCKS = {
    "S": {"num": ["accommodates", "bedrooms", "beds", "cleaned_bathrooms"],
          "cat": ["occupancy_type", "property_group_corrected"]},
    "L": {"num": ["latitude", "longitude", "dist_recreacion_m", "dist_hospital_m",
                  "dist_universidad_m", "dist_colegio_m", "dist_tienda_m", "dist_estacion_bus_m"],
          "cat": ["neighbourhood"]},
    "R": {"num": ["number_of_reviews", "review_scores_rating", "review_scores_cleanliness",
                  "review_scores_accuracy", "review_scores_checkin", "review_scores_communication",
                  "review_scores_location", "review_scores_value"], "cat": []},
    "A": {"num": ["available_amenities_count", "luxury_amenities_count", "basic_amenities_count",
                  "beach_amenities_count", "has_pool_corrected"], "cat": []},
    "T": {"text": ["name", "description"], "num": [], "cat": []},
    "H": {"num": ["host_portfolio_size_snapshot", "multi_listing_host_snapshot"], "cat": []},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")


def _json(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def preprocessing(spec):
    numeric = [col for block in spec for col in BLOCKS[block]["num"]]
    categorical = [col for block in spec for col in BLOCKS[block]["cat"]]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if "T" in spec:
        transformers.append(("text", Pipeline([("join", ListingText()), ("semantic", SafeTextSVD())]),
                             BLOCKS["T"]["text"]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def model_grids(include_xgboost=False):
    families = {
        "Ridge": (Ridge(), {"model__alpha": [.1, 10., 1000.]}),
        "ElasticNet": (ElasticNet(l1_ratio=.5, max_iter=20000, tol=1e-5, selection="cyclic"),
                       {"model__alpha": [.01, .1, 1.]}),
        "GradientBoosting": (GradientBoostingRegressor(learning_rate=.05, random_state=42), [
            {"model__n_estimators": [100], "model__max_depth": [3], "model__min_samples_leaf": [2]},
            {"model__n_estimators": [300], "model__max_depth": [3], "model__min_samples_leaf": [5]},
            {"model__n_estimators": [300], "model__max_depth": [5], "model__min_samples_leaf": [5]},
        ]),
        "RandomForest": (RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1), [
            {"model__min_samples_leaf": [2], "model__max_features": [.7]},
            {"model__min_samples_leaf": [5], "model__max_features": [.7]},
            {"model__min_samples_leaf": [5], "model__max_features": [1.]},
        ]),
        "DecisionTree": (DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
                         {"model__max_depth": [4, 8]}),
        "Bagging": (BaggingRegressor(estimator=DecisionTreeRegressor(min_samples_leaf=3, random_state=42),
                                     n_estimators=100, random_state=42, n_jobs=1),
                    {"model__max_features": [.7, 1.]}),
    }
    if include_xgboost:
        from xgboost import XGBRegressor
        families["XGBoost"] = (XGBRegressor(n_estimators=200, learning_rate=.05, objective="reg:squarederror",
                                           tree_method="hist", n_jobs=1, random_state=42, verbosity=0),
                               {"model__max_depth": [3, 5]})
    return families


def group_values(data, design):
    if design == "random":
        return None
    if design == "host":
        return data["host_group"].astype(str).to_numpy()
    if design == "neighbourhood":
        return data["neighbourhood"].fillna("__missing__").astype(str).to_numpy()
    raise ValueError(f"Unknown design: {design}")


def make_splits(data, design, seed, n_splits=3):
    groups = group_values(data, design)
    if groups is None:
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(data))
    return list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(data, groups=groups))


def assert_isolated(data, train, test, design):
    if set(train) & set(test):
        raise AssertionError("Overlapping row positions")
    if design != "random":
        values = group_values(data, design)
        if set(values[train]) & set(values[test]):
            raise AssertionError(f"Overlapping {design} groups")


def split_manifest(data, splits, design):
    result = []
    for index, (train, test) in enumerate(splits):
        assert_isolated(data, train, test, design)
        result.append({"fold": index, "train_row_id": data.iloc[train].row_id.tolist(),
                       "test_row_id": data.iloc[test].row_id.tolist(),
                       "n_train": len(train), "n_test": len(test),
                       "test_groups": None if design == "random" else sorted(set(group_values(data, design)[test]))})
    return result


def score(y, prediction):
    return {"MAE": float(mean_absolute_error(y, prediction)),
            "RMSE": float(math.sqrt(mean_squared_error(y, prediction))),
            "R2": float(r2_score(y, prediction)), "n": int(len(y))}


def search_family(name, spec, train, splits, families, folder, jobs, cache):
    estimator, grid = families[name]
    pipeline = Pipeline([("preprocess", preprocessing(spec)), ("model", estimator)], memory=cache)
    search = GridSearchCV(pipeline, grid, cv=splits, scoring="neg_mean_absolute_error",
                          refit=True, n_jobs=jobs, error_score="raise", return_train_score=True)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        search.fit(train, train["price"].to_numpy())
    folder.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(search.cv_results_)
    results["params_json"] = results["params"].map(lambda x: json.dumps(x, sort_keys=True))
    results.to_csv(folder / "cv_results.csv", index=False)
    info = {"family": name, "spec": spec, "inner_MAE": float(-search.best_score_),
            "best_params": search.best_params_, "seconds": time.monotonic() - started,
            "candidate_count": len(results), "inner_folds": len(splits),
            "regressor_fits": len(results) * len(splits) + 1}
    dump(info, folder / "selection.json")
    return search.best_estimator_, info


def choose_core(infos):
    # Tie rule is fixed in the protocol; only INNER metrics enter the decision.
    return min(CORE, key=lambda name: (infos[name]["inner_MAE"], CORE.index(name)))


def prediction_frame(test, prediction, design, repeat, fold, method, spec, family):
    return pd.DataFrame({"row_id": test.row_id.to_numpy(), "observed_price": test.price.to_numpy(),
                         "prediction": np.asarray(prediction), "design": design, "repeat": repeat,
                         "fold": fold, "method": method, "spec": spec, "family": family,
                         "occupancy_type": test.occupancy_type.to_numpy(),
                         "number_of_reviews": test.number_of_reviews.to_numpy()})


def calibration_split(train, design, seed):
    groups = group_values(train, design)
    if groups is None:
        fitting, calibration = train_test_split(np.arange(len(train)), test_size=.25, random_state=seed)
    else:
        fitting, calibration = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
                                    .split(train, groups=groups))
    assert_isolated(train, fitting, calibration, design)
    return fitting, calibration


def residual_quantile(errors, alpha=.2):
    """Finite-sample split-conformal rank, rather than an interpolated quantile."""
    errors = np.asarray(errors, dtype=float)
    rank = math.ceil((len(errors) + 1) * (1 - alpha))
    return float(np.partition(errors, rank - 1)[rank - 1]) if rank <= len(errors) else math.inf


def evaluate_intervals(train, test, design, repeat, fold, seed, families, folder, jobs, cache):
    fitting_index, calibration_index = calibration_split(train, design, seed)
    fitting, calibration = train.iloc[fitting_index].copy(), train.iloc[calibration_index].copy()
    inner = make_splits(fitting, design, seed + 1)
    dump({"fitting_row_id": fitting.row_id.tolist(), "calibration_row_id": calibration.row_id.tolist(),
          "inner": split_manifest(fitting, inner, design)}, folder / "splits.json")
    models, infos = {}, {}
    for name in CORE:
        models[name], infos[name] = search_family(name, FULL, fitting, inner, families,
                                                folder / name, jobs, cache)
    selected = choose_core(infos)
    model = models[selected]
    calibration_prediction = model.predict(calibration)
    errors = np.abs(calibration.price.to_numpy() - calibration_prediction)
    quantile = residual_quantile(errors)
    prediction = model.predict(test)
    review_threshold = float(fitting.number_of_reviews.median())
    price_thresholds = fitting.price.quantile([1/3, 2/3]).to_numpy()
    scale_floor = .25 * float(fitting.price.median())
    calibration_scale = np.maximum(calibration_prediction, scale_floor)
    prediction_scale = np.maximum(prediction, scale_floor)
    relative_quantile = residual_quantile(errors / calibration_scale)
    outputs, band_summaries = [], {}
    for method, half_width in [("absolute_residual", np.repeat(quantile, len(test))),
                               ("relative_residual", relative_quantile * prediction_scale)]:
        frame = prediction_frame(test, prediction, design, repeat, fold, "interval_point", FULL, selected)
        frame["interval_method"] = method
        frame["lower"] = np.maximum(prediction - half_width, 0.)
        frame["upper"] = np.maximum(prediction + half_width, 0.)
        frame["covered"] = ((frame.observed_price >= frame.lower) & (frame.observed_price <= frame.upper))
        frame["width"] = frame.upper - frame.lower
        frame["review_band"] = np.where(frame.number_of_reviews <= review_threshold, "lower", "higher")
        # Defined by predicted price and proper-fitting target thresholds; no test outcome enters the bins.
        frame["predicted_price_band"] = np.where(frame.prediction <= price_thresholds[0], "lower",
                                                np.where(frame.prediction <= price_thresholds[1], "middle", "higher"))
        frame["observed_price_band_diagnostic"] = np.where(frame.observed_price <= price_thresholds[0], "lower",
                                                           np.where(frame.observed_price <= price_thresholds[1], "middle", "higher"))
        band_summaries[method] = {"coverage": float(frame.covered.mean()), "mean_width": float(frame.width.mean())}
        outputs.append(frame)
    output = pd.concat(outputs, ignore_index=True)
    if repeat == 1 and fold == 1:
        joblib.dump(model, folder / "selected_interval_model.joblib")
    dump({"selected_family": selected, "proper_fit_n": len(fitting), "calibration_n": len(calibration),
          "test_n": len(test), "nominal_coverage": .8, "absolute_residual_quantile": quantile,
          "relative_residual_quantile": relative_quantile, "scale_floor_proper_fit_only": scale_floor,
          "price_tertile_thresholds_proper_fit_only": price_thresholds,
          "bands": band_summaries,
          "review_threshold_fit_only": review_threshold, "point_metrics": score(test.price, prediction),
          "regressor_fits": sum(info["regressor_fits"] for info in infos.values()),
          "interpretation": "Separate fitting/calibration; exchangeability required for coverage guarantees; grouped designs alone do not establish it."},
         folder / "summary.json")
    pd.DataFrame({"row_id": calibration.row_id.to_numpy(), "observed_price": calibration.price.to_numpy(),
                  "prediction": calibration_prediction, "absolute_residual": errors,
                  "proper_fit_price_scale": calibration_scale,
                  "relative_absolute_residual": errors / calibration_scale}).to_csv(folder / "calibration_residuals.csv", index=False)
    output.to_csv(folder / "predictions.csv", index=False)
    return output, sum(info["regressor_fits"] for info in infos.values())


def run_fold(data, train_index, test_index, design, repeat, fold, seed, config, families, output, fingerprint):
    folder = output / "folds" / f"{design}_r{repeat}_f{fold}"
    checkpoint = folder / "complete.json"
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous["fingerprint"] != fingerprint:
            raise RuntimeError(f"Checkpoint belongs to a different run: {folder}")
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    assert_isolated(data, train_index, test_index, design)
    train, test = data.iloc[train_index].copy(), data.iloc[test_index].copy()
    inner = make_splits(train, design, seed + 100 + fold)
    dump({"outer": split_manifest(data, [(train_index, test_index)], design),
          "inner": split_manifest(train, inner, design)}, folder / "splits.json")
    cache = joblib.Memory(output / "cache", verbose=0)
    rows, predictions, models, infos = [], [], {}, {}
    fits = 0
    for name in families:
        model, info = search_family(name, FULL, train, inner, families, folder / "full" / name, config["jobs"], cache)
        models[name], infos[name] = model, info
        fits += info["regressor_fits"]
        predicted = model.predict(test)
        method = f"{name}_full"
        rows.append({"design": design, "repeat": repeat, "fold": fold, "method": method,
                     "family": name, "spec": FULL, "n_train": len(train), **score(test.price, predicted)})
        predictions.append(prediction_frame(test, predicted, design, repeat, fold, method, FULL, name))
    selected = choose_core(infos)
    if repeat == 1 and fold == 1:
        joblib.dump(models[selected], folder / "selected_full_model.joblib")
    full_prediction = models[selected].predict(test)
    rows.append({"design": design, "repeat": repeat, "fold": fold, "method": "selected_full",
                 "family": selected, "spec": FULL, "n_train": len(train), **score(test.price, full_prediction)})
    predictions.append(prediction_frame(test, full_prediction, design, repeat, fold, "selected_full", FULL, selected))
    for removed in FULL:
        spec = FULL.replace(removed, "")
        method = f"without_{removed}"
        model, info = search_family(selected, spec, train, inner, families, folder / method, config["jobs"], cache)
        fits += info["regressor_fits"]
        predicted = model.predict(test)
        rows.append({"design": design, "repeat": repeat, "fold": fold, "method": method,
                     "family": selected, "spec": spec, "n_train": len(train), **score(test.price, predicted)})
        predictions.append(prediction_frame(test, predicted, design, repeat, fold, method, spec, selected))
    if config["host_sensitivity"]:
        method, spec = "snapshot_H_sensitivity", FULL + "H"
        model, info = search_family(selected, spec, train, inner, families, folder / method, config["jobs"], cache)
        fits += info["regressor_fits"]
        predicted = model.predict(test)
        rows.append({"design": design, "repeat": repeat, "fold": fold, "method": method,
                     "family": selected, "spec": spec, "n_train": len(train), **score(test.price, predicted)})
        predictions.append(prediction_frame(test, predicted, design, repeat, fold, method, spec, selected))
    constant = DummyRegressor(strategy="median").fit(np.zeros((len(train), 1)), train.price)
    predicted = constant.predict(np.zeros((len(test), 1)))
    rows.append({"design": design, "repeat": repeat, "fold": fold, "method": "training_median",
                 "family": "Dummy", "spec": "none", "n_train": len(train), **score(test.price, predicted)})
    predictions.append(prediction_frame(test, predicted, design, repeat, fold, "training_median", "none", "Dummy"))
    fits += 1
    if config["intervals"]:
        interval, interval_fits = evaluate_intervals(train, test, design, repeat, fold, seed + 200 + fold,
                                                     families, folder / "interval", config["jobs"], cache)
        fits += interval_fits
        interval.to_csv(folder / "interval_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(folder / "predictions.csv", index=False)
    summary = {"fingerprint": fingerprint, "design": design, "repeat": repeat, "fold": fold,
               "selected_family_inner": selected, "inner_MAE_by_family": {k: v["inner_MAE"] for k, v in infos.items()},
               "regressor_fits": fits, "complete": True}
    dump(summary, checkpoint)
    return summary


def aggregate(output, expected_n, repeats):
    folders = sorted((output / "folds").glob("*/complete.json"))
    if not folders:
        return
    metrics = pd.concat([pd.read_csv(x.parent / "metrics.csv") for x in folders], ignore_index=True)
    predictions = pd.concat([pd.read_csv(x.parent / "predictions.csv") for x in folders], ignore_index=True)
    keys = ["design", "repeat", "method"]
    pooled = []
    for labels, frame in predictions.groupby(keys):
        if frame.row_id.duplicated().any():
            raise AssertionError(f"Repeated external prediction inside repetition: {labels}")
        if len(frame) != expected_n:
            raise AssertionError(f"Incomplete OOF coverage: {labels}: {len(frame)} != {expected_n}")
        pooled.append(dict(zip(keys, labels)) | score(frame.observed_price, frame.prediction))
    pooled = pd.DataFrame(pooled)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    pooled.to_csv(output / "pooled_metrics_by_repeat.csv", index=False)
    pooled.groupby(["design", "method"])[["MAE", "RMSE", "R2"]].agg(["mean", "min", "max"]).to_csv(output / "pooled_metric_descriptives.csv")
    comparisons = []
    for (design, repeat, fold), frame in metrics.groupby(["design", "repeat", "fold"]):
        full = frame.loc[frame.method == "selected_full"].iloc[0]
        for _, row in frame.loc[frame.method.str.startswith("without_") | (frame.method == "snapshot_H_sensitivity")].iterrows():
            comparisons.append({"design": design, "repeat": repeat, "fold": fold, "method": row.method,
                                "family_selected_inside_training": full.family,
                                "MAE_variant_minus_full": row.MAE - full.MAE,
                                "RMSE_variant_minus_full": row.RMSE - full.RMSE,
                                "R2_full_minus_variant": full.R2 - row.R2})
    pd.DataFrame(comparisons).to_csv(output / "paired_fold_ablations.csv", index=False)
    pooled_comparisons = pooled.merge(pooled[pooled.method == "selected_full"].drop(columns="method"),
                                      on=["design", "repeat"], suffixes=("", "_full"))
    for metric in ["MAE", "RMSE", "R2"]:
        pooled_comparisons[f"{metric}_variant_minus_full"] = pooled_comparisons[metric] - pooled_comparisons[f"{metric}_full"]
    pooled_comparisons.to_csv(output / "paired_pooled_contrasts.csv", index=False)
    interval_files = [x.parent / "interval_predictions.csv" for x in folders if (x.parent / "interval_predictions.csv").exists()]
    if interval_files:
        intervals = pd.concat([pd.read_csv(x) for x in interval_files], ignore_index=True)
        intervals.to_csv(output / "interval_oof_predictions.csv", index=False)
        summaries = []
        for grouping, fields in [("overall", ["design", "repeat", "interval_method"]),
                                 ("occupancy", ["design", "repeat", "interval_method", "occupancy_type"]),
                                 ("review_band", ["design", "repeat", "interval_method", "review_band"]),
                                 ("predicted_price_band", ["design", "repeat", "interval_method", "predicted_price_band"]),
                                 ("observed_price_band_diagnostic", ["design", "repeat", "interval_method", "observed_price_band_diagnostic"])]:
            for labels, frame in intervals.groupby(fields):
                labels = labels if isinstance(labels, tuple) else (labels,)
                summaries.append({"grouping": grouping, **dict(zip(fields, labels)),
                                  "coverage": frame.covered.mean(), "mean_width": frame.width.mean(),
                                  "nominal_coverage": .8, **score(frame.observed_price, frame.prediction)})
        pd.DataFrame(summaries).to_csv(output / "interval_summary.csv", index=False)
    dump({"completed_outer_folds": len(folders), "unique_rows_per_repetition": expected_n,
          "repetitions_planned": repeats, "regressor_fits": sum(json.loads(x.read_text())["regressor_fits"] for x in folders),
          "interpretation": "Internal revalidation of historical data; overlapping folds/repeats are not independent replicates; no conventional fold-based confidence intervals."}, output / "run_summary.json")


def configuration(args, families):
    from sklearn.model_selection import ParameterGrid
    core_cost = sum(len(ParameterGrid(families[name][1])) * 3 + 1 for name in CORE)
    all_cost = sum(len(ParameterGrid(grid)) * 3 + 1 for _, grid in families.values())
    max_ablation_cost = max(len(ParameterGrid(families[name][1])) * 3 + 1 for name in CORE)
    per_fold = all_cost + 5 * max_ablation_cost + int(args.host_sensitivity) * max_ablation_cost + 1
    if args.intervals:
        per_fold += core_cost
    config = {"designs": args.designs, "outer_folds": 3, "inner_folds": 3, "seeds": list(SEEDS[:args.repeats]),
              "repeats": args.repeats, "jobs": args.jobs, "main_spec": FULL, "blocks": BLOCKS,
              "primary_metric": "MAE", "secondary_metrics": ["RMSE", "R2"],
              "core_families": list(CORE), "families": {name: {"estimator": repr(estimator), "grid": grid}
                                                         for name, (estimator, grid) in families.items()},
              "text": {"min_df": 3, "max_df": .95, "max_features": 1500, "n_components": 4},
              "intervals": args.intervals, "host_sensitivity": args.host_sensitivity,
              "estimated_regressor_fits": per_fold * 3 * args.repeats * len(args.designs),
              "selection": "Inner 3-fold MAE; fixed core-family tie order; no outer outcomes enter model or interval selection."}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--source", type=Path, help="Original CSV; invokes preparation and records its hash")
    inputs.add_argument("--prepared-csv", type=Path, help="Portable corrected restricted CSV; no original identifiers required")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--designs", nargs="+", choices=["random", "host", "neighbourhood"], default=["random", "host", "neighbourhood"])
    parser.add_argument("--repeats", type=int, choices=[1, 2], default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--xgboost", action="store_true")
    parser.add_argument("--intervals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host-sensitivity", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.source is None and args.prepared_csv is None:
        args.prepared_csv = Path(__file__).resolve().parents[1] / "data/model_ready_listings_restricted.csv"
    if len(set(args.designs)) != len(args.designs):
        parser.error("Designs may not be repeated")
    if args.jobs < 1:
        parser.error("Use a positive bounded number of worker processes")
    args.output.mkdir(parents=True, exist_ok=True)
    families = model_grids(args.xgboost)
    config = configuration(args, families)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "sklearn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
                "platform": platform.platform()}
    if args.xgboost:
        import xgboost
        versions["xgboost"] = xgboost.__version__
    config["versions"] = versions
    input_csv = args.prepared_csv if args.prepared_csv is not None else args.source
    config["input_mode"] = "prepared_csv" if args.prepared_csv is not None else "original_with_preparation"
    preparation_script = Path(__file__).with_name("prepare_data.py")
    config["hashes"] = {"input_csv": digest(input_csv), "reanalysis_code": digest(__file__),
                        "preparation_code": digest(preparation_script),
                        "feature_code": digest(Path(__file__).with_name("features.py")),
                        "protocol": digest(Path(__file__).with_name("PROTOCOL.md"))}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["fingerprint"] = fingerprint
    config_path = args.output / "frozen_config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError("Output folder already has a different frozen configuration; use another output directory")
    dump(config, config_path)
    dump(versions, args.output / "versions.json")
    print(json.dumps({"execute": args.execute, "estimated_regressor_fits": config["estimated_regressor_fits"],
                      "config": str(config_path), "fingerprint": fingerprint}), flush=True)
    if not args.execute:
        return
    if args.prepared_csv is not None:
        data = pd.read_csv(args.prepared_csv)
        if "source_row_index" in data and not data.source_row_index.is_monotonic_increasing:
            raise AssertionError("Prepared CSV order must retain ascending original source-row order")
    else:
        from prepare_data import prepare_data
        data = prepare_data(args.source, args.output / "data")
    if not data.row_id.is_unique:
        raise AssertionError("Stable row IDs must be unique")
    if not np.isfinite(data.price).all() or not (data.price > 0).all():
        raise AssertionError("Target prices must be finite and positive; adjudicate invalid originals before fitting")
    required = {"row_id", "price", "host_group"} | {col for block in BLOCKS.values() for cols in block.values() for col in cols}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing preparation fields: {sorted(missing)}")
    dump(ListingText().audit(data[["name", "description"]]), args.output / "text_sanitization_audit.json")
    log = args.output / "progress.jsonl"
    completed = []
    for design in args.designs:
        for repeat, seed in enumerate(SEEDS[:args.repeats], 1):
            external = make_splits(data, design, seed)
            dump(split_manifest(data, external, design), args.output / "splits" / f"{design}_repeat{repeat}.json")
            for fold, (train, test) in enumerate(external, 1):
                started = time.monotonic()
                result = run_fold(data, train, test, design, repeat, fold, seed, config, families, args.output, fingerprint)
                event = {**result, "seconds_this_call": time.monotonic() - started}
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, default=_json) + "\n")
                print(json.dumps(event, default=_json), flush=True)
                completed.append(result)
    aggregate(args.output, len(data), args.repeats)
    print(json.dumps({"complete": True, "folds": len(completed),
                      "regressor_fits": sum(x["regressor_fits"] for x in completed), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
