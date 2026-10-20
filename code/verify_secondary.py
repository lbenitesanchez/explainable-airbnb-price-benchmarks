#!/usr/bin/env python3
"""Independent, read-only numerical audit of the Sustainability diagnostics.

Uses Python's standard library, not secondary_diagnostics or sklearn. The only
default write is qa/secondary_independent_verification.json. No refitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

DESIGNS = ("random", "host", "neighbourhood")
REPEATS = (1, 2)
INTERVALS = ("absolute_residual", "relative_residual")
POINT_METHODS = {
    "Ridge_full", "ElasticNet_full", "GradientBoosting_full", "RandomForest_full",
    "DecisionTree_full", "Bagging_full", "XGBoost_full", "selected_full",
    "without_S", "without_L", "without_R", "without_A", "without_T",
    "snapshot_H_sensitivity", "training_median",
}
POINT_METRICS = ("signed_bias", "MAE", "RMSE", "R2", "mean_observed_price", "mean_prediction")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def number(row, field):
    value = float(row[field]) if row[field] != "" else float("nan")
    return value


def boolean(value):
    require(value in ("True", "False", "true", "false", "1", "0"), "Non-boolean field")
    return value.lower() in ("true", "1")


def protect(root):
    path = root / "baseline_primary_sha256.json"
    baseline = json.loads(path.read_text())["files"]
    require(isinstance(baseline, dict) and len(baseline) > 0, "Private-run integrity baseline must contain files")
    differences = [name for name, digest in baseline.items()
                   if not (root / name).is_file() or sha256(root / name) != digest]
    require(not differences, f"Protected files changed: {differences}")
    return {"checked": len(baseline), "mismatches": [], "baseline_sha256": sha256(path)}


def host_lists(rows, data):
    result = defaultdict(list)
    for row in rows:
        result[data[row["row_id"]]["host_group"]].append(row)
    return result


def mean(values):
    return math.fsum(values) / len(values)


def average(rows, data, weighting, value):
    if weighting == "listing":
        return mean([value(row) for row in rows])
    require(weighting == "equal_host", "Unknown weighting")
    # Independent implementation: average within each host, then across hosts.
    return mean([mean([value(row) for row in group])
                 for group in host_lists(rows, data).values()])


def metrics(rows, data, weighting):
    observed_mean = average(rows, data, weighting, lambda row: number(row, "observed_price"))
    squared_error = average(rows, data, weighting,
                            lambda row: (number(row, "prediction") - number(row, "observed_price")) ** 2)
    target_variance = average(rows, data, weighting,
                              lambda row: (number(row, "observed_price") - observed_mean) ** 2)
    return {
        "n_listings": len(rows), "n_hosts": len(host_lists(rows, data)),
        "signed_bias": average(rows, data, weighting,
                               lambda row: number(row, "prediction") - number(row, "observed_price")),
        "MAE": average(rows, data, weighting,
                       lambda row: abs(number(row, "prediction") - number(row, "observed_price"))),
        "RMSE": math.sqrt(squared_error),
        "R2": 1 - squared_error / target_variance if target_variance else float("nan"),
        "mean_observed_price": observed_mean,
        "mean_prediction": average(rows, data, weighting, lambda row: number(row, "prediction")),
    }


def width_median(rows, data, weighting):
    values = [number(row, "upper") - number(row, "lower") for row in rows]
    if weighting == "listing":
        return statistics.median(values)
    counts = Counter(data[row["row_id"]]["host_group"] for row in rows)
    # Rational weights avoid a floating-point ambiguity at exactly half mass.
    pairs = sorted((value, Fraction(1, counts[data[row["row_id"]]["host_group"]]))
                   for row, value in zip(rows, values))
    total = sum((weight for _, weight in pairs), Fraction())
    cumulative = Fraction()
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= total / 2:
            return value
    raise AssertionError("Empty weighted median")


def interval_metrics(rows, data, weighting):
    point = metrics(rows, data, weighting)
    result = {"n_listings": point["n_listings"], "n_hosts": point["n_hosts"], "nominal_coverage": 0.8}
    result.update({"interval_point_" + key: value for key, value in point.items()
                   if key not in ("n_listings", "n_hosts")})
    result.update({
        "coverage": average(rows, data, weighting,
                            lambda row: float(number(row, "lower") <= number(row, "observed_price") <= number(row, "upper"))),
        "mean_width": average(rows, data, weighting,
                              lambda row: number(row, "upper") - number(row, "lower")),
        "median_width": width_median(rows, data, weighting),
    })
    return result


def validate_inputs(root, results, prepared_csv):
    raw = read_csv(prepared_csv)
    data = {row["row_id"]: row for row in raw}
    require(len(raw) == len(data) and len(data) > 0, "Prepared listing IDs/count mismatch")
    require(all(row["row_id"] and row["host_group"] and row["neighbourhood"] for row in raw), "Missing keys")
    host_counts = Counter(row["host_group"] for row in raw)
    neighbourhood_counts = Counter(row["neighbourhood"] for row in raw)
    require(len(host_counts) > 0 and len(neighbourhood_counts) > 0, "Empty host/neighbourhood groups")
    config = json.loads((results / "frozen_config.json").read_text())
    require(config["input_mode"] == "prepared_csv" and config["hashes"]["input_csv"] == sha256(prepared_csv), "Prepared file does not match frozen configuration")
    run = json.loads((results / "run_summary.json").read_text())
    require(run["completed_outer_folds"] == 18 and run["regressor_fits"] == 2916 and run["unique_rows_per_repetition"] == len(data), "Primary run summary mismatch")
    for row in raw:
        count = host_counts[row["host_group"]]
        require(int(row["host_portfolio_size_snapshot"]) == count, "Saved host-size proxy mismatch")
        require(int(row["multi_listing_host_snapshot"]) == int(count >= 2), "Saved multi-listing proxy mismatch")

    all_ids = set(data)
    fold_for = {}
    fingerprint = set()
    fits = 0
    checkpoints = sorted((results / "folds").glob("*/complete.json"))
    require(len(checkpoints) == 18, "Expected 18 completed folds")
    for design in DESIGNS:
        for repeat in REPEATS:
            seen = set()
            for fold in (1, 2, 3):
                folder = results / "folds" / f"{design}_r{repeat}_f{fold}"
                complete = json.loads((folder / "complete.json").read_text())
                require(complete["complete"] is True, "Incomplete checkpoint")
                require((complete["design"], complete["repeat"], complete["fold"]) == (design, repeat, fold), "Checkpoint identity mismatch")
                fingerprint.add(complete["fingerprint"])
                fits += complete["regressor_fits"]
                outer = json.loads((folder / "splits.json").read_text())["outer"]
                require(len(outer) == 1, "Expected one external split in each fold checkpoint")
                split = outer[0]
                train, test = set(split["train_row_id"]), set(split["test_row_id"])
                require(len(train) == split["n_train"] and len(test) == split["n_test"], "Repeated split rows")
                require(not train & test and train | test == all_ids, "External split is not a partition")
                require(not seen & test, "A listing occurs in multiple external test folds")
                if design != "random":
                    field = "host_group" if design == "host" else "neighbourhood"
                    require(not {data[x][field] for x in train} & {data[x][field] for x in test}, "External group overlap")
                seen |= test
                for row_id in test:
                    fold_for[design, repeat, row_id] = fold
            require(seen == all_ids, "External test folds do not cover snapshot")
    require(len(fingerprint) == 1 and fits == 2916, "Frozen numerical fingerprint/fit count mismatch")

    point = read_csv(results / "oof_predictions.csv")
    interval = read_csv(results / "interval_oof_predictions.csv")
    require(len(point) == len(data) * 6 * 15, "Point OOF row count mismatch")
    require(len(interval) == len(data) * 6 * 2, "Interval OOF row count mismatch")
    point_keys, interval_keys = set(), set()
    point_slices, interval_slices = defaultdict(set), defaultdict(set)
    selected = []
    for row in point + interval:
        key = row["row_id"]
        require(key in data, "Prediction key not in prepared data")
        design, repeat = row["design"], int(row["repeat"])
        require(design in DESIGNS and repeat in REPEATS, "Unexpected design/repetition")
        require(int(row["fold"]) == fold_for[design, repeat, key], "Saved external test-fold membership mismatch")
        require(number(row, "observed_price") == number(data[key], "price"), "Observed price linkage mismatch")
        require(math.isfinite(number(row, "prediction")), "Non-finite prediction")
        require(row["occupancy_type"] == data[key]["occupancy_type"], "Accommodation format linkage mismatch")
        require(number(row, "number_of_reviews") == number(data[key], "number_of_reviews"), "Review count linkage mismatch")
        if "interval_method" in row:
            require(row["method"] == "interval_point" and row["interval_method"] in INTERVALS and row["spec"] == "SLRAT", "Incorrect interval point-model identity")
            k = (design, repeat, key, row["interval_method"])
            require(k not in interval_keys, "Duplicated interval key")
            interval_keys.add(k)
            interval_slices[design, repeat, row["interval_method"]].add(key)
            lower, upper, observed = (number(row, name) for name in ("lower", "upper", "observed_price"))
            require(all(math.isfinite(v) for v in (lower, upper)) and 0 <= lower <= upper, "Invalid interval endpoints")
            require(close(number(row, "width"), upper - lower), "Saved interval width mismatch")
            require(boolean(row["covered"]) == (lower <= observed <= upper), "Saved interval coverage mismatch")
        else:
            require(row["method"] in POINT_METHODS, "Unexpected point method")
            k = (design, repeat, key, row["method"])
            require(k not in point_keys, "Duplicated point key")
            point_keys.add(k)
            point_slices[design, repeat, row["method"]].add(key)
            if row["method"] == "selected_full":
                require(row["spec"] == "SLRAT", "Selected-full features changed")
                selected.append(row)
    require(set(point_slices) == {(d, r, m) for d in DESIGNS for r in REPEATS for m in POINT_METHODS}, "Point slice completeness mismatch")
    require(set(interval_slices) == {(d, r, m) for d in DESIGNS for r in REPEATS for m in INTERVALS}, "Interval slice completeness mismatch")
    require(all(ids == all_ids for ids in list(point_slices.values()) + list(interval_slices.values())), "Snapshot coverage mismatch")
    # The two interval constructions share the same proper-fit point model.
    interval_points = {}
    for row in interval:
        k = (row["design"], int(row["repeat"]), row["row_id"])
        value = (row["prediction"], row["family"], row["fold"], row["spec"])
        require(k not in interval_points or interval_points[k] == value, "Interval constructions use different point models")
        interval_points[k] = value
    different = sum(not close(number(row, "prediction"), float(interval_points[row["design"], int(row["repeat"]), row["row_id"]][0]))
                    for row in selected)
    require(different > 0, "Main and interval point models unexpectedly identical throughout")
    return data, selected, interval, {
        "n_listings": len(data), "n_hosts": len(host_counts), "n_neighbourhoods": len(neighbourhood_counts),
        "single_listing_hosts": sum(n == 1 for n in host_counts.values()),
        "multi_listing_hosts": sum(n >= 2 for n in host_counts.values()),
        "multi_host_listing_n": sum(n for n in host_counts.values() if n >= 2),
        "neighbourhoods_n_ge_30": sum(n >= 30 for n in neighbourhood_counts.values()),
        "listings_in_neighbourhoods_n_ge_30": sum(n for n in neighbourhood_counts.values() if n >= 30),
        "point_oof_rows_all_methods": len(point), "selected_point_rows": len(selected),
        "interval_oof_rows": len(interval), "completed_checkpoints": len(checkpoints),
        "regressor_fits_unchanged": fits, "frozen_fingerprint": next(iter(fingerprint)),
        "main_vs_interval_point_different_predictions": different,
    }


def expected_tables(data, point, interval):
    """Rebuild exports from row-level sources without reading their generator."""
    outputs = {}
    for is_interval, source in ((False, point), (True, interval)):
        kind = "interval" if is_interval else "point"
        for context, subgroup_field in (("overall", None), ("host_portfolio", "host_portfolio_group"), ("neighbourhood", "neighbourhood")):
            key_fields = ["design", "repeat"]
            if is_interval:
                key_fields.append("interval_method")
            if subgroup_field:
                key_fields.append(subgroup_field)
            grouped = defaultdict(list)
            for row in source:
                key = [row["design"], row["repeat"]]
                if is_interval:
                    key.append(row["interval_method"])
                if context == "neighbourhood":
                    key.append(data[row["row_id"]]["neighbourhood"])
                elif context == "host_portfolio":
                    key.append("one_observed_listing" if int(data[row["row_id"]]["host_portfolio_size_snapshot"]) == 1 else "multiple_observed_listings")
                grouped[tuple(key)].append(row)
            expected = {}
            for key, rows in grouped.items():
                for weighting in ("listing", "equal_host"):
                    result = (interval_metrics if is_interval else metrics)(rows, data, weighting)
                    if context == "neighbourhood":
                        result["interpretable_n_ge_30"] = len(rows) >= 30
                    expected[(*key, weighting)] = result
            outputs[f"{context}_{kind}_by_repeat"] = (key_fields + ["weighting"], expected)
            summary_keys = [key for key in key_fields if key != "repeat"] + ["weighting"]
            by_repeats = defaultdict(list)
            for key, values in expected.items():
                # Repeat occupies position 1 in the documented input key.
                by_repeats[key[:1] + key[2:]].append((key[1], values))
            summary = {}
            invariant_fields = {"n_listings", "n_hosts", "nominal_coverage", "interpretable_n_ge_30"}
            for key, repetitions in by_repeats.items():
                require({repeat for repeat, _ in repetitions} == {"1", "2"} and len(repetitions) == 2, "Secondary summary pseudoreplication")
                first, second = [values for _, values in repetitions]
                result = {"n_repeats": 2}
                for field in first:
                    if field in invariant_fields:
                        require(first[field] == second[field], f"Repeated group invariant differs: {field}")
                        result[field] = first[field]
                    else:
                        values = [first[field], second[field]]
                        finite = [v for v in values if not math.isnan(v)]
                        result[field + "_mean"] = mean(finite) if finite else float("nan")
                        result[field + "_min"] = min(finite) if finite else float("nan")
                        result[field + "_max"] = max(finite) if finite else float("nan")
                summary[key] = result
            outputs[f"{context}_{kind}_descriptive_summary"] = (summary_keys, summary)
    return outputs


def compare_export(path, key_fields, expected):
    rows = read_csv(path)
    require(len(rows) == len(expected), f"Row count mismatch: {path.name}")
    require(rows, f"Empty export: {path.name}")
    fields = set(key_fields) | set(next(iter(expected.values())))
    require(set(rows[0]) == fields, f"Unexpected/missing columns: {path.name}: {set(rows[0]) ^ fields}")
    seen = set()
    differences = defaultdict(float)
    undefined = 0
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        require(key not in seen, f"Duplicate export key: {path.name}")
        seen.add(key)
        require(key in expected, f"Unknown export key: {path.name}: {key}")
        for field, value in expected[key].items():
            if isinstance(value, bool):
                require(boolean(row[field]) == value, f"Threshold flag mismatch: {path.name}: {key}")
            elif isinstance(value, int):
                require(float(row[field]) == value, f"Count mismatch: {path.name}: {key}: {field}")
            else:
                actual = number(row, field)
                require(close(actual, value), f"Numerical mismatch: {path.name}: {key}: {field}: exported={actual}, independent={value}")
                if math.isnan(value):
                    undefined += 1
                else:
                    differences[field] = max(differences[field], abs(actual - value))
    require(seen == set(expected), f"Export key completeness mismatch: {path.name}")
    # Within a design/repetition, neighbourhoods must appear alphabetically.
    if "neighbourhood" in key_fields:
        label_groups = defaultdict(list)
        for row in rows:
            other = tuple(row[field] for field in key_fields if field != "neighbourhood")
            label_groups[other].append(row["neighbourhood"])
        require(all(labels == sorted(labels) for labels in label_groups.values()), f"Neighbourhood rows ranked rather than alphabetical: {path.name}")
    return {"status": "PASS", "rows": len(rows), "sha256": sha256(path),
            "max_absolute_numerical_difference": dict(differences), "undefined_cells_preserved": undefined}


def verify_tables(root, secondary, data, point, interval):
    expected = expected_tables(data, point, interval)
    table_paths = set((secondary / "tables").glob("*.csv"))
    require({path.stem for path in table_paths} == set(expected), "Expected exactly 12 secondary CSV exports")
    result = {}
    for name, (keys, values) in expected.items():
        result[name] = compare_export(secondary / "tables" / f"{name}.csv", keys, values)
    manifest_path = secondary / "secondary_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["analysis"] == "secondary_descriptive_no_refit", "Incorrect secondary analysis identity")
    require(manifest["point_method"] == "selected_full" and manifest["interval_point_method"] == "interval_point", "Main/interval model conflation in manifest")
    require(manifest["interval_methods"] == list(INTERVALS) and manifest["neighbourhood_interpretability_minimum_listings"] == 30, "Fixed diagnostic choices changed")
    require(manifest["repeats"] == list(REPEATS) and manifest["complete_outer_folds"] == 18, "Repetition/fold manifest mismatch")
    require(set(manifest["designs"]) == set(DESIGNS), "Design manifest mismatch")
    require(manifest["currency"]["code"] == "PEN" and manifest["currency"]["numeric_values_changed"] is False and manifest["currency"]["currency_conversion_applied"] is False, "Currency/unit declaration changed")
    for name, metadata in manifest["input_sha256"].items():
        require(sha256(root / metadata["path"]) == metadata["sha256"], f"Manifest input hash mismatch: {name}")
    for name, metadata in manifest["outputs"].items():
        require(sha256(secondary / name) == metadata["sha256"], f"Manifest output hash mismatch: {name}")
        require(len(read_csv(secondary / name)) == metadata["rows"], f"Manifest row count mismatch: {name}")
    require(set(manifest["outputs"]) == {f"tables/{name}.csv" for name in expected}, "Manifest output coverage mismatch")
    for name, digest in manifest["figures"].items():
        require(sha256(secondary / name) == digest, f"Manifest figure hash mismatch: {name}")
    for phase in ("primary_integrity_before", "primary_integrity_after"):
        require(manifest[phase]["all_match"] is True and manifest[phase]["files_verified"] == len(json.loads((root / "baseline_primary_sha256.json").read_text())["files"]),
                "Generator protected-file check did not pass")
    require(sha256(secondary / "integration_notes_es.json") == manifest["notes_sha256"], "Integration-note hash mismatch")
    # Store publication-ready aggregate values, independently computed.
    overall = expected["overall_point_descriptive_summary"][1]
    host = expected["host_portfolio_point_descriptive_summary"][1]
    neighbourhood = expected["neighbourhood_point_descriptive_summary"][1]
    interval_neighbourhood = expected["neighbourhood_interval_descriptive_summary"][1]
    findings = {}
    for design in DESIGNS:
        row = {"overall": {weight: overall[(design, weight)] for weight in ("listing", "equal_host")},
               "host_portfolio": {group: {weight: host[(design, group, weight)] for weight in ("listing", "equal_host")}
                                  for group in ("one_observed_listing", "multiple_observed_listings")}}
        eligible = [v for k, v in neighbourhood.items() if k[0] == design and k[-1] == "listing" and v["interpretable_n_ge_30"]]
        row["eligible_neighbourhood_listing_weight_two_repeat_mean_ranges"] = {
            metric: [min(v[metric + "_mean"] for v in eligible), max(v[metric + "_mean"] for v in eligible)]
            for metric in ("MAE", "signed_bias")}
        row["eligible_neighbourhood_interval_coverage_two_repeat_mean_ranges"] = {
            method: [min(v["coverage_mean"] for k, v in interval_neighbourhood.items()
                         if k[0] == design and k[1] == method and k[-1] == "listing" and v["interpretable_n_ge_30"]),
                     max(v["coverage_mean"] for k, v in interval_neighbourhood.items()
                         if k[0] == design and k[1] == method and k[-1] == "listing" and v["interpretable_n_ge_30"])]
            for method in INTERVALS}
        findings[design] = row
    return {"exports": result, "independently_recalculated_rows": sum(v["rows"] for v in result.values()),
            "manifest_sha256": sha256(manifest_path), "findings": findings}


def self_test():
    data = {"a": {"host_group": "h1"}, "b": {"host_group": "h1"}, "c": {"host_group": "h2"}}
    rows = [{"row_id": key, "observed_price": str(y), "prediction": str(p), "lower": "0", "upper": str(width)}
            for key, y, p, width in (("a", 1, 2, 2), ("b", 2, 4, 4), ("c", 3, 0, 6))]
    m = metrics(rows, data, "equal_host")
    require(close(m["MAE"], 2.25) and close(m["signed_bias"], -0.75), "Host/sign unit test failed")
    require(close(m["RMSE"], math.sqrt(5.75)), "Host RMSE must average MSE before taking square root")
    require(close(m["R2"], 1 - 5.75 / 0.6875), "Weighted R2 denominator test failed")
    require(width_median(rows, data, "equal_host") == 4, "Lower inverse-CDF median test failed")
    require(width_median(rows[:2], data, "listing") == 3, "Conventional sample median test failed")
    constant = [dict(row, observed_price="1") for row in rows]
    require(math.isnan(metrics(constant, data, "listing")["R2"]), "Constant-target R2 must be undefined")
    # A host spanning subgroups gets total unit weight within the reported subset.
    require(close(metrics(rows[1:], data, "equal_host")["MAE"], 2.5), "Subgroup host counts must be recomputed locally")
    try:
        require(False, "intentional negative control")
    except AssertionError:
        pass
    else:
        raise AssertionError("Negative control failed")
    return {"status": "PASS", "checks": 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path, help="Saved primary results directory")
    parser.add_argument("--data", type=Path, help="Prepared restricted listing CSV")
    parser.add_argument("--secondary", type=Path, help="Secondary export directory to verify")
    parser.add_argument("--output", type=Path, help="Verification JSON; default qa/secondary_independent_verification.json")
    parser.add_argument("--self-test", action="store_true", help="Run independent metric unit tests only; no output file")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    root = args.root.resolve()
    results = (args.results or root / "results").resolve()
    prepared_csv = (args.data or root / "data/model_ready_listings_restricted.csv").resolve()
    secondary = (args.secondary or root / "publication_secondary").resolve()
    output = (args.output or root / "qa/secondary_independent_verification.json").resolve()
    report = {"status": "FAIL", "utc": datetime.now(timezone.utc).isoformat(),
              "implementation": "independent Python standard-library recalculation; no imported analysis implementation; no model fitting",
              "verifier_sha256": sha256(Path(__file__)), "limitations": [
                  "Post-protocol secondary description of an already examined historical snapshot; not a new independent sample.",
                  "Two repetitions reuse listings and are descriptive, not independent inferential replications.",
                  "Observed host portfolio is not platform-wide scale, professionalism, residence, or local ownership.",
                  "Neighbourhood n>=30 is a reporting threshold, not a guarantee of reliability or social representativeness.",
                  "Prices and widths are PEN asking prices; no demand, revenue, fairness, housing or environmental effect is estimated.",
              ]}
    try:
        report["metric_self_tests"] = self_test()
        report["protected_before"] = protect(root)
        data, point, interval, report["input_checks"] = validate_inputs(root, results, prepared_csv)
        report["tables"] = verify_tables(root, secondary, data, point, interval)
        report["protected_after"] = protect(root)
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "error": report.get("error")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
