#!/usr/bin/env python3
"""Descriptive secondary diagnostics from preserved out-of-fold predictions.

No estimator is imported, fitted, selected or recalibrated. See the prespecified
secondary protocol for estimands and the fixed neighbourhood reporting rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apply_currency_metadata import read_currency_metadata


DESIGNS = ("random", "host", "neighbourhood")
DESIGN_LABELS = {"random": "Random listings", "host": "Held-out hosts",
                 "neighbourhood": "Held-out neighbourhoods"}
REPEATS = (1, 2)
INTERVAL_METHODS = ("absolute_residual", "relative_residual")
WEIGHTINGS = ("listing", "equal_host")
MIN_NEIGHBOURHOOD_LISTINGS = 30
POINT_METRICS = ("signed_bias", "MAE", "RMSE", "R2", "mean_observed_price", "mean_prediction")
INTERVAL_METRICS = ("coverage", "mean_width", "median_width",
                    *(f"interval_point_{m}" for m in POINT_METRICS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_primary(root: Path, baseline: Path) -> dict:
    """Read every protected file; secondary outputs never join this baseline."""
    raw = json.loads(baseline.read_text(encoding="utf-8"))
    entries = raw.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Primary baseline must contain a nonempty files mapping.")
    for relative, expected in entries.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Primary baseline contains a path outside the project.")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Protected primary file missing or changed: {relative}")
    return {"files_verified": len(entries), "all_match": True,
            "baseline_sha256": sha256(baseline)}


def validate_prediction_slices(frame: pd.DataFrame, data: pd.DataFrame,
                               splits: dict, interval: bool = False) -> None:
    keys = ["design", "repeat"] + (["interval_method"] if interval else [])
    if set(frame.design) != set(DESIGNS) or set(frame["repeat"]) != set(REPEATS):
        raise ValueError("Predictions must contain every prespecified design and repetition.")
    if interval and set(frame.interval_method) != set(INTERVAL_METHODS):
        raise ValueError("Both preserved interval methods must be present.")
    if not np.isfinite(frame[["observed_price", "prediction"]].to_numpy()).all():
        raise ValueError("Nonfinite observed prices or predictions.")
    target = data.set_index("row_id").price
    linked = frame.row_id.map(target)
    if linked.isna().any() or not np.allclose(frame.observed_price, linked, rtol=0, atol=1e-10):
        raise ValueError("Prediction targets do not match the preserved prepared data.")
    expected_rows = set(data.row_id)
    expected_cells = len(DESIGNS) * len(REPEATS) * (len(INTERVAL_METHODS) if interval else 1)
    groups = list(frame.groupby(keys, sort=False))
    if len(groups) != expected_cells:
        raise ValueError("A design/repetition prediction cell is missing.")
    for key, part in groups:
        if part.row_id.duplicated().any() or set(part.row_id) != expected_rows:
            raise ValueError(f"Incomplete or duplicate OOF row coverage in {key}.")
        if set(part.fold) != {1, 2, 3}:
            raise ValueError("Each prediction slice must contain all three outer folds.")
        for fold, cell in part.groupby("fold"):
            if set(cell.row_id) != splits[(key[0], int(key[1]), int(fold))]:
                raise ValueError("Prediction rows disagree with saved external test folds.")


def validate_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    columns = ["observed_price", "prediction", "lower", "upper", "width"]
    if not np.isfinite(result[columns].to_numpy()).all() or (result.upper < result.lower).any():
        raise ValueError("Nonfinite or reversed interval endpoints.")
    covered = (result.observed_price >= result.lower) & (result.observed_price <= result.upper)
    raw = result.covered.astype(str).str.lower()
    if not raw.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Unrecognized saved coverage flag.")
    if not np.array_equal(covered, raw.isin(["true", "1"])):
        raise ValueError("Saved coverage disagrees with interval endpoints.")
    widths = result.upper - result.lower
    if not np.allclose(widths, result.width, rtol=1e-10, atol=1e-8):
        raise ValueError("Saved interval widths disagree with endpoints.")
    result["covered"], result["width"] = covered, widths
    return result


def load_inputs(root: Path, results: Path, data_path: Path) -> dict:
    data = pd.read_csv(data_path)
    required = ["row_id", "host_group", "neighbourhood", "price", "host_portfolio_size_snapshot"]
    if any(field not in data for field in required) or data[required].isna().any().any():
        raise ValueError("Prepared snapshot lacks required grouping/target fields.")
    if not data.row_id.is_unique:
        raise ValueError("Prepared snapshot contains duplicate listing IDs.")
    counts = data.groupby("host_group").row_id.transform("size")
    if not np.array_equal(counts.to_numpy(), data.host_portfolio_size_snapshot.to_numpy()):
        raise ValueError("Saved host portfolio size differs from observed snapshot rows.")
    data["host_portfolio_group"] = np.where(counts == 1, "one_observed_listing", "multiple_observed_listings")
    config = json.loads((results / "frozen_config.json").read_text(encoding="utf-8"))
    if (set(config["designs"]) != set(DESIGNS) or config["repeats"] != 2 or
            config["outer_folds"] != 3 or config.get("main_spec") != "SLRAT"):
        raise ValueError("This extension requires the original three-design, two-repeat SLRAT analysis.")
    if config.get("input_mode") != "prepared_csv" or config["hashes"]["input_csv"] != sha256(data_path):
        raise ValueError("Prepared snapshot hash differs from the frozen fitting input.")
    names = {f"{d}_r{r}_f{f}" for d in DESIGNS for r in REPEATS for f in (1, 2, 3)}
    actual = {p.parent.name for p in (results / "folds").glob("*/complete.json")}
    if names != actual:
        raise ValueError("Exactly 18 completed outer folds are required.")
    splits = {}
    ids = set(data.row_id)
    indexed = data.set_index("row_id")
    for design in DESIGNS:
        for repeat in REPEATS:
            used = set()
            for fold in (1, 2, 3):
                name = f"{design}_r{repeat}_f{fold}"
                checkpoint = json.loads((results / "folds" / name / "complete.json").read_text())
                if (checkpoint.get("complete") is not True or checkpoint["fingerprint"] != config["fingerprint"]
                        or (checkpoint["design"], checkpoint["repeat"], checkpoint["fold"]) != (design, repeat, fold)):
                    raise ValueError(f"Mismatched completed checkpoint: {name}")
                saved = json.loads((results / "folds" / name / "splits.json").read_text())["outer"][0]
                train, test = set(saved["train_row_id"]), set(saved["test_row_id"])
                if (len(train) != len(saved["train_row_id"]) or len(test) != len(saved["test_row_id"])
                        or train & test or train | test != ids or used & test):
                    raise ValueError(f"Invalid external partition: {name}")
                field = {"host": "host_group", "neighbourhood": "neighbourhood"}.get(design)
                if field and set(indexed.loc[list(train), field]) & set(indexed.loc[list(test), field]):
                    raise ValueError(f"Saved external groups overlap: {name}")
                splits[(design, repeat, fold)] = test
                used |= test
            if used != ids:
                raise ValueError("External folds do not partition the full observed snapshot.")
    point = pd.read_csv(results / "oof_predictions.csv")
    point = point.loc[point.method.eq("selected_full")].copy()
    if set(point.spec) != {"SLRAT"} or not set(point.family).issubset(set(config["core_families"])):
        raise ValueError("Selected-full predictions differ from the fixed core SLRAT procedure.")
    intervals = pd.read_csv(results / "interval_oof_predictions.csv")
    if set(intervals.method) != {"interval_point"} or set(intervals.spec) != {"SLRAT"}:
        raise ValueError("Unexpected preserved interval predictor.")
    validate_prediction_slices(point, data, splits)
    validate_prediction_slices(intervals, data, splits, interval=True)
    intervals = validate_intervals(intervals)
    meta = data[["row_id", "host_group", "neighbourhood", "host_portfolio_group"]]
    point = point.merge(meta, on="row_id", how="left", validate="many_to_one")
    intervals = intervals.merge(meta, on="row_id", how="left", validate="many_to_one")
    return {"data": data, "point": point, "intervals": intervals, "config": config,
            "data_path": data_path, "results": results, "root": root}


def listing_weights(frame: pd.DataFrame, weighting: str) -> np.ndarray:
    if weighting == "listing":
        return np.ones(len(frame), dtype=float)
    if weighting == "equal_host":
        return 1.0 / frame.groupby("host_group").host_group.transform("size").to_numpy(dtype=float)
    raise ValueError(f"Unknown weighting: {weighting}")


def point_metrics(frame: pd.DataFrame, weighting: str) -> dict:
    weights = listing_weights(frame, weighting)
    observed = frame.observed_price.to_numpy(dtype=float)
    predicted = frame.prediction.to_numpy(dtype=float)
    error = predicted - observed
    mean_y = float(np.average(observed, weights=weights))
    denominator = float(np.sum(weights * (observed - mean_y) ** 2))
    squared_error = float(np.sum(weights * error ** 2))
    return {"n_listings": int(len(frame)), "n_hosts": int(frame.host_group.nunique()),
            "signed_bias": float(np.average(error, weights=weights)),
            "MAE": float(np.average(np.abs(error), weights=weights)),
            "RMSE": float(np.sqrt(np.average(error ** 2, weights=weights))),
            "R2": float(1 - squared_error / denominator) if denominator > 0 else np.nan,
            "mean_observed_price": mean_y,
            "mean_prediction": float(np.average(predicted, weights=weights))}


def weighted_host_median(values: np.ndarray, host_listing_counts: np.ndarray) -> float:
    """Lower inverse-CDF median with exact rational1/n_h host weights.

    Floating cumulative sums can miss an exact50% boundary and select the next
    value. The integer within-group host counts identify the exact estimand.
    """
    order = np.argsort(values, kind="stable")
    weights = [Fraction(1, int(n)) for n in host_listing_counts]
    threshold = sum(weights, Fraction(0)) / 2
    cumulative = Fraction(0)
    for index in order:
        cumulative += weights[index]
        if cumulative >= threshold:
            return float(values[index])
    raise ValueError("Empty or invalid host-weighted median input.")


def interval_metrics(frame: pd.DataFrame, weighting: str) -> dict:
    base = point_metrics(frame, weighting)
    weights = listing_weights(frame, weighting)
    widths = frame.width.to_numpy(dtype=float)
    counts = frame.groupby("host_group").host_group.transform("size").to_numpy(dtype=int)
    return {"n_listings": base.pop("n_listings"), "n_hosts": base.pop("n_hosts"),
            "nominal_coverage": .8,
            "coverage": float(np.average(frame.covered.to_numpy(dtype=float), weights=weights)),
            "mean_width": float(np.average(widths, weights=weights)),
            "median_width": float(np.median(widths)) if weighting == "listing" else weighted_host_median(widths, counts),
            **{f"interval_point_{key}": value for key, value in base.items()}}


def aggregate(frame: pd.DataFrame, context: str, interval: bool = False) -> pd.DataFrame:
    keys = ["design", "repeat"] + (["interval_method"] if interval else [])
    if context != "overall":
        keys.append(context)
    records = []
    for key, part in frame.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, key))
        if context == "neighbourhood":
            base["interpretable_n_ge_30"] = bool(len(part) >= MIN_NEIGHBOURHOOD_LISTINGS)
        for weighting in WEIGHTINGS:
            calculator = interval_metrics if interval else point_metrics
            records.append({**base, "weighting": weighting, **calculator(part, weighting)})
    return pd.DataFrame(records)


def descriptive_summary(frame: pd.DataFrame, context: str, interval: bool = False) -> pd.DataFrame:
    keys = ["design", "weighting"] + (["interval_method"] if interval else [])
    if context != "overall":
        keys.append(context)
    invariants = ["n_listings", "n_hosts"] + (["nominal_coverage"] if interval else [])
    if context == "neighbourhood":
        invariants.append("interpretable_n_ge_30")
    metrics = INTERVAL_METRICS if interval else POINT_METRICS
    rows = []
    for key, part in frame.groupby(keys, sort=True, dropna=False):
        if set(part["repeat"]) != set(REPEATS) or len(part) != len(REPEATS):
            raise ValueError("Descriptive summaries require exactly the two saved repetitions.")
        if any(part[field].nunique(dropna=False) != 1 for field in invariants):
            raise ValueError("Group sizes or fixed definitions changed across repetitions.")
        row = {**dict(zip(keys, key)), "n_repeats": 2,
               **{field: part[field].iloc[0] for field in invariants}}
        for field in metrics:
            values = part[field]
            row.update({f"{field}_mean": float(values.mean()),
                        f"{field}_min": float(values.min()), f"{field}_max": float(values.max())})
        rows.append(row)
    return pd.DataFrame(rows)


def create_figures(tables: dict, output: Path) -> list[Path]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    paths = []
    point = tables["overall_point_by_repeat"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"listing": "#2166ac", "equal_host": "#b2182b"}
    for ax, metric, title in zip(axes, ["MAE", "signed_bias"], ["Mean absolute error", "Signed bias (prediction − observed)"]):
        for index, design in enumerate(DESIGNS):
            for weighting, offset in [("listing", -.1), ("equal_host", .1)]:
                part = point.loc[point.design.eq(design) & point.weighting.eq(weighting)].sort_values("repeat")
                x = index + offset + np.array([-.02, .02])
                ax.scatter(x, part[metric], color=colors[weighting], s=38, alpha=.85,
                           marker="o" if weighting == "listing" else "s",
                           label=("Listing weights" if weighting == "listing" else "Equal host weights") if index == 0 else None)
        ax.set_xticks(range(3), [DESIGN_LABELS[d].replace("Held-out ", "Held-out\n") for d in DESIGNS])
        ax.set_title(title)
        ax.set_ylabel("PEN per advertised night")
        ax.grid(axis="y", alpha=.2)
        if metric == "signed_bias":
            ax.axhline(0, color=".5", linewidth=.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Selected-core OOF predictions: listing and equal-host weighting\nTwo descriptive repetition points per weighting; no confidence intervals", fontsize=11)
    for suffix in ["png", "pdf"]:
        path = figures / f"figure_S_secondary_host_weighting.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    point = tables["neighbourhood_point_by_repeat"]
    intervals = tables["neighbourhood_interval_by_repeat"]
    eligible = sorted(point.loc[point.interpretable_n_ge_30, "neighbourhood"].unique())
    if not eligible:
        return paths
    # Portrait layout keeps12pt type above8pt when printed at7.2in height.
    # A design occupies each row: bias at left and interval coverage at right.
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(3, 2, figsize=(7.3, 10.3), sharey=True)
    fig.subplots_adjust(left=.285, right=.985, bottom=.06, top=.89,
                        wspace=.24, hspace=.44)
    y = np.arange(len(eligible))
    for row, design in enumerate(DESIGNS):
        ax = axes[row, 0]
        part = point.loc[point.design.eq(design) & point.weighting.eq("listing")]
        for repeat, marker, offset in [(1, "o", -.09), (2, "s", .09)]:
            rows = part.loc[part["repeat"].eq(repeat)].set_index("neighbourhood").loc[eligible]
            ax.scatter(rows.signed_bias, y + offset, marker=marker, s=25, color="#2166ac")
        ax.axvline(0, color=".5", linewidth=.8)
        ax.set_title(f"{chr(65 + row * 2)}  {DESIGN_LABELS[design]}", fontsize=12.5, loc="left", pad=8)
        ax.set_xlabel("Signed bias (PEN)", fontsize=12)
        ax.set_yticks(y, eligible, fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(axis="x", alpha=.2)
        ax = axes[row, 1]
        for method, color, base_offset in [("absolute_residual", "#2166ac", -.16),
                                            ("relative_residual", "#b2182b", .16)]:
            part = intervals.loc[intervals.design.eq(design) & intervals.weighting.eq("listing") & intervals.interval_method.eq(method)]
            for repeat, marker, offset in [(1, "o", -.04), (2, "s", .04)]:
                rows = part.loc[part["repeat"].eq(repeat)].set_index("neighbourhood").loc[eligible]
                ax.scatter(rows.coverage * 100, y + base_offset + offset, marker=marker, color=color, s=23)
        ax.axvline(80, color=".5", linewidth=.8, linestyle="--")
        ax.set_title(f"{chr(66 + row * 2)}  Interval coverage", fontsize=12.5, loc="left", pad=8)
        ax.set_xlabel("Coverage (%)", fontsize=12)
        ax.set_xlim(0, 103)
        ax.set_xticks([0, 40, 80, 100])
        ax.tick_params(axis="both", labelsize=12, labelleft=False)
        ax.grid(axis="x", alpha=.2)
    axes[0, 0].set_ylim(len(eligible) - .5, -.5)
    handles = [Line2D([], [], marker="o", linestyle="none", color=".25", label="Repetition 1"),
               Line2D([], [], marker="s", linestyle="none", color=".25", label="Repetition 2"),
               Line2D([], [], color="#2166ac", linewidth=2, label="Absolute"),
               Line2D([], [], color="#b2182b", linewidth=2, label="Relative")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .963), ncol=4,
               frameon=False, fontsize=12, handlelength=1.0, columnspacing=1.0,
               handletextpad=.4)
    fig.suptitle("Neighbourhood diagnostics (at least 30 observed listings)", fontsize=13, y=.99)
    for suffix in ["png", "pdf"]:
        path = figures / f"figure_S_secondary_neighbourhood_diagnostics.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def integration_notes(bundle: dict, tables: dict) -> dict:
    data = bundle["data"]
    counts = data.groupby("host_portfolio_group").agg(n_listings=("row_id", "size"), n_hosts=("host_group", "nunique"))
    overall = tables["overall_point_descriptive_summary"]
    group = tables["host_portfolio_point_descriptive_summary"]
    neighbourhood = tables["neighbourhood_point_descriptive_summary"]
    interval = tables["neighbourhood_interval_descriptive_summary"]
    eligible = neighbourhood.loc[neighbourhood.interpretable_n_ge_30]
    listings_eligible = data.neighbourhood.value_counts().loc[lambda x: x >= MIN_NEIGHBOURHOOD_LISTINGS]
    findings = []
    for design in DESIGNS:
        records = overall.loc[overall.design.eq(design)].set_index("weighting")
        listing, host = records.loc["listing"], records.loc["equal_host"]
        findings.append({"design": design,
                         "MAE_listing_mean_PEN": float(listing.MAE_mean),
                         "MAE_equal_host_mean_PEN": float(host.MAE_mean),
                         "equal_host_minus_listing_MAE_PEN": float(host.MAE_mean - listing.MAE_mean),
                         "signed_bias_listing_mean_PEN": float(listing.signed_bias_mean),
                         "signed_bias_equal_host_mean_PEN": float(host.signed_bias_mean),
                         "MAE_listing_repeat_range_PEN": [float(listing.MAE_min), float(listing.MAE_max)],
                         "MAE_equal_host_repeat_range_PEN": [float(host.MAE_min), float(host.MAE_max)]})
    eligible_ranges = []
    for design in DESIGNS:
        part = eligible.loc[eligible.design.eq(design) & eligible.weighting.eq("listing")]
        row = {"design": design,
               "signed_bias_range_across_neighbourhood_two_repeat_means_PEN": [float(part.signed_bias_mean.min()), float(part.signed_bias_mean.max())],
               "MAE_range_across_neighbourhood_two_repeat_means_PEN": [float(part.MAE_mean.min()), float(part.MAE_mean.max())]}
        for method in INTERVAL_METHODS:
            cell = interval.loc[interval.design.eq(design) & interval.weighting.eq("listing") & interval.interval_method.eq(method) & interval.interpretable_n_ge_30]
            row[method] = {"coverage_range_across_neighbourhood_two_repeat_means": [float(cell.coverage_mean.min()), float(cell.coverage_mean.max())],
                           "mean_width_range_across_neighbourhood_two_repeat_means_PEN": [float(cell.mean_width_mean.min()), float(cell.mean_width_mean.max())]}
        eligible_ranges.append(row)
    return {"idioma": "es", "tipo_analisis": "Extensión secundaria exploratoria y descriptiva; sin reajuste de modelos.",
            "units": {"code": "PEN", "price_error_width": "Peruvian soles (PEN) per advertised night", "coverage": "proportion", "R2": "dimensionless"},
            "prespecified_limitations": ["Extensión secundaria posterior al análisis principal; umbral de30 fijado antes de calcular los diagnósticos secundarios.",
                                         "Dos repeticiones de particiones de la misma muestra, sin independencia ni intervalos de confianza.",
                                         "Conteos de anuncios por anfitrión observados en el archivo, no en toda la plataforma.",
                                         "Todos los barrios se conservan; solo n>=30 se incluye en rangos narrativos o figuras.",
                                         "Sin significación estadística, inferencias causales o interpretación de daños socioeconómicos."],
            "unidad": "Soles peruanos (PEN) por noche anunciada; sin conversión monetaria.",
            "muestra": {"anuncios": int(len(data)), "anfitriones_observados": int(data.host_group.nunique()),
                        "barrios": int(data.neighbourhood.nunique()),
                        "barrios_n_ge_30": int(len(listings_eligible)),
                        "anuncios_en_barrios_n_ge_30": int(listings_eligible.sum()),
                        "grupos_anfitrion": counts.reset_index().to_dict(orient="records")},
            "definiciones": {"error_con_signo": "predicción menos precio observado; valores positivos indican sobrepredicción",
                             "MAE_equal_host": "media entre anfitriones de la media del error absoluto de sus anuncios dentro del grupo analizado",
                             "portfolio": "uno frente a dos o más anuncios observados en este archivo; no es el portafolio total de la plataforma",
                             "regla_barrios": "Todos los barrios se conservan en CSV. Se describen e ilustran solo los de n>=30 anuncios, umbral fijado antes de calcular estos diagnósticos.",
                             "repeticiones": "Las dos repeticiones reutilizan la misma muestra. Promedios y rangos son descriptivos, no intervalos de confianza.",
                             "intervalos": "Cobertura y anchura de los dos procedimientos80% guardados; usan las predicciones de su propio ajuste proper-training/calibración, distintas del selected_full refit."},
            "comparacion_ponderacion": findings,
            "host_portfolio_listing_weight_summary": group.loc[group.weighting.eq("listing")].to_dict(orient="records"),
            "barrios_interpretables_rangos": eligible_ranges,
            "alcance": "Los diagnósticos describen desempeño predictivo y cobertura en la muestra. No prueban desigualdad socioeconómica, daño, desplazamiento, asequibilidad, causalidad o profesionalización. No contienen pruebas de significación ni rankings de barrios."}


def run(root: Path, results: Path, data_path: Path, output: Path, baseline: Path,
        make_figures: bool = True) -> dict:
    root, results, data_path, output, baseline = [p.resolve() for p in [root, results, data_path, output, baseline]]
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Secondary outputs must stay in a dedicated directory within this project.")
    reserved = {"data", "results", "publication", "code", "tests", "editorial", "submission"}
    if output.relative_to(root).parts[0] in reserved:
        raise ValueError("Secondary outputs cannot overwrite primary inputs, results or documents.")
    before = verify_primary(root, baseline)
    bundle = load_inputs(root, results, data_path)
    currency = read_currency_metadata(data_path)
    if currency.get("status") != "author_confirmed" or currency.get("code") != "PEN":
        raise ValueError("Explicit author-confirmed PEN metadata is required; currency is never inferred.")
    tables = {}
    for context, stem in [("overall", "overall"), ("host_portfolio_group", "host_portfolio"), ("neighbourhood", "neighbourhood")]:
        for interval, label, frame in [(False, "point", bundle["point"]), (True, "interval", bundle["intervals"])]:
            table = aggregate(frame, context, interval)
            tables[f"{stem}_{label}_by_repeat"] = table
            tables[f"{stem}_{label}_descriptive_summary"] = descriptive_summary(table, context, interval)
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = table_dir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.12g")
        written.append(path)
    figure_paths = create_figures(tables, output) if make_figures else []
    notes = integration_notes(bundle, tables)
    notes_path = output / "integration_notes_es.json"
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary_path = output / "secondary_summary.json"
    summary_path.write_text(notes_path.read_text(encoding="utf-8"), encoding="utf-8")
    after = verify_primary(root, baseline)
    if before != after:
        raise AssertionError("Protected primary baseline verification changed during execution.")
    code_path = Path(__file__).resolve()
    protocol = code_path.with_name("SECONDARY_ANALYSIS_PROTOCOL.md")
    inputs = {"data": data_path, "point_predictions": results / "oof_predictions.csv",
              "interval_predictions": results / "interval_oof_predictions.csv",
              "frozen_config": results / "frozen_config.json", "secondary_protocol": protocol,
              "secondary_code": code_path, "primary_baseline": baseline,
              "currency_metadata": Path(currency["metadata_file"])}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "analysis": "secondary_descriptive_no_refit", "point_method": "selected_full",
                "interval_point_method": "interval_point", "interval_methods": list(INTERVAL_METHODS),
                "designs": list(DESIGNS), "repeats": list(REPEATS), "complete_outer_folds": 18,
                "currency": currency, "neighbourhood_interpretability_minimum_listings": MIN_NEIGHBOURHOOD_LISTINGS,
                "primary_integrity_before": before, "primary_integrity_after": after,
                "weighted_estimands": {"listing": "Every observed listing has equal weight.",
                                       "equal_host": "Every host within the reported group has equal total weight; each listing weight is1/n_h in that group.",
                                       "RMSE": "Square root of weighted mean squared error, not mean host RMSE.",
                                       "R2": "One minus weighted SSE / weighted target SST; undefined if SST=0.",
                                       "signed_bias": "prediction minus observed price"},
                "limits": ["Secondary extension after primary analysis; thresholds fixed before secondary calculation.",
                           "Two repeated partitions of one snapshot; descriptive ranges are not confidence intervals.",
                           "Observed snapshot portfolios; no platform-wide portfolio, professionalism or causal inference.",
                           "All neighbourhoods exported; n>=30 rule only controls interpretive summaries and figures.",
                           "Interval-specific point fits are distinct from the main refitted selected-full predictor.",
                           "No refitting, recalibration, hypothesis tests, direct host identifiers, listing text or coordinates exported."],
                "input_sha256": {name: {"path": os.path.relpath(path, root), "sha256": sha256(path)} for name, path in inputs.items()},
                "outputs": {str(path.relative_to(output)): {"sha256": sha256(path), "rows": int(len(tables[path.stem]))}
                            for path in written},
                "figures": {str(path.relative_to(output)): sha256(path) for path in figure_paths},
                "notes_sha256": sha256(notes_path), "secondary_summary_sha256": sha256(summary_path),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}}
    (output / "secondary_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "complete", "tables": len(tables), "figure_files": len(figure_paths),
            "protected_primary_files_unchanged": after["files_verified"], "sample": notes["muestra"],
            "weighting_findings": notes["comparacion_ponderacion"], "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(run(root, args.results or root / "results",
                         args.data or root / "data/model_ready_listings_restricted.csv",
                         args.output or root / "publication_secondary",
                         args.baseline or root / "baseline_primary_sha256.json",
                         not args.no_figures), ensure_ascii=False, indent=2))
