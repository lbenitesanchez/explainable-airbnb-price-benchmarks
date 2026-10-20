#!/usr/bin/env python3
"""Article tables/figures from complete shared-fold out-of-fold predictions.

No models are fitted. Requires all 18 folds, two repeats, three designs and seven
benchmark families. Recomputes pooled R2 from predictions, never averages fold R2.
Two-repeat ranges are descriptive, not confidence intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from apply_currency_metadata import read_currency_metadata,attach_publication_currency


DESIGNS = ("random", "host", "neighbourhood")
DESIGN_LABELS = {"random": "Random listings", "host": "Held-out hosts", "neighbourhood": "Held-out neighbourhoods"}
FAMILIES = ("Ridge", "ElasticNet", "DecisionTree", "Bagging", "RandomForest", "GradientBoosting", "XGBoost")
CORE = ("Ridge", "ElasticNet", "GradientBoosting", "RandomForest")
MODEL_LABELS = {"training_median": "Training median", "Ridge_full": "Ridge",
    "ElasticNet_full": "Elastic net", "DecisionTree_full": "Decision tree", "Bagging_full": "Bagging",
    "RandomForest_full": "Random forest", "GradientBoosting_full": "Gradient boosting",
    "XGBoost_full": "XGBoost", "selected_full": "Selected core"}
BENCHMARK_METHODS = ("training_median", *(f"{name}_full" for name in FAMILIES), "selected_full")
BLOCK_LABELS = {"S": "Structure", "L": "Location", "R": "Reviews", "A": "Amenities", "T": "Text", "H": "Snapshot host proxy"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_data(path: str | Path) -> Path:
    path = Path(path)
    return path / "model_ready_listings_restricted.csv" if path.is_dir() else path


def metrics(y, pred) -> dict:
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)), "n": int(len(y))}


def require_complete(results: str | Path, data: str | Path) -> dict:
    """Reject missing folds, mismatched snapshots, and partial OOF coverage."""
    results, data_path = Path(results), resolve_data(data)
    config = json.loads((results / "frozen_config.json").read_text())
    if set(config["designs"]) != set(DESIGNS) or config["repeats"] != 2 or config["outer_folds"] != 3:
        raise ValueError("Article summaries require three designs, two repeats and three outer folds (18 complete folds).")
    if set(config["families"]) != set(FAMILIES):
        raise ValueError("Article benchmarking requires all seven predeclared families, including XGBoost.")
    input_hash=config.get("hashes",{}).get("input_csv")
    if config.get("input_mode")=="prepared_csv" and input_hash!=sha256(data_path):
        raise ValueError("Prepared CSV hash does not match the frozen model-fitting input.")
    if config.get("input_mode")=="original_with_preparation":
        preparation=json.loads(data_path.with_name("preparation_manifest.json").read_text())
        if preparation["source_sha256"]!=input_hash or preparation["output_sha256"].get(data_path.name)!=sha256(data_path):
            raise ValueError("Prepared data manifest does not match frozen original input and export.")
    prepared = pd.read_csv(data_path)
    if not prepared.row_id.is_unique:
        raise ValueError("Prepared data has duplicate row_id values.")
    all_rows = set(prepared.row_id)
    indexed=prepared.set_index("row_id")
    checkpoints, split_sets = [], {}
    expected_folders = {f"{design}_r{repeat}_f{fold}" for design in DESIGNS for repeat in (1,2) for fold in (1,2,3)}
    actual_folders = {p.parent.name for p in (results / "folds").glob("*/complete.json")}
    if actual_folders != expected_folders:
        raise ValueError(f"Expected 18 completed folds. Missing: {sorted(expected_folders-actual_folders)}; unexpected: {sorted(actual_folders-expected_folders)}")
    for name in sorted(expected_folders):
        folder = results / "folds" / name
        checkpoint = json.loads((folder / "complete.json").read_text())
        if not checkpoint.get("complete") or checkpoint["fingerprint"] != config["fingerprint"]:
            raise ValueError(f"Incomplete or mismatched checkpoint: {name}")
        if checkpoint["selected_family_inner"] not in CORE:
            raise ValueError(f"Selected family outside fixed core: {name}")
        outer = json.loads((folder / "splits.json").read_text())["outer"][0]
        train, test = set(outer["train_row_id"]), set(outer["test_row_id"])
        if train & test or train | test != all_rows or len(train)!=len(outer["train_row_id"]) or len(test)!=len(outer["test_row_id"]):
            raise ValueError(f"Outer row partition does not match prepared data: {name}")
        grouping={"host":"host_group","neighbourhood":"neighbourhood"}.get(checkpoint["design"])
        if grouping and set(indexed.loc[list(train),grouping]) & set(indexed.loc[list(test),grouping]):
            raise ValueError(f"Train/test groups overlap in saved outer partition: {name}")
        split_sets[(checkpoint["design"], checkpoint["repeat"], checkpoint["fold"])] = test
        checkpoints.append(checkpoint)
    predictions = pd.read_csv(results / "oof_predictions.csv")
    if set(predictions.design)!=set(DESIGNS) or set(predictions["repeat"])!={1,2} or set(predictions.fold)!={1,2,3}:
        raise ValueError("Unexpected design, repetition or fold labels in OOF predictions.")
    expected_methods = set(BENCHMARK_METHODS) | {f"without_{b}" for b in "SLRAT"}
    if config.get("host_sensitivity"):
        expected_methods.add("snapshot_H_sensitivity")
    for design in DESIGNS:
        for repeat in (1,2):
            portion = predictions.loc[(predictions.design == design) & (predictions["repeat"] == repeat)]
            if set(portion.method) != expected_methods:
                raise ValueError(f"Missing or unexpected prediction methods: {design}, repeat {repeat}")
            for method, frame in portion.groupby("method"):
                if frame.row_id.duplicated().any() or set(frame.row_id) != all_rows:
                    raise ValueError(f"Incomplete OOF row coverage: {design}/{repeat}/{method}")
                for fold, part in frame.groupby("fold"):
                    if set(part.row_id) != split_sets[(design,repeat,int(fold))]:
                        raise ValueError("Methods do not share the saved external test rows.")
    observed = prepared.set_index("row_id").price
    if not np.isfinite(predictions[["observed_price", "prediction"]].to_numpy()).all():
        raise ValueError("Nonfinite point predictions or observed prices.")
    if not np.allclose(predictions.observed_price, predictions.row_id.map(observed), rtol=0, atol=1e-10):
        raise ValueError("Observed prices do not match prepared snapshot.")
    return {"results": results, "data_path": data_path, "config": config,
            "data": prepared, "predictions": predictions, "checkpoints": checkpoints,
            "test_sets": split_sets}


def descriptive(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, sort=False)[values].agg(["mean","min","max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    output = grouped.reset_index()
    count = frame.groupby(keys, sort=False).size().reset_index(name="n_repeats")
    return output.merge(count, on=keys, validate="one_to_one")


def pooled_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"design": design, "repeat": repeat, "method": method,
                          **metrics(part.observed_price, part.prediction)}
                         for (design,repeat,method), part in frame.groupby(["design","repeat","method"])])


def interval_tables(bundle: dict) -> tuple[pd.DataFrame,pd.DataFrame]:
    path = bundle["results"] / "interval_oof_predictions.csv"
    if not bundle["config"].get("intervals") or not path.exists():
        raise ValueError("Both predeclared interval methods must be available for article summaries.")
    frame = pd.read_csv(path)
    expected_methods = {"absolute_residual", "relative_residual"}
    if set(frame.interval_method) != expected_methods:
        raise ValueError("Unexpected or missing interval calibration methods.")
    if not np.isfinite(frame[["lower","upper","prediction","observed_price"]].to_numpy()).all():
        raise ValueError("Nonfinite interval values.")
    if (frame.upper < frame.lower).any():
        raise ValueError("Interval endpoint order violated.")
    recomputed = ((frame.observed_price >= frame.lower) & (frame.observed_price <= frame.upper))
    saved_covered = frame.covered.astype(str).str.lower().isin(["true","1"])
    if not np.array_equal(recomputed.to_numpy(),saved_covered.to_numpy()):
        raise ValueError("Stored coverage flags disagree with interval endpoints.")
    frame = frame.assign(covered=recomputed, width=frame.upper-frame.lower)
    overall, segmented = [], []
    for (design,repeat,method), part in frame.groupby(["design","repeat","interval_method"]):
        if part.row_id.duplicated().any() or set(part.row_id) != set(bundle["data"].row_id):
            raise ValueError("Incomplete interval OOF row coverage.")
        base = {"design":design,"repeat":repeat,"interval_method":method,"nominal_coverage":.8}
        overall.append({**base,"coverage":float(part.covered.mean()),"mean_width":float(part.width.mean()),
                        "median_width":float(part.width.median()),**metrics(part.observed_price,part.prediction)})
        for field in ["occupancy_type","review_band","predicted_price_band","observed_price_band_diagnostic"]:
            if field not in part:
                continue
            for group, cell in part.groupby(field,dropna=False):
                segmented.append({**base,"grouping":field,"group":group,
                    "coverage":float(cell.covered.mean()),"mean_width":float(cell.width.mean()),
                    "median_width":float(cell.width.median()),**metrics(cell.observed_price,cell.prediction)})
    if len(overall) != 12:
        raise ValueError("Expected 3 designs x 2 repeats x 2 interval methods.")
    return pd.DataFrame(overall),pd.DataFrame(segmented)


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"),dpi=300,bbox_inches="tight",facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def plot_benchmark(frame: pd.DataFrame, output: Path, currency: dict | None = None) -> None:
    fig, axes = plt.subplots(1,3,figsize=(12.2,6.7),sharey=True,sharex=True)
    colors, markers = {1:"#174a72",2:"#b45622"},{1:"o",2:"s"}
    order = list(reversed(BENCHMARK_METHODS))
    for ax,design in zip(axes,DESIGNS):
        part = frame[frame.design==design]
        for i,method in enumerate(order):
            vals = part[part.method==method].set_index("repeat").MAE
            ax.plot([vals.loc[1],vals.loc[2]],[i-.08,i+.08],color="#c6c6c6",lw=1,zorder=1)
            for repeat,offset in [(1,-.08),(2,.08)]:
                ax.scatter(vals.loc[repeat],i+offset,s=32,color=colors[repeat],marker=markers[repeat],
                           label=f"Repeat {repeat}" if i==0 else None,zorder=2)
        unit=(currency or {}).get("code") or "recorded-price units"
        ax.set(title=DESIGN_LABELS[design],xlabel=f"MAE ({unit})")
        ax.grid(axis="x",color="#e4e4e4",lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_xlim(0,1.07*float(frame.MAE.max()))
    axes[0].set_yticks(range(len(order)),[MODEL_LABELS[m] for m in order])
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.955),ncol=2,frameon=False)
    fig.suptitle("Out-of-fold price errors across evaluation designs",fontsize=16,y=.99)
    fig.text(.5,.015,"Each point pools all held-out predictions in one repeat. Two repeats; no confidence intervals.",ha="center",fontsize=10.5)
    fig.tight_layout(rect=(0,.05,1,.91))
    save_figure(fig,output/"figure_01_benchmark_mae")


def plot_ablations(frame: pd.DataFrame, output: Path, currency: dict | None = None) -> None:
    fig,axes=plt.subplots(1,3,figsize=(12.2,4.9),sharex=True,sharey=True)
    order=list(reversed("SLRAT"))
    for ax,design in zip(axes,DESIGNS):
        for i,block in enumerate(order):
            part=frame[(frame.design==design)&(frame.method==f"without_{block}")].set_index("repeat")
            vals=part.MAE_variant_minus_full
            ax.plot([vals.loc[1],vals.loc[2]],[i-.08,i+.08],color="#c6c6c6",lw=1)
            for repeat,offset,color,marker in [(1,-.08,"#174a72","o"),(2,.08,"#b45622","s")]:
                ax.scatter(vals.loc[repeat],i+offset,s=32,color=color,marker=marker,
                           label=f"Repeat {repeat}" if i==0 else None)
        ax.axvline(0,color="#555555",lw=.9)
        code=(currency or {}).get("code")
        xlabel=f"MAE change on removal ({code})" if code else "MAE without block - MAE full"
        ax.set(title=DESIGN_LABELS[design],xlabel=xlabel)
        ax.grid(axis="x",color="#e4e4e4",lw=.6)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_yticks(range(len(order)),[BLOCK_LABELS[b] for b in order])
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.suptitle("Paired predictive contribution of signal blocks",fontsize=16,y=.99)
    fig.text(.5,.015,"Positive values: omission increases error. Family chosen inside training; no causal interpretation or confidence intervals.",ha="center",fontsize=10.5)
    fig.tight_layout(rect=(0,.07,1,.88))
    save_figure(fig,output/"figure_02_paired_ablations")


def summarize(results: str|Path,data: str|Path,output: str|Path,currency_metadata_path: str|Path|None=None) -> dict:
    bundle=require_complete(results,data)
    currency=read_currency_metadata(bundle["data_path"],currency_metadata_path)
    pooled=pooled_predictions(bundle["predictions"])
    archived=pd.read_csv(bundle["results"]/"pooled_metrics_by_repeat.csv")
    check=pooled.merge(archived,on=["design","repeat","method"],suffixes=("","_saved"),validate="one_to_one")
    if len(check)!=len(pooled) or not all(np.allclose(check[m],check[f"{m}_saved"],rtol=1e-9,atol=1e-9) for m in ["MAE","RMSE","R2"]):
        raise ValueError("Pooled metrics do not reproduce the saved run.")
    benchmark=pooled[pooled.method.isin(BENCHMARK_METHODS)].copy()
    benchmark["model_label"]=benchmark.method.map(MODEL_LABELS)
    benchmark_summary=descriptive(benchmark,["design","method","model_label"],["MAE","RMSE","R2"])
    full=pooled[pooled.method=="selected_full"].drop(columns="method")
    variants=pooled[pooled.method.str.startswith("without_")|(pooled.method=="snapshot_H_sensitivity")]
    paired=variants.merge(full,on=["design","repeat"],suffixes=("","_full"),validate="many_to_one")
    for metric in ["MAE","RMSE","R2"]:
        paired[f"{metric}_variant_minus_full"]=paired[metric]-paired[f"{metric}_full"]
        paired[f"{metric}_full_minus_variant"]=-paired[f"{metric}_variant_minus_full"]
    paired["contrast"]=np.where(paired.method=="snapshot_H_sensitivity","Adding snapshot H","Omitting "+paired.method.str.replace("without_","",regex=False))
    paired_summary=descriptive(paired,["design","method","contrast"],["MAE_variant_minus_full","RMSE_variant_minus_full","R2_variant_minus_full"])
    intervals,segments=interval_tables(bundle)
    interval_descriptive=descriptive(intervals,["design","interval_method"],["coverage","mean_width","median_width","MAE","RMSE","R2"])
    selected=pd.DataFrame([{k:c[k] for k in ["design","repeat","fold","selected_family_inner"]} for c in bundle["checkpoints"]])
    selection_counts=selected.groupby(["design","selected_family_inner"]).size().reindex(
        pd.MultiIndex.from_product([DESIGNS,CORE],names=["design","selected_family_inner"]),fill_value=0
    ).reset_index(name="selected_outer_folds")
    selection_counts["outer_folds_per_design"]=6
    selection_counts["selection_fraction"]=selection_counts.selected_outer_folds/6
    output=Path(output);tables=output/"tables";figures=output/"figures"
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    sample=bundle["data"]
    continuous=[]
    for column in ["price","accommodates","bedrooms","beds","cleaned_bathrooms","number_of_reviews"]:
        values=sample[column]
        continuous.append({"variable":column,"n_observed":int(values.notna().sum()),"n_missing":int(values.isna().sum()),
            "mean":values.mean(),"sd_sample":values.std(ddof=1),"min":values.min(),"q1":values.quantile(.25),
            "median":values.median(),"q3":values.quantile(.75),"max":values.max()})
    categories=pd.concat([sample[column].value_counts(dropna=False).rename_axis("category").reset_index(name="n").assign(variable=column)
        for column in ["occupancy_type","property_group_corrected"]],ignore_index=True)
    frames={"table_00_sample_continuous":pd.DataFrame(continuous),"table_00_sample_categories":categories,
        "table_01_benchmark_oof_by_repeat":benchmark,"table_02_benchmark_descriptive":benchmark_summary,
        "table_03_ablations_paired_by_repeat":paired,"table_04_ablations_descriptive":paired_summary,
        "table_05_intervals_by_repeat":intervals,"table_06_intervals_by_segment":segments,
        "table_07_core_family_selection_counts":selection_counts,"table_08_intervals_descriptive":interval_descriptive,
        "table_s01_fold_metrics":pd.read_csv(bundle["results"]/"fold_metrics.csv"),
        "table_s02_selected_families_by_fold":selected}
    for name,frame in frames.items():frame.to_csv(tables/f"{name}.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":14,"pdf.fonttype":42,"ps.fonttype":42})
    plot_benchmark(benchmark,figures,currency);plot_ablations(paired,figures,currency)
    manifest={"complete_outer_folds":18,"n_rows":len(bundle["data"]),"repeats":2,
        "source_results_fingerprint":bundle["config"]["fingerprint"],"prepared_data_sha256":sha256(bundle["data_path"]),
        "aggregation":"MAE/RMSE/R2 recomputed by pooling each repeat's out-of-fold predictions over all observations; not fold averages.",
        "ranges":"Mean/min/max over two overlapping repeats are descriptive. No confidence intervals or tests of independent folds.",
        "ablation_sign":"MAE_variant_minus_full > 0 means omitted block worsens prediction. H contrast adds H instead of omitting it.",
        "interval_groups":"Predicted-price groups use proper-fitting thresholds; observed-price diagnostic groups, when present, use outcomes for descriptive coverage only.",
        "price_unit":"Original posted-price scale; currency unresolved.",
        "files":[str(p.relative_to(output)) for p in sorted(tables.glob("table_*.csv"))]+[str(p.relative_to(output)) for p in sorted(figures.glob("figure_0[12]*"))]}
    manifest=attach_publication_currency(manifest,currency)
    manifest["column_units"]={"MAE":"PEN" if currency.get("code")=="PEN" else "recorded-price units",
        "RMSE":"same as MAE","interval_width":"same as MAE","ablation_MAE_difference":"same as MAE",
        "R2":"dimensionless","coverage":"proportion","attribution_share":"dimensionless proportion"}
    (output/"summary_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def refresh_currency_only(results: str|Path,data: str|Path,output: str|Path,currency_metadata_path: str|Path|None=None) -> dict:
    """Relabel existing figures/sidecars; never rewrite CSVs or recompute SHAP."""
    bundle=require_complete(results,data)
    currency=read_currency_metadata(bundle["data_path"],currency_metadata_path)
    if currency.get("status")!="author_confirmed":
        raise ValueError("A confirmed local currency sidecar is required for this metadata-only refresh.")
    output=Path(output);figures=output/"figures";tables=output/"tables"
    numeric_before={p:sha256(p) for p in output.rglob("*.csv")}
    arrays_before={p:sha256(p) for p in output.rglob("*.npz")}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":14,"pdf.fonttype":42,"ps.fonttype":42})
    plot_benchmark(pd.read_csv(tables/"table_01_benchmark_oof_by_repeat.csv"),figures,currency)
    plot_ablations(pd.read_csv(tables/"table_03_ablations_paired_by_repeat.csv"),figures,currency)
    for name in ["summary_manifest.json","explanation_manifest.json"]:
        path=output/name
        if not path.exists():continue
        manifest=attach_publication_currency(json.loads(path.read_text()),currency)
        manifest["column_units"]={"MAE":"PEN","RMSE":"PEN","interval_width":"PEN",
            "ablation_MAE_difference":"PEN","additive_SHAP_value":"PEN",
            "R2":"dimensionless","coverage":"proportion","attribution_share":"dimensionless proportion"}
        if name=="explanation_manifest.json":
            for fit in manifest.get("fits",[]):
                fit["price_unit"]=currency["price_unit"];fit["shap_value_unit"]="PEN"
        path.write_text(json.dumps(manifest,indent=2)+"\n")
    for path in (output/"explanations").glob("*_metadata.json"):
        manifest=attach_publication_currency(json.loads(path.read_text()),currency)
        manifest["shap_value_unit"]="PEN";manifest["attribution_share_unit"]="dimensionless proportion"
        path.write_text(json.dumps(manifest,indent=2)+"\n")
    assert all(sha256(p)==before for p,before in numeric_before.items())
    assert all(sha256(p)==before for p,before in arrays_before.items())
    return {"code":"PEN","numeric_csvs_verified_unchanged":len(numeric_before),
            "shap_arrays_verified_unchanged":len(arrays_before),"regression_fits":0,"SHAP_recomputed":False,
            "updated":"Two figure labels and unit metadata; percentage SHAP figure unchanged."}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results",type=Path,required=True);cli.add_argument("--data",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--currency-metadata",type=Path)
    cli.add_argument("--refresh-currency-only",action="store_true")
    args=cli.parse_args()
    operation=refresh_currency_only if args.refresh_currency_only else summarize
    print(json.dumps(operation(args.results,args.data,args.output,args.currency_metadata),indent=2))
