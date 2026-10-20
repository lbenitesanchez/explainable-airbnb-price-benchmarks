#!/usr/bin/env python3
"""SHAP diagnostics for exactly three preselected saved outer-fold models.

Explains repeat 1/fold 1 for each design, irrespective of its error. Only held-out
test rows are explained. Linear backgrounds are sampled from outer training.
TreeExplainer uses stored tree training counts, with no external background.
Attributions describe the fitted prediction function, not causal effects or
shares of explained variance. Three fits do not establish attribution stability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge,ElasticNet

# Provides stable paths for the transformers embedded in joblib artifacts.
from features import ListingText,SafeTextSVD  # noqa: F401
from reanalyse import BLOCKS
from summarize_results import DESIGNS,DESIGN_LABELS,BLOCK_LABELS,require_complete,save_figure,sha256
from apply_currency_metadata import read_currency_metadata,attach_publication_currency


def transformed_names(preprocessor) -> tuple[np.ndarray,list[str]]:
    """Use public name APIs in ColumnTransformer output order.

    ListingText joins two strings and has no name API; its downstream SVD does.
    Numeric and categorical names include imputer indicators and one-hot levels.
    """
    names=[]
    numeric_owner={col:block for block,definition in BLOCKS.items() for col in definition["num"]}
    categorical_owner={col:block for block,definition in BLOCKS.items() for col in definition["cat"]}
    owners=[]
    for transformer_name,transformer,columns in preprocessor.transformers_:
        if transformer_name=="remainder" or isinstance(transformer,str):
            continue
        if transformer_name=="text":
            local=transformer.named_steps["semantic"].get_feature_names_out()
            names.extend(f"text__{name}" for name in local)
            owners.extend(["T"]*len(local))
        elif transformer_name=="numeric":
            local=transformer.get_feature_names_out(columns)
            for name in local:
                original=str(name).removeprefix("missingindicator_")
                if original not in numeric_owner:
                    raise ValueError(f"Cannot assign numeric feature to a block: {name}")
                names.append(f"numeric__{name}");owners.append(numeric_owner[original])
        elif transformer_name=="categorical":
            local=transformer.get_feature_names_out(columns)
            for name in local:
                matches=[column for column in categorical_owner if str(name).startswith(f"{column}_")]
                if len(matches)!=1:
                    raise ValueError(f"Cannot uniquely assign categorical feature to a block: {name}")
                names.append(f"categorical__{name}");owners.append(categorical_owner[matches[0]])
        else:
            raise ValueError(f"Unexpected preprocessing branch: {transformer_name}")
    if len(names)!=len(set(names)):
        raise ValueError("Transformed feature names are not unique.")
    return np.asarray(names,dtype=str),owners


def explain_one(bundle:dict,design:str,output:Path,background_n:int=200) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    folder=bundle["results"]/"folds"/f"{design}_r1_f1"
    model_path=folder/"selected_full_model.joblib"
    checkpoint=json.loads((folder/"complete.json").read_text())
    partition=json.loads((folder/"splits.json").read_text())["outer"][0]
    frame=bundle["data"].set_index("row_id",drop=False)
    train=frame.loc[partition["train_row_id"]].copy()
    test=frame.loc[partition["test_row_id"]].copy()
    if set(train.row_id)&set(test.row_id):
        raise ValueError("Training and explained test rows overlap.")
    fitted=joblib.load(model_path)
    preprocessor,regressor=fitted.named_steps["preprocess"],fitted.named_steps["model"]
    names,blocks=transformed_names(preprocessor)
    transformed=np.asarray(preprocessor.transform(test),dtype=float)
    if transformed.shape[1]!=len(names):
        raise ValueError("Transformed features and names differ in width.")
    background_ids=[]
    if isinstance(regressor,(GradientBoostingRegressor,RandomForestRegressor)):
        explainer=shap.TreeExplainer(regressor,feature_perturbation="tree_path_dependent",model_output="raw")
        values=np.asarray(explainer.shap_values(transformed,check_additivity=True),dtype=float)
        explanation_type="TreeExplainer tree_path_dependent; stored training path counts, no external background"
    elif isinstance(regressor,(Ridge,ElasticNet)):
        positions=np.random.RandomState(42).choice(len(train),size=min(background_n,len(train)),replace=False)
        background=train.iloc[positions]
        background_ids=background.row_id.tolist()
        matrix=np.asarray(preprocessor.transform(background),dtype=float)
        masker=shap.maskers.Independent(matrix,max_samples=len(matrix))
        explainer=shap.LinearExplainer(regressor,masker)
        values=np.asarray(explainer.shap_values(transformed),dtype=float)
        explanation_type="LinearExplainer with independent masker; deterministic outer-training-only background"
    else:
        raise TypeError(f"Predeclared core regressor not supported for SHAP: {type(regressor).__name__}")
    if values.shape!=transformed.shape:
        raise ValueError(f"Unexpected SHAP array shape: {values.shape} vs {transformed.shape}")
    expected=float(np.asarray(explainer.expected_value).reshape(-1)[0])
    prediction=np.asarray(fitted.predict(test),dtype=float)
    saved=bundle["predictions"]
    saved=saved[(saved.design==design)&(saved["repeat"]==1)&(saved.fold==1)&(saved.method=="selected_full")].set_index("row_id")
    if set(saved.index)!=set(test.row_id) or not np.allclose(prediction,saved.loc[test.row_id,"prediction"],rtol=1e-8,atol=1e-8):
        raise ValueError("Saved model predictions do not reproduce the selected-core OOF predictions.")
    residual=prediction-(expected+values.sum(axis=1))
    if not np.allclose(prediction,expected+values.sum(axis=1),rtol=1e-5,atol=1e-5):
        raise ValueError(f"SHAP additivity does not reproduce predictions; max error={np.max(np.abs(residual))}")
    abs_mean=np.abs(values).mean(axis=0)
    feature_table=pd.DataFrame({"design":design,"repeat":1,"fold":1,
        "selected_family":checkpoint["selected_family_inner"],"feature":names,"block":blocks,
        "mean_abs_shap":abs_mean,"mean_signed_shap":values.mean(axis=0),"explained_test_n":len(test)})
    total=float(abs_mean.sum())
    if not np.isfinite(total) or total<=0:
        raise ValueError("Attribution shares are undefined when total absolute SHAP mass is zero or nonfinite.")
    block_table=feature_table.groupby(["design","repeat","fold","selected_family","block"],sort=False).agg(
        sum_mean_abs_shap=("mean_abs_shap","sum"),n_transformed_features=("feature","size")).reset_index()
    block_table["attribution_share"]=block_table.sum_mean_abs_shap/total
    if not np.isclose(block_table.attribution_share.sum(),1.0):
        raise ValueError("Block attribution shares do not reconcile to the total.")
    block_table["explained_test_n"]=len(test)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/f"{design}_r1_f1_shap_values_restricted.npz",
        values=values,base_values=np.full(len(test),expected),transformed_features=transformed,
        row_id=test.row_id.to_numpy(dtype=str),feature_names=names,blocks=np.asarray(blocks,dtype=str),
        model_prediction=prediction)
    metadata={"design":design,"repeat":1,"fold":1,"selection":"First repeat and first outer fold fixed before errors were reviewed; core family chosen by inner MAE.",
        "selected_family":checkpoint["selected_family_inner"],"model_path":str(model_path),"model_sha256":sha256(model_path),
        "explainer":explanation_type,"train_n":len(train),"explained_test_n":len(test),
        "linear_background_n":len(background_ids),"linear_background_row_ids":background_ids,
        "background_disjoint_from_explained_test":not bool(set(background_ids)&set(test.row_id)),
        "expected_prediction":expected,"max_additivity_error":float(np.max(np.abs(residual))),
        "total_sum_mean_absolute_shap":total,"shap_version":shap.__version__,
        "interpretation":"Block concentration is sum of feature mean absolute SHAP divided by the total. It is not share of variance, a causal effect, exclusive block value, or evidence of stability across all folds.",
        "restricted":"Saved arrays contain pseudonymous row IDs and transformed exact-location features; local controlled research use only."}
    (output/f"{design}_r1_f1_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return feature_table,block_table,metadata


def plot_concentration(frame:pd.DataFrame,output:Path) -> None:
    fig,ax=plt.subplots(figsize=(9.3,5.4))
    order=list(reversed("SLRAT"));colors=["#174a72","#4f866a","#b45622"]
    for j,(design,color) in enumerate(zip(DESIGNS,colors)):
        part=frame[frame.design==design].set_index("block")
        heights=[100*part.loc[b,"attribution_share"] if b in part.index else 0 for b in order]
        family=part.selected_family.iloc[0]
        ax.barh(np.arange(len(order))+(j-1)*.23,heights,height=.21,color=color,
                label=f"{DESIGN_LABELS[design]}: {family}")
    ax.set_yticks(range(len(order)),[BLOCK_LABELS[b] for b in order])
    ax.set(xlabel="Share of sum of mean absolute SHAP (%)",title="Attribution concentration in three preselected outer fits")
    ax.grid(axis="x",color="#e4e4e4",lw=.6);ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=10.5,loc="lower center",bbox_to_anchor=(.5,.07),ncol=1)
    fig.text(.5,.015,"Repeat 1 / fold 1 only; held-out test explanations. Predictive attribution, not explained variance or causal effects.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.23,1,1));save_figure(fig,output/"figure_s01_shap_block_concentration")


def explain(results:str|Path,data:str|Path,output:str|Path,background_n:int=200) -> dict:
    bundle=require_complete(results,data)
    currency=read_currency_metadata(bundle["data_path"])
    if sha256(Path(__file__).with_name("features.py"))!=bundle["config"]["hashes"]["feature_code"]:
        raise ValueError("Custom transformer source changed after model fitting.")
    output=Path(output);arrays=output/"explanations";tables=output/"tables";figures=output/"figures"
    if background_n<1:raise ValueError("Linear background size must be positive.")
    for design in DESIGNS:
        if not (bundle["results"]/"folds"/f"{design}_r1_f1"/"selected_full_model.joblib").exists():
            raise FileNotFoundError(f"Missing preselected model artifact: {design}, repeat 1/fold 1")
    feature_tables,block_tables,metadata=[],[],[]
    for design in DESIGNS:
        feature,block,info=explain_one(bundle,design,arrays,background_n)
        info["price_unit"]=currency["price_unit"]
        info["shap_value_unit"]=currency.get("code") or "recorded-price units"
        info["attribution_share_unit"]="dimensionless proportion"
        (arrays/f"{design}_r1_f1_metadata.json").write_text(json.dumps(info,indent=2)+"\n")
        feature_tables.append(feature);block_tables.append(block);metadata.append(info)
    feature=pd.concat(feature_tables,ignore_index=True);block=pd.concat(block_tables,ignore_index=True)
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    feature.to_csv(tables/"table_s03_shap_feature_attributions.csv",index=False)
    block.to_csv(tables/"table_s04_shap_block_concentration.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"pdf.fonttype":42,"ps.fonttype":42})
    plot_concentration(block,figures)
    manifest={"source_results_fingerprint":bundle["config"]["fingerprint"],"n_preselected_fits":3,
        "fits":metadata,"explanations_scope":"Held-out cases in three fixed outer folds. No best-outer-fold selection and no general attribution-stability claim.",
        "price_unit":"Original posted-price scale; currency unresolved."}
    manifest=attach_publication_currency(manifest,currency)
    (output/"explanation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results",type=Path,required=True);cli.add_argument("--data",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True);cli.add_argument("--background-n",type=int,default=200)
    args=cli.parse_args()
    print(json.dumps(explain(args.results,args.data,args.output,args.background_n),indent=2))
