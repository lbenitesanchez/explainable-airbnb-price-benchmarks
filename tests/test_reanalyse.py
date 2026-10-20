"""Checks that exercise separation and persisted transformers without heavy fits."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import reanalyse as r
from features import SafeTextSVD, ListingText


def dataset(n=72):
    rng = np.random.default_rng(901)
    data = {column: rng.uniform(1, 5, n) for block in r.BLOCKS.values() for column in block["num"]}
    data.update({"row_id": [f"r{i:04}" for i in range(n)],
                 "host_group": [f"h{i//3}" for i in range(n)],
                 "neighbourhood": [f"district{i//6}" for i in range(n)],
                 "occupancy_type": np.where(np.arange(n) % 2, "whole", "room"),
                 "property_group_corrected": np.where(np.arange(n) % 2, "apartment", "room_unspecified"),
                 "name": [f"urban{i%5} garden{i%7}" for i in range(n)],
                 "description": [f"details{i%8} furnished rental city{i%6}" for i in range(n)],
                 "price": rng.uniform(20, 800, n)})
    data = pd.DataFrame(data)
    data.loc[::7, "cleaned_bathrooms"] = np.nan
    return data


@pytest.mark.parametrize("design", ["random", "host", "neighbourhood"])
def test_outer_folds_cover_each_row_once_and_keep_groups_out(design):
    data = dataset()
    splits = r.make_splits(data, design, 20261018)
    tested = []
    for train, test in splits:
        r.assert_isolated(data, train, test, design)
        tested.extend(test)
    assert sorted(tested) == list(range(len(data)))


def test_isolation_rejects_shared_rows_and_hosts():
    data = dataset()
    with pytest.raises(AssertionError, match="row"):
        r.assert_isolated(data, [0, 1], [1, 2], "random")
    with pytest.raises(AssertionError, match="host"):
        r.assert_isolated(data, [0], [1], "host")


@pytest.mark.parametrize("design", ["random", "host", "neighbourhood"])
def test_calibration_and_inner_fitting_are_disjoint(design):
    data = dataset()
    fit, cal = r.calibration_split(data, design, 19)
    r.assert_isolated(data, fit, cal, design)
    fitting = data.iloc[fit]
    inner = r.make_splits(fitting, design, 20)
    for train, validation in inner:
        r.assert_isolated(fitting, train, validation, design)
        assert set(fitting.iloc[train].row_id).isdisjoint(data.iloc[cal].row_id)
        assert set(fitting.iloc[validation].row_id).isdisjoint(data.iloc[cal].row_id)


def test_text_vocabulary_and_imputation_use_only_fitting_data():
    data = dataset()
    train, test = data.iloc[:54].copy(), data.iloc[54:].copy()
    train["cleaned_bathrooms"] = 2.
    test["cleaned_bathrooms"] = np.nan
    test["name"] = "forbiddenuniquetestword"
    test["occupancy_type"] = "never_seen_type"
    prep = r.preprocessing(r.FULL)
    prep.fit(train, train.price)
    transformed = prep.transform(test)
    assert np.isfinite(transformed).all()
    text = prep.named_transformers_["text"].named_steps["semantic"]
    assert "forbiddenuniquetestword" not in text.vectorizer_.vocabulary_
    numeric = prep.named_transformers_["numeric"].named_steps["impute"]
    bathroom_index = [c for block in r.FULL for c in r.BLOCKS[block]["num"]].index("cleaned_bathrooms")
    assert numeric.statistics_[bathroom_index] == 2.


def test_empty_vocabulary_outputs_fixed_width_zeroes():
    fitted = SafeTextSVD().fit(np.array(["always same"] * 8))
    transformed = fitted.transform(np.array(["unseen", "", "different"]))
    assert transformed.shape == (3, 4)
    assert not transformed.any()


def test_currency_mask_preserves_capacity_numbers_without_target_access():
    values = pd.DataFrame({"name": ["4 guests S/ 120 por noche USD90", "3 beds 100 soles US$40 $50 PEN 60"],
                           "description": ["write a@b.com https://example.com", "2 bathrooms"]})
    cleaned = ListingText().transform(values)
    assert "4 guests" in cleaned[0]
    assert "2 bathrooms" in cleaned[1]
    assert "120" not in cleaned[0] and "USD90" not in cleaned[0]
    for amount in ["100", "40", "50", "60"]:
        assert amount not in cleaned[1]
    assert "a@b.com" not in cleaned[0] and "https" not in cleaned[0]
    audit = ListingText().audit(values)
    assert audit["rows_with_explicit_currency_amount"] == 2
    assert audit["explicit_currency_amount_matches"] == 6


def test_conformal_quantile_uses_finite_sample_rank():
    assert r.residual_quantile([1, 4, 3, 2], .2) == 4.
    assert np.isinf(r.residual_quantile([1.], .2))


def test_primary_spec_excludes_transductive_host_and_inherited_sentiment():
    assert r.FULL == "SLRAT"
    columns = [col for block in r.FULL for cols in r.BLOCKS[block].values() for col in cols]
    assert "host_portfolio_size_snapshot" not in columns
    assert "sentiment_score_listing" not in columns
    assert "has_pool_corrected" in columns
    assert "cleaned_bathrooms" in columns


def test_budget_matches_frozen_full_plan():
    args = argparse.Namespace(designs=["random", "host", "neighbourhood"], repeats=2,
                              jobs=2, intervals=True, host_sensitivity=True)
    families = r.model_grids(False)
    # Avoid making an optional platform library a prerequisite for this arithmetic test.
    families["XGBoost"] = (Ridge(), {"model__alpha": [1., 2.]})
    assert r.configuration(args, families)["estimated_regressor_fits"] == 2916


def test_small_nested_parallel_search_and_persistence(tmp_path):
    data = dataset()
    train, test = data.iloc[:54], data.iloc[54:]
    original_prices = data.price.copy()
    inner = r.make_splits(train, "host", 21)
    pipeline, info = r.search_family("Ridge", r.FULL, train, inner, r.model_grids(),
                                   tmp_path / "search", 2, joblib.Memory(tmp_path / "cache", verbose=0))
    assert info["regressor_fits"] == 10
    assert np.isfinite(pipeline.predict(test)).all()
    pd.testing.assert_series_equal(original_prices, data.price)
    assert len(pd.read_csv(tmp_path / "search/cv_results.csv")) == 3
    joblib.dump(pipeline, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(loaded.predict(test), pipeline.predict(test))
    assert loaded.named_steps["preprocess"].named_transformers_["text"].named_steps["semantic"].__class__.__module__ == "features"


def test_exact_original_prices_not_changed_by_metrics_or_prediction_records():
    data = dataset()
    before = data.price.to_numpy().copy()
    frame = r.prediction_frame(data, before + .01, "host", 1, 1, "probe", r.FULL, "Ridge")
    assert np.array_equal(frame.observed_price.to_numpy(), before)
    assert np.array_equal(data.price.to_numpy(), before)
    assert r.score(before, before)["MAE"] == 0.
