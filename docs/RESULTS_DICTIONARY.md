# Reading the aggregate results

Each results CSV contains aggregate metrics, not individual predictions. Retained
cells preserve their original textual numeric precision, units, column order and
row order. Copies without filters are byte-identical to their source CSVs.
[`selection_manifest.json`](../results_public/selection_manifest.json) records every source/output hash and row loss.

- `design`: `random` is random-listing cross-fitting; `host` withholds host
  groups; `neighbourhood` withholds recorded neighbourhood groups.
- `repeat`: one of two complete cross-fitting repetitions on the same inventory.
  The repetitions do not add independent observations.
- `method`: the named family/comparator or the pre-specified `selected_full`
  core-selection procedure. Comparisons do not select the best observed outer
  family after evaluation.
- `MAE`, `RMSE`, signed bias, and interval widths are in PEN. Signed bias is
  prediction minus observed advertised price. `R2` is dimensionless.
- Primary complete-repetition point metrics are computed from pooled
  out-of-fold predictions for all 1,698 observations, rather than averaging
  fold-specific R² values. Each plotted benchmark point is a repetition-level
  aggregate, not an advertisement.
- `_mean`, `_min`, and `_max` describe the two repetition values. They are not
  confidence limits or independent-replication uncertainty estimates.
- Ablation differences are variant minus full MAE: positive values mean greater
  error after removing a block. Only `without_S`, `without_L`, `without_R`,
  `without_A`, and `without_T` are retained. The optional transductive host
  sensitivity is omitted from this minimal selection.
- S: structural attributes; L: location; R: reviews; A: recorded available
  amenities; T: train-fitted latent title/description representation. No raw text
  or exact location values accompany these block labels.
- `coverage` is a proportion; the nominal target is 0.80. Widths are in PEN.
  `absolute_residual` and `relative_residual` are separate calibration methods.
  Calibration-model point predictions are separate from the principal
  selected-core point predictions.
- `weighting=listing` gives each observed advertisement equal influence.
  `equal_host` averages within-host errors and then across hosts in the reported
  group. Only the broad overall/portfolio point displays retain both weightings.
- `one_observed_listing` and `multiple_observed_listings` describe the complete
  observed snapshot, not total platform portfolios or professional status. The
  broad portfolio cohorts contain 1,311 listings/1,311 hosts and 387 listings/119
  hosts respectively; no individual host results are supplied.
- Neighbourhood files include only rows with at least 30 listings and 5 hosts,
  and only listing weights. `n_listings` and `n_hosts` are subgroup counts, not
  additional samples. Secondary comparisons are exploratory and do not refit or
  alter the primary models.
- `attribution_share` is a block's share of the sum of mean absolute SHAP
  attributions in one of three preselected repeat-1/fold-1 fits. It is neither
  explained variance nor an independent or causal effect. No observation-level
  attributions or category-specific feature table is supplied.

The original Figure 1/2/S1/S2 PNG and PDF bytes are retained. Figure S2 contains
the same 13 eligible neighbourhoods as the filtered CSVs. The other figures show
whole-inventory performance, block removals, and block-level attribution shares.
All figures were visually inspected for individual record labels or values.
