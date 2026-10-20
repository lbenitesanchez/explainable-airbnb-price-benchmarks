# Prespecified secondary diagnostic analysis

This protocol was written on 19 October 2026 before calculating the secondary summaries. The analysis is a descriptive extension for a potential Sustainability submission. It does not change the prepared data, fitted models, external folds, predictions, primary tables, or main analysis protocol.

## Fixed scope and estimands

- Use only `selected_full` from the saved main out-of-fold point predictions. This is the core-family selection procedure selected within training folds; do not retrospectively choose the best benchmark family.
- Report each of the three saved validation designs and both saved repetitions separately. Pool the external test-fold predictions within each design and repetition before computing metrics. The two repetitions reuse the same snapshot and are not independent replications.
- Compare listing weighting (every observed listing has equal weight) with host weighting (every observed host has equal total weight). Within each reported group, a listing's host weight is the inverse of that host's number of listings in that group. Host-weighted MAE and signed bias are averages of per-host mean errors; host-weighted RMSE is the square root of the average of per-host mean squared errors. Weighted R2 uses the correspondingly weighted target mean and weighted squared-error/total-sum-of-squares ratio. It is undefined if target variation is zero.
- Define signed bias as `prediction - observed_price`: positive values indicate overprediction. Prices, errors and interval widths are in Peruvian soles (PEN) per advertised night, as confirmed by the author in `currency_metadata.json`. No currency conversion or price adjustment is applied.
- Define host portfolio groups from the entire observed input snapshot: exactly one observed listing versus at least two observed listings. These are observed snapshot counts, not verified platform-wide host portfolios or a measure of host professionalism. Verify the saved snapshot-size field against row counts by `host_group`.
- Use all saved neighbourhood labels, including small groups, in machine-readable tables. **Prespecified interpretability threshold: a neighbourhood must contain at least 30 observed listings (`n_listings >= 30`) to be included in narrative ranges or the neighbourhood diagnostic figure.** Every neighbourhood table retains its listing count, distinct observed host count and threshold flag. This is a descriptive reporting rule, not a reliability guarantee or a significance threshold.
- Neighbourhood rows and plots are ordered by label, never by error, coverage, or a punitive performance ranking. Do not infer deprivation, displacement, housing affordability, welfare, or causal effects from these diagnostics.

## Interval diagnostics

- Use the existing saved interval predictions, separately for `absolute_residual` and `relative_residual`, at their original nominal 80% level. Do not recalibrate, refit, change endpoints, or choose a method after observing diagnostics.
- Interval predictions come from the corresponding saved proper-training/calibration procedure and need not equal the refitted main `selected_full` point predictions. Report interval coverage and mean/median width from these interval-specific predictions; keep their point-error diagnostics separately labelled.
- Recompute coverage from endpoints inclusively and width as upper minus lower; verify consistency with saved fields and observed targets. Apply listing and equal-host weighting separately. Weighted medians use the first ordered value whose cumulative weight reaches at least half of total weight (the lower inverse-CDF median); use exact rational1/n_h weights to preserve the intended50% boundary. For listing weights, report the conventional sample median.
- Report aggregate, observed host-portfolio-group, and neighbourhood diagnostics for each design, repetition, interval method and weighting. Group size always means distinct observed listings in that repetition, not duplicated observations across repetitions.

## Summaries, validation and outputs

- Across the two repetitions, provide means and descriptive minima/maxima of each diagnostic. These ranges are not confidence intervals. Do not conduct significance tests, bootstrap inference, treat repetitions as independent observations, or make causal claims.
- Validate uniqueness of listing IDs in prepared data and within each prediction slice, complete coverage of the same snapshot in all six design/repetition cells, fold labels and saved external test-row membership, finite predictions, exact observed-target linkage, and all 18 completed checkpoints. Verify all paths listed in `baseline_primary_sha256.json` before and after execution and fail if any protected file changes.
- Do not export direct identifiers, raw listing text, coordinates, or row-level errors. Secondary outputs contain aggregate tables only. The observed neighbourhood labels are retained solely to join the descriptive context tables.
- Write to `publication_secondary/` for the released analysis outputs; a separate directory within the project is accepted for reproduction, excluding primary data, results, publication, code, tests and document directories. Include per-repetition CSVs, descriptive two-repeat summary CSVs, a machine-readable manifest with input/code/protocol hashes and definitions, Spanish integration notes in JSON, and a scientific PNG (300 dpi) plus PDF if it adds explanatory value.

The neighbourhood diagnostic figure, if produced, will show signed bias and 80% interval coverage only for labels meeting the fixed 30-listing rule. It will include a zero-bias reference and nominal-coverage reference and show the two repetitions as descriptive points, with no inferred confidence bars.
