# Explainable Airbnb asking price benchmarks

Research code and selected aggregate results accompanying the manuscript **Explainable Airbnb Asking Price Benchmarks across Hosts and Neighbourhoods**.

The study evaluates advertised nightly prices for 1,698 Airbnb listings, 1,430 observed hosts and 38 recorded neighbourhoods in Lima–Callao, Peru. Prices are recorded in Peruvian soles (PEN); no currency conversion was performed. These are asking prices rather than observed transactions or revenue.

## Study design and findings

The primary analysis uses nested evaluation under random-listing, host-grouped and neighbourhood-grouped partitions. Train-fitted preprocessing, paired feature-block removals and calibrated prediction intervals assess information content and prediction reliability. The pre-specified core selector chooses among Ridge, Elastic Net, gradient boosting and random forest within the training data; the seven-family benchmark is a separate comparison.

The pre-specified family selector achieves mean absolute errors of 46.73, 49.00 and 53.52 PEN across the three designs. A separate exploratory analysis gives each observed host equal total weight, increasing those errors to 50.13, 51.58 and 56.76 PEN. Aggregate coverage of nominal 80% intervals does not ensure adequate coverage in every predicted-price band.

The secondary diagnostics were defined after the primary results were available. They describe host inventories, weighting choices and neighbourhood variation without selecting or refitting the primary models. They should be distinguished from the original primary protocol.

These results are internal validation of a previously examined historical snapshot. They do not establish optimal rates, causal effects, profitability, discrimination or environmental or social sustainability outcomes.

## Contents and access

- `code/` contains the analytical workflow and verification code.
- `tests/` contains checks using synthetic inputs.
- `data/` documents the required input schema; the listing-level dataset is excluded.
- `results_public/` contains a deliberately limited selection of aggregate tables and figures.
- `docs/` describes reproduction, diffusion filters, citation and licensing status.

The included selection of tables is not the full analytical output. Small neighbourhood groups and other disclosive or unnecessarily detailed summaries are excluded. The diffusion rules do not change the underlying scientific estimates or constitute a guarantee of anonymisation.

The code can be inspected and its synthetic checks can be run without the restricted dataset. Full numerical reproduction requires authorised access to the corrected input. Postprocessing also requires the corresponding private predictions and fitted models. Restricted inputs and generated individual-level artifacts must remain outside version control.

The processed source records scraping dates of 17–18 December 2024. Upstream collection and inherited distance calculations are not fully reconstructed; see [data provenance and access](data/README.md).

See [reproduction instructions](docs/REPRODUCIBILITY.md) for installation and execution, [diffusion rules](docs/DATA_DIFFUSION.md) for inclusion and exclusion criteria, and [result definitions](docs/RESULTS_DICTIONARY.md) for interpreting the tables.

## Citation and licensing

The accompanying manuscript is cited as an unpublished work in [citation information](docs/CITATION.md). No journal acceptance, publication DOI or software contribution roles are inferred.

A reuse licence has not yet been selected. The original platform data have separate permissions. See [licensing status](docs/LICENSING.md) before redistributing or reusing the materials.
