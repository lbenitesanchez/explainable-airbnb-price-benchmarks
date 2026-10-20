# Reproducing the asking-price benchmark analyses

This directory contains analytical source, protocols, locked dependencies and tests. It contains no listing data, saved predictions, fitted models, SHAP arrays, manuscript files, author forms or execution metadata. Public aggregate results and figures are selected separately; they are not sufficient inputs to reconstruct the analyses.

The procedure predicts advertised nightly prices. It does not estimate realised transactions, revenue, demand, optimal prices, causal effects or measured sustainability outcomes. The two repeated partitions reuse one inventory and do not constitute independent replications.

## Environment

Run the commands from the repository root. Create an isolated environment at a private location outside this checkout, activate it, and install the recorded versions:

```bash
python -m pip install -r requirements.lock.txt -r requirements-test.txt
```


Use Python 3.12 (the study used 3.12.14), preferably in a fresh isolated environment outside the tracked checkout. Install `requirements.lock.txt`; install `requirements-test.txt` to run tests. These are the study's recorded versions, not a claim of compatibility with arbitrary newer releases or every operating system. Numerical results may vary across platform, BLAS/OpenMP and package builds.

The fixed complete study uses NumPy, pandas, SciPy, scikit-learn, joblib, threadpoolctl, XGBoost, matplotlib and, for optional explanations, SHAP. The full workflow requires XGBoost; `--skip-shap` skips explanations only. On systems needing an explicit OpenMP library location, pass `--openmp-library-dir /path/to/library-directory`. This affects child-process environment only. The launcher sets OMP/OpenBLAS/MKL thread limits to one and controls search parallelism with `--jobs`.

## Required private inputs

Use only data you are authorised to process. Keep every input and generated raw artifact outside the source repository.

The preferred input is an authorised copy of `model_ready_listings_restricted.csv`, supplied with `--prepared-csv`. Its schema is defined by the feature blocks in `code/reanalyse.py` and the export list in `code/prepare_data.py`. It includes a positive finite `price`, unique stable `row_id`, `host_group`, `neighbourhood`, corrected structural and amenity fields, reviews, precomputed distances, coordinates and the original `name`/`description` text. These files remain restricted: pseudonyms do not make text and exact locations anonymous. Preserve row order and the `source_row_index` ordering when present.

Alternatively, `--source-csv` runs the supplied deterministic preparation rules on an authorised original CSV. The original schema requires `listings_id`, `host_id`, `bathrooms_text`, `bathrooms`, `property_type`, `ammenities` (the inherited spelling), `neighbourhood`, `price`, `accommodates`, `bedrooms`, `beds`, `latitude`, `longitude`, `name`, `description`, and the review/distance columns enumerated by `REVIEW_COLUMNS` and `DISTANCE_COLUMNS`. These parsers and nominal mappings were designed for the study's Spanish-language source. They are not a universal Airbnb scraper or a complete source-capture reconstruction. Inherited distances must already exist; no geocoding or external data collection is performed.

Supply an explicit private currency confirmation with `--currency-metadata`. The study-specific reporting currently supports PEN, without conversion. The sidecar must truthfully record the unit already verified for the supplied study data; this repository does not manufacture an author confirmation. The accepted schema is:

```json
{
  "code": "PEN",
  "name": "Peruvian sol",
  "confirmed_by": "Author",
  "price_unit": "Peruvian soles (PEN) per advertised night",
  "numeric_values_changed": false,
  "currency_conversion_applied": false
}
```

No private sidecar is distributed here. This unit declaration concerns the data only; it is not a publication approval or author declaration form. Other currencies require an explicitly revised reporting protocol and labels, not relabelling unchanged amounts as PEN.

## Run a plan first

All paths below are examples to replace with your own private locations. Omitting an execution mode prints the commands and expected budget without copying inputs, creating output directories or fitting models:

```bash
python run_reproduction.py \
  --prepared-csv /private/input/model_ready_listings_restricted.csv \
  --currency-metadata /private/input/currency_metadata.json \
  --work-dir /private/reproduction-run \
  --jobs 2
```

Add `--execute` to run the complete fixed procedure. Use a new empty private work directory; the launcher refuses a tracked source directory, an ancestor of it, an unrelated nonempty directory, or replacement of an existing input with different bytes. The intended full configuration has three designs, three external folds, two fixed repetitions, seven benchmark families, core-family selection, five ablations, the snapshot-H sensitivity and two interval constructions: 2,916 regressor fits. Runtime depends on hardware and data; no fixed completion time is promised.

The launcher deliberately prepares data first and then fits through the portable corrected-CSV route, even with `--source-csv`. When resuming, pass `--prepared-csv` pointing to the corrected private file. Source, preparation, protocol, configuration and environment hashes govern checkpoint reuse. Changing the environment, jobs or any fingerprinted source requires a different results directory; do not rewrite hashes to reuse incompatible checkpoints.

For complete existing results in the same private work directory:

```bash
python run_reproduction.py \
  --prepared-csv /private/reproduction-run/data/model_ready_listings_restricted.csv \
  --currency-metadata /private/reproduction-run/currency_metadata.json \
  --work-dir /private/reproduction-run --postprocess-only
```

`--secondary-only` recalculates and independently verifies the host-weighted, portfolio and neighbourhood summaries without fitting regressors. `--postprocess-only` also audits the primary computations, regenerates primary tables/figures and, unless `--skip-shap`, loads the three models selected in the predefined first outer folds. Load only trusted models produced by your own run; joblib deserialisation executes Python objects.

## Outputs stay private

The launcher creates `data/`, `results/`, `publication/` and `publication_secondary/` beneath the external `--work-dir`. They include restricted row-level data, OOF predictions, split identifiers, calibration residuals, CV records, fitted models, SHAP arrays and local execution metadata. They must not be copied wholesale to a public repository. Small-cell aggregates and descriptive metadata may also disclose information.

`baseline_primary_sha256.json` is generated for that private run before secondary processing and checked afterwards. It protects the current inputs and saved primary outputs, excluding disposable estimator cache. It does **not** require, distribute or purport to reproduce the historical 845-file private archive. It is an execution integrity check, not independent external validation.

Use the launcher rather than relying on the low-level study scripts' historical directory defaults. `.gitignore` supplies additional guards for typical restricted artifacts; it does not review or anonymise new outputs. Public publication of any regenerated table requires a separate disclosure review. The analysis itself retains small neighbourhoods in private CSVs; the predeclared n≥30 threshold is a descriptive presentation rule, not an anonymisation guarantee.

## Verification and scope

```bash
python code/tests_data.py
python code/verify_secondary.py --self-test
python -m pytest tests
```

The tests use synthetic records and hand-calculated estimands. One small Ridge search exercises nested parallel preprocessing and persisted transformers; it is not the full empirical analysis. The independent verifiers require your private data and complete saved prediction/split/CV artifacts. No claim is made that the paper's numerical results can be regenerated without the authorised study inputs.

`prepare_data.py`, `features.py`, `reanalyse.py`, both protocols and the model/estimand tests are retained from the analysed version. Public-source adaptations concern the launcher, supplied-snapshot row-count checks, private-run integrity baselines and relative audit-manifest paths. They do not alter fitting grids, seeds, transformations, selection, metrics, conformal formulas or weighted estimands. Dataset cardinalities are now checked against the supplied CSV rather than hard-coded to 1,698/1,430/38; full postprocessing still requires the fixed 18-fold, two-repeat, all-family design.

The secondary reporting assumes both observed host-portfolio groups exist and at least one neighbourhood has n≥30. Other inventories may need a separately documented reporting adaptation; a new inventory is a new analysis, not a reproduction of the historical study. Missing groups, insufficient training/calibration data, zero target variance or unavailable dependencies may correctly prevent the study-specific audits from passing.

No licence choice or permission to redistribute the private inputs is implied by this source preparation. Licence and repository release decisions remain with the rights holders.
