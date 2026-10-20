# Data scope and access

This selection contains scientific aggregate results and schema documentation.
It contains no listing-level or host-level dataset. Field names in
[`data/restricted_input_schema.csv`](../data/restricted_input_schema.csv), such as `host_id`, `row_id`, `latitude` and
`description`, describe the excluded input schema; their values are not supplied.
The empirical `missing_n` column was omitted from the source dictionary. All
other retained schema cells are unchanged.

The study analyses a processed derivative of archived Airbnb listings for
Lima–Callao, described in the manuscript as a December 2024 snapshot. The archived
numeric advertised nightly prices are in Peruvian soles (PEN), as confirmed by
the authors. The data do not establish realised transactions, occupancy,
profitability, a standardised fee-and-stay quote, or sustainability effects.
Upstream discovery/capture and inherited distance-variable provenance are not
fully reconstructed by these materials.

Direct identifiers, pseudonymous identifiers, exact coordinates, source URLs,
original titles/descriptions, listing-level outcomes/predictions, split
membership, fitted models and individual SHAP arrays are excluded. No examples
from real source records are inserted into the schema documentation.

Neighbourhood outputs retain a row only when `n_listings >= 30 AND n_hosts >= 5`.
All retained groups happen to have at least 32 observed listings and 27 observed
hosts. Only listing-weighted neighbourhood rows are included, matching the
supplementary figure. This is a conservative dissemination decision and does not
guarantee anonymisation or protection against arbitrary linkage with external
data. Names, individual counts and outcome aggregates of the excluded
neighbourhoods are not supplied.

The original full neighbourhood outputs contain small groups, including groups
with one listing, so they are not a substitute for this filtered selection.
Likewise, the prior aggregate ZIP is not the reviewed publication selection.
Entire rare-category/segment tables, exact observed minima/maxima, fine
partition-margin tables and feature-level SHAP categories are omitted. This
avoids publishing the remaining margins of a suppressed small category.

The retained neighbourhoods cover 1,419 listings. The omitted set contains 25
neighbourhoods and 279 listings. Subtracting the retained neighbourhood aggregates
from a complete-inventory aggregate therefore does not isolate one omitted
listing or one omitted neighbourhood. No named residual outcome table or
cross-tabulation of omitted neighbourhoods is released. This review addresses
the simple margins present in this selection, not a formal disclosure-risk
proof.

Full numerical refitting requires separately authorised restricted inputs; these
aggregates cannot reproduce source records or replace the restricted data.
Publication of code and aggregate results does not itself grant rights to
redistribute third-party listing content or access restricted files. This repository
provides reviewed code and aggregate results only. It does not grant a licence or
third-party permission to redistribute restricted listing data, approve access to
them, or provide a public deposit of those records.
