#!/usr/bin/env python3
"""Prepare the Lima Airbnb snapshot without modifying or imputing source data.

The returned DataFrame preserves all source columns and adds corrected variables.
CSV exports remove direct platform IDs/URLs but retain exact location and free
text for controlled local research use. They are RESTRICTED, not anonymous.
Missing bathrooms and invalid amenities must be imputed in each training fold.
No observation is removed, no price is changed, and no global imputation occurs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_SHA256_AUDITED = "236ef3811f51c6b78e98d91e5259a741abaec6c3c2cb0b9229a5cf6c85023b01"
ID_NAMESPACE = "airbnb-lima-dec2024-revision-2026-v1"
DIRECT_IDENTIFIERS = ("listings_id", "listing_url", "host_id", "host_url")

# These nominal labels describe physical accommodation format, independently of
# whether the offer is a whole property or a room. Ambiguous labels stay unknown.
PROPERTY_GROUP_MAPPING = {
    "Alojamiento entero: vivienda rentada": "apartment",
    "Alojamiento entero: condo": "apartment",
    "Alojamiento entero: apartamento con servicios incluidos": "apartment",
    "Alojamiento entero: loft": "apartment",
    "Alojamiento entero: alojamiento": "house",
    "Alojamiento entero: casa de campo": "house",
    "Alojamiento entero: villa": "house",
    "Alojamiento entero: casa adosada": "house",
    "Alojamiento entero: casa de huéspedes": "guest_unit",
    "Alojamiento entero: suite de invitados": "guest_unit",
    "Alojamiento entero: cabaña": "cabin_or_bungalow",
    "Alojamiento entero: bungalow": "cabin_or_bungalow",
    "Habitación": "room_unspecified",
    "Habitación privada": "room_unspecified",
    "Minicasa": "other_or_unspecified",
    "Casa particular": "other_or_unspecified",
    "Alojamiento entero: alojamiento vacacional": "other_or_unspecified",
    "Contenedor de transporte": "other_or_unspecified",
    "Choza": "other_or_unspecified",
    "Arquitectura subterránea": "other_or_unspecified",
    "Todo el lugar": "other_or_unspecified",
}

# Historical keyword definitions are retained so corrections can be audited.
# Each family is counted independently. Counts need not partition amenities.
LUXURY_KEYWORDS = [
    "Jacuzzi", "Sauna", "Gimnasio", "Cargador para autos eléctricos",
    "Sistema de sonido", "Televisor de alta definición",
    "Televisor con pantalla de 55\u00a0pulgadas",
    "Televisor con pantalla de 60\u00a0pulgadas",
    "Televisor con pantalla de 65\u00a0pulgadas",
    "Televisor con pantalla de 70\u00a0pulgadas",
    "Televisor con Disney+", "Televisor con HBO Max",
    "Televisor con Netflix, servicio de cable prémium",
    "Televisor con Reproductor de DVD", "Piscina cubierta privada",
    "Chimenea interior",
]
BASIC_KEYWORDS = [
    "Cocina", "Lavadora", "Secadora", "Refrigerador", "Microondas",
    "Aire acondicionado", "Wifi", "TV",
]
BEACH_KEYWORDS = [
    "Playa", "Costa", "Acceso al lago", "Vista al mar", "Vista al océano",
    "Vista a la bahía", "Vista a la playa",
]
REVIEW_COLUMNS = [
    "number_of_reviews", "review_scores_rating", "review_scores_cleanliness",
    "review_scores_accuracy", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
]
DISTANCE_COLUMNS = [
    "dist_recreacion_m", "dist_hospital_m", "dist_parque_m",
    "dist_universidad_m", "dist_colegio_m", "dist_tienda_m", "dist_estacion_bus_m",
]
AMENITY_COLUMNS = [
    "available_amenities_count", "luxury_amenities_count",
    "basic_amenities_count", "beach_amenities_count", "has_pool_corrected",
]
STRUCTURAL_FLAGS = [
    "flag_bedrooms_gt_capacity", "flag_beds_gt_capacity", "flag_bedrooms_gt_beds",
    "flag_bedrooms_ge_10", "flag_bathrooms_gt_capacity",
]


def normalize(value: object) -> str:
    """Normalize spelling/space without changing source fields."""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text.replace("\\xa0", " ").replace("\u00a0", " ")).strip()


def parse_bathrooms(value: object) -> float:
    """Read bathroom counts from explicit Spanish text; unknown stays NaN."""
    if pd.isna(value):
        return np.nan
    text = normalize(value).casefold()
    if text in {"", "-", "nan", "none", "null", "n/a"}:
        return np.nan
    if "no hay baño" in text:
        return 0.0
    if "medio baño" in text:
        return 0.5
    number = re.search(r"\d+(?:[,.]\d+)?", text)
    return float(number.group().replace(",", ".")) if number else np.nan


def occupancy_from_type(value: object) -> str:
    """Infer occupancy only from an explicit whole/room label in raw type."""
    text = normalize(value).casefold()
    if text.startswith("alojamiento entero:") or text == "todo el lugar":
        return "whole"
    if text.startswith("habitación"):
        return "room"
    return "other"


def parse_amenity_list(value: object) -> list[str] | None:
    try:
        values = ast.literal_eval(str(value))
    except (ValueError, SyntaxError, TypeError):
        return None
    if not isinstance(values, (list, tuple)) or any(not isinstance(x, str) for x in values):
        return None
    return list(values)


def amenity_features(value: object) -> dict[str, float]:
    """Independent availability and category checks fix pool short-circuiting."""
    values = parse_amenity_list(value)
    if values is None:
        return {**dict.fromkeys(AMENITY_COLUMNS, np.nan), "amenities_parse_invalid": 1}
    available = [normalize(x).casefold() for x in values if "no disponible" not in normalize(x).casefold()]
    families = {
        "luxury_amenities_count": LUXURY_KEYWORDS,
        "basic_amenities_count": BASIC_KEYWORDS,
        "beach_amenities_count": BEACH_KEYWORDS,
    }
    result = {key: sum(any(normalize(k).casefold() in x for k in keys) for x in available)
              for key, keys in families.items()}
    result.update({"available_amenities_count": len(available),
                   "has_pool_corrected": int(any("piscina" in x for x in available)),
                   "amenities_parse_invalid": 0})
    return result


def _legacy_pool(value: object) -> int:
    """Historical definition, used only to count changed rows in the audit."""
    values = parse_amenity_list(value) or []
    has_pool = 0
    for item in values:
        if "No disponible" in item:
            continue
        if any(k in item for k in LUXURY_KEYWORDS):
            continue
        if any(k in item for k in BASIC_KEYWORDS):
            continue
        if any(k in item for k in BEACH_KEYWORDS):
            continue
        if "Piscina" in item:
            has_pool = 1
    return has_pool


def _stable_id(value: object, domain: str) -> str:
    digest = hashlib.sha256(f"{ID_NAMESPACE}|{domain}|{value}".encode("utf-8")).hexdigest()
    return f"{domain}_{digest[:24]}"


def _write_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _dictionary(raw: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    derived = {
        "row_id": ("grouping_only", "SHA256 of namespace and listing ID, truncated to 24 hex; stable under row reordering."),
        "host_group": ("grouping_only", "Domain-separated SHA256 of host ID; grouping only, never a predictor."),
        "source_row_index": ("audit_only", "Zero-based physical row position in source CSV; never a predictor."),
        "cleaned_bathrooms": ("structural", "Parse explicit decimal/comma, half=0.5, no bath=0; '-' remains missing. No imputation."),
        "bathroom_text_missing": ("missingness", "1 when bathroom text cannot supply a numeric count."),
        "bathroom_value_changed": ("audit_only", "1 when known parsed bathroom count differs from original numeric count."),
        "occupancy_type": ("structural_nominal", "whole: explicit Alojamiento entero prefix or Todo el lugar; room: Habitación prefix; other: no explicit occupancy."),
        "property_group_corrected": ("structural_nominal", "Explicit fixed map to apartment, house, guest_unit, cabin_or_bungalow, room_unspecified, other_or_unspecified. Separate from occupancy."),
        "property_mapping_unrecognized": ("audit_only", "1 when raw type is absent from the fixed map; nominal fallback is other_or_unspecified."),
        "available_amenities_count": ("amenities", "Count observed list entries excluding No disponible. The scraped list may not represent all platform amenities."),
        "luxury_amenities_count": ("amenities", "Independent case/space-normalized match to archived luxury keyword family, at most one count per entry."),
        "basic_amenities_count": ("amenities", "Independent match to archived basic keyword family; category counts can overlap."),
        "beach_amenities_count": ("amenities", "Independent match to archived beach keyword family; category counts can overlap."),
        "has_pool_corrected": ("amenities", "1 if an available list entry contains piscina, independent of luxury/basic/beach classification."),
        "amenities_parse_invalid": ("audit_only", "Malformed or non-string list => feature counts missing; no false zero."),
        "host_portfolio_size_snapshot": ("host_snapshot_only", "Count host's listings in full observed snapshot; not platform-wide inventory or strictly ex ante deployment feature."),
        "multi_listing_host_snapshot": ("host_snapshot_only", "1 when host_portfolio_size_snapshot exceeds one."),
        "flag_structural_extreme_any": ("audit_only", "Union of specified structural inconsistency flags. No price threshold or automatic deletion."),
    }
    for flag in STRUCTURAL_FLAGS:
        derived[flag] = ("audit_only", "Deterministic structural plausibility flag; retains row and raw values, never based on price.")
    records = []
    for column in frame.columns:
        if column in raw.columns:
            role = ("direct_identifier_not_exported" if column in DIRECT_IDENTIFIERS else "raw_preserved")
            definition = "Original source column preserved without value changes."
            if column == "price":
                role, definition = "target", "Posted nightly price in original numeric scale; currency is unresolved. No winsorization or transformation."
            elif column == "bathrooms":
                role, definition = "raw_audit_only", "Archived processed count, known to contain conversion errors. Use cleaned_bathrooms in models."
            elif column in ("name", "description"):
                role, definition = "text_restricted", "Raw free text; fit all text transformations inside each training fold only."
            elif column in ("latitude", "longitude", "geometry"):
                role, definition = "location_restricted", "Exact inherited location, not de-identified for public release."
            elif column == "sentiment_score_listing":
                role, definition = "unverified_auxiliary", "Inherited score; upstream model and score semantics unavailable. Exclude from confirmatory core until documented."
        else:
            role, definition = derived[column]
        records.append({"column": column, "dtype": str(frame[column].dtype),
                        "origin": "raw" if column in raw.columns else "derived",
                        "role": role, "definition": definition,
                        "missing_n": int(frame[column].isna().sum()),
                        "in_restricted_export": column not in DIRECT_IDENTIFIERS})
    return pd.DataFrame(records)


def _check_parser_examples() -> None:
    examples = {"2,5 baños": 2.5, "Medio baño": 0.5, "No hay baño": 0.0,
                "2\\xa0baños compartidos": 2.0, "3,5\u00a0baños compartidos": 3.5}
    assert all(parse_bathrooms(k) == v for k, v in examples.items())
    assert np.isnan(parse_bathrooms("-"))
    pool = amenity_features("['Piscina cubierta privada', 'No disponible: Piscina']")
    assert pool["has_pool_corrected"] == 1 and pool["luxury_amenities_count"] == 1
    assert amenity_features("['No disponible: Piscina']")["has_pool_corrected"] == 0
    assert np.isnan(amenity_features("malformed")["available_amenities_count"])
    assert occupancy_from_type("Alojamiento entero: suite de invitados") == "whole"
    assert occupancy_from_type("Habitación privada") == "room"


def prepare_data(source_csv: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Return complete raw+corrected frame and write restricted exports/audits.

    Existing original IDs are preserved in memory for conservation checks but
    omitted from exports. Exact coordinates and listing text remain restricted.
    Stable hashes are pseudonyms, not a guarantee against re-identification.
    """
    _check_parser_examples()
    source, output = Path(source_csv).resolve(), Path(output_dir).resolve()
    if output == source.parent or output in source.parents:
        raise ValueError("Use a new output directory separate from the source directory.")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = pd.read_csv(source, dtype={"listings_id": str, "host_id": str})
    required = {"listings_id", "host_id", "bathrooms_text", "bathrooms", "property_type",
                "ammenities", "neighbourhood", "price", "accommodates", "bedrooms", "beds",
                "latitude", "longitude", "name", "description", *REVIEW_COLUMNS, *DISTANCE_COLUMNS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    if raw.listings_id.isna().any() or raw.host_id.isna().any() or raw.listings_id.duplicated().any():
        raise ValueError("Source needs unique nonmissing listing IDs and nonmissing host IDs; no silent deletion allowed.")
    if not np.isfinite(raw.price).all() or not (raw.price > 0).all():
        raise ValueError("Nonpositive/nonfinite prices require explicit source review, not automatic filtering.")
    frame = raw.copy(deep=True)
    frame["source_row_index"] = np.arange(len(frame), dtype=int)
    frame["row_id"] = frame.listings_id.map(lambda x: _stable_id(x, "listing"))
    frame["host_group"] = frame.host_id.map(lambda x: _stable_id(x, "host"))
    frame["cleaned_bathrooms"] = frame.bathrooms_text.map(parse_bathrooms).astype(float)
    frame["bathroom_text_missing"] = frame.cleaned_bathrooms.isna().astype(int)
    known = frame.cleaned_bathrooms.notna()
    frame["bathroom_value_changed"] = (known & frame.cleaned_bathrooms.ne(frame.bathrooms)).astype(int)
    frame["occupancy_type"] = frame.property_type.map(occupancy_from_type)
    frame["property_group_corrected"] = frame.property_type.map(PROPERTY_GROUP_MAPPING).fillna("other_or_unspecified")
    frame["property_mapping_unrecognized"] = (~frame.property_type.isin(PROPERTY_GROUP_MAPPING)).astype(int)
    features = pd.DataFrame(frame.ammenities.map(amenity_features).tolist(), index=frame.index)
    frame = pd.concat([frame, features], axis=1)
    frame["host_portfolio_size_snapshot"] = frame.groupby("host_group")["row_id"].transform("size").astype(int)
    frame["multi_listing_host_snapshot"] = (frame.host_portfolio_size_snapshot > 1).astype(int)
    frame["flag_bedrooms_gt_capacity"] = (frame.bedrooms > frame.accommodates).astype(int)
    frame["flag_beds_gt_capacity"] = (frame.beds > frame.accommodates).astype(int)
    frame["flag_bedrooms_gt_beds"] = (frame.bedrooms > frame.beds).astype(int)
    frame["flag_bedrooms_ge_10"] = (frame.bedrooms >= 10).astype(int)
    frame["flag_bathrooms_gt_capacity"] = (frame.cleaned_bathrooms > frame.accommodates).astype(int)
    frame["flag_structural_extreme_any"] = frame[STRUCTURAL_FLAGS].any(axis=1).astype(int)

    # Conservation checks detect unintended alteration or loss before any export.
    pd.testing.assert_frame_equal(frame[raw.columns], raw, check_exact=True)
    assert len(frame) == len(raw) and frame.row_id.is_unique
    assert frame.host_group.nunique() == raw.host_id.nunique()
    assert frame.source_row_index.tolist() == list(range(len(raw)))
    assert frame.cleaned_bathrooms.isna().sum() == frame.bathroom_text_missing.sum()
    assert frame.price.equals(raw.price)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    legacy_pool = raw.ammenities.map(_legacy_pool)
    legacy_room = raw.property_type.isin(["Habitación", "Habitación privada",
        "Alojamiento entero: suite de invitados", "Alojamiento entero: casa de huéspedes"])
    counts = {
        "rows": len(frame), "raw_columns": len(raw.columns),
        "distinct_listings": int(raw.listings_id.nunique()), "distinct_hosts": int(raw.host_id.nunique()),
        "neighbourhoods": int(raw.neighbourhood.nunique()), "rows_removed": 0, "price_values_changed": 0,
        "bathrooms_known_text_corrected": int(frame.bathroom_value_changed.sum()),
        "bathrooms_unknown_text_left_missing": int(frame.bathroom_text_missing.sum()),
        "pool_available_corrected_total": int(frame.has_pool_corrected.sum()),
        "pool_flags_changed_from_legacy": int(frame.has_pool_corrected.ne(legacy_pool).sum()),
        "legacy_room_only_total": int(legacy_room.sum()),
        "corrected_explicit_room_total": int(frame.occupancy_type.eq("room").sum()),
        "legacy_room_reclassified_whole": int((legacy_room & frame.occupancy_type.eq("whole")).sum()),
        "amenities_parse_invalid": int(frame.amenities_parse_invalid.sum()),
        "property_mapping_unrecognized": int(frame.property_mapping_unrecognized.sum()),
        "structural_flag_any": int(frame.flag_structural_extreme_any.sum()),
    }
    if source_hash == SOURCE_SHA256_AUDITED:
        expected = {"rows": 1698, "raw_columns": 37, "distinct_hosts": 1430,
                    "bathrooms_known_text_corrected": 142, "bathrooms_unknown_text_left_missing": 218,
                    "pool_available_corrected_total": 535, "pool_flags_changed_from_legacy": 36,
                    "legacy_room_only_total": 310, "corrected_explicit_room_total": 288,
                    "legacy_room_reclassified_whole": 22}
        assert all(counts[k] == v for k, v in expected.items()), (counts, expected)

    output.mkdir(parents=True, exist_ok=True)
    audit = output / "audit_tables"
    audit.mkdir(exist_ok=True)
    restricted = frame.drop(columns=list(DIRECT_IDENTIFIERS), errors="ignore")
    assert not set(DIRECT_IDENTIFIERS) & set(restricted.columns)
    restricted.to_csv(output / "cleaned_listings_restricted.csv", index=False)
    model_columns = ["row_id", "host_group", "source_row_index", "neighbourhood", "price",
        "name", "description", "latitude", "longitude", "accommodates", "bedrooms", "beds",
        "cleaned_bathrooms", "bathroom_text_missing", "occupancy_type", "property_group_corrected",
        *REVIEW_COLUMNS, *DISTANCE_COLUMNS, *AMENITY_COLUMNS,
        "host_portfolio_size_snapshot", "multi_listing_host_snapshot",
        "sentiment_score_listing", "flag_structural_extreme_any", *STRUCTURAL_FLAGS]
    model_columns = [c for c in model_columns if c in frame.columns]
    frame[model_columns].to_csv(output / "model_ready_listings_restricted.csv", index=False)
    _dictionary(raw, frame).to_csv(output / "variable_dictionary.csv", index=False)
    pd.DataFrame([{"metric": k, "value": v} for k, v in counts.items()]).to_csv(audit / "preparation_summary.csv", index=False)
    grouped = frame.assign(legacy_bathrooms=raw.bathrooms).groupby(
        ["bathrooms_text", "legacy_bathrooms", "cleaned_bathrooms"], dropna=False).size().reset_index(name="n")
    grouped.to_csv(audit / "bathroom_text_mapping.csv", index=False)
    prop = frame.groupby(["property_type", "occupancy_type", "property_group_corrected"], dropna=False).size().reset_index(name="n")
    prop.to_csv(audit / "property_type_mapping.csv", index=False)
    pd.DataFrame({"legacy_has_pool": legacy_pool, "has_pool_corrected": frame.has_pool_corrected}).value_counts().reset_index(name="n").to_csv(audit / "pool_flag_changes.csv", index=False)
    pd.DataFrame([{"flag": f, "n": int(frame[f].sum())} for f in STRUCTURAL_FLAGS + ["flag_structural_extreme_any"]]).to_csv(audit / "structural_flags_summary.csv", index=False)
    row_changes = frame[["row_id", "source_row_index", "bathroom_value_changed", "bathroom_text_missing",
                         "flag_structural_extreme_any", *STRUCTURAL_FLAGS]].copy()
    row_changes["pool_value_changed"] = frame.has_pool_corrected.ne(legacy_pool).astype(int)
    row_changes["legacy_room_reclassified_whole"] = (legacy_room & frame.occupancy_type.eq("whole")).astype(int)
    row_changes.to_csv(audit / "row_change_flags_restricted.csv", index=False)
    descriptions = frame[["price", "accommodates", "bedrooms", "beds", "cleaned_bathrooms", "number_of_reviews"]].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    descriptions.index.name = "variable"
    descriptions.to_csv(audit / "descriptive_statistics.csv")
    raw.neighbourhood.value_counts().rename_axis("neighbourhood").reset_index(name="n").to_csv(audit / "neighbourhood_counts.csv", index=False)
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob("*.csv"))}
    _write_json({"source_csv": str(source), "source_sha256": source_hash,
        "source_unchanged_verified": True, "counts": counts,
        "identifier_namespace": ID_NAMESPACE,
        "identity_method": "Domain-separated SHA256, first 24 hex digits; deterministic pseudonyms, not anonymous.",
        "privacy": "RESTRICTED local research exports: exact locations and free text retained; direct platform IDs/URLs excluded.",
        "currency": "unresolved: do not infer or label currency until source capture is confirmed",
        "missingness": "No global imputation. Unknown bathroom text remains NaN; impute only inside training folds.",
        "structural_flags": "Plausibility review flags only. No deletions, price trimming, or target-dependent cleaning.",
        "host_proxy": "Full observed snapshot inventory; not platform-wide or strictly ex ante. Omit H block for new-host claims.",
        "amenities": "Observed list only; independent family matches, categories may overlap; scope may be truncated upstream.",
        "occupancy_counts": {k: int(v) for k,v in frame.occupancy_type.value_counts().items()},
        "property_group_counts": {k: int(v) for k,v in frame.property_group_corrected.value_counts().items()},
        "model_ready_columns": model_columns, "output_sha256": hashes}, output / "preparation_manifest.json")
    return frame


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-csv", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    arguments = cli.parse_args()
    result = prepare_data(arguments.source_csv, arguments.output_dir)
    print(json.dumps({"rows": len(result), "output_dir": str(arguments.output_dir.resolve()),
                      "bathrooms_missing": int(result.cleaned_bathrooms.isna().sum()),
                      "pool_total": int(result.has_pool_corrected.sum())}))
