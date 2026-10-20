#!/usr/bin/env python3
"""Independent behavioral checks and aggregate-only text leakage screening.

Run checks: python tests_data.py
Also audit source text: python tests_data.py --source /path/listings.csv
    --audit-output /path/audit_tables/text_target_leakage_screening.json

The text screen writes counts only, never listing text, IDs, URLs, or examples.
Currency expressions in listing descriptions do not establish the currency used
for the target, and numerical matches are screening indicators, not proof of
outcome leakage. Actual source files are read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import (
    AMENITY_COLUMNS, DIRECT_IDENTIFIERS, DISTANCE_COLUMNS, REVIEW_COLUMNS,
    amenity_features, occupancy_from_type, parse_bathrooms, prepare_data,
)


def synthetic_source() -> pd.DataFrame:
    """Small independently specified cases with distinct known meanings."""
    records = []
    for i, (bath, old, typ, amenities, host) in enumerate([
        ("2,5 baños", 1, "Alojamiento entero: suite de invitados",
         "['Piscina cubierta privada', 'Wifi']", "host-A"),
        ("-", 0, "Habitación privada", "['No disponible: Piscina', 'Cocina']", "host-A"),
        ("No hay baño", 1, "Todo el lugar", "['TV']", "host-B"),
        ("Medio baño compartido", 1, "Casa particular", "[]", "host-C"),
    ]):
        record = {
            "listings_id": f"listing-{i}", "host_id": host,
            "listing_url": f"https://example.invalid/room/{i}",
            "host_url": f"https://example.invalid/host/{host}",
            "bathrooms_text": bath, "bathrooms": old, "property_type": typ,
            "ammenities": amenities, "neighbourhood": "area-A" if i < 2 else "area-B",
            "price": [34.0, 110.5, 2100.0, 99.25][i],
            "accommodates": 2, "bedrooms": [1, 1, 22, 1][i], "beds": 1,
            "latitude": -12.0, "longitude": -77.0,
            "name": "Synthetic title", "description": "Synthetic description",
        }
        record.update({column: 4.5 for column in REVIEW_COLUMNS})
        record["number_of_reviews"] = 3
        record.update({column: 100.0 for column in DISTANCE_COLUMNS})
        records.append(record)
    return pd.DataFrame(records)


class BathroomMeaningTests(unittest.TestCase):
    def test_explicit_counts_decimals_half_and_zero(self):
        for text, expected in [
            ("0 baños", 0.0), ("No hay baño", 0.0), ("1 baño privado", 1.0),
            ("2,5 baños", 2.5), ("3.5 baños", 3.5),
            ("Medio baño", 0.5), ("Medio baño compartido", 0.5),
            ("2\u00a0baños compartidos", 2.0), ("3,5\\xa0baños compartidos", 3.5),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_bathrooms(text), expected)

    def test_unknown_is_not_a_zero_bathroom(self):
        for text in ["-", "", None, np.nan, "texto sin cantidad"]:
            with self.subTest(text=text):
                self.assertTrue(math.isnan(parse_bathrooms(text)))
        self.assertFalse(math.isnan(parse_bathrooms("No hay baño")))


class AmenityMeaningTests(unittest.TestCase):
    def test_private_luxury_pool_is_also_a_pool(self):
        features = amenity_features("['Piscina cubierta privada', 'Wifi']")
        self.assertEqual(features["has_pool_corrected"], 1)
        self.assertEqual(features["luxury_amenities_count"], 1)
        self.assertEqual(features["available_amenities_count"], 2)

    def test_unavailable_pool_is_excluded(self):
        features = amenity_features("['No disponible: Piscina', 'Cocina']")
        self.assertEqual(features["has_pool_corrected"], 0)
        self.assertEqual(features["available_amenities_count"], 1)

    def test_normalized_spelling_and_bad_input_do_not_create_false_zero(self):
        self.assertEqual(amenity_features("['PISCINA cubierta privada']")["has_pool_corrected"], 1)
        features = amenity_features("not a list")
        self.assertEqual(features["amenities_parse_invalid"], 1)
        self.assertTrue(all(math.isnan(features[c]) for c in AMENITY_COLUMNS))


class OccupancyMeaningTests(unittest.TestCase):
    def test_whole_guest_units_are_not_room_offers(self):
        for text in ["Alojamiento entero: suite de invitados",
                     "Alojamiento entero: casa de huéspedes", "Todo el lugar"]:
            self.assertEqual(occupancy_from_type(text), "whole")
        self.assertEqual(occupancy_from_type("Habitación privada"), "room")
        self.assertEqual(occupancy_from_type("Habitación"), "room")
        self.assertEqual(occupancy_from_type("Casa particular"), "other")


class ConservationAndIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="airbnb-preparation-tests-")
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.csv"
        self.raw = synthetic_source()
        self.raw.to_csv(self.source, index=False)
        self.original_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_preparation_preserves_prices_rows_and_raw_identifiers(self):
        prepared = prepare_data(self.source, self.root / "prepared")
        self.assertEqual(len(prepared), len(self.raw))
        self.assertEqual(prepared.price.tolist(), self.raw.price.tolist())
        self.assertEqual(prepared.listings_id.tolist(), self.raw.listings_id.tolist())
        self.assertEqual(prepared.host_id.tolist(), self.raw.host_id.tolist())
        self.assertEqual(prepared.cleaned_bathrooms.iloc[0], 2.5)
        self.assertTrue(pd.isna(prepared.cleaned_bathrooms.iloc[1]))
        self.assertEqual(prepared.cleaned_bathrooms.iloc[2], 0.0)
        self.assertEqual(prepared.cleaned_bathrooms.iloc[3], 0.5)
        self.assertEqual(prepared.loc[2, "bedrooms"], 22)
        self.assertEqual(prepared.loc[2, "flag_bedrooms_gt_capacity"], 1)
        self.assertEqual(prepared.host_portfolio_size_snapshot.tolist(), [2, 2, 1, 1])
        exported = pd.read_csv(self.root / "prepared" / "cleaned_listings_restricted.csv")
        self.assertFalse(set(DIRECT_IDENTIFIERS) & set(exported.columns))
        self.assertEqual(exported.price.tolist(), self.raw.price.tolist())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_hash)

    def test_identity_and_grouping_are_stable_under_reordering(self):
        first = prepare_data(self.source, self.root / "first")
        shuffled_source = self.root / "shuffled.csv"
        self.raw.sample(frac=1, random_state=19).to_csv(shuffled_source, index=False)
        second = prepare_data(shuffled_source, self.root / "second")
        for field in ["row_id", "host_group", "host_portfolio_size_snapshot", "price"]:
            a = first.set_index("listings_id")[field].sort_index()
            b = second.set_index("listings_id")[field].sort_index()
            pd.testing.assert_series_equal(a, b)
        self.assertEqual(first.row_id.nunique(), len(self.raw))
        self.assertEqual(first.host_group.nunique(), self.raw.host_id.nunique())

    def test_duplicate_ids_fail_without_silent_deletion(self):
        duplicate = pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True)
        duplicate.to_csv(self.source, index=False)
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "unique nonmissing listing IDs"):
            prepare_data(self.source, self.root / "duplicate-output")
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse((self.root / "duplicate-output").exists())

    def test_invalid_prices_fail_instead_of_trimming_the_sample(self):
        self.raw.loc[0, "price"] = 0
        self.raw.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "Nonpositive/nonfinite prices"):
            prepare_data(self.source, self.root / "invalid-price-output")


# Fixed, target-independent screen; it does not learn vocabulary from outcomes.
CURRENCY_MARKER = re.compile(r"(?i)(?<!\w)(?:s\s*/\.?|pen\b|usd\b|us\s*\$)|\$|\b(?:sol(?:es)?|d[oó]lar(?:es)?)\b")
AMOUNT_AFTER_MARKER = re.compile(r"(?i)(?:s\s*/\.?\s*|\bpen\s*|\busd\s*|us\s*\$\s*|\$\s*)(\d+(?:[.,]\d+)*)")
AMOUNT_BEFORE_MARKER = re.compile(r"(?i)(\d+(?:[.,]\d+)*)\s*(?:sol(?:es)?\b|d[oó]lar(?:es)?\b|pen\b|usd\b)")
PRICING_LANGUAGE = re.compile(r"(?i)\b(?:precio(?:s)?|tarifa(?:s)?|costo(?:s)?|nightly|mensual)\b|\b(?:por|per|x)\s+(?:noche|night|d[ií]a|mes|month)\b")
NIGHTLY_LANGUAGE = re.compile(r"(?i)\b(?:por|per|x)\s+(?:noche|night)\b|\bnightly\b")


def _amount_candidates(token: str) -> set[float]:
    """Keep decimal/thousands ambiguity rather than guessing a currency locale."""
    candidates = set()
    for value in {token, token.replace(",", ""), token.replace(".", ""),
                  token.replace(",", ".")}:
        try:
            candidates.add(float(value))
        except ValueError:
            pass
    return candidates


def screen_text_target(source: str | Path, output: str | Path) -> dict:
    source, output = Path(source), Path(output)
    frame = pd.read_csv(source, usecols=["name", "description", "price"])
    rows = {}
    aggregate = {}
    for column in ["name", "description", "combined"]:
        text = ((frame.name.fillna("") + " " + frame.description.fillna(""))
                if column == "combined" else frame[column].fillna("").astype(str))
        markers = text.map(lambda x: bool(CURRENCY_MARKER.search(x)))
        amounts = text.map(lambda x: [*AMOUNT_AFTER_MARKER.findall(x), *AMOUNT_BEFORE_MARKER.findall(x)])
        with_amount = amounts.map(bool)
        pricing = text.map(lambda x: bool(PRICING_LANGUAGE.search(x)))
        nightly = text.map(lambda x: bool(NIGHTLY_LANGUAGE.search(x)))
        matches = [any(np.isclose(candidate, target, rtol=1e-8, atol=1e-8)
                       for token in found for candidate in _amount_candidates(token))
                   for found, target in zip(amounts, frame.price)]
        aggregate[column] = {
            "rows_with_currency_marker": int(markers.sum()),
            "rows_with_monetary_amount_expression": int(with_amount.sum()),
            "rows_with_pricing_language": int(pricing.sum()),
            "rows_with_monetary_amount_and_nightly_language": int((with_amount & nightly).sum()),
            "rows_with_amount_numeric_match_to_target_any_locale": int(sum(matches)),
        }
        rows[column] = int(len(text))
    result = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n_rows": int(len(frame)), "counts": aggregate,
        "screen": "Fixed regex counts; each category counts rows, categories may overlap. No listing text, IDs, or examples exported.",
        "numeric_match_rule": "Any plausible decimal/thousands parse numerically equal to the same row's price; no FX conversion and no currency assumptions.",
        "interpretation": "Monetary expressions can be legitimate predictors or quoted ancillary charges. Numeric matches flag possible target echo, not proven leakage. Absence of regex matches does not prove absence of all price information.",
        "currency": "Listing text mentions do not identify the capture/display currency of the numeric price target. Confirm upstream metadata separately.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path)
    cli.add_argument("--audit-output", type=Path)
    options = cli.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    if options.source:
        destination = options.audit_output or (Path(__file__).resolve().parent.parent / "data" / "audit_tables" / "text_target_leakage_screening.json")
        print(json.dumps(screen_text_target(options.source, destination), ensure_ascii=False))
