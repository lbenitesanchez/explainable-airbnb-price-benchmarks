#!/usr/bin/env python3
"""Apply author-confirmed price units to metadata without changing model inputs.

The frozen preparation code can be rerun unchanged. Follow it with this helper
to attach the later confirmation and update the descriptive price dictionary.
No numeric data, model, prediction, or analysis result is converted or rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_currency_metadata(data_csv: str | Path, metadata_path: str | Path | None = None) -> dict:
    """Read a local confirmation sidecar; never infer currency from numbers/text.

    Lookup is bounded to the data directory and its parent project directory.
    Explicit metadata paths are supported for portable regenerated data folders.
    Absence preserves a qualified historical declaration, not an assumed unit.
    """
    data_csv=Path(data_csv).resolve()
    candidates=([Path(metadata_path).resolve()] if metadata_path is not None else
                [data_csv.parent/"currency_metadata.json",data_csv.parent.parent/"currency_metadata.json"])
    source=next((p for p in candidates if p.is_file()),None)
    if source is None:
        if metadata_path is not None:
            raise FileNotFoundError(f"Currency confirmation sidecar is missing: {metadata_path}")
        return {"status":"historically_unresolved_no_confirmation_sidecar",
                "code":None,"name":None,"confirmed_by":None,
                "price_unit":"Recorded nightly-price units; no author-confirmation sidecar found.",
                "numeric_values_changed":False,"currency_conversion_applied":False}
    raw=json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw,dict) or raw.get("code")!="PEN" or raw.get("confirmed_by")!="Author":
        raise ValueError("Expected an explicit Author confirmation of PEN in the local currency metadata.")
    if raw.get("name")!="Peruvian sol" or not isinstance(raw.get("price_unit"),str):
        raise ValueError("Currency name/price unit do not match the author-confirmed Peruvian sol metadata.")
    if raw.get("numeric_values_changed") is not False or raw.get("currency_conversion_applied") is not False:
        raise ValueError("This metadata-only step does not support conversions or changes to numeric values.")
    return {**raw,"status":"author_confirmed",
            "metadata_file":str(source),"metadata_sha256":file_sha256(source),
            "historical_note":"Currency-unresolved statements in frozen fitting/preparation files predate the author confirmation; they remain historical provenance, not the current unit declaration."}


def attach_publication_currency(manifest: dict, currency: dict) -> dict:
    """Preserve the earlier unit declaration while attaching the current one."""
    result=dict(manifest)
    if currency.get("status")=="author_confirmed" and "price_unit" in result:
        result.setdefault("historical_price_unit_declaration",result.get("price_unit"))
    result["price_unit"]=currency["price_unit"]
    result["currency_confirmation"]=currency
    return result


def apply_currency_metadata(data_dir: str | Path, metadata_path: str | Path) -> dict:
    data_dir=Path(data_dir).resolve()
    currency=read_currency_metadata(data_dir/"model_ready_listings_restricted.csv",metadata_path)
    manifest_path=data_dir/"preparation_manifest.json"
    dictionary_path=data_dir/"variable_dictionary.csv"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    numeric_files={p:file_sha256(p) for p in data_dir.rglob("*.csv") if p!=dictionary_path}
    if not numeric_files or not (data_dir/"model_ready_listings_restricted.csv").is_file():
        raise ValueError("Expected preserved prepared model data before applying currency metadata.")
    with dictionary_path.open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    prices=[row for row in rows if row["column"]=="price"]
    if len(prices)!=1:
        raise ValueError("The variable dictionary must have exactly one price row.")
    earlier_confirmation=manifest.get("currency_confirmation",{})
    historical_definition=earlier_confirmation.get("historical_dictionary_price_definition",prices[0]["definition"])
    prices[0]["definition"]=("Posted nightly price in Peruvian soles (PEN), confirmed by the author. "
                              "Original recorded values are unchanged; no currency conversion, winsorization or transformation.")
    with dictionary_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    # Keep the original manifest['currency'] field intact for historical fidelity.
    confirmation={**currency,
        "metadata_reference":os.path.relpath(Path(currency["metadata_file"]),data_dir),
        "historical_preparation_currency_declaration":manifest.get("currency"),
        "historical_dictionary_price_definition":historical_definition,
        "verified_numeric_csvs_unchanged":True}
    manifest["currency_confirmation"]=confirmation
    manifest.setdefault("output_sha256",{})["variable_dictionary.csv"]=file_sha256(dictionary_path)
    for path,before in numeric_files.items():
        if file_sha256(path)!=before:
            raise AssertionError(f"A numeric CSV changed during metadata-only update: {path}")
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"data_dir":str(data_dir),"code":"PEN","dictionary_price_rows_updated":1,
            "numeric_csvs_verified_unchanged":len(numeric_files),
            "historical_preparation_currency_field_preserved":True,
            "metadata_sha256":currency["metadata_sha256"]}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir",type=Path,required=True)
    cli.add_argument("--metadata",type=Path,required=True)
    args=cli.parse_args()
    print(json.dumps(apply_currency_metadata(args.data_dir,args.metadata),indent=2))
