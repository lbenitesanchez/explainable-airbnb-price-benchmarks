#!/usr/bin/env python3
"""Run the fixed study procedure in an explicitly selected private directory.

No input data, predictions, fitted models or execution metadata belong in the
tracked source repository. Without an execution mode this prints a plan only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_work_directory(path: Path, repository: Path = ROOT) -> Path:
    path, repository = path.expanduser().resolve(), repository.resolve()
    if path == repository or path.is_relative_to(repository) or repository.is_relative_to(path):
        raise ValueError("--work-dir must be a dedicated private directory outside the source repository and its ancestors")
    return path


def copy_identical_or_new(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"Refusing to replace a different private input: {destination}; use another work directory")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def initialize_private_work(work: Path) -> None:
    marker = work / ".airbnb-private-workspace"
    if work.exists() and any(work.iterdir()) and not marker.is_file():
        raise ValueError("Use a new empty private work directory; existing unrelated directories are not modified")
    work.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text("Private analytical workspace; not a public release directory.\n")


def primary_snapshot(work: Path) -> dict:
    """Guard this private run, without requiring the historical 845-file archive."""
    paths = [p for base in (work / "data", work / "results") for p in base.rglob("*")
             if p.is_file() and "cache" not in p.relative_to(work).parts
             and "__pycache__" not in p.parts]
    paths.append(work / "currency_metadata.json")
    if not (work / "results/run_summary.json").is_file():
        raise ValueError("A complete primary run is required before secondary diagnostics")
    return {"scope": "Private reproduction inputs and saved primary outputs, excluding disposable estimator cache",
            "files": {p.relative_to(work).as_posix(): sha256(p) for p in sorted(set(paths))}}


def record_or_check_snapshot(work: Path) -> Path:
    path = work / "baseline_primary_sha256.json"
    current = primary_snapshot(work)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("files") != current["files"]:
            raise ValueError("Private primary files changed after their integrity snapshot; use a separate work directory")
    else:
        path.write_text(json.dumps(current, indent=2) + "\n")
    return path


def command_plan(work: Path, jobs: int, skip_shap: bool) -> list[list[str]]:
    data = work / "data/model_ready_listings_restricted.csv"
    results = work / "results"
    publication = work / "publication"
    secondary = work / "publication_secondary"
    def command(script: str, *args: object) -> list[str]:
        return [sys.executable, str(ROOT / "code" / script), *map(str, args)]
    commands = [
        command("reanalyse.py", "--prepared-csv", data, "--output", results,
                "--jobs", jobs, "--xgboost", "--execute"),
        command("verify_results.py", "--results", results, "--data", data,
                "--output", publication / "primary_independent_verification.json"),
        command("summarize_results.py", "--results", results, "--data", data,
                "--output", publication),
    ]
    if not skip_shap:
        commands.append(command("explain_models.py", "--results", results, "--data", data,
                                "--output", publication))
    commands.extend([
        command("secondary_diagnostics.py", "--root", work, "--results", results,
                "--data", data, "--output", secondary,
                "--baseline", work / "baseline_primary_sha256.json"),
        command("verify_secondary.py", "--root", work, "--results", results,
                "--data", data, "--secondary", secondary,
                "--output", secondary / "independent_verification.json"),
    ])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prepared-csv", type=Path, help="Authorized corrected listing input matching the documented schema")
    source.add_argument("--source-csv", type=Path, help="Authorized original listing CSV matching prepare_data.py's schema")
    parser.add_argument("--work-dir", type=Path, required=True, help="Private output directory outside this source repository")
    parser.add_argument("--currency-metadata", type=Path, required=True, help="Private explicit PEN unit-confirmation sidecar; never inferred from prices")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--skip-shap", action="store_true", help="Skip optional explanations; all primary and secondary numerical diagnostics still run")
    parser.add_argument("--openmp-library-dir", type=Path, help="Optional macOS OpenMP library directory, passed only to child-process environment")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run the full fixed design (2916 regressor fits) and postprocessing")
    mode.add_argument("--postprocess-only", action="store_true", help="Audit and summarize complete private outputs; no new regression fits")
    mode.add_argument("--secondary-only", action="store_true", help="Recalculate and independently audit secondary diagnostics only; no model fitting")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        work = private_work_directory(args.work_dir)
    except ValueError as error:
        parser.error(str(error))
    commands = command_plan(work, args.jobs, args.skip_shap)
    if not (args.execute or args.postprocess_only or args.secondary_only):
        print(json.dumps({"plan_only": True, "estimated_regressor_fits": 2916,
                          "private_work_directory": str(work), "commands": commands}, indent=2))
        return
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(work / "tmp")
    env["JOBLIB_TEMP_FOLDER"] = str(work / "tmp")
    env["MPLCONFIGDIR"] = str(work / "matplotlib_cache")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    if args.openmp_library_dir:
        library = args.openmp_library_dir.expanduser().resolve()
        if not library.is_dir():
            parser.error("--openmp-library-dir does not exist")
        env["DYLD_LIBRARY_PATH"] = str(library) + (os.pathsep + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    initialize_private_work(work)
    (work / "tmp").mkdir(exist_ok=True)
    (work / "matplotlib_cache").mkdir(exist_ok=True)
    data = work / "data/model_ready_listings_restricted.csv"
    copy_identical_or_new(args.currency_metadata.expanduser().resolve(), work / "currency_metadata.json")
    if args.prepared_csv:
        copy_identical_or_new(args.prepared_csv.expanduser().resolve(), data)
    elif args.execute:
        if data.exists():
            raise ValueError("Raw-source preparation needs a fresh work directory; resume using --prepared-csv with the saved corrected CSV")
        subprocess.run([sys.executable, str(ROOT / "code/prepare_data.py"),
                        "--source-csv", str(args.source_csv.expanduser().resolve()),
                        "--output-dir", str(work / "data")], cwd=work, env=env, check=True)
    elif not data.is_file():
        raise FileNotFoundError("Postprocessing raw-source mode requires its previously prepared private data file")
    sys.path.insert(0, str(ROOT / "code"))
    from apply_currency_metadata import read_currency_metadata
    read_currency_metadata(data, work / "currency_metadata.json")
    # All models are generated by this local run. Never load untrusted joblib files.
    if args.execute:
        subprocess.run(commands[0], cwd=work, env=env, check=True)
    record_or_check_snapshot(work)
    start = len(commands) - 2 if args.secondary_only else 1
    for command in commands[start:]:
        subprocess.run(command, cwd=work, env=env, check=True)
    record_or_check_snapshot(work)
    print(json.dumps({"completed": True, "private_work_directory": str(work),
                      "note": "No outputs have been selected or approved for public release."}))


if __name__ == "__main__":
    main()
