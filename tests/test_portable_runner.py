"""Path, immutability and command-plan guards; no regression pipeline run."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portable_runner", ROOT / "run_reproduction.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_private_work_rejects_repository_children_and_ancestors(tmp_path):
    repo = tmp_path / "repository"
    for path in (repo, repo / "private", tmp_path):
        with pytest.raises(ValueError):
            runner.private_work_directory(path, repo)
    assert runner.private_work_directory(tmp_path / "private-run", repo) == tmp_path / "private-run"


def test_existing_unrelated_directory_is_not_modified(tmp_path):
    original = tmp_path / "existing.txt"
    original.write_text("do not replace")
    with pytest.raises(ValueError, match="unrelated"):
        runner.initialize_private_work(tmp_path)
    assert original.read_text() == "do not replace"
    assert not (tmp_path / ".airbnb-private-workspace").exists()


def test_private_input_copy_refuses_different_existing_bytes(tmp_path):
    source, dest = tmp_path / "source.csv", tmp_path / "work/input.csv"
    source.write_text("price\n100\n")
    runner.copy_identical_or_new(source, dest)
    runner.copy_identical_or_new(source, dest)
    source.write_text("price\n101\n")
    with pytest.raises(ValueError, match="different private input"):
        runner.copy_identical_or_new(source, dest)
    assert dest.read_text() == "price\n100\n"


def test_run_integrity_snapshot_detects_changed_or_added_outputs(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "results/cache").mkdir(parents=True)
    (tmp_path / "data/input.csv").write_text("price\n100\n")
    (tmp_path / "currency_metadata.json").write_text("{}")
    (tmp_path / "results/run_summary.json").write_text("{}")
    cache = tmp_path / "results/cache/disposable"
    cache.write_text("first")
    baseline = runner.record_or_check_snapshot(tmp_path)
    assert len(json.loads(baseline.read_text())["files"]) == 3
    cache.write_text("second")
    runner.record_or_check_snapshot(tmp_path)
    (tmp_path / "results/run_summary.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        runner.record_or_check_snapshot(tmp_path)


def test_command_plan_keeps_all_generated_paths_in_explicit_private_work(tmp_path):
    work = tmp_path / "private-run"
    commands = runner.command_plan(work, 2, False)
    assert len(commands) == 6
    assert "--xgboost" in commands[0] and "--execute" in commands[0]
    for command in commands:
        for flag in ("--output", "--results", "--data", "--baseline", "--root", "--secondary"):
            if flag in command:
                path = Path(command[command.index(flag) + 1])
                assert path == work or path.is_relative_to(work)
    assert not any("explain_models.py" in item for command in runner.command_plan(work, 2, True) for item in command)
