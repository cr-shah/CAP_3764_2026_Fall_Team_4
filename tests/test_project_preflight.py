"""Tests for read-only local project preflight checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.project_preflight import (
    EXPECTED_PYTHON_VERSION,
    EXPECTED_RUNTIME_VERSIONS,
    inspect_processed_file,
    inspect_raw_file,
    inspect_runtime,
    main,
    run_preflight,
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_raw_file_accepts_exact_canonical_bytes_without_rewriting(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    content = b"a,b\n1,2\n"
    source.write_bytes(content)

    result = inspect_raw_file(source, sha256(content))

    assert result["status"] == "ok"
    assert result["sha256"] == sha256(content)
    assert result["normalized_newline_sha256"] == sha256(content)
    assert result["ok"] is True
    assert source.read_bytes() == content


def test_raw_file_identifies_crlf_only_mismatch_without_rewriting(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    canonical = b"a,b\n1,2\n"
    crlf = b"a,b\r\n1,2\r\n"
    source.write_bytes(crlf)

    result = inspect_raw_file(source, sha256(canonical))

    assert result["status"] == "crlf_only_mismatch"
    assert result["sha256"] == sha256(crlf)
    assert result["normalized_newline_sha256"] == sha256(canonical)
    assert result["ok"] is False
    assert "line endings" in result["message"].lower()
    assert source.read_bytes() == crlf


def test_raw_file_rejects_content_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    source.write_bytes(b"wrong data\n")

    result = inspect_raw_file(source, sha256(b"expected data\n"))

    assert result["status"] == "content_mismatch"
    assert result["ok"] is False
    assert "different dataset" in result["message"].lower()


def test_raw_file_reports_missing_path(tmp_path: Path) -> None:
    source = tmp_path / "missing.csv"

    result = inspect_raw_file(source, sha256(b"expected\n"))

    assert result["status"] == "missing"
    assert result["ok"] is False
    assert result["path"] == str(source)


def test_processed_file_is_optional_when_absent(tmp_path: Path) -> None:
    result = inspect_processed_file(tmp_path / "missing.csv")

    assert result == {
        "path": str(tmp_path / "missing.csv"),
        "status": "missing_optional",
        "ok": True,
        "message": "Processed data is absent; regenerate it with python -m src.preprocessing.",
    }


def test_processed_file_requires_order_id_and_target_headers(tmp_path: Path) -> None:
    source = tmp_path / "processed.csv"
    pd.DataFrame({"order_id": [1], "other": [0]}).to_csv(source, index=False)

    result = inspect_processed_file(source)

    assert result["status"] == "invalid_schema"
    assert result["ok"] is False
    assert result["missing_columns"] == ["late_delivery_risk"]


def test_runtime_reports_every_required_package_and_version(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.project_preflight.platform.python_version", lambda: EXPECTED_PYTHON_VERSION
    )
    monkeypatch.setattr(
        "src.project_preflight.metadata.version",
        lambda distribution: EXPECTED_RUNTIME_VERSIONS[distribution],
    )
    monkeypatch.setattr("src.project_preflight.import_module", lambda name: object())

    result = inspect_runtime()

    assert result["ok"] is True
    assert result["python"]["installed"] is True
    assert result["python"]["version"]
    for package in [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "scipy",
        "pytest",
        "jupyterlab",
        "ipykernel",
        "plotly",
    ]:
        assert package in result["packages"]
        assert set(result["packages"][package]) == {
            "installed",
            "version",
            "expected_version",
            "matches_expected",
            "importable",
            "import_error",
        }


def test_runtime_requires_declared_versions_and_successful_imports(monkeypatch) -> None:
    monkeypatch.setattr("src.project_preflight.platform.python_version", lambda: "3.11.18")
    monkeypatch.setattr(
        "src.project_preflight.metadata.version",
        lambda distribution: {
            "pandas": "2.2.3",
            "numpy": "2.1.3",
            "scikit-learn": "1.5.2",
            "matplotlib": "3.9.4",
            "seaborn": "0.13.2",
            "scipy": "1.17.1",
            "pytest": "8.3.5",
            "jupyterlab": "4.2.7",
            "ipykernel": "6.29.5",
            "plotly": "5.24.1",
        }[distribution],
    )
    monkeypatch.setattr("src.project_preflight.import_module", lambda name: object())

    assert inspect_runtime()["ok"] is True

    monkeypatch.setattr("src.project_preflight.metadata.version", lambda distribution: "0.0.0")
    mismatched = inspect_runtime()
    assert mismatched["ok"] is False
    assert mismatched["packages"]["pandas"]["matches_expected"] is False

    def broken_import(name: str) -> object:
        if name == "pandas":
            raise ImportError("broken binary")
        return object()

    monkeypatch.setattr("src.project_preflight.import_module", broken_import)
    unimportable = inspect_runtime()
    assert unimportable["ok"] is False
    assert unimportable["packages"]["pandas"]["importable"] is False
    assert "broken binary" in unimportable["packages"]["pandas"]["import_error"]


def test_run_preflight_combines_check_statuses(monkeypatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_bytes(b"canonical\n")
    processed = tmp_path / "processed.csv"
    pd.DataFrame({"order_id": [1], "late_delivery_risk": [0]}).to_csv(
        processed, index=False
    )
    monkeypatch.setattr("src.project_preflight.EXPECTED_RAW_SHA256", sha256(b"canonical\n"))
    monkeypatch.setattr(
        "src.project_preflight.inspect_runtime",
        lambda: {"python": {}, "packages": {}, "ok": True},
    )

    result = run_preflight(raw, processed)

    assert result["ok"] is True
    assert result["raw_data"]["status"] == "ok"
    assert result["processed_data"]["status"] == "ok"


def test_run_preflight_none_paths_use_project_defaults(monkeypatch) -> None:
    seen = {}

    def inspect_raw(path: Path, expected: str) -> dict:
        seen["raw"] = path
        return {"ok": True}

    def inspect_processed(path: Path) -> dict:
        seen["processed"] = path
        return {"ok": True}

    monkeypatch.setattr("src.project_preflight.inspect_raw_file", inspect_raw)
    monkeypatch.setattr("src.project_preflight.inspect_processed_file", inspect_processed)
    monkeypatch.setattr(
        "src.project_preflight.inspect_runtime",
        lambda: {"python": {}, "packages": {}, "ok": True},
    )

    run_preflight(None, None)

    assert seen["raw"] == Path("data/raw/DataCoSupplyChainDataset.csv").resolve()
    assert seen["processed"] == Path("data/processed/dataco_clean.csv").resolve()


def test_cli_returns_nonzero_and_prints_json_for_failed_preflight(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "src.project_preflight.run_preflight",
        lambda raw_path, processed_path: {
            "ok": False,
            "raw_data": {"status": "missing"},
            "processed_data": {"status": "missing_optional"},
            "runtime": {"ok": True},
        },
    )
    monkeypatch.setattr("sys.argv", ["project_preflight"])

    exit_code = main()

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["raw_data"]["status"] == "missing"
