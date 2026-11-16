"""Read-only checks for the local DataCo source and project environment."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd

from .baseline_evaluation import EXPECTED_RAW_SHA256
from .data_loader import DEFAULT_RAW_DATA_PATH
from .preprocessing import DEFAULT_PROCESSED_DATA_PATH


REQUIRED_DISTRIBUTIONS = (
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
)
PROCESSED_REQUIRED_COLUMNS = frozenset({"order_id", "late_delivery_risk"})


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file without changing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_newlines(path: Path) -> str:
    """Hash bytes after replacing CRLF pairs with LF, without rewriting input."""
    digest = hashlib.sha256()
    pending = b""
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = pending + chunk
            if data.endswith(b"\r"):
                pending, data = b"\r", data[:-1]
            else:
                pending = b""
            digest.update(data.replace(b"\r\n", b"\n"))
    digest.update(pending)
    return digest.hexdigest()


def inspect_raw_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Classify raw-source integrity, including a CRLF-only mismatch."""
    source = Path(path)
    if not source.is_file():
        return {
            "path": str(source),
            "status": "missing",
            "ok": False,
            "message": "Raw DataCo CSV is missing; download it into data/raw/.",
        }

    actual = sha256_file(source)
    normalized = sha256_normalized_newlines(source)
    result: dict[str, Any] = {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": actual,
        "normalized_newline_sha256": normalized,
        "expected_sha256": expected_sha256,
    }
    if actual == expected_sha256:
        return {
            **result,
            "status": "ok",
            "ok": True,
            "message": "Raw DataCo CSV matches the canonical project source.",
        }
    if normalized == expected_sha256:
        return {
            **result,
            "status": "crlf_only_mismatch",
            "ok": False,
            "message": (
                "Raw CSV content matches after normalizing CRLF line endings. "
                "Convert the local file to LF without committing it, then rerun preflight."
            ),
        }
    return {
        **result,
        "status": "content_mismatch",
        "ok": False,
        "message": (
            "Raw CSV is a different dataset or has changed content; obtain the canonical "
            "DataCo source before running preprocessing or evaluation."
        ),
    }


def inspect_processed_file(path: Path) -> dict[str, Any]:
    """Check optional processed-data headers without loading customer rows."""
    source = Path(path)
    if not source.is_file():
        return {
            "path": str(source),
            "status": "missing_optional",
            "ok": True,
            "message": "Processed data is absent; regenerate it with python -m src.preprocessing.",
        }
    try:
        columns = set(pd.read_csv(source, nrows=0).columns)
    except Exception as error:  # pandas supplies the concrete parser failure.
        return {
            "path": str(source),
            "status": "unreadable",
            "ok": False,
            "message": f"Processed CSV headers could not be read: {error}",
        }
    missing = sorted(PROCESSED_REQUIRED_COLUMNS - columns)
    if missing:
        return {
            "path": str(source),
            "status": "invalid_schema",
            "ok": False,
            "missing_columns": missing,
            "message": f"Processed CSV is missing required columns: {missing}",
        }
    return {
        "path": str(source),
        "status": "ok",
        "ok": True,
        "column_count": len(columns),
        "message": "Processed CSV contains the required structural columns.",
    }


def inspect_runtime() -> dict[str, Any]:
    """Report Python and required distribution versions without importing them."""
    packages: dict[str, dict[str, Any]] = {}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = {"installed": False, "version": None}
        else:
            packages[distribution] = {"installed": True, "version": version}
    return {
        "python": {"installed": True, "version": platform.python_version()},
        "packages": packages,
        "ok": all(value["installed"] for value in packages.values()),
    }


def run_preflight(
    raw_path: Path = DEFAULT_RAW_DATA_PATH,
    processed_path: Path = DEFAULT_PROCESSED_DATA_PATH,
) -> dict[str, Any]:
    """Run all read-only project checks and return one machine-readable result."""
    raw_data = inspect_raw_file(Path(raw_path), EXPECTED_RAW_SHA256)
    processed_data = inspect_processed_file(Path(processed_path))
    runtime = inspect_runtime()
    return {
        "ok": all(check["ok"] for check in (raw_data, processed_data, runtime)),
        "raw_data": raw_data,
        "processed_data": processed_data,
        "runtime": runtime,
    }


def main() -> int:
    """Print preflight JSON and return a shell-friendly success status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_DATA_PATH)
    arguments = parser.parse_args()
    result = run_preflight(arguments.raw_path, arguments.processed_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
