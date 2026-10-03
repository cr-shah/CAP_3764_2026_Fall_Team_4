"""Portable loading helpers for the DataCo source files."""

from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_PATH = (
    REPOSITORY_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
)
DEFAULT_DATA_DICTIONARY_PATH = (
    REPOSITORY_ROOT / "data" / "raw" / "DescriptionDataCoSupplyChain.csv"
)
RAW_DATA_ENCODING = "latin-1"


def _resolve_path(path: str | PathLike[str] | None, default: Path) -> Path:
    """Resolve an optional path, treating relative paths as repository-relative."""
    if path is None:
        return default

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def raw_data_path(path: str | PathLike[str] | None = None) -> Path:
    """Return the expected raw-data path without loading the file."""
    return _resolve_path(path, DEFAULT_RAW_DATA_PATH)


def data_dictionary_path(path: str | PathLike[str] | None = None) -> Path:
    """Return the expected data-dictionary path without loading the file."""
    return _resolve_path(path, DEFAULT_DATA_DICTIONARY_PATH)


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found at {path}. "
            "Download the DataCo files and place them in data/raw/."
        )


def load_raw_data(
    path: str | PathLike[str] | None = None,
    **read_csv_options: Any,
) -> pd.DataFrame:
    """Load the immutable DataCo raw CSV using its required Latin-1 encoding.

    Relative paths are resolved from the repository root. The caller may pass
    additional ``pandas.read_csv`` options, but the source encoding is fixed so
    that loading behaves consistently across operating systems.
    """
    source = raw_data_path(path)
    _require_file(source, "Raw DataCo dataset")

    if "encoding" in read_csv_options and read_csv_options["encoding"] != RAW_DATA_ENCODING:
        raise ValueError(
            f"The DataCo source must be read with encoding={RAW_DATA_ENCODING!r}."
        )
    read_csv_options["encoding"] = RAW_DATA_ENCODING
    return pd.read_csv(source, **read_csv_options)


def load_data_dictionary(
    path: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Load the small DataCo field-description file."""
    source = data_dictionary_path(path)
    _require_file(source, "DataCo data dictionary")
    return pd.read_csv(source, encoding=RAW_DATA_ENCODING)


def sha256_checksum(path: str | PathLike[str]) -> str:
    """Compute a file's SHA-256 checksum without modifying it."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_file_metadata(path: str | PathLike[str] | None = None) -> dict[str, str | int]:
    """Return non-sensitive integrity metadata for the raw dataset."""
    source = raw_data_path(path)
    _require_file(source, "Raw DataCo dataset")
    return {
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_checksum(source),
        "encoding": RAW_DATA_ENCODING,
    }
