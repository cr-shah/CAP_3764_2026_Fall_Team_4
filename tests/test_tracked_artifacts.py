"""Static consistency checks for tracked reproducibility artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import REPOSITORY_ROOT
from src.project_preflight import EXPECTED_PYTHON_VERSION, EXPECTED_RUNTIME_VERSIONS


def test_preflight_expected_versions_match_environment_file() -> None:
    declared = {}
    for line in (REPOSITORY_ROOT / "environment.yml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and "=" in stripped:
            name, version = stripped[2:].split("=", maxsplit=1)
            declared[name] = version

    assert declared["python"] == EXPECTED_PYTHON_VERSION
    assert {name: declared[name] for name in EXPECTED_RUNTIME_VERSIONS} == (
        EXPECTED_RUNTIME_VERSIONS
    )


def test_order_level_report_runtime_matches_declared_environment() -> None:
    report = json.loads(
        (REPOSITORY_ROOT / "reports/order_level_evaluation.json").read_text(encoding="utf-8")
    )

    assert report["runtime"] == {
        "python": EXPECTED_PYTHON_VERSION,
        "pandas": EXPECTED_RUNTIME_VERSIONS["pandas"],
        "numpy": EXPECTED_RUNTIME_VERSIONS["numpy"],
        "scikit_learn": EXPECTED_RUNTIME_VERSIONS["scikit-learn"],
    }


def test_order_level_json_and_csv_results_agree() -> None:
    report = json.loads(
        (REPOSITORY_ROOT / "reports/order_level_evaluation.json").read_text(encoding="utf-8")
    )
    table = pd.read_csv(REPOSITORY_ROOT / "reports/order_level_evaluation.csv")

    assert table["partition"].tolist() == [row["partition"] for row in report["results"]]
    for index, row in enumerate(report["results"]):
        for field, value in row.items():
            actual = table.loc[index, field]
            if isinstance(value, float):
                assert np.isclose(actual, value, rtol=0, atol=1e-12)
            else:
                assert actual == value


def test_numerical_notebook_has_no_volatile_execution_timestamps() -> None:
    notebook = json.loads(
        (REPOSITORY_ROOT / "notebooks/02_numerical_eda.ipynb").read_text(encoding="utf-8")
    )

    for cell in notebook["cells"]:
        assert "execution" not in cell.get("metadata", {})
