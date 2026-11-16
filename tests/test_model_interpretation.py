"""Verify interpreted evidence against tracked sources without customer data."""

import csv
import io
import json
import shutil

import pytest

from src.model_interpretation import (
    ROOT, SOURCES, build_report, comparison_csv, interpretation_markdown, write_figure,
)


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def resolve(path, pointer):
    value = load(path)
    for key in pointer.strip("/").split("/"):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def test_every_comparison_metric_has_matching_source():
    report = build_report()
    assert len(report["rows"]) == 12
    for row in report["rows"]:
        source = resolve(row["source"], row["source_pointer"])
        ap_key = ("selected_cv_average_precision" if row["stage"] == "inner_training_cv"
                  else "validation_average_precision" if row["stage"] == "validation"
                  else "average_precision")
        assert row["average_precision"] == source[ap_key]
        if row["operating_source"] is None:
            assert all(row[key] is None for key in (
                "precision", "recall", "threshold", "true_alerts", "false_alerts",
                "missed_late", "correct_non_alerts", "total_alerts", "alert_rate"))
            continue
        outcome = resolve(row["operating_source"], row["operating_pointer"])
        for key in ("precision", "recall", "true_alerts", "false_alerts",
                    "correct_non_alerts", "total_alerts", "alert_rate"):
            assert row[key] == outcome[key]
        missed = outcome.get("missed_late_items", outcome.get("missed_late_orders"))
        assert row["missed_late"] == missed
        tp, fp, fn, tn = (row[key] for key in (
            "true_alerts", "false_alerts", "missed_late", "correct_non_alerts"))
        assert tp + fp + fn + tn == row["observations"]
        assert row["precision"] == pytest.approx(tp / (tp + fp))
        assert row["recall"] == pytest.approx(tp / (tp + fn))
        assert row["alert_rate"] == pytest.approx((tp + fp) / row["observations"])
        assert row["threshold"] == report["frozen_configuration"]["alert_threshold"]


def test_stages_ties_and_missing_results_are_honest():
    rows = build_report()["rows"]
    inner = [r for r in rows if r["stage"] == "inner_training_cv"]
    assert {r["model"] for r in inner} == {"logistic_regression", "decision_tree", "random_forest"}
    assert all(r["partition"] == "train_expanding_folds" and r["observations"] is None for r in inner)
    validation = [r for r in rows if r["stage"] == "validation" and r["model"] != "always_predict_late"]
    assert len({r["average_precision"] for r in validation}) == 1
    assert next(r for r in validation if r["model"] == "random_forest")["precision"] is None
    assert {r["model"] for r in rows if r["stage"] == "final_frozen_test"} == {
        "always_predict_late", "frozen_depth_3_tree"}
    assert all(r["variant"] == "post_hoc_reporting_only" for r in rows if r["unit"] == "unique_order")


def test_profiles_and_logistic_threshold_match_tracked_evidence():
    report = build_report()
    profiles = report["shipping_profiles"]
    assert [p["shipping_mode"] for p in profiles] == ["First Class", "Second Class", "Same Day", "Standard Class"]
    assert [p["alerts_at_frozen_threshold"] for p in profiles] == [True, True, False, False]
    candidate_profiles = load(SOURCES[2])["training_shipping_profiles_and_candidate_probabilities"]
    for model in ("decision_tree", "logistic_regression", "random_forest"):
        assert [p["shipping_mode"] for p in sorted(candidate_profiles, key=lambda p: p[model], reverse=True)] == [
            p["shipping_mode"] for p in profiles]
    config = report["frozen_configuration"]
    assert {key: config["estimator_parameters"][key] for key in (
        "max_depth", "min_samples_leaf", "random_state")} == {
            "max_depth": 3, "min_samples_leaf": 100, "random_state": 42}
    assert config["alert_threshold"] == .5
    assert load(SOURCES[1])["model"]["leaves"] == 4
    for profile in profiles:
        matching = next(p for p in candidate_profiles if p["shipping_mode"] == profile["shipping_mode"])
        assert matching["decision_tree"] == profile["late_probability"]
        assert matching["days_for_shipment_scheduled"] == profile["days_for_shipment_scheduled"]
    logistic = next(r for r in report["rows"] if r["stage"] == "validation" and r["model"] == "logistic_regression")
    policy = resolve(logistic["operating_source"], logistic["operating_pointer"])
    assert policy["policy_id"] == "P2"
    assert policy["threshold_interval"]["lower_exclusive"] < logistic["threshold"] <= policy["threshold_interval"]["upper_inclusive"]


def test_generated_tables_and_prose_are_current():
    report = build_report()
    assert load("reports/model_comparison.json") == report
    assert (ROOT / "reports/model_comparison.csv").read_text(encoding="utf-8") == comparison_csv(report)
    assert (ROOT / "docs/MODEL_INTERPRETATION.md").read_text(encoding="utf-8") == interpretation_markdown(report)
    csv_rows = list(csv.DictReader(io.StringIO(comparison_csv(report))))
    for original, exported in zip(report["rows"], csv_rows):
        for key, value in original.items():
            assert exported[key] == ("" if value is None else str(value))


@pytest.mark.parametrize("json_path,csv_path,pointer", [
    (SOURCES[2], "reports/automl_leaderboard.csv", "/validation_leaderboard"),
    (SOURCES[4], "reports/final_evaluation.csv", "/results"),
    (SOURCES[5], "reports/order_level_evaluation.csv", "/results"),
])
def test_paired_source_tables_agree(json_path, csv_path, pointer):
    originals = resolve(json_path, pointer)
    with (ROOT / csv_path).open(encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert len(originals) == len(exported)
    for original, row in zip(originals, exported):
        for key, value in original.items():
            actual = row[key]
            if isinstance(value, float):
                assert float(actual) == pytest.approx(value, rel=0, abs=1e-12)
            elif isinstance(value, (dict, list)):
                assert json.loads(actual) == value
            else:
                assert actual == ("" if value is None else str(value))


def test_alert_policy_csv_agrees_with_json():
    models = load(SOURCES[3])["models"]
    with (ROOT / "reports/validation_alert_policies.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == sum(len(model["policies"]) for model in models.values())
    for row in rows:
        original = next(p for p in models[row["model"]]["policies"] if p["policy_id"] == row["policy_id"])
        for key, value in original.items():
            if isinstance(value, float):
                assert float(row[key]) == pytest.approx(value, rel=0, abs=1e-12)
            elif isinstance(value, (dict, list)):
                assert json.loads(row[key]) == value
            else:
                assert row[key] == ("" if value is None else str(value))


def test_report_needs_only_tracked_json_and_preserves_sources(tmp_path):
    for source in SOURCES:
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    before = {path: (tmp_path / path).read_bytes() for path in SOURCES}
    report = build_report(tmp_path)
    assert report == build_report()
    assert before == {path: (tmp_path / path).read_bytes() for path in SOURCES}


def test_figure_annotations_use_correct_counts_and_units(tmp_path, monkeypatch):
    import matplotlib.figure

    captured = {}
    def capture(figure, path, **kwargs):
        captured["figure"] = figure
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    write_figure(build_report(), tmp_path / "figure.png")
    left, right = captured["figure"].axes
    assert left.images[0].get_array().tolist() == [[6747, 5743], [1223, 9065]]
    assert right.images[0].get_array().tolist() == [[3883, 3345], [698, 5225]]
    assert "22,778 items (primary)" in left.get_title()
    assert "13,151 unique orders (supplemental)" in right.get_title()
    assert [tick.get_text() for tick in left.get_xticklabels()] == ["Alert", "No alert"]
    assert [tick.get_text() for tick in left.get_yticklabels()] == ["Late", "Not late"]
