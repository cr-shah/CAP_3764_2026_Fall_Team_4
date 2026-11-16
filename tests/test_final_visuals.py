from src import final_visuals as fv


def test_plotted_values_match_tracked_reports():
    comparison = {(d["stage"], d["model"]): d["average_precision"] for d in fv.comparison_data()}
    assert round(comparison[("final_frozen_test", "frozen_depth_3_tree")], 6) == 0.749683
    tied = {comparison[("validation", m)] for m in
            ("logistic_regression", "random_forest", "decision_tree")}
    assert len(tied) == 1 and round(tied.pop(), 6) == 0.742

    tradeoff = fv.tradeoff_data()
    frozen = next(p for p in tradeoff["policies"] if p["policy"] == "P2")
    assert (round(frozen["precision"], 4), round(frozen["recall"], 4)) == (0.8419, 0.5314)
    assert round(tradeoff["final_test"]["order"][1], 4) == 0.5372

    profiles = fv.profile_data()
    by_name = {d["profile"]: d for d in profiles}
    assert by_name["Standard Class"]["missed_late"] == 9121
    assert by_name["Second Class"]["false_alerts"] == 1824
    assert [d["alert"] for d in profiles] == [True, True, False, False]


def test_build_figures_writes_three_pngs(tmp_path):
    paths = fv.build_figures(output_dir=tmp_path)
    assert [p.name for p in paths] == list(fv.FILES.values())
    assert all(p.stat().st_size > 10_000 for p in paths)
