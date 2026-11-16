"""Draw the three final-report figures from tracked report artifacts only.

No customer data is read and no model is fitted. Every plotted value comes from
the CSV files listed in SOURCES, which are produced by the frozen evaluation and
teammate analysis runners.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "figures" / "final"
SOURCES = {
    "comparison": "reports/model_comparison.csv",
    "policies": "reports/validation_alert_policies.csv",
    "calibration": "reports/calibration_analysis.csv",
    "orders": "reports/order_level_evaluation.csv",
}
FILES = {
    "comparison": "fig1_model_comparison.png",
    "tradeoff": "fig2_alert_policy_tradeoff.png",
    "profiles": "fig3_profile_risk_and_errors.png",
}
BLUE, LIGHT_BLUE, GREY, ORANGE, INK = "#1570EF", "#B2CCFF", "#98A2B3", "#E04F16", "#15283B"
MODEL_LABELS = {
    "always_predict_late": "Always late (reference)",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest (depth 3)",
    "decision_tree": "Depth-3 decision tree (selected)",
    "frozen_depth_3_tree": "Frozen depth-3 tree",
}
PROFILE_ORDER = ["First Class", "Second Class", "Same Day", "Standard Class"]


def _rows(name: str, root: Path) -> list[dict[str, str]]:
    with (root / SOURCES[name]).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def comparison_data(root: Path = ROOT) -> list[dict]:
    """Validation candidates and the frozen final-test result, item-weighted AP."""
    rows = _rows("comparison", root)
    keep = [r for r in rows if r["unit"] == "order_item" and (
        (r["stage"] == "validation" and r["variant"] in {"tuned_candidate", "reference"}
         and r["model"] != "depth_3_tree")
        or r["stage"] == "final_frozen_test")]
    return [{"stage": r["stage"], "model": r["model"], "label": MODEL_LABELS[r["model"]],
             "observations": int(r["observations"]),
             "average_precision": float(r["average_precision"])} for r in keep]


def tradeoff_data(root: Path = ROOT) -> dict:
    """Validation tree alert policies plus the frozen final-test operating points."""
    policies = [{"policy": r["policy_id"], "precision": float(r["precision"]),
                 "recall": float(r["recall"]), "alert_rate": float(r["alert_rate"]),
                 "profiles": r["alerted_shipping_profiles"].count("shipping_mode")}
                for r in _rows("policies", root)
                if r["model"] == "depth_3_tree" and r["precision"]]
    final = {}
    for r in _rows("comparison", root):
        if r["stage"] == "final_frozen_test" and r["model"] == "frozen_depth_3_tree":
            final["item"] = (float(r["precision"]), float(r["recall"]))
    for r in _rows("orders", root):
        if r["partition"] == "test":
            final["order"] = (float(r["precision"]), float(r["recall"]))
    return {"policies": policies, "final_test": final}


def profile_data(root: Path = ROOT) -> list[dict]:
    """Person 1's validation calibration and error table, one row per profile."""
    rows = {r["shipping_mode"]: r for r in _rows("calibration", root)}
    return [{"profile": mode,
             "days": int(rows[mode]["days_for_shipment_scheduled"]),
             "items": int(rows[mode]["items"]),
             "predicted": float(rows[mode]["predicted_late_probability"]),
             "observed": float(rows[mode]["observed_late_rate"]),
             "ci_low": float(rows[mode]["observed_rate_ci95_lower"]),
             "ci_high": float(rows[mode]["observed_rate_ci95_upper"]),
             "alert": rows[mode]["alerts"] == "True",
             "false_alerts": int(rows[mode]["false_alerts"]),
             "missed_late": int(rows[mode]["missed_late_items"])}
            for mode in PROFILE_ORDER]


def _footer(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, color="#475467")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, facecolor="white", metadata={"Software": "final_visuals"})
    plt.close(fig)


def draw_comparison(data: list[dict], path: Path) -> None:
    order = ["always_predict_late", "logistic_regression", "random_forest", "decision_tree"]
    validation = sorted((d for d in data if d["stage"] == "validation"),
                        key=lambda d: order.index(d["model"]))
    final = sorted((d for d in data if d["stage"] == "final_frozen_test"),
                   key=lambda d: d["model"] != "always_predict_late")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for axis, rows, title in [
        (axes[0], validation, f"Validation ({validation[0]['observations']:,} items)"),
        (axes[1], final, f"Final test, scored once ({final[0]['observations']:,} items)"),
    ]:
        values = [r["average_precision"] for r in rows]
        colors = [GREY if r["model"] == "always_predict_late" else
                  BLUE if r["model"] in {"decision_tree", "frozen_depth_3_tree"} else LIGHT_BLUE
                  for r in rows]
        bars = axis.barh([r["label"] for r in rows], values, color=colors)
        for bar, value in zip(bars, values):
            axis.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                      va="center", fontsize=9, color=INK)
        axis.set_xlim(0, 1)
        axis.set_title(title, fontsize=11, color=INK)
        axis.set_xlabel("Average precision (late class)")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Figure 1. Model comparison: tuned models tie on validation AP; "
                 "all beat the always-late reference", fontsize=12, color=INK, x=0.01, ha="left")
    _footer(fig, f"Source: {SOURCES['comparison']}. Item-weighted. The three tuned models rank the "
                 "four shipping profiles identically. Always-late AP equals the late-item prevalence.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _save(fig, path)


def draw_tradeoff(data: dict, path: Path) -> None:
    policies = data["policies"]
    names = {1: "First Class only", 3: "+ Same Day", 4: "+ Standard Class (= always late)"}
    fig, axis = plt.subplots(figsize=(8.5, 5))
    axis.plot([p["recall"] for p in policies], [p["precision"] for p in policies],
              color=GREY, linewidth=1.5, zorder=1)
    for p in policies:
        frozen = p["policy"] == "P2"
        axis.scatter(p["recall"], p["precision"], s=150 if frozen else 70,
                     color=BLUE if frozen else LIGHT_BLUE, edgecolor=INK, zorder=3)
        label = ("Frozen 0.50 rule: First + Second Class" if frozen
                 else names[p["profiles"]]) + f"\nalerts {p['alert_rate']:.0%} of items"
        axis.annotate(label, (p["recall"], p["precision"]), xytext=(8, 8),
                      textcoords="offset points", fontsize=8.5, color=INK,
                      fontweight="bold" if frozen else "normal")
    item, order = data["final_test"]["item"], data["final_test"]["order"]
    axis.scatter(item[1], item[0], marker="D", s=60, facecolor="white", edgecolor=ORANGE,
                 linewidth=1.8, zorder=4, label="Final test, items (reported once, not used for tuning)")
    axis.scatter(order[1], order[0], marker="s", s=60, facecolor="white", edgecolor=INK,
                 linewidth=1.5, zorder=4, label="Final test, unique orders (supplemental)")
    axis.set_xlim(0.2, 1.12)
    axis.set_ylim(0.5, 1.0)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Recall: share of late items that receive an alert")
    axis.set_ylabel("Precision: share of alerts that are late")
    axis.set_title("Figure 2. Alert-policy trade-off: each lower threshold adds a whole "
                   "shipping profile", fontsize=11, color=INK, loc="left")
    axis.legend(loc="lower left", fontsize=8, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _footer(fig, f"Sources: {SOURCES['policies']} (validation, depth-3 tree, items); "
                 f"{SOURCES['comparison']}; {SOURCES['orders']}.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def draw_profiles(data: list[dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    x = list(range(len(data)))
    observed = [d["observed"] for d in data]
    errors = [[d["observed"] - d["ci_low"] for d in data],
              [d["ci_high"] - d["observed"] for d in data]]
    axis.bar(x, observed, color=[BLUE if d["alert"] else LIGHT_BLUE for d in data], width=0.6,
             yerr=errors, capsize=4, ecolor=INK, label="Observed late rate (95% Wilson CI)")
    axis.scatter(x, [d["predicted"] for d in data], marker="_", s=900, linewidth=2.5,
                 color=ORANGE, zorder=3, label="Model late probability")
    axis.axhline(0.5, color=INK, linestyle="--", linewidth=1, label="Frozen 0.50 alert threshold")
    for i, d in enumerate(data):
        decision = "ALERT" if d["alert"] else "no alert"
        burden = (f"{d['false_alerts']:,} false alerts" if d["alert"]
                  else f"{d['missed_late']:,} missed late")
        axis.text(i, 1.02, f"{decision}\n{burden}", ha="center", fontsize=8.5, color=INK)
    axis.set_xticks(x, [f"{d['profile']}\n{d['days']} scheduled day{'' if d['days'] == 1 else 's'}"
                        f"\n{d['items']:,} items" for d in data])
    axis.set_ylim(0, 1.2)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Late-delivery rate (validation items)")
    axis.set_title("Figure 3. Validation profile risk: probabilities track observed rates; "
                   "errors concentrate", fontsize=11, color=INK, loc="left")
    axis.legend(loc="center right", fontsize=8, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _footer(fig, f"Source: {SOURCES['calibration']} (Person 1 validation analysis). "
                 "Every item in a profile gets the same score and decision.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def build_figures(root: Path = ROOT, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / FILES[key] for key in ("comparison", "tradeoff", "profiles")]
    draw_comparison(comparison_data(root), paths[0])
    draw_tradeoff(tradeoff_data(root), paths[1])
    draw_profiles(profile_data(root), paths[2])
    return paths


def main() -> None:
    for path in build_figures():
        print(f"Saved {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
