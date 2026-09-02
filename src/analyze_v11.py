"""Coverage-space diagnosis for the v1.1 selective-classification plan.

The route-validation split selects confidence cutoffs. Official-valid only
applies those cutoffs and is never used for fitting or threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COVERAGE_POINTS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"label", "p1_positive", "p2_positive", "p1_uncertain", "entropy"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[df["label"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError(f"{path} has no known binary labels")
    df["p_fusion"] = (df["p1_positive"] + df["p2_positive"]) / 2
    df["confidence"] = np.maximum(df["p1_positive"], 1 - df["p1_positive"])
    df["p1_pred"] = (df["p1_positive"] >= .5).astype(int)
    df["p2_pred"] = (df["p2_positive"] >= .5).astype(int)
    df["fusion_pred"] = (df["p_fusion"] >= .5).astype(int)
    return df.reset_index(drop=True)


def error_rate(df: pd.DataFrame, prob: str) -> float:
    return float(((df[prob] >= .5).astype(int) != df["label"]).mean()) if len(df) else float("nan")


def top_fraction(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    n = max(1, int(round(len(df) * fraction)))
    return df.nlargest(n, "confidence", keep="first")


def curves(route: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in COVERAGE_POINTS:
        selected = top_fraction(route, target)
        cutoff = float(selected["confidence"].min())
        official_selected = official[official["confidence"] >= cutoff]
        for split, frame, observed in (("route_validation", selected, len(selected) / len(route)),
                                       ("official_valid", official_selected, len(official_selected) / len(official))):
            for method, prob in (("densenet121", "p1_positive"), ("resnet50", "p2_positive"), ("fusion", "p_fusion")):
                rows.append({"target_coverage": target, "confidence_cutoff_route": cutoff,
                             "split": split, "method": method, "rows": len(frame),
                             "observed_coverage": observed, "selective_risk": error_rate(frame, prob)})
    return pd.DataFrame(rows)


def bootstrap_difference(df: pd.DataFrame, left: str, right: str, n_boot: int, seed: int, unit: str) -> dict:
    rng = np.random.default_rng(seed)
    work = df.copy()
    if unit == "patient" and "Patient" in work.columns:
        groups = work.groupby("Patient", sort=False).indices
        keys = np.array(list(groups), dtype=object)
        samples = []
        for _ in range(n_boot):
            chosen = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([groups[k] for k in chosen])
            x = work.iloc[idx]
            samples.append(error_rate(x, left) - error_rate(x, right))
    else:
        samples = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(work), len(work))
            x = work.iloc[idx]
            samples.append(error_rate(x, left) - error_rate(x, right))
        unit = "image"
    samples = np.asarray(samples)
    point = error_rate(work, left) - error_rate(work, right)
    return {"left": left, "right": right, "point_estimate": point,
            "ci_lower": float(np.quantile(samples, .025)),
            "ci_upper": float(np.quantile(samples, .975)),
            "n_bootstrap": n_boot, "unit": unit, "rows": len(work)}


def bootstrap_risk(df: pd.DataFrame, prob: str, n_boot: int, seed: int, unit: str) -> dict:
    """Bootstrap a selective-risk estimate for a fixed accepted subset."""
    rng = np.random.default_rng(seed)
    work = df.copy()
    samples = []
    if unit == "patient" and "Patient" in work.columns:
        groups = work.groupby("Patient", sort=False).indices
        keys = np.array(list(groups), dtype=object)
        for _ in range(n_boot):
            chosen = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([groups[k] for k in chosen])
            samples.append(error_rate(work.iloc[idx], prob))
    else:
        unit = "image"
        for _ in range(n_boot):
            idx = rng.integers(0, len(work), len(work))
            samples.append(error_rate(work.iloc[idx], prob))
    samples = np.asarray(samples)
    return {"probability": prob, "point_estimate": error_rate(work, prob),
            "ci_lower": float(np.quantile(samples, .025)),
            "ci_upper": float(np.quantile(samples, .975)),
            "n_bootstrap": n_boot, "unit": unit, "rows": len(work)}


def distribution(df: pd.DataFrame, split: str) -> pd.DataFrame:
    records = []
    for col in ("confidence", "p1_positive", "p1_uncertain", "entropy", "p2_positive", "disagreement"):
        if col not in df:
            continue
        x = df[col].dropna()
        records.append({"split": split, "feature": col, "rows": len(x),
                        "mean": float(x.mean()), "std": float(x.std()),
                        "q05": float(x.quantile(.05)), "q25": float(x.quantile(.25)),
                        "median": float(x.median()), "q75": float(x.quantile(.75)),
                        "q95": float(x.quantile(.95))})
    return pd.DataFrame(records)


def plot_outputs(curve: pd.DataFrame, route: pd.DataFrame, official: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figures")
        return
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for split in ("route_validation", "official_valid"):
        for method in ("densenet121", "resnet50", "fusion"):
            x = curve[(curve.split == split) & (curve.method == method)]
            ax.plot(x.observed_coverage, x.selective_risk, marker="o", label=f"{split}:{method}")
    ax.set(xlabel="Coverage", ylabel="Selective risk", title="Coverage-risk curves")
    ax.grid(alpha=.25); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "coverage_risk.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ("confidence", "p1_positive", "p1_uncertain")):
        axes_data = [(route, "route_validation"), (official, "official_valid")]
        for frame, name in axes_data:
            ax.hist(frame[col], bins=30, alpha=.5, density=True, label=name)
        ax.set_title(col); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "confidence_distribution.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for frame, name in ((route, "route_validation"), (official, "official_valid")):
        ax.hist(frame["disagreement"], bins=30, alpha=.5, density=True, label=name)
    ax.set(title="Model disagreement distribution", xlabel="abs(p1_positive - p2_positive)"); ax.legend(); fig.tight_layout(); fig.savefig(out / "model_disagreement.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for frame, name in ((route, "route_validation"), (official, "official_valid")):
        frame = frame.copy(); frame["bin"] = pd.qcut(frame["confidence"], 10, duplicates="drop")
        grouped = frame.groupby("bin", observed=True).apply(lambda x: pd.Series({"confidence": x.confidence.mean(), "accuracy": 1 - error_rate(x, "p1_positive")}), include_groups=False)
        ax.plot(grouped.confidence, grouped.accuracy, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4); ax.set(xlabel="Mean confidence", ylabel="Accuracy", title="DenseNet reliability"); ax.legend(); fig.tight_layout(); fig.savefig(out / "reliability_densenet121.png", dpi=180); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-unit", choices=["image", "patient"], default="patient")
    args = p.parse_args()
    route, official = prepare(args.route), prepare(args.official)
    for frame in (route, official):
        frame["disagreement"] = (frame["p1_positive"] - frame["p2_positive"]).abs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve = curves(route, official)
    curve.to_csv(args.output_dir / "coverage_risk.csv", index=False)
    pd.concat([distribution(route, "route_validation"), distribution(official, "official_valid")], ignore_index=True).to_csv(args.output_dir / "confidence_distribution.csv", index=False)
    comparisons = {
        "route_validation": {
            "fusion_minus_densenet": bootstrap_difference(route, "p_fusion", "p1_positive", args.bootstrap, args.seed, args.bootstrap_unit),
            "fusion_minus_resnet": bootstrap_difference(route, "p_fusion", "p2_positive", args.bootstrap, args.seed + 1, args.bootstrap_unit),
            "densenet_minus_resnet": bootstrap_difference(route, "p1_positive", "p2_positive", args.bootstrap, args.seed + 2, args.bootstrap_unit),
        },
        "official_valid": {
            "fusion_minus_densenet": bootstrap_difference(official, "p_fusion", "p1_positive", args.bootstrap, args.seed, args.bootstrap_unit),
            "fusion_minus_resnet": bootstrap_difference(official, "p_fusion", "p2_positive", args.bootstrap, args.seed + 1, args.bootstrap_unit),
            "densenet_minus_resnet": bootstrap_difference(official, "p1_positive", "p2_positive", args.bootstrap, args.seed + 2, args.bootstrap_unit),
        },
    }
    # Primary v1.1 comparison: use the 75% cutoff selected on route-validation
    # and evaluate the corresponding accepted subsets in both splits.
    route_75 = top_fraction(route, .75)
    cutoff_75 = float(route_75["confidence"].min())
    official_75 = official[official["confidence"] >= cutoff_75]
    comparisons["matched_coverage_75"] = {
        "route_validation": {
            "fusion_minus_densenet": bootstrap_difference(route_75, "p_fusion", "p1_positive", args.bootstrap, args.seed + 10, args.bootstrap_unit),
            "fusion_minus_resnet": bootstrap_difference(route_75, "p_fusion", "p2_positive", args.bootstrap, args.seed + 11, args.bootstrap_unit),
        },
        "official_valid": {
            "fusion_minus_densenet": bootstrap_difference(official_75, "p_fusion", "p1_positive", args.bootstrap, args.seed + 12, args.bootstrap_unit),
            "fusion_minus_resnet": bootstrap_difference(official_75, "p_fusion", "p2_positive", args.bootstrap, args.seed + 13, args.bootstrap_unit),
        },
        "route_cutoff": cutoff_75,
        "route_coverage": len(route_75) / len(route),
        "official_coverage": len(official_75) / len(official),
    }
    comparisons["matched_coverage_75"]["risk_ci"] = {
        "route_validation_densenet": bootstrap_risk(route_75, "p1_positive", args.bootstrap, args.seed + 20, args.bootstrap_unit),
        "route_validation_resnet": bootstrap_risk(route_75, "p2_positive", args.bootstrap, args.seed + 21, args.bootstrap_unit),
        "route_validation_fusion": bootstrap_risk(route_75, "p_fusion", args.bootstrap, args.seed + 22, args.bootstrap_unit),
        "official_valid_densenet": bootstrap_risk(official_75, "p1_positive", args.bootstrap, args.seed + 23, args.bootstrap_unit),
        "official_valid_resnet": bootstrap_risk(official_75, "p2_positive", args.bootstrap, args.seed + 24, args.bootstrap_unit),
        "official_valid_fusion": bootstrap_risk(official_75, "p_fusion", args.bootstrap, args.seed + 25, args.bootstrap_unit),
    }
    (args.output_dir / "bootstrap_risk_ci.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    complementarity = {"route_validation": {"densenet_error": error_rate(route, "p1_positive"), "resnet_error": error_rate(route, "p2_positive"), "fusion_error": error_rate(route, "p_fusion")}, "official_valid": {"densenet_error": error_rate(official, "p1_positive"), "resnet_error": error_rate(official, "p2_positive"), "fusion_error": error_rate(official, "p_fusion")}}
    (args.output_dir / "model_complementarity.json").write_text(json.dumps(complementarity, indent=2), encoding="utf-8")
    selected = curve[(curve.split == "route_validation") & (curve.target_coverage == .75) & (curve.method == "densenet121")].iloc[0]
    official_selected = curve[(curve.split == "official_valid") & (curve.target_coverage == .75) & (curve.method == "densenet121")].iloc[0]
    (args.output_dir / "selected_operating_points.json").write_text(json.dumps({"selection_split": "route_validation", "target_coverage": .75, "method": "densenet121", "confidence_cutoff_on_route_validation": float(selected.confidence_cutoff_route), "route_observed_coverage": float(selected.observed_coverage), "route_selective_risk": float(selected.selective_risk), "official_observed_coverage": float(official_selected.observed_coverage), "official_selective_risk": float(official_selected.selective_risk)}, indent=2), encoding="utf-8")
    curve.to_json(args.output_dir / "coverage_risk.json", orient="records", indent=2)
    plot_outputs(curve, route, official, args.output_dir / "figures")
    print(json.dumps({"route_rows": len(route), "official_rows": len(official), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
