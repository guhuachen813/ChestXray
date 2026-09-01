"""Diagnose model complementarity, disagreement, and matched-coverage risk."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def risk(frame, prob_col, coverage=None):
    x = frame.copy()
    if coverage is not None:
        n = max(1, int(round(len(x) * coverage)))
        conf = np.maximum(x["p1_positive"], 1 - x["p1_positive"])
        x = x.loc[conf.nlargest(n).index]
    pred = (x[prob_col] >= 0.5).astype(int)
    return len(x), float((pred != x["label"]).mean()) if len(x) else float("nan")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    df = df[df["label"].isin([0, 1])].copy()
    df["p1_pred"] = (df["p1_positive"] >= .5).astype(int)
    df["p2_pred"] = (df["p2_positive"] >= .5).astype(int)
    df["p_fused"] = (df["p1_positive"] + df["p2_positive"]) / 2
    df["p_fused_pred"] = (df["p_fused"] >= .5).astype(int)
    df["m1_error"] = (df.p1_pred != df.label)
    df["m2_error"] = (df.p2_pred != df.label)
    df["fused_error"] = (df.p_fused_pred != df.label)
    complementarity = {
        "rows": int(len(df)),
        "model1_error_rate": float(df.m1_error.mean()),
        "model2_error_rate": float(df.m2_error.mean()),
        "both_correct": int((~df.m1_error & ~df.m2_error).sum()),
        "model1_wrong_model2_correct": int((df.m1_error & ~df.m2_error).sum()),
        "model1_correct_model2_wrong": int((~df.m1_error & df.m2_error).sum()),
        "both_wrong": int((df.m1_error & df.m2_error).sum()),
        "fused_error_rate": float(df.fused_error.mean()),
    }
    bins = [-np.inf, .05, .10, .20, .30, np.inf]
    d = df.dropna(subset=["disagreement"]).copy()
    d["disagreement_bin"] = pd.cut(d.disagreement, bins=bins, right=False)
    disagreement = d.groupby("disagreement_bin", observed=False).agg(rows=("label", "size"), error_rate=("m1_error", "mean"), fused_error_rate=("fused_error", "mean")).reset_index()
    coverage = {}
    for c in [.60, .70, .80, .90]:
        n, r = risk(df, "p1_positive", c)
        a = df[df.action == "ACCEPT"].copy()
        ar = float((a.p_fused >= .5).astype(int).ne(a.label).mean()) if len(a) else float("nan")
        coverage[str(c)] = {"n": n, "baseline_selective_risk": r, "agent_accept_coverage": float(len(a) / len(df)), "agent_accept_risk": ar}
    result = {"input": str(args.input), "complementarity": complementarity, "disagreement_bins": disagreement.to_dict(orient="records"), "matched_coverage": coverage}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
