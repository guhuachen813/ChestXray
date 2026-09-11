"""Create deterministic, patient-level stratified splits for Agent experiments.

The split is stratified before assigning patients to the four Agent subsets.
It must not sort patients by label and then slice the concatenated list.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_MAP = {0: 0, 1: 1, -1: 2}
SPLITS = ("model_train", "calibration", "route_validation", "model_selection")


def _allocate_counts(n: int, fractions: Iterable[float]) -> list[int]:
    """Allocate n items using largest remainders so counts sum exactly to n."""
    values = np.asarray(list(fractions), dtype=float) * n
    counts = np.floor(values).astype(int)
    remainder = int(n - counts.sum())
    if remainder:
        order = np.argsort(-(values - counts))
        counts[order[:remainder]] += 1
    return counts.tolist()


def _patient_label(series: pd.Series) -> int:
    """Return a deterministic mapped-label mode for one patient."""
    counts = series.value_counts().sort_index()
    return int(counts.index[0])


def _patient_overlap(frame: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for i, left in enumerate(SPLITS):
        left_ids = set(frame.loc[frame["agent_split"].eq(left), "Patient"].astype(str))
        for right in SPLITS[i + 1 :]:
            right_ids = set(frame.loc[frame["agent_split"].eq(right), "Patient"].astype(str))
            result[f"{left}__{right}"] = len(left_ids & right_ids)
    return result


def _split_summary(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split, group in frame.groupby("agent_split", sort=False):
        raw = group["Cardiomegaly_raw"]
        known = raw.isin([0, 1])
        summary[split] = {
            "patients": int(group["Patient"].nunique()),
            "rows": int(len(group)),
            "mapped_label_counts": {str(k): int(v) for k, v in group["Cardiomegaly"].value_counts().sort_index().items()},
            "known_binary_rows": int(known.sum()),
            "known_binary_positive": int((raw.eq(1) & known).sum()),
            "known_binary_positive_rate": float((raw[known].eq(1)).mean()) if known.any() else None,
            "raw_uncertain_rows": int(raw.eq(-1).sum()),
            "raw_missing_rows": int(raw.isna().sum()),
            "view_counts": {str(k): int(v) for k, v in group["Frontal/Lateral"].value_counts(dropna=False).items()},
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration-fraction", type=float, default=0.1)
    p.add_argument("--route-fraction", type=float, default=0.1)
    p.add_argument("--model-selection-fraction", type=float, default=0.1)
    args = p.parse_args()

    fractions = [
        1.0 - args.calibration_fraction - args.route_fraction - args.model_selection_fraction,
        args.calibration_fraction,
        args.route_fraction,
        args.model_selection_fraction,
    ]
    if any(value <= 0 for value in fractions):
        raise ValueError("All split fractions must be positive and sum to 1.")

    df = pd.read_csv(args.manifest)
    required = {"split", "Patient", "Cardiomegaly", "Frontal/Lateral"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    # Keep the original label for auditing, while exposing the U-MultiClass
    # mapping expected by the training script: negative=0, positive=1, U=2.
    df["Cardiomegaly_raw"] = df["Cardiomegaly"]
    df["Cardiomegaly"] = df["Cardiomegaly"].map(LABEL_MAP).fillna(0).astype(int)
    df["Patient"] = df["Patient"].astype(str)

    lateral_rows = int((df["Frontal/Lateral"] != "Frontal").sum())
    train = df[df["split"].eq("train") & df["Frontal/Lateral"].eq("Frontal")].copy()
    official = df[df["split"].eq("valid") & df["Frontal/Lateral"].eq("Frontal")].copy()
    if train.empty or official.empty:
        raise ValueError("Expected non-empty frontal train and valid subsets.")

    patient_labels = train.groupby("Patient")["Cardiomegaly"].agg(_patient_label)
    # Missing Cardiomegaly entries are mapped to negative under U-MultiClass.
    # They must therefore remain in the negative patient stratum rather than
    # being assigned wholesale to model_train.
    eligible = patient_labels
    unassigned_patients: set[str] = set()

    assignments: dict[str, str] = {}
    for label_value, group in eligible.groupby(eligible):
        shuffled = group.sample(frac=1.0, random_state=args.seed + 1009 * int(label_value))
        counts = _allocate_counts(len(shuffled), fractions)
        start = 0
        for split, count in zip(SPLITS, counts):
            assignments.update({str(patient): split for patient in shuffled.index[start : start + count]})
            start += count

    train["agent_split"] = train["Patient"].map(assignments)
    if train["agent_split"].isna().any():
        raise RuntimeError("Some training patients were not assigned to an Agent split.")
    official["agent_split"] = "official_valid"
    out = pd.concat([train, official], ignore_index=True)

    # Shuffle rows after assignment so CSV order cannot encode label blocks.
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_name = "cardiomegaly_agent_split.csv"
    out.to_csv(args.output_dir / out_name, index=False)

    split_summary = _split_summary(out)
    pooled_rate = float(train["Cardiomegaly"].eq(1).mean())
    split_rate_deviation = {
        split: abs(float(details["mapped_label_counts"].get("1", 0)) / details["rows"] - pooled_rate)
        for split, details in split_summary.items()
        if split in SPLITS and details["known_binary_positive_rate"] is not None
    }
    overlap_ok = all(value == 0 for value in _patient_overlap(out).values())
    prevalence_ok = all(value <= 0.015 for value in split_rate_deviation.values())
    official_overlap_ok = len(set(official["Patient"]) & set(train["Patient"])) == 0
    report = {
        "script_version": "stratified_patient_split_v2",
        "seed": args.seed,
        "fractions": dict(zip(SPLITS, fractions)),
        "output_file": out_name,
        "rows": int(len(out)),
        "lateral_rows_excluded_from_agent": lateral_rows,
        "unassigned_patients_with_no_observed_label": len(unassigned_patients),
        "pooled_train_mapped_positive_rate": pooled_rate,
        "split_mapped_positive_rate_deviation": split_rate_deviation,
        "split_summary": split_summary,
        "patient_overlap": _patient_overlap(out),
        "official_valid_patient_overlap_with_train": len(
            set(official["Patient"]) & set(train["Patient"])
        ),
        "validation": {
            "patient_overlap_ok": overlap_ok,
            "mapped_positive_rate_deviation_le_1_5pp_ok": prevalence_ok,
            "official_valid_overlap_ok": official_overlap_ok,
            "status": "PASS" if overlap_ok and prevalence_ok and official_overlap_ok else "CHECK",
        },
    }
    (args.output_dir / "agent_split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
