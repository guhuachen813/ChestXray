"""Create a deterministic patient-level train/validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--label-policy", choices=["clean", "u-ignore"], default="clean")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    label_column = "Cardiomegaly"
    if args.label_policy == "clean":
        clean = frame[frame[label_column].isin([0, 1])].copy()
    else:
        # U-Ignore masks explicit uncertain labels (-1), while an unmentioned
        # finding (blank/NaN) is treated as negative, matching CheXpert's 0/1/u setup.
        clean = frame[frame[label_column].ne(-1)].copy()
        clean[label_column] = clean[label_column].fillna(0).astype(int)
    train = clean[clean["split"].eq("train")].copy()
    official_valid = clean[clean["split"].eq("valid")].copy()

    # Split patients while preserving the positive-patient proportion.
    patient_labels = train.groupby("Patient")[label_column].max()
    val_patients = set()
    for label_value, group in patient_labels.groupby(patient_labels):
        shuffled = group.sample(frac=1.0, random_state=args.seed + int(label_value))
        val_count = max(1, round(len(shuffled) * args.val_fraction))
        val_patients.update(shuffled.iloc[:val_count].index.astype(str))
    train["internal_split"] = train["Patient"].astype(str).map(lambda p: "internal_val" if p in val_patients else "internal_train")
    official_valid["internal_split"] = "official_valid"
    result = pd.concat([train, official_valid], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or ("cardiomegaly_clean_patient_split.csv" if args.label_policy == "clean" else "cardiomegaly_u_ignore_patient_split.csv")
    result.to_csv(args.output_dir / output_name, index=False)
    report = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "label_policy": args.label_policy,
        "output_file": output_name,
        "rows": len(result),
        "patient_counts": result.groupby("internal_split")["Patient"].nunique().to_dict(),
        "row_counts": result["internal_split"].value_counts().to_dict(),
        "positive_counts": result.groupby("internal_split")[label_column].sum().to_dict(),
        "patient_overlap_train_val": len(set(train.loc[train.internal_split.eq("internal_train"), "Patient"]) & set(train.loc[train.internal_split.eq("internal_val"), "Patient"])),
    }
    (args.output_dir / "patient_split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
