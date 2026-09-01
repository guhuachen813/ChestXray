"""Build a traceable Cardiomegaly manifest and check patient leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_split(data_root: Path, split: str) -> pd.DataFrame:
    csv_path = data_root / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"Path", "Frontal/Lateral", "Cardiomegaly"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    frame["split"] = split
    # Patient/study identifiers are encoded in CheXpert's path column.
    parts = frame["Path"].astype(str).str.replace("\\", "/").str.split("/")
    frame["Patient"] = parts.str[-3]
    frame["Study"] = parts.str[-2]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="Cardiomegaly")
    parser.add_argument("--include-lateral", action="store_true", help="Keep lateral rows for QC rejection-rate reporting.")
    args = parser.parse_args()

    root = args.data_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ("train", "valid"):
        frame = read_split(root, split)
        if not args.include_lateral:
            frame = frame[frame["Frontal/Lateral"].eq("Frontal")].copy()
        frame["label_status"] = frame[args.label].map({1: "positive", 0: "negative", -1: "uncertain"})
        def resolve_image(value: str) -> Path:
            normalized = str(value).replace("\\", "/")
            prefix = "CheXpert-v1.0-small/"
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
            return (root / normalized).resolve()

        frame["image_path"] = frame["Path"].map(lambda value: str(resolve_image(value)))
        frame = frame[~frame["image_path"].map(lambda value: Path(value).name.startswith("._"))].copy()
        frame["exists"] = frame["image_path"].map(lambda value: Path(value).exists())
        frames.append(frame)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(out / "cardiomegaly_all.csv", index=False)

    train = all_rows[all_rows["split"].eq("train")]
    valid = all_rows[all_rows["split"].eq("valid")]
    train_patients = set(train["Patient"].astype(str))
    valid_patients = set(valid["Patient"].astype(str))
    overlap = sorted(train_patients & valid_patients)

    summary = {
        "data_root": str(root),
        "label": args.label,
        "rows": int(len(all_rows)),
        "frontal_rows": int(len(all_rows)),
        "missing_files": int((~all_rows["exists"]).sum()),
        "split_rows": all_rows["split"].value_counts().to_dict(),
        "patient_counts": {key: int(value) for key, value in all_rows.groupby("split")["Patient"].nunique().items()},
        "label_counts_by_split": {
            split: {str(key): int(value) for key, value in group[args.label].value_counts(dropna=False).items()}
            for split, group in all_rows.groupby("split")
        },
        "view_counts_by_split": {
            split: {str(key): int(value) for key, value in group["Frontal/Lateral"].value_counts(dropna=False).items()}
            for split, group in all_rows.groupby("split")
        },
    }
    leakage = {
        "train_valid_patient_overlap_count": len(overlap),
        "train_valid_patient_overlap_examples": overlap[:20],
        "duplicate_image_paths": int(all_rows["image_path"].duplicated().sum()),
        "duplicate_studies": int(all_rows["Study"].duplicated().sum()),
        "status": "PASS" if not overlap and summary["missing_files"] == 0 else "CHECK",
    }
    (out / "cardiomegaly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "leakage_report.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")

    print(json.dumps({"summary": summary, "leakage": leakage}, indent=2))


if __name__ == "__main__":
    main()
