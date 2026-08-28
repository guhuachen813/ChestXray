"""Run basic checks on a generated CheXpert manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    required = {"image_path", "Patient", "Study", "split", "exists"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    train = set(frame.loc[frame.split.eq("train"), "Patient"].astype(str))
    valid = set(frame.loc[frame.split.eq("valid"), "Patient"].astype(str))
    report = {
        "rows": len(frame),
        "missing_files": int((~frame.exists.astype(bool)).sum()),
        "duplicate_paths": int(frame.image_path.duplicated().sum()),
        "train_valid_patient_overlap": sorted(train & valid),
        "split_rows": frame.split.value_counts().to_dict(),
    }
    report["status"] = "PASS" if not report["missing_files"] and not report["duplicate_paths"] and not report["train_valid_patient_overlap"] else "CHECK"
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
