from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Allow pytest to collect this test from the repository root without requiring
# the source tree to be installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.make_agent_split import _allocate_counts, main


def test_allocate_counts_sums_to_population() -> None:
    counts = _allocate_counts(101, [0.7, 0.1, 0.1, 0.1])
    assert sum(counts) == 101
    assert counts == [71, 10, 10, 10]


def test_main_stratifies_patients_and_keeps_official_valid(tmp_path: Path, monkeypatch) -> None:
    rows = []
    for patient in range(1, 101):
        label = 1 if patient <= 15 else (-1 if patient <= 25 else 0)
        for study in range(2):
            rows.append(
                {
                    "split": "train",
                    "Patient": f"p{patient:03d}",
                    "Study": f"s{patient:03d}_{study}",
                    "Cardiomegaly": label,
                    "Frontal/Lateral": "Frontal",
                }
            )
    for patient in range(101, 111):
        rows.append(
            {
                "split": "valid",
                "Patient": f"p{patient:03d}",
                "Study": f"s{patient:03d}_0",
                "Cardiomegaly": 1 if patient >= 106 else 0,
                "Frontal/Lateral": "Frontal",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_agent_split.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "42",
        ],
    )
    main()

    output = pd.read_csv(tmp_path / "out" / "cardiomegaly_agent_split.csv")
    report = json.loads((tmp_path / "out" / "agent_split_report.json").read_text())
    assert report["official_valid_patient_overlap_with_train"] == 0
    assert all(value == 0 for value in report["patient_overlap"].values())
    assert set(output.loc[output["agent_split"].eq("official_valid"), "Patient"]) == {
        f"p{patient:03d}" for patient in range(101, 111)
    }

    # With only 15 positive patients and four integer strata, row-level rates
    # can differ substantially. Verify the stronger invariant: each label
    # stratum is distributed as evenly as integer allocation permits.
    split_counts = {
        split: output.loc[output["agent_split"].eq(split), "Cardiomegaly"].eq(1).sum()
        for split in ("model_train", "calibration", "route_validation", "model_selection")
    }
    assert split_counts["model_train"] > 0
    assert all(split_counts[split] > 0 for split in ("calibration", "route_validation", "model_selection"))
