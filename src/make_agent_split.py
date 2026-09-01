"""Create patient-level model/calibration/route splits for Agent experiments."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--seed", type=int, default=42); p.add_argument("--calibration-fraction", type=float, default=.1); p.add_argument("--route-fraction", type=float, default=.1); p.add_argument("--model-selection-fraction", type=float, default=.1); args = p.parse_args()
    df = pd.read_csv(args.manifest); df["Cardiomegaly"] = df["Cardiomegaly"].fillna(0).replace(-1, 2).astype(int); lateral_rows = int((df["Frontal/Lateral"] != "Frontal").sum()); train = df[df.split.eq("train") & df["Frontal/Lateral"].eq("Frontal")].copy(); official = df[df.split.eq("valid") & df["Frontal/Lateral"].eq("Frontal")].copy()
    patient_label = train.groupby("Patient")["Cardiomegaly"].agg(lambda x: int(x.value_counts().index[0])); groups = []
    for value, g in patient_label.groupby(patient_label): groups.append(g.sample(frac=1, random_state=args.seed + int(value)))
    patients = pd.concat(groups).index.astype(str).tolist(); n_cal = max(1, round(len(patients)*args.calibration_fraction)); n_route = max(1, round(len(patients)*args.route_fraction)); n_sel = max(1, round(len(patients)*args.model_selection_fraction)); cal = set(patients[:n_cal]); route = set(patients[n_cal:n_cal+n_route]); selection = set(patients[n_cal+n_route:n_cal+n_route+n_sel])
    train["agent_split"] = train.Patient.astype(str).map(lambda x: "calibration" if x in cal else "route_validation" if x in route else "model_selection" if x in selection else "model_train"); official["agent_split"] = "official_valid"; out = pd.concat([train, official], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True); out_name = "cardiomegaly_agent_split.csv"; out.to_csv(args.output_dir/out_name, index=False)
    report = {"seed": args.seed, "fractions": {"calibration": args.calibration_fraction, "route_validation": args.route_fraction, "model_selection": args.model_selection_fraction}, "rows": len(out), "patient_counts": out.groupby("agent_split").Patient.nunique().to_dict(), "row_counts": out.agent_split.value_counts().to_dict(), "lateral_rows_excluded_from_agent": lateral_rows, "patient_overlap": int(len(set(train.loc[train.agent_split.eq("model_train"),"Patient"]) & set(train.loc[train.agent_split.ne("model_train"),"Patient"]))) }
    (args.output_dir/"agent_split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
