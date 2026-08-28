"""Evaluate a trained checkpoint on internal validation or official valid."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from train_baseline import CheXpertDataset, make_model, predict

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--split", choices=["internal_val", "official_valid"], default="official_valid"); parser.add_argument("--image-size", type=int, default=320); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--threshold", type=float); parser.add_argument("--output", type=Path); args = parser.parse_args()
    frame = pd.read_csv(args.manifest); frame = frame[frame["Cardiomegaly"].isin([0, 1]) & frame.internal_split.eq(args.split)].copy(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device); checkpoint = torch.load(args.checkpoint, map_location=device); model.load_state_dict(checkpoint["model"]); loader = DataLoader(CheXpertDataset(frame, args.data_root, args.image_size, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    probs, labels = predict(model, loader, device); threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5)); pred = (probs >= threshold).astype(int); tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    metrics = {"split": args.split, "rows": len(frame), "threshold": threshold, "auroc": float(roc_auc_score(labels, probs)), "auprc": float(average_precision_score(labels, probs)), "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)), "f1": float(f1_score(labels, pred, zero_division=0)), "brier": float(brier_score_loss(labels, probs)), "positive_rate": float(labels.mean())}; print(json.dumps(metrics, indent=2))
    if args.output: args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
    