"""Evaluate binary or U-MultiClass checkpoints with calibrated metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from train_baseline import CheXpertDataset, make_model, predict_outputs, probabilities

def ece(y, p, bins=10):
    out = 0.0; conf = np.maximum(p, 1-p); pred = (p >= .5).astype(int)
    for lo, hi in zip(np.linspace(0,1,bins,endpoint=False), np.linspace(0,1,bins+1)[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.any(): out += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(out)

def multiclass_brier(labels, probs):
    onehot = np.eye(3)[labels.astype(int)]; return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--split", choices=["internal_val", "official_valid", "calibration", "route_validation"], default="official_valid"); parser.add_argument("--split-column", default="internal_split"); parser.add_argument("--image-size", type=int, default=320); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--threshold", type=float); parser.add_argument("--temperature", type=float, default=1.0); parser.add_argument("--output", type=Path); args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu"); num_classes = int(ckpt.get("num_classes", 1)); arch = ckpt.get("arch", "densenet121"); frame = pd.read_csv(args.manifest); frame = frame[frame[args.split_column].eq(args.split)].copy(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(num_classes, arch).to(device); model.load_state_dict(ckpt["model"]); loader = DataLoader(CheXpertDataset(frame, args.data_root, args.image_size, False, num_classes), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers); logits, labels = predict_outputs(model, loader, device); probs_all = probabilities(logits / args.temperature, num_classes)
    if num_classes == 3:
        p = probs_all[:, 1]; known = labels != 2; y = labels[known].astype(int); pp = p[known]; metrics = {"multiclass_brier": multiclass_brier(labels, probs_all), "multiclass_ece": float(np.mean([ece((labels == c).astype(int), probs_all[:, c]) for c in range(3)])), "uncertain_rate": float((labels == 2).mean())}
    else: p = probs_all[:, 0]; y = labels.astype(int); pp = p; metrics = {}
    threshold = float(args.threshold if args.threshold is not None else ckpt.get("threshold", .5)); pred = (pp >= threshold).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel(); metrics.update({"split": args.split, "arch": arch, "num_classes": num_classes, "rows": len(frame), "threshold": threshold, "temperature": args.temperature, "auroc": float(roc_auc_score(y, pp)), "auprc": float(average_precision_score(y, pp)), "sensitivity": float(tp / max(tp+fn,1)), "specificity": float(tn / max(tn+fp,1)), "f1": float(f1_score(y, pred, zero_division=0)), "brier": float(brier_score_loss(y, pp)), "ece": ece(y, pp), "positive_rate": float(y.mean())}); print(json.dumps(metrics, indent=2))
    if args.output: args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
