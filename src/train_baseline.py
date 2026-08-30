"""Train a single-label DenseNet baseline on the patient-level CheXpert split."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resolve_image(row: pd.Series, data_root: Path) -> Path:
    relative = str(row["Path"]).replace("\\", "/")
    prefix = "CheXpert-v1.0-small/"
    if relative.startswith(prefix):
        relative = relative[len(prefix):]
    for path in (Path(str(row.get("image_path", ""))), data_root / relative):
        if path.is_file(): return path
    raise FileNotFoundError(f"Image not found: {row.get('Path')} (root={data_root})")

class CheXpertDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path, image_size: int, train: bool, num_classes: int = 1):
        self.frame = frame.reset_index(drop=True); self.data_root = data_root; self.num_classes = num_classes
        ops = [transforms.Resize((image_size, image_size))]
        if train: ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(ops)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]; image = Image.open(resolve_image(row, self.data_root)).convert("RGB")
        dtype = torch.long if self.num_classes == 3 else torch.float32
        return self.transform(image), torch.tensor(int(row["Cardiomegaly"]) if self.num_classes == 3 else float(row["Cardiomegaly"]), dtype=dtype)

def make_model(num_classes: int = 1) -> nn.Module:
    model = models.densenet121(weights=None); model.classifier = nn.Linear(model.classifier.in_features, num_classes); return model

@torch.no_grad()
def predict(model, loader, device, num_classes: int = 1):
    model.eval(); probs, labels = [], []
    for images, target in loader:
        logits = model(images.to(device))
        probs.append((torch.softmax(logits, dim=1)[:, 1] if num_classes == 3 else torch.sigmoid(logits.flatten())).cpu().numpy()); labels.append(target.numpy())
    return np.concatenate(probs), np.concatenate(labels)

def choose_threshold(y_true, probs) -> float:
    best = (0.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = probs >= threshold; tp = np.sum(pred & (y_true == 1)); fp = np.sum(pred & (y_true == 0)); fn = np.sum(~pred & (y_true == 1))
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        if f1 > best[0] or (f1 == best[0] and threshold > best[1]): best = (float(f1), float(threshold))
    return best[1]

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, default=Path("outputs")); parser.add_argument("--image-size", type=int, default=320); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--num-classes", type=int, choices=[1, 3], default=1); args = parser.parse_args()
    seed_everything(args.seed); frame = pd.read_csv(args.manifest); frame = frame[frame["Cardiomegaly"].isin([0, 1, 2] if args.num_classes == 3 else [0, 1])]
    train_frame = frame[frame.internal_split.eq("internal_train")]; val_frame = frame[frame.internal_split.eq("internal_val")]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(CheXpertDataset(train_frame, args.data_root, args.image_size, True, args.num_classes), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(CheXpertDataset(val_frame, args.data_root, args.image_size, False, args.num_classes), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = make_model(args.num_classes).to(device)
    if args.num_classes == 3:
        counts = train_frame["Cardiomegaly"].value_counts().reindex([0, 1, 2], fill_value=0).to_numpy(dtype=float); weights = counts.sum() / np.maximum(counts * 3, 1); criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        positives = float(train_frame["Cardiomegaly"].sum()); negatives = float(len(train_frame) - positives); criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / max(positives, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True); best_loss = float("inf"); history = []; started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for images, target in train_loader:
            optimizer.zero_grad(set_to_none=True); logits = model(images.to(device)); loss = criterion(logits if args.num_classes == 3 else logits.flatten(), target.to(device)); loss.backward(); optimizer.step(); running += loss.item() * len(target)
        val_probs, val_labels = predict(model, val_loader, device, args.num_classes)
        if args.num_classes == 3:
            valid = val_labels != 2
            val_loss = float(-(np.log(np.clip(1 - val_probs, 1e-7, 1 - 1e-7)) * (val_labels == 0) + np.log(np.clip(val_probs, 1e-7, 1 - 1e-7)) * (val_labels == 1)).sum() / max(valid.sum(), 1))
        else: val_loss = float(-(val_labels * np.log(np.clip(val_probs, 1e-7, 1 - 1e-7)) + (1 - val_labels) * np.log(np.clip(1 - val_probs, 1e-7, 1 - 1e-7))).mean())
        record = {"epoch": epoch, "train_loss": running / len(train_frame), "val_loss": val_loss}; history.append(record); print(json.dumps(record))
        if val_loss < best_loss:
            best_loss = val_loss; threshold_labels = val_labels[val_labels != 2] if args.num_classes == 3 else val_labels; threshold_probs = val_probs[val_labels != 2] if args.num_classes == 3 else val_probs; torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed, "num_classes": args.num_classes, "threshold": choose_threshold(threshold_labels, threshold_probs)}, args.output_dir / "densenet121_cardiomegaly_best.pt")
    (args.output_dir / "train_metadata.json").write_text(json.dumps({"device": str(device), "epochs": args.epochs, "seed": args.seed, "train_rows": len(train_frame), "val_rows": len(val_frame), "elapsed_seconds": time.time() - started, "history": history}, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
