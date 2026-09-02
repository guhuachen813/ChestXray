"""Train DenseNet-121 or ResNet-50 on CheXpert labels."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resolve_image(row, data_root):
    relative = str(row["Path"]).replace("\\", "/"); prefix = "CheXpert-v1.0-small/"
    if relative.startswith(prefix): relative = relative[len(prefix):]
    for path in (Path(str(row.get("image_path", ""))), data_root / relative):
        if path.is_file(): return path
    raise FileNotFoundError(f"Image not found: {row.get('Path')} (root={data_root})")

class CheXpertDataset(Dataset):
    def __init__(self, frame, data_root, image_size, train, num_classes=1):
        self.frame = frame.reset_index(drop=True); self.data_root = data_root; self.num_classes = num_classes
        ops = [transforms.Resize((image_size, image_size))]
        if train: ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]; self.transform = transforms.Compose(ops)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]; image = Image.open(resolve_image(row, self.data_root)).convert("RGB"); value = int(row["Cardiomegaly"]) if self.num_classes == 3 else float(row["Cardiomegaly"])
        return self.transform(image), torch.tensor(value, dtype=torch.long if self.num_classes == 3 else torch.float32)

def make_model(num_classes=1, arch="densenet121"):
    if arch == "resnet50": model = models.resnet50(weights=None); model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "densenet121": model = models.densenet121(weights=None); model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else: raise ValueError(f"Unsupported architecture: {arch}")
    return model

@torch.no_grad()
def predict_outputs(model, loader, device):
    model.eval(); logits, labels = [], []
    for images, target in loader: logits.append(model(images.to(device)).cpu()); labels.append(target)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()

def probabilities(logits, num_classes):
    x = torch.from_numpy(logits); return torch.softmax(x, dim=1).numpy() if num_classes == 3 else torch.sigmoid(x).reshape(-1, 1).numpy()

def choose_threshold(y_true, probs):
    best = (0.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = probs >= threshold; tp = np.sum(pred & (y_true == 1)); fp = np.sum(pred & (y_true == 0)); fn = np.sum(~pred & (y_true == 1)); f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        if f1 > best[0] or (f1 == best[0] and threshold > best[1]): best = (float(f1), float(threshold))
    return best[1]

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, default=Path("outputs")); parser.add_argument("--image-size", type=int, default=320); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--num-classes", type=int, choices=[1, 3], default=1); parser.add_argument("--arch", choices=["densenet121", "resnet50"], default="densenet121"); parser.add_argument("--split-column", default="internal_split"); parser.add_argument("--train-split", default="internal_train"); parser.add_argument("--val-split", default="internal_val"); args = parser.parse_args()
    seed_everything(args.seed); frame = pd.read_csv(args.manifest); frame = frame[frame["Cardiomegaly"].isin([0, 1, 2] if args.num_classes == 3 else [0, 1])]; train_frame = frame[frame[args.split_column].eq(args.train_split)]; val_frame = frame[frame[args.split_column].eq(args.val_split)]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(CheXpertDataset(train_frame, args.data_root, args.image_size, True, args.num_classes), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda"); val_loader = DataLoader(CheXpertDataset(val_frame, args.data_root, args.image_size, False, args.num_classes), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = make_model(args.num_classes, args.arch).to(device)
    if args.num_classes == 3:
        counts = train_frame["Cardiomegaly"].value_counts().reindex([0, 1, 2], fill_value=0).to_numpy(float); criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / np.maximum(counts * 3, 1), dtype=torch.float32, device=device))
    else:
        positives = float(train_frame["Cardiomegaly"].sum()); criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(len(train_frame) - positives, 1) / max(positives, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); args.output_dir.mkdir(parents=True, exist_ok=True); best_loss = float("inf"); history = []; started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for images, target in train_loader:
            optimizer.zero_grad(set_to_none=True); logits = model(images.to(device)); loss = criterion(logits if args.num_classes == 3 else logits.flatten(), target.to(device)); loss.backward(); optimizer.step(); running += loss.item() * len(target)
        val_logits, val_labels = predict_outputs(model, val_loader, device)
        # Report validation loss with the same criterion used for optimization.
        val_logits_tensor = torch.from_numpy(val_logits).to(device)
        if args.num_classes == 3:
            val_labels_tensor = torch.from_numpy(val_labels).long().to(device)
            val_loss = float(criterion(val_logits_tensor, val_labels_tensor).item())
        else:
            val_labels_tensor = torch.from_numpy(val_labels).float().to(device)
            val_loss = float(criterion(val_logits_tensor.flatten(), val_labels_tensor).item())
        val_probs = probabilities(val_logits, args.num_classes); p = val_probs[:, 1] if args.num_classes == 3 else val_probs[:, 0]; known = val_labels != 2 if args.num_classes == 3 else np.ones(len(val_labels), bool); yy, pp = val_labels[known], p[known]; record = {"epoch": epoch, "train_loss": running / len(train_frame), "val_loss": val_loss}; history.append(record); print(json.dumps(record))
        if val_loss < best_loss:
            best_loss = val_loss; state = {"model": model.state_dict(), "epoch": epoch, "seed": args.seed, "num_classes": args.num_classes, "arch": args.arch, "threshold": choose_threshold(yy, pp)}; torch.save(state, args.output_dir / "best.pt"); torch.save(state, args.output_dir / f"{args.arch}_cardiomegaly_best.pt")
    (args.output_dir / "train_metadata.json").write_text(json.dumps({"device": str(device), "arch": args.arch, "num_classes": args.num_classes, "epochs": args.epochs, "seed": args.seed, "train_rows": len(train_frame), "val_rows": len(val_frame), "elapsed_seconds": time.time() - started, "history": history}, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
